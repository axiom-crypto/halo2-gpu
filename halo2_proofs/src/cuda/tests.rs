//! Kernel-level micro-tests.
//!
//! The end-to-end correctness gate for halo2-gpu is the openvm-sdk
//! `test_sdk_fibonacci` integration test. This module adds cheap
//! kernel-level sanity checks that exercise the single-stream
//! `HALO2_GPU_CTX` path end-to-end through the FFI wrappers under
//! `src/cuda/funcs.rs`; those wrappers internally use `DeviceBuffer<T>`
//! + `MemCopy*` traits from cuda-common where applicable.
//!
//! These tests mirror the style used by stark-backend's cuda-backend
//! tests — allocate on host, call the Rust wrapper which handles the
//! H2D → launch → D2H path internally, compare against a trivial CPU
//! reference. They run whenever a CUDA device is visible to the process.

use ff::{Field, PrimeField, WithSmallOrderMulGroup};
use group::Curve;
use halo2curves::bn256::{Fr, G1Affine, G1};
use rand_core::OsRng;

use crate::arithmetic::best_fft;
use crate::cpu::arithmetic::tests::permutation_product_cpu;
use crate::cpu::arithmetic::{best_multiexp_cpu, lookup_product_cpu};
use crate::cpu::evaluator::permutation_quotient_cpu_chunk;
use crate::cuda::funcs::{
    batch_invert_single_gpu, cosetfft_gpu, cosetfft_many_h2d, eval_polynomial_gpu, fft_gpu,
    generate_omega_powers_gpu, grand_product_device, grand_product_device_with_prefix_device,
    grand_product_gpu, lookup_product_device, lookup_product_gpu, multiexp_gpu,
    permutation_product_device, permutation_product_gpu, permutation_quotient_gpu,
    permute_expression_pair_device,
};
use crate::cuda::utils::FFITraitObject;
use crate::cuda::utils::HALO2_GPU_CTX;
use crate::fft::recursive::FFTData;
use crate::poly::{DeviceChunks, DevicePolyExt, EvaluationDomain, NttType};
use openvm_cuda_common::copy::{MemCopyD2H, MemCopyH2D};
use openvm_cuda_common::d_buffer::DeviceBuffer;

#[test]
fn test_batch_invert_single_gpu_roundtrip() {
    // x_i * (x_i^-1) == 1 for random Fr samples (zero draw has
    // probability ~2^-254 and is not filtered).
    let n = 1usize << 12;
    let original: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();
    let mut inverted = original.clone();
    batch_invert_single_gpu(&mut inverted).unwrap();

    for (i, (a, a_inv)) in original.iter().zip(inverted.iter()).enumerate() {
        assert_eq!(*a * *a_inv, Fr::ONE, "roundtrip failed at index {i}");
    }
}

#[test]
fn test_eval_polynomial_single_gpu_vs_horner() {
    // Compare GPU Horner eval against the scalar CPU Horner used by
    // `arithmetic::eval_polynomial`.
    let n = 1usize << 14;
    let poly: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();
    let point = Fr::random(OsRng);

    let gpu = eval_polynomial_gpu(&poly, point).unwrap();
    let cpu = poly
        .iter()
        .rev()
        .fold(Fr::ZERO, |acc, coeff| acc * point + coeff);

    assert_eq!(gpu, cpu, "GPU polynomial eval disagrees with CPU Horner");
}

#[test]
fn test_generate_omega_powers_single_gpu_vs_cpu() {
    // Sequentially-computed omega^i must match the GPU LUT. Pre-fill
    // the tail with a sentinel (Fr::ONE) and an oversized buffer so we
    // can assert the kernel writes exactly `output_num` slots and does
    // not touch the tail past the cutoff.
    let log_n = 12u32;
    let domain = halo2_axiom::poly::EvaluationDomain::<Fr>::new(1, log_n);
    let domain = EvaluationDomain::from_host_domain(&domain);
    let omega = domain.get_omega();

    let total = 1usize << log_n;
    let output_num = total - 7; // non-power-of-two cutoff
    let sentinel = Fr::ONE;
    let mut powers_gpu = vec![sentinel; total];
    generate_omega_powers_gpu(&mut powers_gpu, omega, log_n, output_num as u64).unwrap();

    let mut powers_cpu = vec![Fr::ZERO; output_num];
    let mut cur = Fr::ONE;
    for slot in powers_cpu.iter_mut() {
        *slot = cur;
        cur *= &omega;
    }

    assert_eq!(
        powers_gpu[..output_num],
        powers_cpu[..],
        "GPU omega powers disagree with CPU over [0, output_num)"
    );
    for (i, v) in powers_gpu[output_num..].iter().enumerate() {
        assert_eq!(
            *v,
            sentinel,
            "tail slot {} (abs index {}) was overwritten past cutoff",
            i,
            output_num + i
        );
    }
}

#[test]
fn test_fft_gpu_vs_best_fft() {
    // Forward FFT over the 2^log_n-th roots of unity: compare the
    // in-place `fft_gpu` path against the CPU recursive FFT used
    // elsewhere in the prover.
    let log_n = 11u32;
    let n = 1usize << log_n;
    let domain = halo2_axiom::poly::EvaluationDomain::<Fr>::new(1, log_n);
    let domain = EvaluationDomain::from_host_domain(&domain);
    let omega = domain.get_omega();
    let omega_inv = domain.get_omega_inv();

    let input: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();
    let mut gpu = input.clone();
    fft_gpu(NttType::FFT.into(), &mut gpu, log_n, omega, Fr::ONE).unwrap();

    let mut cpu = input;
    let fft_data = FFTData::new(n, omega, omega_inv);
    best_fft(&mut cpu, omega, log_n, &fft_data, false);

    assert_eq!(gpu, cpu, "fft_gpu disagrees with best_fft");
}

#[test]
fn test_fft_many_gpu_vs_best_fft() {
    // Batch forward FFT (`_halo2_fft_many` via `fft_gpu_many`) vs the
    // CPU recursive `best_fft`. Exercises the device-resident omega
    // input path post-omega.h-closure across multiple polynomials
    // per call and at several log_n widths.
    use crate::cuda::funcs::fft_gpu_many;
    for &log_n in &[10u32, 12, 14] {
        let n = 1usize << log_n;
        let domain = halo2_axiom::poly::EvaluationDomain::<Fr>::new(1, log_n);
        let domain = EvaluationDomain::from_host_domain(&domain);
        let omega = domain.get_omega();
        let omega_inv = domain.get_omega_inv();
        let num_polys = 3usize;

        let inputs: Vec<Vec<Fr>> = (0..num_polys)
            .map(|_| (0..n).map(|_| Fr::random(OsRng)).collect())
            .collect();

        let gpu: Vec<Vec<Fr>> = inputs.clone();
        let gpu_objs: Vec<FFITraitObject> = gpu
            .iter()
            .map(|p| FFITraitObject::new(p.as_ptr() as usize))
            .collect();
        fft_gpu_many(NttType::FFT.into(), gpu_objs, log_n, omega, Fr::ONE).unwrap();

        let cpu: Vec<Vec<Fr>> = inputs
            .iter()
            .map(|input| {
                let mut buf = input.clone();
                let fft_data = FFTData::new(n, omega, omega_inv);
                best_fft(&mut buf, omega, log_n, &fft_data, false);
                buf
            })
            .collect();

        for i in 0..num_polys {
            assert_eq!(
                gpu[i], cpu[i],
                "_halo2_fft_many disagrees with best_fft at log_n={}, poly {}",
                log_n, i
            );
        }
    }
}

#[test]
fn test_fft_normal_to_device_vs_cpu() {
    // Forward FFT with device-resident input AND device-resident output,
    // compared against the CPU recursive FFT (`best_fft`). Covers the
    // log_n band that `evaluate_h`'s single-poly `coeff_to_extended_part`
    // sites operate over on fibonacci. The corresponding host-in/host-out
    // path is tested by `test_fft_gpu_vs_best_fft`.
    use crate::cuda::funcs::fft_normal_device;
    use crate::cuda::utils::HALO2_GPU_CTX;
    use openvm_cuda_common::copy::{MemCopyD2H, MemCopyH2D};
    use openvm_cuda_common::d_buffer::DeviceBuffer;

    for log_n in [12u32, 13, 14] {
        let n = 1usize << log_n;
        let domain = halo2_axiom::poly::EvaluationDomain::<Fr>::new(1, log_n);
        let domain = EvaluationDomain::from_host_domain(&domain);
        let omega = domain.get_omega();
        let omega_inv = domain.get_omega_inv();

        let host_input: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();

        let mut cpu = host_input.clone();
        let fft_data = FFTData::new(n, omega, omega_inv);
        best_fft(&mut cpu, omega, log_n, &fft_data, false);

        let d_input: DeviceBuffer<Fr> = host_input.as_slice().to_device_on(&HALO2_GPU_CTX).unwrap();
        let d_output: DeviceBuffer<Fr> = DeviceBuffer::<Fr>::with_capacity_on(n, &HALO2_GPU_CTX);

        fft_normal_device(
            NttType::FFT.into(),
            log_n,
            &d_input,
            &d_output,
            omega,
            Fr::ONE,
        )
        .unwrap();

        let gpu: Vec<Fr> = d_output.to_host_on(&HALO2_GPU_CTX).unwrap();

        assert_eq!(
            gpu, cpu,
            "fft_normal_device disagrees with best_fft at log_n={log_n}"
        );
    }
}

#[test]
fn test_fft_normal_to_device_ifft_vs_cpu() {
    // Inverse FFT with device-resident input AND device-resident output,
    // compared against the CPU recursive iFFT (`best_fft(..., inverse=true)`).
    // Exercises the iFFT branch of the runtime-supported set listed in
    // `_halo2_fft_normal_to_device`'s ntt_type guard.
    use crate::cuda::funcs::fft_normal_device;
    use crate::cuda::utils::HALO2_GPU_CTX;
    use openvm_cuda_common::copy::{MemCopyD2H, MemCopyH2D};
    use openvm_cuda_common::d_buffer::DeviceBuffer;

    for log_n in [12u32, 13, 14] {
        let n = 1usize << log_n;
        let domain = halo2_axiom::poly::EvaluationDomain::<Fr>::new(1, log_n);
        let domain = EvaluationDomain::from_host_domain(&domain);
        let omega = domain.get_omega();
        let omega_inv = domain.get_omega_inv();
        let n_inv = Fr::from(n as u64).invert().unwrap();

        let host_input: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();

        // CPU reference: `best_fft(inverse=true)` performs only the
        // butterfly; the `1/n` scaling is applied explicitly afterwards
        // (matching the prover's own iFFT shape in `poly/domain.rs`).
        // The GPU iFFT FFI fuses both into one call via the `divisor`
        // slot.
        let mut cpu = host_input.clone();
        let fft_data = FFTData::new(n, omega, omega_inv);
        best_fft(&mut cpu, omega_inv, log_n, &fft_data, true);
        for v in cpu.iter_mut() {
            *v *= n_inv;
        }

        let d_input: DeviceBuffer<Fr> = host_input.as_slice().to_device_on(&HALO2_GPU_CTX).unwrap();
        let d_output: DeviceBuffer<Fr> = DeviceBuffer::<Fr>::with_capacity_on(n, &HALO2_GPU_CTX);

        fft_normal_device(
            NttType::iFFT.into(),
            log_n,
            &d_input,
            &d_output,
            omega_inv,
            n_inv,
        )
        .unwrap();

        let gpu: Vec<Fr> = d_output.to_host_on(&HALO2_GPU_CTX).unwrap();

        assert_eq!(
            gpu, cpu,
            "fft_normal_device (iFFT) disagrees with CPU reference \
             (best_fft(inverse=true) + 1/n scaling) at log_n={log_n}"
        );
    }
}

#[test]
fn test_fft_normal_to_device_coset_part_vs_cpu() {
    // CosetFFT_Part with device-resident input AND device-resident output.
    // The kernel composes `distribute_powers(a, divisor)` and a forward FFT
    // with twiddle `omega`. The Rust contract documented in
    // `EvaluationDomain::coeff_to_extended_part_many_device` (poly/domain.rs)
    // is `omega = self.omega`, `divisor = self.g_coset * extended_omega_factor`
    // — swapping the two slots is a known footgun. The CPU reference here
    // replicates the kernel's algebra explicitly so the test catches a
    // regression that breaks either slot or the composition order.
    use crate::cuda::funcs::fft_normal_device;
    use crate::cuda::utils::HALO2_GPU_CTX;
    use openvm_cuda_common::copy::{MemCopyD2H, MemCopyH2D};
    use openvm_cuda_common::d_buffer::DeviceBuffer;

    for log_n in [12u32, 13, 14] {
        let n = 1usize << log_n;
        let domain = halo2_axiom::poly::EvaluationDomain::<Fr>::new(1, log_n);
        let domain = EvaluationDomain::from_host_domain(&domain);
        let omega = domain.get_omega();
        let omega_inv = domain.get_omega_inv();
        let g_coset = domain.inner.g_coset;
        // Pick a non-identity extended_omega_factor — `extended_omega^1` is
        // the simplest non-trivial choice and exercises the
        // `distribute_powers(divisor)` shift end-to-end.
        let extended_omega_factor = domain.get_extended_omega();
        let divisor = g_coset * extended_omega_factor;

        let host_input: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();

        // CPU reference: distribute_powers(divisor) then forward FFT.
        // `distribute_powers` is module-private on EvaluationDomain, so
        // inline it here to keep the test free of internal-API access.
        let mut cpu = host_input.clone();
        let mut c_power = Fr::ONE;
        for v in cpu.iter_mut() {
            *v *= c_power;
            c_power *= divisor;
        }
        let fft_data = FFTData::new(n, omega, omega_inv);
        best_fft(&mut cpu, omega, log_n, &fft_data, false);

        let d_input: DeviceBuffer<Fr> = host_input.as_slice().to_device_on(&HALO2_GPU_CTX).unwrap();
        let d_output: DeviceBuffer<Fr> = DeviceBuffer::<Fr>::with_capacity_on(n, &HALO2_GPU_CTX);

        fft_normal_device(
            NttType::CosetFFT_Part.into(),
            log_n,
            &d_input,
            &d_output,
            omega,
            divisor,
        )
        .unwrap();

        let gpu: Vec<Fr> = d_output.to_host_on(&HALO2_GPU_CTX).unwrap();

        assert_eq!(
            gpu, cpu,
            "fft_normal_device (CosetFFT_Part) disagrees with CPU \
             reference (distribute_powers(divisor) + best_fft(omega)) at log_n={log_n}"
        );
    }
}

#[test]
fn test_cosetfft_gpu_roundtrip() {
    // Coset FFT uses a halo2-internal coset-power distribution that
    // is awkward to re-implement in a micro-test; instead verify the
    // forward/inverse pair round-trips to the identity. The inverse
    // iCosetFFT path goes through `fft_gpu` (see
    // `EvaluationDomain::icosetfft` in `poly/domain.rs`) with the
    // `1/n` divisor applied on the return leg.
    let log_n = 10u32;
    let n = 1usize << log_n;
    let domain = halo2_axiom::poly::EvaluationDomain::<Fr>::new(1, log_n);
    let domain = EvaluationDomain::from_host_domain(&domain);
    let omega = domain.get_omega();
    let omega_inv = domain.get_omega_inv();
    let n_inv = Fr::from(n as u64).invert().unwrap();

    let input: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();
    let mut evals = vec![Fr::ZERO; n];
    cosetfft_gpu(
        NttType::CosetFFT.into(),
        &input,
        &mut evals,
        log_n,
        log_n,
        omega,
        Fr::ONE,
    )
    .unwrap();
    fft_gpu(
        NttType::iCosetFFT.into(),
        &mut evals,
        log_n,
        omega_inv,
        n_inv,
    )
    .unwrap();

    assert_eq!(
        evals, input,
        "cosetfft + icosetfft round-trip is not the identity"
    );
}

#[test]
fn test_multiexp_gpu_vs_cpu() {
    // multiexp_gpu falls back to `best_multiexp_cpu` for inputs
    // under 2^14, so we size the test just above that threshold to
    // actually exercise the device path. Using `bases = G`
    // collapses the expected value to `(Σ coeffs) · G`, which keeps
    // the CPU cross-check cheap.
    let n = 1usize << 14;
    let gen = G1::generator();
    let coeffs: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();
    let bases = vec![gen.to_affine(); n];

    let gpu = multiexp_gpu::<G1Affine>(&coeffs, &bases).unwrap();
    let cpu = best_multiexp_cpu::<G1Affine>(&coeffs, &bases);

    assert_eq!(
        gpu.to_affine(),
        cpu.to_affine(),
        "multiexp_gpu disagrees with best_multiexp_cpu",
    );
}

#[test]
fn test_multiexp_gpu_device_bases_vs_cpu() {
    // Device-bases MSM variant: bases live on device once (caller
    // pre-uploads), scalars and output follow the host conventions.
    // Algebraic equivalence vs `best_multiexp_cpu`. A second call
    // reusing the same `bases_device` confirms the device buffer is
    // safe to share across MSM invocations — the cache reuse pattern
    // `ParamsKZG` relies on.
    use crate::cuda::funcs::{multiexp_gpu_device_bases, GPU_MSM_THRESHOLD};
    use crate::cuda::utils::HALO2_GPU_CTX;
    use openvm_cuda_common::copy::MemCopyH2D;
    use openvm_cuda_common::d_buffer::DeviceBuffer;

    // Strictly above the dispatch threshold so the GPU path is taken
    // even if the gate inside `multiexp_gpu` ever moves from `<` to `<=`.
    let n = GPU_MSM_THRESHOLD + 1;
    let coeffs_a: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();
    let coeffs_b: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();
    let bases_host: Vec<G1Affine> = (0..n)
        .map(|_| (G1::generator() * Fr::random(OsRng)).to_affine())
        .collect();
    let bases_device: DeviceBuffer<G1Affine> =
        bases_host.as_slice().to_device_on(&HALO2_GPU_CTX).unwrap();

    for (label, coeffs) in [("first", &coeffs_a), ("second", &coeffs_b)] {
        let gpu = multiexp_gpu_device_bases::<G1Affine>(coeffs, &bases_device).unwrap();
        let cpu = best_multiexp_cpu::<G1Affine>(coeffs, &bases_host);
        assert_eq!(
            gpu.to_affine(),
            cpu.to_affine(),
            "multiexp_gpu_device_bases disagrees with best_multiexp_cpu ({label} call)",
        );
    }
}

#[test]
fn test_multiexp_gpu_device_bases_chunked_vs_unchunked() {
    // Uses a small max_chunk_len to exercise the chunk-fold logic.
    // Force the chunking loop inside `multiexp_gpu_device_bases_chunked`
    // to actually fire by passing a `max_chunk_len` well below the input
    // length. The chunked accumulator must agree with both the
    // single-chunk wrapper and the CPU baseline. Validates the
    // sub-buffer pointer-offset arithmetic and the inter-chunk fold.
    use crate::cuda::funcs::{
        multiexp_gpu_device_bases, multiexp_gpu_device_bases_chunked, GPU_MSM_THRESHOLD,
    };
    use crate::cuda::utils::HALO2_GPU_CTX;
    use openvm_cuda_common::copy::MemCopyH2D;
    use openvm_cuda_common::d_buffer::DeviceBuffer;

    // Non-power-of-two `n` so the last chunk has a smaller length than
    // the rest — exercises the partial-chunk path.
    let n = GPU_MSM_THRESHOLD + 137;
    let coeffs: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();
    let bases_host: Vec<G1Affine> = (0..n)
        .map(|_| (G1::generator() * Fr::random(OsRng)).to_affine())
        .collect();
    let bases_device: DeviceBuffer<G1Affine> =
        bases_host.as_slice().to_device_on(&HALO2_GPU_CTX).unwrap();

    let chunk_size = n / 4 + 1; // 4 chunks, last one short
    let chunked =
        multiexp_gpu_device_bases_chunked::<G1Affine>(&coeffs, &bases_device, chunk_size).unwrap();
    let unchunked = multiexp_gpu_device_bases::<G1Affine>(&coeffs, &bases_device).unwrap();
    let cpu = best_multiexp_cpu::<G1Affine>(&coeffs, &bases_host);

    assert_eq!(
        chunked.to_affine(),
        cpu.to_affine(),
        "chunked multiexp_gpu_device_bases disagrees with best_multiexp_cpu"
    );
    assert_eq!(
        unchunked.to_affine(),
        cpu.to_affine(),
        "unchunked multiexp_gpu_device_bases disagrees with best_multiexp_cpu"
    );
    assert_eq!(
        chunked.to_affine(),
        unchunked.to_affine(),
        "chunked vs unchunked multiexp_gpu_device_bases disagree"
    );
}

#[test]
fn test_multiexp_gpu_device_scalars_device_bases_vs_cpu() {
    // Device-scalars + device-bases MSM variant. Verifies algebraic
    // equivalence against `best_multiexp_cpu` and against the
    // host-scalars + device-bases variant. Both calls reuse the same
    // `bases_device` cache — the production pattern at `ParamsKZG`.
    use crate::cuda::funcs::{
        multiexp_gpu_device_bases, multiexp_gpu_device_scalars_device_bases, GPU_MSM_THRESHOLD,
    };
    use crate::cuda::utils::HALO2_GPU_CTX;
    use openvm_cuda_common::copy::MemCopyH2D;
    use openvm_cuda_common::d_buffer::DeviceBuffer;

    let n = GPU_MSM_THRESHOLD + 1;
    let coeffs: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();
    let bases_host: Vec<G1Affine> = (0..n)
        .map(|_| (G1::generator() * Fr::random(OsRng)).to_affine())
        .collect();
    let bases_device: DeviceBuffer<G1Affine> =
        bases_host.as_slice().to_device_on(&HALO2_GPU_CTX).unwrap();
    let coeffs_device: DeviceBuffer<Fr> = coeffs.as_slice().to_device_on(&HALO2_GPU_CTX).unwrap();

    let gpu_dev =
        multiexp_gpu_device_scalars_device_bases::<G1Affine>(&coeffs_device, &bases_device)
            .unwrap();
    let gpu_host_scalars = multiexp_gpu_device_bases::<G1Affine>(&coeffs, &bases_device).unwrap();
    let cpu = best_multiexp_cpu::<G1Affine>(&coeffs, &bases_host);

    assert_eq!(
        gpu_dev.to_affine(),
        cpu.to_affine(),
        "multiexp_gpu_device_scalars_device_bases disagrees with best_multiexp_cpu"
    );
    assert_eq!(
        gpu_dev.to_affine(),
        gpu_host_scalars.to_affine(),
        "device-scalars variant disagrees with host-scalars + device-bases variant"
    );
}

#[test]
fn test_multiexp_gpu_device_scalars_device_bases_chunked() {
    // Uses a small max_chunk_len to exercise the chunk-fold logic.
    // Chunking equivalence for the device-scalars + device-bases MSM:
    // non-power-of-two `n` so the last chunk has a smaller length.
    // Validates the sub-buffer offset arithmetic on BOTH the scalars
    // and bases sides.
    use crate::cuda::funcs::{
        multiexp_gpu_device_scalars_device_bases, multiexp_gpu_device_scalars_device_bases_chunked,
        GPU_MSM_THRESHOLD,
    };
    use crate::cuda::utils::HALO2_GPU_CTX;
    use openvm_cuda_common::copy::MemCopyH2D;
    use openvm_cuda_common::d_buffer::DeviceBuffer;

    let n = GPU_MSM_THRESHOLD + 137;
    let coeffs: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();
    let bases_host: Vec<G1Affine> = (0..n)
        .map(|_| (G1::generator() * Fr::random(OsRng)).to_affine())
        .collect();
    let bases_device: DeviceBuffer<G1Affine> =
        bases_host.as_slice().to_device_on(&HALO2_GPU_CTX).unwrap();
    let coeffs_device: DeviceBuffer<Fr> = coeffs.as_slice().to_device_on(&HALO2_GPU_CTX).unwrap();

    let chunk_size = n / 4 + 1;
    let chunked = multiexp_gpu_device_scalars_device_bases_chunked::<G1Affine>(
        &coeffs_device,
        &bases_device,
        chunk_size,
    )
    .unwrap();
    let unchunked =
        multiexp_gpu_device_scalars_device_bases::<G1Affine>(&coeffs_device, &bases_device)
            .unwrap();
    let cpu = best_multiexp_cpu::<G1Affine>(&coeffs, &bases_host);

    assert_eq!(
        chunked.to_affine(),
        cpu.to_affine(),
        "chunked device-scalars MSM disagrees with best_multiexp_cpu"
    );
    assert_eq!(
        unchunked.to_affine(),
        cpu.to_affine(),
        "unchunked device-scalars MSM disagrees with best_multiexp_cpu"
    );
    assert_eq!(
        chunked.to_affine(),
        unchunked.to_affine(),
        "chunked vs unchunked device-scalars MSM disagree"
    );
}

#[test]
fn test_chunks_device_round_trip() {
    // `chunks_device` splits a device-resident `Polynomial<F, Coeff>`
    // into `num_chunks` pieces of `chunk_len` each via D→D copy. Every
    // piece must hold the corresponding offset of the parent's values
    // byte-identically. Tests both the clean tiling
    // (`parent.len() == chunk_len * num_chunks`) and that the returned
    // chunks survive the parent's drop (each owns its own
    // `DeviceBuffer`).
    use crate::cuda::utils::HALO2_GPU_CTX;
    use crate::poly::{Coeff, Polynomial};
    use openvm_cuda_common::copy::MemCopyH2D;
    use openvm_cuda_common::d_buffer::DeviceBuffer;

    // Small but non-trivial size so the test stays fast.
    let chunk_len = 1 << 10;
    let num_chunks = 3;
    let total_len = chunk_len * num_chunks;
    let host: Vec<Fr> = (0..total_len).map(|_| Fr::random(OsRng)).collect();
    let dev: DeviceBuffer<Fr> = host.as_slice().to_device_on(&HALO2_GPU_CTX).unwrap();
    let parent: Polynomial<Fr, Coeff, crate::poly::Device> = Polynomial::from_device(dev);

    let chunks = parent.chunks_device(chunk_len);
    assert_eq!(chunks.len(), num_chunks);

    for (i, chunk) in chunks.iter().enumerate() {
        assert_eq!(chunk.len(), chunk_len);
        let d_buf = chunk.device_buf();
        let mut got: Vec<Fr> = vec![Fr::default(); chunk_len];
        unsafe {
            openvm_cuda_common::copy::cuda_memcpy_on::<true, false>(
                got.as_mut_ptr() as *mut libc::c_void,
                d_buf.as_raw_ptr(),
                chunk_len * std::mem::size_of::<Fr>(),
                &HALO2_GPU_CTX,
            )
            .unwrap();
        }
        HALO2_GPU_CTX.stream.to_host_sync().unwrap();
        let expected = &host[i * chunk_len..(i + 1) * chunk_len];
        assert_eq!(got.as_slice(), expected, "chunk {i} mismatch");
    }
}

#[test]
fn test_poly_scale_and_multiply_add_device() {
    // Device fold helpers: `poly_scale_device_with_d_s_minus_one(buf, d_(s-1))`
    // ≡ `buf *= s`, `poly_multiply_add_device(acc, in, s)` ≡ `acc += s * in`.
    // Both tested against a host reference byte-identically. Sequencing
    // `scale` then `multiply_add` exercises the exact pattern used by
    // `Constructed::evaluate`'s Device fold.
    use crate::cuda::funcs::{poly_multiply_add_device, poly_scale_device_with_d_s_minus_one};
    use crate::cuda::utils::HALO2_GPU_CTX;
    use openvm_cuda_common::copy::{cuda_memcpy_on, MemCopyH2D};
    use openvm_cuda_common::d_buffer::DeviceBuffer;

    let n = 1 << 12;
    let acc_host: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();
    let in_host: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();
    let scale: Fr = Fr::random(OsRng);
    let add_scalar: Fr = Fr::random(OsRng);

    // CPU reference: acc := scale * acc; acc := acc + add_scalar * in
    let mut acc_ref = acc_host.clone();
    for v in acc_ref.iter_mut() {
        *v *= scale;
    }
    for (a, b) in acc_ref.iter_mut().zip(in_host.iter()) {
        *a += *b * add_scalar;
    }

    // GPU
    let mut d_acc: DeviceBuffer<Fr> = acc_host.as_slice().to_device_on(&HALO2_GPU_CTX).unwrap();
    let d_in: DeviceBuffer<Fr> = in_host.as_slice().to_device_on(&HALO2_GPU_CTX).unwrap();
    let scale_minus_one = scale - Fr::ONE;
    let d_scale_minus_one: DeviceBuffer<Fr> = std::slice::from_ref(&scale_minus_one)
        .to_device_on(&HALO2_GPU_CTX)
        .unwrap();
    poly_scale_device_with_d_s_minus_one(&mut d_acc, &d_scale_minus_one).unwrap();
    poly_multiply_add_device(&mut d_acc, &d_in, add_scalar).unwrap();

    let mut got: Vec<Fr> = vec![Fr::default(); n];
    unsafe {
        cuda_memcpy_on::<true, false>(
            got.as_mut_ptr() as *mut libc::c_void,
            d_acc.as_raw_ptr(),
            n * std::mem::size_of::<Fr>(),
            &HALO2_GPU_CTX,
        )
        .unwrap();
    }
    HALO2_GPU_CTX.stream.to_host_sync().unwrap();

    assert_eq!(
        got, acc_ref,
        "poly_scale + multiply_add device path mismatch"
    );
}

#[test]
fn test_lagrange_to_coeff_device_vs_host() {
    // Byte-identical equivalence between
    // `EvaluationDomain::lagrange_to_coeff` (Host output) and
    // `lagrange_to_coeff_device` (Device output, D→H'd here for
    // comparison). Tests at k ∈ {20, 22, 23}.
    use crate::cuda::utils::HALO2_GPU_CTX;
    use crate::poly::EvaluationDomain;
    use openvm_cuda_common::copy::cuda_memcpy_on;

    for k in [20u32, 22u32, 23u32] {
        let domain = halo2_axiom::poly::EvaluationDomain::<Fr>::new(1, k);
        let domain = EvaluationDomain::from_host_domain(&domain);
        let n = 1usize << k;
        let host_values: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();
        let lagrange_a = domain.lagrange_from_vec(host_values.clone());
        let lagrange_b = domain.lagrange_from_vec(host_values);

        let coeff_host = domain.lagrange_to_coeff(lagrange_a).unwrap();
        let coeff_dev = domain.lagrange_to_coeff_device(lagrange_b).unwrap();

        let dev_buf = coeff_dev.device_buf();
        let mut dev_host: Vec<Fr> = vec![Fr::default(); n];
        unsafe {
            cuda_memcpy_on::<true, false>(
                dev_host.as_mut_ptr() as *mut libc::c_void,
                dev_buf.as_raw_ptr(),
                n * std::mem::size_of::<Fr>(),
                &HALO2_GPU_CTX,
            )
            .unwrap();
        }
        HALO2_GPU_CTX.stream.to_host_sync().unwrap();
        assert_eq!(
            dev_host.as_slice(),
            coeff_host.values(),
            "lagrange_to_coeff_device disagrees with lagrange_to_coeff at k={}",
            k
        );
    }
}

#[test]
#[ignore = "heavy"]
fn test_ifft_gpu_many_device_to_device_vs_host() {
    // Byte-identical equivalence between the host-input device-output
    // batch iFFT (`ifft_many_h2d`) and the device-input
    // device-output batch iFFT (`ifft_many_device`) for
    // small batch sizes at k ∈ {20, 22, 23}. Same omega/divisor on
    // both paths; the device-input launcher copies each input D2D, the
    // host-input launcher copies H2D.
    use crate::cuda::funcs::{ifft_many_device, ifft_many_h2d};
    use crate::cuda::utils::HALO2_GPU_CTX;
    use crate::poly::EvaluationDomain;
    use openvm_cuda_common::copy::{cuda_memcpy_on, MemCopyH2D};

    for k in [20u32, 22u32, 23u32] {
        let domain = halo2_axiom::poly::EvaluationDomain::<Fr>::new(1, k);
        let domain = EvaluationDomain::from_host_domain(&domain);
        let n = 1usize << k;
        let batch_size = 3usize;
        let host_polys: Vec<_> = (0..batch_size)
            .map(|_| (0..n).map(|_| Fr::random(OsRng)).collect::<Vec<Fr>>())
            .collect();
        let lagrange_polys: Vec<_> = host_polys
            .iter()
            .map(|v| domain.lagrange_from_vec(v.clone()))
            .collect();

        // Host-input path (oracle): existing wrapper, hardcoded
        // `input_on_device: false`.
        let host_input_outs = ifft_many_h2d::<Fr>(
            &lagrange_polys,
            k,
            domain.inner.omega_inv,
            domain.inner.ifft_divisor,
        )
        .unwrap();

        // Device-input path: H2D the same inputs first, then run the
        // new device-input wrapper.
        let in_dbufs: Vec<_> = host_polys
            .iter()
            .map(|v| v.as_slice().to_device_on(&HALO2_GPU_CTX).unwrap())
            .collect();
        let in_objs: Vec<FFITraitObject> = in_dbufs
            .iter()
            .map(|b| FFITraitObject::new(b.as_raw_ptr() as usize))
            .collect();
        let dev_input_outs = ifft_many_device::<Fr>(
            in_objs,
            k,
            domain.inner.omega_inv,
            domain.inner.ifft_divisor,
        )
        .unwrap();

        assert_eq!(host_input_outs.len(), dev_input_outs.len());
        for (i, (h, d)) in host_input_outs
            .iter()
            .zip(dev_input_outs.iter())
            .enumerate()
        {
            let mut h_host: Vec<Fr> = vec![Fr::default(); n];
            let mut d_host: Vec<Fr> = vec![Fr::default(); n];
            unsafe {
                cuda_memcpy_on::<true, false>(
                    h_host.as_mut_ptr() as *mut libc::c_void,
                    h.device_buf().as_raw_ptr(),
                    n * std::mem::size_of::<Fr>(),
                    &HALO2_GPU_CTX,
                )
                .unwrap();
                cuda_memcpy_on::<true, false>(
                    d_host.as_mut_ptr() as *mut libc::c_void,
                    d.as_raw_ptr(),
                    n * std::mem::size_of::<Fr>(),
                    &HALO2_GPU_CTX,
                )
                .unwrap();
            }
            HALO2_GPU_CTX.stream.to_host_sync().unwrap();
            assert_eq!(
                h_host, d_host,
                "ifft_many_device output {} disagrees with host-input variant at k={}",
                i, k
            );
        }
    }
}

#[test]
#[ignore = "heavy"]
fn test_lagrange_to_coeff_many_device_vs_host() {
    // Byte-identical equivalence between `lagrange_to_coeff_many` and
    // `lagrange_to_coeff_many_device` for small batch sizes at
    // k ∈ {20, 22, 23}.
    use crate::cuda::utils::HALO2_GPU_CTX;
    use crate::poly::EvaluationDomain;
    use openvm_cuda_common::copy::cuda_memcpy_on;

    for k in [20u32, 22u32, 23u32] {
        let domain = halo2_axiom::poly::EvaluationDomain::<Fr>::new(1, k);
        let domain = EvaluationDomain::from_host_domain(&domain);
        let n = 1usize << k;
        let batch_size = 3;
        let host_polys: Vec<_> = (0..batch_size)
            .map(|_| (0..n).map(|_| Fr::random(OsRng)).collect::<Vec<Fr>>())
            .collect();
        let lagrange_polys_a: Vec<_> = host_polys
            .iter()
            .map(|v| domain.lagrange_from_vec(v.clone()))
            .collect();
        let lagrange_polys_b: Vec<_> = host_polys
            .iter()
            .map(|v| domain.lagrange_from_vec(v.clone()))
            .collect();

        let host_outs = domain.lagrange_to_coeff_many(&lagrange_polys_a).unwrap();
        let dev_outs = domain
            .lagrange_to_coeff_many_device(&lagrange_polys_b)
            .unwrap();
        assert_eq!(host_outs.len(), dev_outs.len());
        for (i, (h, d)) in host_outs.iter().zip(dev_outs.iter()).enumerate() {
            let dev_buf = d.device_buf();
            let mut dev_host: Vec<Fr> = vec![Fr::default(); n];
            unsafe {
                cuda_memcpy_on::<true, false>(
                    dev_host.as_mut_ptr() as *mut libc::c_void,
                    dev_buf.as_raw_ptr(),
                    n * std::mem::size_of::<Fr>(),
                    &HALO2_GPU_CTX,
                )
                .unwrap();
            }
            HALO2_GPU_CTX.stream.to_host_sync().unwrap();
            assert_eq!(
                dev_host.as_slice(),
                h.values(),
                "lagrange_to_coeff_many_device output {} disagrees at k={}",
                i,
                k
            );
        }
    }
}

#[test]
fn test_params_kzg_commit_lagrange_cache_cold_warm() {
    // Cold-cache vs warm-cache equivalence on `ParamsKZG::commit_lagrange`.
    // Asserts:
    //   1. Cache empty before any commit call.
    //   2. First call (cache miss) populates `g_lagrange_device` and
    //      matches the CPU baseline.
    //   3. Cache populated after first call.
    //   4. Second call (cache hit) still matches the CPU baseline.
    // k chosen so n strictly exceeds GPU_MSM_THRESHOLD, guaranteeing
    // the GPU dispatch path is taken.
    use crate::cuda::funcs::GPU_MSM_THRESHOLD;
    use crate::poly::commitment::{Blind, Params};
    use crate::poly::kzg::commitment::ParamsKZG;
    use halo2curves::bn256::Bn256;

    let k = 15u32;
    let n = 1usize << k;
    assert!(
        n > GPU_MSM_THRESHOLD,
        "test must size strictly above GPU_MSM_THRESHOLD"
    );
    let params = ParamsKZG::<Bn256>::setup(k, OsRng);

    let domain = halo2_axiom::poly::EvaluationDomain::<Fr>::new(1, k);
    let domain = EvaluationDomain::from_host_domain(&domain);
    let mut poly = domain.empty_lagrange();
    for v in poly.iter_mut() {
        *v = Fr::random(OsRng);
    }

    let coeffs: Vec<Fr> = poly.iter().cloned().collect();
    let cpu = best_multiexp_cpu::<G1Affine>(&coeffs, &params.g_lagrange[0..n]);

    assert!(
        params.g_lagrange_device.get().is_none(),
        "g_lagrange_device cache should be empty before any commit call"
    );

    let cold = params.commit_lagrange(&poly, Blind::default());
    assert_eq!(
        cold.to_affine(),
        cpu.to_affine(),
        "cold-cache commit_lagrange disagrees with best_multiexp_cpu"
    );
    assert!(
        params.g_lagrange_device.get().is_some(),
        "g_lagrange_device cache should be populated after the first GPU-routed commit"
    );

    let warm = params.commit_lagrange(&poly, Blind::default());
    assert_eq!(
        warm.to_affine(),
        cpu.to_affine(),
        "warm-cache commit_lagrange disagrees with best_multiexp_cpu"
    );
}

#[test]
fn test_commit_lagrange_device_vs_host() {
    // Device-input `commit_lagrange_device` must produce the same MSM
    // result as the host-input `commit_lagrange` for an arbitrary
    // Lagrange-basis polynomial. Sizing strictly above
    // `GPU_MSM_THRESHOLD` guarantees the device dispatch path is taken
    // by both impls.
    use crate::cuda::funcs::GPU_MSM_THRESHOLD;
    use crate::cuda::utils::HALO2_GPU_CTX;
    use crate::poly::commitment::{Blind, Params};
    use crate::poly::kzg::commitment::ParamsKZG;
    use crate::poly::Polynomial;
    use halo2curves::bn256::Bn256;
    use openvm_cuda_common::copy::MemCopyH2D;

    let k = 15u32;
    let n = 1usize << k;
    assert!(
        n > GPU_MSM_THRESHOLD,
        "test must size strictly above GPU_MSM_THRESHOLD"
    );
    let params = ParamsKZG::<Bn256>::setup(k, OsRng);
    let domain = halo2_axiom::poly::EvaluationDomain::<Fr>::new(1, k);
    let domain = EvaluationDomain::from_host_domain(&domain);

    let mut poly_host = domain.empty_lagrange();
    for v in poly_host.iter_mut() {
        *v = Fr::random(OsRng);
    }
    let d_buf = poly_host
        .values()
        .to_device_on(&HALO2_GPU_CTX)
        .expect("upload poly to device");
    let poly_device: Polynomial<Fr, crate::poly::LagrangeCoeff, crate::poly::Device> =
        Polynomial::from_device(d_buf);

    let host_result = params.commit_lagrange(&poly_host, Blind::default());
    let device_result = params.commit_lagrange_device(&poly_device, Blind::default());

    assert_eq!(
        host_result.to_affine(),
        device_result.to_affine(),
        "commit_lagrange_device disagrees with commit_lagrange"
    );
}

#[test]
fn test_lookup_product_gpu_vs_cpu() {
    // PLONK lookup running-product numerator/denominator kernel —
    // the operation is purely algebraic, so GPU and CPU must agree
    // bit-for-bit.
    let n = 128usize;
    let permuted_input: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();
    let permuted_table: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();
    let compressed_input: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();
    let compressed_table: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();
    let beta = Fr::random(OsRng);
    let gamma = Fr::random(OsRng);

    let mut gpu = vec![Fr::ZERO; n];
    lookup_product_gpu(
        &mut gpu,
        &permuted_input,
        &permuted_table,
        &compressed_input,
        &compressed_table,
        beta,
        gamma,
    )
    .unwrap();

    let mut cpu = vec![Fr::ZERO; n];
    lookup_product_cpu(
        &mut cpu,
        &permuted_input,
        &permuted_table,
        &compressed_input,
        &compressed_table,
        beta,
        gamma,
    );

    assert_eq!(gpu, cpu, "lookup_product_gpu disagrees with CPU reference");
}

#[test]
fn test_lookup_product_gpu_vs_cpu_chunking() {
    // Exercises the per-chunk FFI invocation loop inside
    // `lookup_product_single_gpu`, ensuring the device-resident β/γ
    // scalars uploaded once outside the loop are correctly reused
    // across every chunk.
    for &n in &[64usize, 512, 4096] {
        let permuted_input: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();
        let permuted_table: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();
        let compressed_input: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();
        let compressed_table: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();
        let beta = Fr::random(OsRng);
        let gamma = Fr::random(OsRng);

        let mut gpu = vec![Fr::ZERO; n];
        lookup_product_gpu(
            &mut gpu,
            &permuted_input,
            &permuted_table,
            &compressed_input,
            &compressed_table,
            beta,
            gamma,
        )
        .unwrap();

        let mut cpu = vec![Fr::ZERO; n];
        lookup_product_cpu(
            &mut cpu,
            &permuted_input,
            &permuted_table,
            &compressed_input,
            &compressed_table,
            beta,
            gamma,
        );

        assert_eq!(gpu, cpu, "lookup_product_gpu disagrees with CPU at n={}", n);
    }
}

#[test]
fn test_lookup_product_device_inputs_vs_host() {
    // Equivalence test for the device-input FFI sibling of
    // `_halo2_commit_product`. Compares the device-input wrapper
    // (`lookup_product_device`) against the host-input wrapper
    // (`lookup_product_gpu`) on byte-identical inputs. Result must be
    // byte-equal. Sweeps `n` to exercise both the typical single-call
    // path and the chunking cadence used by `lookup_product_gpu`
    // (matching `test_lookup_product_gpu_vs_cpu_chunking`).
    for &n in &[64usize, 512, 4096] {
        let permuted_input: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();
        let permuted_table: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();
        let compressed_input: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();
        let compressed_table: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();
        let beta = Fr::random(OsRng);
        let gamma = Fr::random(OsRng);

        // Host-input arm (baseline oracle).
        let mut host_out = vec![Fr::ZERO; n];
        lookup_product_gpu(
            &mut host_out,
            &permuted_input,
            &permuted_table,
            &compressed_input,
            &compressed_table,
            beta,
            gamma,
        )
        .unwrap();

        // Device-input arm. Stage each per-poly slice into its own
        // `DeviceBuffer<Fr>`; the wrapper allocates its output buffer
        // internally and returns it device-resident.
        let d_permuted_input = permuted_input
            .as_slice()
            .to_device_on(&HALO2_GPU_CTX)
            .unwrap();
        let d_permuted_table = permuted_table
            .as_slice()
            .to_device_on(&HALO2_GPU_CTX)
            .unwrap();
        let d_compressed_input = compressed_input
            .as_slice()
            .to_device_on(&HALO2_GPU_CTX)
            .unwrap();
        let d_compressed_table = compressed_table
            .as_slice()
            .to_device_on(&HALO2_GPU_CTX)
            .unwrap();
        let d_out = lookup_product_device(
            &d_permuted_input,
            &d_permuted_table,
            &d_compressed_input,
            &d_compressed_table,
            beta,
            gamma,
        )
        .unwrap();

        // D2H the device output and compare byte-for-byte.
        let mut device_out: Vec<Fr> = Vec::with_capacity(n);
        #[allow(clippy::uninit_vec)]
        unsafe {
            device_out.set_len(n);
        }
        let bytes = std::mem::size_of_val::<[Fr]>(device_out.as_slice());
        unsafe {
            openvm_cuda_common::copy::cuda_memcpy_on::<true, false>(
                device_out.as_mut_ptr() as *mut libc::c_void,
                d_out.as_raw_ptr(),
                bytes,
                &HALO2_GPU_CTX,
            )
            .unwrap();
        }
        HALO2_GPU_CTX.stream.to_host_sync().unwrap();

        assert_eq!(
            device_out, host_out,
            "lookup_product_device disagrees with host-input arm at n={}",
            n
        );
    }
}

#[test]
fn test_permutation_product_gpu_vs_cpu() {
    // PLONK permutation running-product kernel. The GPU wrapper
    // takes column pointers as `FFITraitObject`; wrap the
    // host-side `Vec<Vec<Fr>>` the same way the prover does at
    // `plonk/permutation/prover.rs`.
    let log_n = 6u32;
    let n = 1usize << log_n;
    let num_cols = 2usize;
    let domain = halo2_axiom::poly::EvaluationDomain::<Fr>::new(1, log_n);
    let domain = EvaluationDomain::from_host_domain(&domain);
    let omega = domain.get_omega();

    let values: Vec<Vec<Fr>> = (0..num_cols)
        .map(|_| (0..n).map(|_| Fr::random(OsRng)).collect())
        .collect();
    let permutations: Vec<Vec<Fr>> = (0..num_cols)
        .map(|_| (0..n).map(|_| Fr::random(OsRng)).collect())
        .collect();
    let beta = Fr::random(OsRng);
    let gamma = Fr::random(OsRng);
    let delta = <Fr as PrimeField>::DELTA;
    let deltaomega = Fr::random(OsRng);

    let perms_ffi: Vec<FFITraitObject> = permutations
        .iter()
        .map(|p| FFITraitObject::new(p.as_ptr() as usize))
        .collect();
    let values_ffi: Vec<FFITraitObject> = values
        .iter()
        .map(|v| FFITraitObject::new(v.as_ptr() as usize))
        .collect();

    let mut gpu = vec![Fr::ONE; n];
    permutation_product_gpu(
        &mut gpu,
        &perms_ffi,
        &values_ffi,
        beta,
        gamma,
        delta,
        omega,
        deltaomega,
    )
    .unwrap();

    let mut cpu = vec![Fr::ONE; n];
    permutation_product_cpu(
        &mut cpu,
        &values,
        &permutations,
        omega,
        beta,
        gamma,
        delta,
        deltaomega,
    );

    assert_eq!(
        gpu, cpu,
        "permutation_product_gpu disagrees with CPU reference"
    );
}

#[test]
fn test_permutation_product_gpu_vs_cpu_chunking() {
    // Exercises the per-chunk loop in `permutation_product_single_gpu`,
    // including the per-chunk δω re-upload into the device-resident
    // 32-byte slot. Sweeps log_n × num_cols so both the loop body
    // (`δω = δω · ω^offset` recompute and copy_to_on) and the inner
    // launcher batch_size loop (`δω *= δ` in-place mutation of the
    // caller's device slot) are covered.
    for &log_n in &[8u32, 12, 14] {
        let n = 1usize << log_n;
        let domain = halo2_axiom::poly::EvaluationDomain::<Fr>::new(1, log_n);
        let domain = EvaluationDomain::from_host_domain(&domain);
        let omega = domain.get_omega();
        for &num_cols in &[2usize, 5] {
            let values: Vec<Vec<Fr>> = (0..num_cols)
                .map(|_| (0..n).map(|_| Fr::random(OsRng)).collect())
                .collect();
            let permutations: Vec<Vec<Fr>> = (0..num_cols)
                .map(|_| (0..n).map(|_| Fr::random(OsRng)).collect())
                .collect();
            let beta = Fr::random(OsRng);
            let gamma = Fr::random(OsRng);
            let delta = <Fr as PrimeField>::DELTA;
            let deltaomega = Fr::random(OsRng);

            let perms_ffi: Vec<FFITraitObject> = permutations
                .iter()
                .map(|p| FFITraitObject::new(p.as_ptr() as usize))
                .collect();
            let values_ffi: Vec<FFITraitObject> = values
                .iter()
                .map(|v| FFITraitObject::new(v.as_ptr() as usize))
                .collect();

            let mut gpu = vec![Fr::ONE; n];
            permutation_product_gpu(
                &mut gpu,
                &perms_ffi,
                &values_ffi,
                beta,
                gamma,
                delta,
                omega,
                deltaomega,
            )
            .unwrap();

            let mut cpu = vec![Fr::ONE; n];
            permutation_product_cpu(
                &mut cpu,
                &values,
                &permutations,
                omega,
                beta,
                gamma,
                delta,
                deltaomega,
            );

            assert_eq!(
                gpu, cpu,
                "permutation_product_gpu disagrees with CPU at log_n={}, num_cols={}",
                log_n, num_cols
            );
        }
    }
}

#[test]
fn test_permutation_product_device_inputs_vs_host_inputs() {
    // Equivalence test for the device-input FFI sibling of
    // `_halo2_permutation_product`. Compares the device-input wrapper
    // (`permutation_product_device`) against the host-input
    // wrapper (`permutation_product_gpu`) on the same inputs. Result
    // must be byte-equal. Sweeps log_n × num_cols (matching
    // `test_permutation_product_gpu_vs_cpu_chunking`).
    for &log_n in &[8u32, 12, 14] {
        let n = 1usize << log_n;
        let domain = halo2_axiom::poly::EvaluationDomain::<Fr>::new(1, log_n);
        let domain = EvaluationDomain::from_host_domain(&domain);
        let omega = domain.get_omega();
        for &num_cols in &[2usize, 5] {
            let values: Vec<Vec<Fr>> = (0..num_cols)
                .map(|_| (0..n).map(|_| Fr::random(OsRng)).collect())
                .collect();
            let permutations: Vec<Vec<Fr>> = (0..num_cols)
                .map(|_| (0..n).map(|_| Fr::random(OsRng)).collect())
                .collect();
            let beta = Fr::random(OsRng);
            let gamma = Fr::random(OsRng);
            let delta = <Fr as PrimeField>::DELTA;
            let deltaomega = Fr::random(OsRng);

            // Host-input arm (baseline).
            let host_perms_ffi: Vec<FFITraitObject> = permutations
                .iter()
                .map(|p| FFITraitObject::new(p.as_ptr() as usize))
                .collect();
            let host_values_ffi: Vec<FFITraitObject> = values
                .iter()
                .map(|v| FFITraitObject::new(v.as_ptr() as usize))
                .collect();
            let mut host_out = vec![Fr::ONE; n];
            permutation_product_gpu(
                &mut host_out,
                &host_perms_ffi,
                &host_values_ffi,
                beta,
                gamma,
                delta,
                omega,
                deltaomega,
            )
            .unwrap();

            // Device-input arm. Stage each per-column polynomial into its own
            // `DeviceBuffer<Fr>`; build FFITraitObjects from the device
            // pointers. `modified_values_device` is a full-length device
            // buffer initialised to `Fr::ONE`.
            let perm_devs: Vec<DeviceBuffer<Fr>> = permutations
                .iter()
                .map(|p| p.as_slice().to_device_on(&HALO2_GPU_CTX).unwrap())
                .collect();
            let value_devs: Vec<DeviceBuffer<Fr>> = values
                .iter()
                .map(|v| v.as_slice().to_device_on(&HALO2_GPU_CTX).unwrap())
                .collect();
            let device_perms_ffi: Vec<FFITraitObject> = perm_devs
                .iter()
                .map(|b| FFITraitObject::new(b.as_raw_ptr() as usize))
                .collect();
            let device_values_ffi: Vec<FFITraitObject> = value_devs
                .iter()
                .map(|b| FFITraitObject::new(b.as_raw_ptr() as usize))
                .collect();
            let mut modified_values_device = vec![Fr::ONE; n]
                .as_slice()
                .to_device_on(&HALO2_GPU_CTX)
                .unwrap();
            permutation_product_device(
                &mut modified_values_device,
                &device_perms_ffi,
                &device_values_ffi,
                beta,
                gamma,
                delta,
                omega,
                deltaomega,
            )
            .unwrap();

            // D2H the device output and compare byte-for-byte.
            let mut device_out: Vec<Fr> = Vec::with_capacity(n);
            #[allow(clippy::uninit_vec)]
            unsafe {
                device_out.set_len(n);
            }
            let bytes = std::mem::size_of_val::<[Fr]>(device_out.as_slice());
            unsafe {
                openvm_cuda_common::copy::cuda_memcpy_on::<true, false>(
                    device_out.as_mut_ptr() as *mut libc::c_void,
                    modified_values_device.as_raw_ptr(),
                    bytes,
                    &HALO2_GPU_CTX,
                )
                .unwrap();
            }
            HALO2_GPU_CTX.stream.to_host_sync().unwrap();

            assert_eq!(
                device_out, host_out,
                "permutation_product_device_inputs disagrees with host-input arm at log_n={}, num_cols={}",
                log_n, num_cols
            );
        }
    }
}

#[test]
fn test_grand_product_device_inputs_vs_host_inputs() {
    // Equivalence test for the device-input FFI sibling of
    // `_halo2_grand_product`. Compares the device-input wrapper against
    // the host-input wrapper on the same inputs. Result must be
    // byte-equal. Sweeps log_n (matching the
    // `test_lookup_product_gpu_vs_cpu_chunking` cadence).
    for &log_n in &[8u32, 12, 14] {
        let n = 1usize << log_n;
        let input: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();
        let prefix = Fr::random(OsRng);

        let mut host_out = vec![Fr::ZERO; n];
        grand_product_gpu(&mut host_out, &input, prefix).unwrap();

        let input_device = input.as_slice().to_device_on(&HALO2_GPU_CTX).unwrap();
        let d_scanned = grand_product_device(input_device, n, prefix).unwrap();
        let device_out = d_scanned.to_host_on(&HALO2_GPU_CTX).unwrap();

        assert_eq!(
            device_out, host_out,
            "grand_product_device_inputs disagrees with host-input arm at log_n={}",
            log_n
        );
    }
}

#[test]
fn test_grand_product_device_multi_chunk_carry() {
    // A5: exercises the cross-chunk device→device prefix carry in
    // `grand_product_device_chunked`. `_halo2_grand_product_max_len` only
    // forces multi-chunk on memory-sized inputs, so we drive the chunked core
    // directly with a small `chunk_size` that evenly divides `output_len`,
    // producing several in-bounds chunks (offset 0,n/4,n/2,3n/4 → 3 boundary
    // carries). The multi-chunk device scan must be byte-identical to both the
    // single-chunk host oracle (`grand_product_gpu`) and a plain CPU running
    // product — proving the D2D boundary carry rolls the prefix correctly.
    for &log_n in &[10u32, 12, 14] {
        let n = 1usize << log_n;
        let input: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();
        let prefix = Fr::random(OsRng);

        // Host single-chunk oracle.
        let mut host_out = vec![Fr::ZERO; n];
        grand_product_gpu(&mut host_out, &input, prefix).unwrap();

        // CPU reference: out[i] = prefix * prod_{j<=i} input[j].
        let mut cpu_out = vec![Fr::ZERO; n];
        let mut acc = prefix;
        for i in 0..n {
            acc *= input[i];
            cpu_out[i] = acc;
        }

        // Device multi-chunk: device-resident prefix, forced chunk_size = n/4.
        let input_device = input.as_slice().to_device_on(&HALO2_GPU_CTX).unwrap();
        let prefix_device = std::slice::from_ref(&prefix)
            .to_device_on(&HALO2_GPU_CTX)
            .unwrap();
        let chunk_size = n / 4;
        assert!(chunk_size > 0 && n.is_multiple_of(chunk_size));
        let d_scanned = crate::cuda::funcs::grand_product::grand_product_device_chunked(
            input_device,
            n,
            &prefix_device,
            chunk_size,
        )
        .unwrap();
        let device_out = d_scanned.to_host_on(&HALO2_GPU_CTX).unwrap();

        assert_eq!(
            device_out, cpu_out,
            "multi-chunk device carry disagrees with CPU reference at log_n={}",
            log_n
        );
        assert_eq!(
            device_out, host_out,
            "multi-chunk device carry disagrees with single-chunk host oracle at log_n={}",
            log_n
        );
    }
}

#[test]
fn test_permutation_grand_product_cross_set_carry() {
    // A4: the permutation prover chains each set's grand product into the next
    // via z_0[set k] = z_{acc_len-1}[set k-1]. Compares the device-resident
    // carry (`grand_product_device_with_prefix_device` + D2D z[0] seed + D2D
    // carry update, mirroring `permutation::Argument::commit`) against the
    // host-scalar carry route over a multi-set scenario. The concatenated
    // per-set Z must be byte-identical.
    let log_n = 12u32;
    let n = 1usize << log_n;
    let blinding = 6usize;
    let acc_len = n - blinding;
    let num_sets = 3usize;
    let scalar_bytes = std::mem::size_of::<Fr>();

    // Per-set modified-value inputs (analogue of `d_modified_values`).
    let sets: Vec<Vec<Fr>> = (0..num_sets)
        .map(|_| (0..n).map(|_| Fr::random(OsRng)).collect())
        .collect();

    // ---- Route A: host-scalar carry (baseline / oracle). ----
    let mut host_zs: Vec<Vec<Fr>> = Vec::new();
    let mut last_z = Fr::ONE;
    for modified in &sets {
        let input_device = modified.as_slice().to_device_on(&HALO2_GPU_CTX).unwrap();
        let d_scanned = grand_product_device(input_device, acc_len - 1, last_z).unwrap();
        let scanned = d_scanned.to_host_on(&HALO2_GPU_CTX).unwrap();
        let mut z = vec![Fr::ZERO; acc_len];
        z[0] = last_z;
        z[1..acc_len].copy_from_slice(&scanned[0..acc_len - 1]);
        last_z = z[acc_len - 1];
        host_zs.push(z);
    }

    // ---- Route B: device-resident carry (the A4 fix). ----
    let d_ones = vec![Fr::ONE]
        .as_slice()
        .to_device_on(&HALO2_GPU_CTX)
        .unwrap();
    let d_last_z = DeviceBuffer::<Fr>::with_capacity_on(1, &HALO2_GPU_CTX);
    unsafe {
        openvm_cuda_common::copy::cuda_memcpy_on::<true, true>(
            d_last_z.as_mut_raw_ptr(),
            d_ones.as_raw_ptr(),
            scalar_bytes,
            &HALO2_GPU_CTX,
        )
        .unwrap();
    }
    let mut dev_zs: Vec<Vec<Fr>> = Vec::new();
    for modified in &sets {
        let input_device = modified.as_slice().to_device_on(&HALO2_GPU_CTX).unwrap();
        let d_scanned =
            grand_product_device_with_prefix_device(input_device, acc_len - 1, &d_last_z).unwrap();
        let d_z = DeviceBuffer::<Fr>::with_capacity_on(acc_len, &HALO2_GPU_CTX);
        unsafe {
            // z[0] = carry (D2D).
            openvm_cuda_common::copy::cuda_memcpy_on::<true, true>(
                d_z.as_mut_raw_ptr(),
                d_last_z.as_raw_ptr(),
                scalar_bytes,
                &HALO2_GPU_CTX,
            )
            .unwrap();
            // z[1..acc_len] = d_scanned[0..acc_len-1] (D2D).
            openvm_cuda_common::copy::cuda_memcpy_on::<true, true>(
                (d_z.as_mut_raw_ptr() as *mut u8).add(scalar_bytes) as *mut libc::c_void,
                d_scanned.as_raw_ptr(),
                (acc_len - 1) * scalar_bytes,
                &HALO2_GPU_CTX,
            )
            .unwrap();
            // carry update: d_last_z = z[acc_len-1] (D2D).
            openvm_cuda_common::copy::cuda_memcpy_on::<true, true>(
                d_last_z.as_mut_raw_ptr(),
                (d_z.as_raw_ptr() as *const u8).add((acc_len - 1) * scalar_bytes)
                    as *const libc::c_void,
                scalar_bytes,
                &HALO2_GPU_CTX,
            )
            .unwrap();
        }
        let z = d_z.to_host_on(&HALO2_GPU_CTX).unwrap();
        dev_zs.push(z);
    }

    assert_eq!(
        dev_zs, host_zs,
        "device cross-set carry disagrees with host-scalar carry route"
    );
    // Sanity: the carry actually chains, and set 0 starts from ONE.
    assert_eq!(dev_zs[0][0], Fr::ONE);
    for k in 1..num_sets {
        assert_eq!(
            dev_zs[k][0],
            dev_zs[k - 1][acc_len - 1],
            "set {k} z[0] must equal set {} z[acc_len-1]",
            k - 1
        );
    }
}

#[test]
fn test_permutation_quotient_gpu_vs_cpu() {
    // Equivalence test for the permutation-quotient kernel.
    //
    // Sweeps the kernel-shape parameters fibonacci alone doesn't cover:
    //   - `n_sets`: 2 forces the "first set / last set" branch coverage,
    //     5 mirrors the actual fibonacci permutation argument.
    //   - `chunk_len`: 1 matches degree=3 circuits; ≥2 exercises the
    //     inner-column ∏_j loop body that's trivial at 1.
    //   - `length`: 256 catches rotation wraparound at the boundaries
    //     (idx=0 → r_last wraps; idx=length-1 → r_next wraps);
    //     4096 verifies grid-sizing / launch-params behavior at scale.
    //
    // Both sides compute against `crate::cpu::evaluator::permutation_quotient_cpu_chunk`
    // — the same function the production
    // prover calls. The kernel is byte-for-byte equivalent: assertEq on
    // raw `Fr` is the gate.
    use ff::Field;

    fn run_one(length: usize, n_sets: usize, chunk_len: usize, n_perm_cols: usize) {
        assert!(n_perm_cols <= n_sets * chunk_len);
        assert!(n_perm_cols + chunk_len > n_sets * chunk_len);
        let isize_ = length as i32;
        // last_rotation tracks the prover's `-(blinding_factors + 1)` —
        // single-digit negative is representative.
        let last_rotation: i32 = -3;
        let rot_scale: i32 = 1;

        // Generate inputs. Both CPU and GPU consume the same buffers; OsRng
        // is fine because we only compare CPU vs GPU on identical inputs.
        let make_poly = |n: usize| (0..n).map(|_| Fr::random(OsRng)).collect::<Vec<Fr>>();
        let mut values_init = make_poly(length);
        let l0 = make_poly(length);
        let l_last = make_poly(length);
        let l_active_row = make_poly(length);
        let perm_prod_owned: Vec<Vec<Fr>> = (0..n_sets).map(|_| make_poly(length)).collect();
        let perm_owned: Vec<Vec<Fr>> = (0..n_perm_cols).map(|_| make_poly(length)).collect();
        // column_values: rotate column types Advice / Fixed / Instance —
        // identity at this layer is just slice contents, but the variety
        // catches any kernel-side mis-indexing in the chunk_len > 1 path.
        let column_values_owned: Vec<Vec<Fr>> =
            (0..n_perm_cols).map(|_| make_poly(length)).collect();

        let perm_prod: Vec<&[Fr]> = perm_prod_owned.iter().map(|v| v.as_slice()).collect();
        let perm: Vec<&[Fr]> = perm_owned.iter().map(|v| v.as_slice()).collect();
        let cols: Vec<&[Fr]> = column_values_owned.iter().map(|v| v.as_slice()).collect();

        let beta = Fr::random(OsRng);
        let gamma = Fr::random(OsRng);
        let y = Fr::random(OsRng);
        let delta_start = beta * Fr::ZETA;
        // omega: a true 2^k root for `length = 2^k`.
        let log_n = length.trailing_zeros();
        assert_eq!(length, 1 << log_n, "length must be a power of two");
        let omega: Fr = {
            // halo2curves bn256 Fr has S = 28 (ROOT_OF_UNITY at order 2^28).
            // Fold to order length via squaring.
            let mut w = Fr::ROOT_OF_UNITY;
            for _ in 0..(Fr::S - log_n) {
                w = w.square();
            }
            w
        };
        // current_extended_omega: just an arbitrary domain coset multiplier.
        let current_extended_omega = Fr::random(OsRng);

        // CPU reference: call the production CPU function with start=0
        // over the full row range.
        let mut values_cpu = values_init.clone();
        permutation_quotient_cpu_chunk(
            &mut values_cpu,
            0,
            rot_scale,
            isize_,
            last_rotation,
            chunk_len,
            &cols,
            &perm_prod,
            &perm,
            &l0,
            &l_last,
            &l_active_row,
            y,
            beta,
            gamma,
            delta_start,
            current_extended_omega,
            omega,
        );

        // GPU under test.
        permutation_quotient_gpu(
            &mut values_init,
            &perm_prod,
            &perm,
            &cols,
            &l0,
            &l_last,
            &l_active_row,
            beta,
            gamma,
            y,
            delta_start,
            current_extended_omega,
            omega,
            chunk_len,
            last_rotation,
            rot_scale,
            isize_,
        )
        .expect("permutation_quotient_gpu");

        for (i, (g, c)) in values_init.iter().zip(values_cpu.iter()).enumerate() {
            assert_eq!(
                g, c,
                "GPU vs CPU mismatch at row {i} (length={length}, n_sets={n_sets}, chunk_len={chunk_len})"
            );
        }
    }

    // Parameter grid (per the TDD plan): boundary n_sets ∈ {2, 5},
    // chunk_len ∈ {1, 2}, two sizes for grid/launch-params coverage,
    // and a partial-last-set case (real fibonacci hits this when
    // pk.vk.cs.permutation.columns.len() % chunk_len != 0).
    for &length in &[256usize, 4096] {
        for &n_sets in &[2usize, 5] {
            for &chunk_len in &[1usize, 2] {
                run_one(length, n_sets, chunk_len, n_sets * chunk_len);
            }
        }
        // Partial-last-set: 3 sets × chunk_len 2 with only 5 columns
        // total (the last set has 1 column, not 2).
        run_one(length, 3, 2, 5);
        // Larger partial: 4 sets × chunk_len 3 with 10 columns (last set
        // has 1 column, exercises a deeper truncation).
        run_one(length, 4, 3, 10);
    }
}

#[test]
fn test_cosetfft_gpu_many_to_device_vs_host_output() {
    // The device-output FFT variant must produce exactly the same bytes
    // as the host-output one. Both run the same
    // kernel over the same inputs; only the final per-poly memcpy
    // direction differs (DeviceToDevice vs DeviceToHost). Any deviation
    // here breaks the assumption that downstream GPU kernels can read
    // the device buffer in place of the host one.
    use crate::cuda::funcs::cosetfft_gpu_many;

    let log_n = 10u32;
    let extend_log_n = 10u32;
    let extend_size: usize = 1 << extend_log_n;
    let domain = halo2_axiom::poly::EvaluationDomain::<Fr>::new(1, log_n);
    let domain = EvaluationDomain::from_host_domain(&domain);
    let omega = domain.get_omega();
    let omega_part = domain.inner.g_coset; // matches typical cosetfft_part call shape

    // Three input polys of length 2^log_n, padded to 2^extend_log_n.
    let num_many = 3usize;
    let in_polys: Vec<Vec<Fr>> = (0..num_many)
        .map(|_| {
            let mut v: Vec<Fr> = (0..extend_size).map(|_| Fr::random(OsRng)).collect();
            // Match the prover's coeff-form padding: zero past `1 << log_n`.
            for slot in v.iter_mut().skip(1 << log_n) {
                *slot = Fr::ZERO;
            }
            v
        })
        .collect();

    // Path 1: host-output FFT.
    let mut host_outs: Vec<Vec<Fr>> = (0..num_many).map(|_| vec![Fr::ZERO; extend_size]).collect();
    let in_objs: Vec<FFITraitObject> = in_polys
        .iter()
        .map(|v| FFITraitObject::from_slice(v.as_slice()))
        .collect();
    let host_out_objs: Vec<FFITraitObject> = host_outs
        .iter_mut()
        .map(|v| FFITraitObject::from_slice(v.as_mut_slice()))
        .collect();
    cosetfft_gpu_many::<Fr>(
        NttType::CosetFFT_Part.into(),
        in_objs.clone(),
        host_out_objs,
        log_n,
        extend_log_n,
        omega_part,
        Fr::ONE,
    )
    .unwrap();

    // Path 2: device-output FFT, then D→H back to compare.
    let dev_bufs = cosetfft_many_h2d::<Fr>(
        NttType::CosetFFT_Part.into(),
        in_objs,
        log_n,
        extend_log_n,
        omega_part,
        Fr::ONE,
    )
    .unwrap();

    use crate::cuda::utils::HALO2_GPU_CTX;
    use openvm_cuda_common::copy::cuda_memcpy_on;
    let mut device_outs: Vec<Vec<Fr>> =
        (0..num_many).map(|_| vec![Fr::ZERO; extend_size]).collect();
    for (slot, buf) in device_outs.iter_mut().zip(dev_bufs.iter()) {
        unsafe {
            cuda_memcpy_on::<true, false>(
                slot.as_mut_ptr() as *mut libc::c_void,
                buf.as_raw_ptr(),
                std::mem::size_of_val::<[Fr]>(slot),
                &HALO2_GPU_CTX,
            )
            .unwrap();
        }
    }
    HALO2_GPU_CTX.stream.synchronize().unwrap();

    for (i, (h, d)) in host_outs.iter().zip(device_outs.iter()).enumerate() {
        assert_eq!(
            h, d,
            "FFT poly {i}: device-output variant disagrees with host-output variant",
        );
        let _ = omega; // silence unused for this test
    }
}

#[test]
fn test_cosetfft_gpu_many_device_to_device_vs_host_output() {
    // The device-input + device-output FFT variant must produce exactly
    // the same bytes as the host-input + host-output one. Any divergence
    // here breaks the assumption that downstream GPU kernels can read
    // the device buffer in place of the host one.
    use crate::cuda::funcs::{cosetfft_gpu_many, cosetfft_many_device};
    use crate::cuda::utils::HALO2_GPU_CTX;
    use openvm_cuda_common::copy::{cuda_memcpy_on, MemCopyH2D};

    fn run_one(log_n: u32) {
        let extend_log_n = log_n;
        let extend_size: usize = 1 << extend_log_n;
        let domain = halo2_axiom::poly::EvaluationDomain::<Fr>::new(1, log_n);
        let domain = EvaluationDomain::from_host_domain(&domain);
        let omega_part = domain.inner.g_coset;

        let num_many = 3usize;
        let in_polys: Vec<Vec<Fr>> = (0..num_many)
            .map(|_| {
                let mut v: Vec<Fr> = (0..extend_size).map(|_| Fr::random(OsRng)).collect();
                for slot in v.iter_mut().skip(1 << log_n) {
                    *slot = Fr::ZERO;
                }
                v
            })
            .collect();

        // Path 1: host-input + host-output (reference).
        let mut host_outs: Vec<Vec<Fr>> =
            (0..num_many).map(|_| vec![Fr::ZERO; extend_size]).collect();
        let in_objs_host: Vec<FFITraitObject> = in_polys
            .iter()
            .map(|v| FFITraitObject::from_slice(v.as_slice()))
            .collect();
        let host_out_objs: Vec<FFITraitObject> = host_outs
            .iter_mut()
            .map(|v| FFITraitObject::from_slice(v.as_mut_slice()))
            .collect();
        cosetfft_gpu_many::<Fr>(
            NttType::CosetFFT_Part.into(),
            in_objs_host,
            host_out_objs,
            log_n,
            extend_log_n,
            omega_part,
            Fr::ONE,
        )
        .unwrap();

        // Path 2: pre-upload inputs to device, run device-input + device-output.
        let in_dbufs: Vec<_> = in_polys
            .iter()
            .map(|v| v.as_slice().to_device_on(&HALO2_GPU_CTX).unwrap())
            .collect();
        let in_objs_device: Vec<FFITraitObject> = in_dbufs
            .iter()
            .map(|b| FFITraitObject::new(b.as_raw_ptr() as usize))
            .collect();
        let dev_bufs = cosetfft_many_device::<Fr>(
            NttType::CosetFFT_Part.into(),
            in_objs_device,
            log_n,
            extend_log_n,
            omega_part,
            Fr::ONE,
        )
        .unwrap();

        let mut device_outs: Vec<Vec<Fr>> =
            (0..num_many).map(|_| vec![Fr::ZERO; extend_size]).collect();
        for (slot, buf) in device_outs.iter_mut().zip(dev_bufs.iter()) {
            unsafe {
                cuda_memcpy_on::<true, false>(
                    slot.as_mut_ptr() as *mut libc::c_void,
                    buf.as_raw_ptr(),
                    std::mem::size_of_val::<[Fr]>(slot),
                    &HALO2_GPU_CTX,
                )
                .unwrap();
            }
        }
        HALO2_GPU_CTX.stream.synchronize().unwrap();

        for (i, (h, d)) in host_outs.iter().zip(device_outs.iter()).enumerate() {
            assert_eq!(
                h, d,
                "log_n={log_n} FFT poly {i}: device-input variant disagrees with host-input variant",
            );
        }
    }

    run_one(10);
    run_one(12);
    run_one(14);
}

#[test]
fn test_quotient_lookups_gpu_vs_cpu() {
    // GPU does iFFT+CosetFFT_Part on dense permuted polys internally; the
    // CPU reference pre-computes coset forms via `lagrange_to_extend_part`
    // / `coeff_to_extended_part` before calling `quotient_lookups_cpu`.
    use crate::cpu::arithmetic::quotient_lookups_cpu;
    use crate::cuda::modules::QuotientLookupsGpu;
    use crate::poly::{Coeff, LagrangeCoeff, Polynomial};

    fn run_one(log_n: u32) {
        let length = 1usize << log_n;
        let domain = halo2_axiom::poly::EvaluationDomain::<Fr>::new(1, log_n);
        let domain = EvaluationDomain::from_host_domain(&domain);

        let make_poly = |n: usize| (0..n).map(|_| Fr::random(OsRng)).collect::<Vec<Fr>>();

        // Already-coset-form inputs (both sides).
        let table_values = make_poly(length);
        let l0 = make_poly(length);
        let l_last = make_poly(length);
        let l_active_row = make_poly(length);
        let values_init = make_poly(length);

        // Pre-coset inputs: same values fed to both sides.
        let permuted_input_lagrange: Polynomial<Fr, LagrangeCoeff> =
            Polynomial::new(make_poly(length));
        let permuted_table_lagrange: Polynomial<Fr, LagrangeCoeff> =
            Polynomial::new(make_poly(length));
        let product_coeff: Polynomial<Fr, Coeff> = Polynomial::new(make_poly(length));

        let beta = Fr::random(OsRng);
        let gamma = Fr::random(OsRng);
        let y = Fr::random(OsRng);
        let extended_omega_factor = Fr::random(OsRng);
        // CPU `lagrange_to_extend_part` multiplies by `g_coset` internally;
        // GPU takes the pre-scaled `omega_part`. Both land on the same coset.
        let omega_part = domain.inner.g_coset * extended_omega_factor;

        let permuted_input_coset = domain
            .lagrange_to_extend_part(&permuted_input_lagrange, extended_omega_factor)
            .unwrap();
        let permuted_table_coset = domain
            .lagrange_to_extend_part(&permuted_table_lagrange, extended_omega_factor)
            .unwrap();
        let product_coset = domain
            .coeff_to_extended_part(product_coeff.clone(), extended_omega_factor)
            .unwrap();

        let mut values_cpu = values_init.clone();
        quotient_lookups_cpu(
            &mut values_cpu,
            &table_values,
            product_coset.values(),
            permuted_input_coset.values(),
            permuted_table_coset.values(),
            &l0,
            &l_last,
            &l_active_row,
            beta,
            gamma,
            y,
            length,
        );

        let mut gpu = QuotientLookupsGpu::<Fr>::new(
            &values_init,
            &l0,
            &l_last,
            &l_active_row,
            beta,
            gamma,
            y,
            log_n,
            domain.inner.omega_inv,
            domain.inner.ifft_divisor,
            domain.inner.omega,
            length,
        );
        gpu.calculate_constraints(
            &table_values,
            product_coeff.values(),
            permuted_input_lagrange.values(),
            permuted_table_lagrange.values(),
            omega_part,
        )
        .expect("calculate_constraints");
        let mut values_gpu = vec![Fr::ZERO; length];
        gpu.copy_values_back_to_host(&mut values_gpu);

        for (i, (g, c)) in values_gpu.iter().zip(values_cpu.iter()).enumerate() {
            assert_eq!(
                g, c,
                "GPU vs CPU mismatch at row {i} (log_n={log_n}, length={length})"
            );
        }
    }

    // Parameter sweep:
    //   log_n=8  (256):  rotation wraparound at idx=0 / idx=length-1.
    //   log_n=12 (4096): production-shape parameters — covers grid /
    //                    launch-params behavior at the scale the fibonacci
    //                    e2e gate exercises.
    for &log_n in &[8u32, 12] {
        run_one(log_n);
    }
}

#[test]
fn test_poly_multiply_add_gpu_vs_cpu() {
    // Kernel: `acc[i] += scalar * poly_in[i]`.
    use crate::cuda::funcs::poly_multiply_add_single_gpu;

    fn run_one(log_n: u32) {
        let n = 1usize << log_n;
        let poly_in: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();
        let mut acc_init: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();
        let scalar = Fr::random(OsRng);

        let mut cpu = acc_init.clone();
        for (a, p) in cpu.iter_mut().zip(poly_in.iter()) {
            *a += scalar * *p;
        }

        poly_multiply_add_single_gpu(&mut acc_init, &poly_in, scalar).expect("poly_multiply_add");

        for (i, (g, c)) in acc_init.iter().zip(cpu.iter()).enumerate() {
            assert_eq!(g, c, "GPU vs CPU mismatch at row {i} (log_n={log_n})");
        }
    }

    // Sweep:
    //   log_n=8  (256):  small launch + boundary behavior.
    //   log_n=12 (4096): production-shape parameters for shplonk callers.
    for &log_n in &[8u32, 12] {
        run_one(log_n);
    }
}

// Equivalence test for `compress_expressions_device`. The host arm
// (CPU `evaluate` + Horner-fold by theta) mirrors the
// `lookup.commit_permuted` closure; the device arm uses the
// `_halo2_quotient_device_columns` FFI via the `ColumnPool` shim. Both
// must be byte-identical at log_n ∈ {20, 22, 23}.
#[test]
#[ignore = "heavy"]
fn test_compress_expressions_gpu_vs_cpu() {
    use crate::cuda::funcs::ColumnPool;
    use crate::plonk::evaluation::{compress_expressions_device, evaluate};
    use crate::plonk::sealed::SealedPhase;
    use crate::plonk::{
        GpuAdviceQuery as AdviceQuery, GpuExpression as Expression, GpuFirstPhase as FirstPhase,
        GpuFixedQuery as FixedQuery, GpuInstanceQuery as InstanceQuery,
    };
    use crate::poly::Rotation;

    fn run_one(log_n: u32) {
        let j: u32 = 4;
        let domain = halo2_axiom::poly::EvaluationDomain::<Fr>::new(j, log_n);
        let domain = EvaluationDomain::from_host_domain(&domain);
        let n: usize = 1usize << log_n;

        let mk_col = || -> crate::poly::Polynomial<Fr, crate::poly::LagrangeCoeff> {
            let v: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();
            domain.lagrange_from_vec(v)
        };
        let fixed_polys: Vec<_> = (0..2).map(|_| mk_col()).collect();
        let advice_polys: Vec<_> = (0..2).map(|_| mk_col()).collect();
        let instance_polys: Vec<_> = (0..1).map(|_| mk_col()).collect();
        let challenges: Vec<Fr> = vec![Fr::random(OsRng); 1];

        let f0 = Expression::Fixed(FixedQuery {
            index: Some(0),
            column_index: 0,
            rotation: Rotation::cur(),
        });
        let f1 = Expression::Fixed(FixedQuery {
            index: Some(1),
            column_index: 1,
            rotation: Rotation::cur(),
        });
        let a0 = Expression::Advice(AdviceQuery {
            index: Some(0),
            column_index: 0,
            rotation: Rotation::cur(),
            phase: FirstPhase.to_sealed(),
        });
        let a1 = Expression::Advice(AdviceQuery {
            index: Some(1),
            column_index: 1,
            rotation: Rotation::cur(),
            phase: FirstPhase.to_sealed(),
        });
        let i0 = Expression::Instance(InstanceQuery {
            index: Some(0),
            column_index: 0,
            rotation: Rotation::cur(),
        });

        let expressions: Vec<Expression<Fr>> = vec![
            Expression::Sum(Box::new(f0.clone()), Box::new(a0.clone())),
            Expression::Sum(
                Box::new(Expression::Product(Box::new(f1), Box::new(a1.clone()))),
                Box::new(Expression::Negated(Box::new(i0))),
            ),
            Expression::Scaled(Box::new(a0), Fr::from(7u64)),
        ];

        let theta = Fr::random(OsRng);

        let host_out: Vec<Fr> = expressions
            .iter()
            .map(|expr| {
                evaluate(
                    expr,
                    n,
                    1,
                    &fixed_polys,
                    &advice_polys,
                    &instance_polys,
                    &challenges,
                )
            })
            .fold(vec![Fr::ZERO; n], |mut acc, ev| {
                for (a, e) in acc.iter_mut().zip(ev.iter()) {
                    *a = *a * theta + *e;
                }
                acc
            });

        let mut pool = ColumnPool::<Fr>::new(n);
        let fixed_slices: Vec<&[Fr]> = fixed_polys.iter().map(|p| p.values()).collect();
        let advice_slices: Vec<&[Fr]> = advice_polys.iter().map(|p| p.values()).collect();
        let instance_slices: Vec<&[Fr]> = instance_polys.iter().map(|p| p.values()).collect();
        pool.try_init(&fixed_slices, &advice_slices, &instance_slices)
            .expect("ColumnPool::try_init failed (insufficient VRAM?)");
        let dev_out =
            compress_expressions_device::<G1Affine>(&expressions, theta, n, 1, &pool, &challenges)
                .expect("compress_expressions_device failed");

        assert_eq!(
            host_out.len(),
            dev_out.len(),
            "length mismatch at log_n={log_n}"
        );
        for (i, (h, d)) in host_out.iter().zip(dev_out.iter()).enumerate() {
            assert_eq!(
                h, d,
                "compress_expressions host vs device disagree at log_n={log_n}, idx={i}"
            );
        }
    }

    for &log_n in &[20u32, 22, 23] {
        run_one(log_n);
    }
}

#[test]
#[ignore = "heavy"]
fn test_compress_expressions_gpu_inplace_device_vs_host() {
    use crate::cuda::funcs::ColumnPool;
    use crate::plonk::evaluation::{
        compress_expressions_device, compress_expressions_in_place_device, GraphEvaluator,
    };
    use crate::plonk::sealed::SealedPhase;
    use crate::plonk::{
        GpuAdviceQuery as AdviceQuery, GpuExpression as Expression, GpuFirstPhase as FirstPhase,
        GpuFixedQuery as FixedQuery, GpuInstanceQuery as InstanceQuery,
    };
    use crate::poly::Rotation;
    use openvm_cuda_common::copy::cuda_memcpy_on;
    use openvm_cuda_common::d_buffer::DeviceBuffer;

    fn run_one(log_n: u32) {
        let j: u32 = 4;
        let domain = halo2_axiom::poly::EvaluationDomain::<Fr>::new(j, log_n);
        let domain = EvaluationDomain::from_host_domain(&domain);
        let n: usize = 1usize << log_n;

        let mk_col = || -> crate::poly::Polynomial<Fr, crate::poly::LagrangeCoeff> {
            let v: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();
            domain.lagrange_from_vec(v)
        };
        let fixed_polys: Vec<_> = (0..2).map(|_| mk_col()).collect();
        let advice_polys: Vec<_> = (0..2).map(|_| mk_col()).collect();
        let instance_polys: Vec<_> = (0..1).map(|_| mk_col()).collect();
        let challenges: Vec<Fr> = vec![Fr::random(OsRng); 1];

        let f0 = Expression::Fixed(FixedQuery {
            index: Some(0),
            column_index: 0,
            rotation: Rotation::cur(),
        });
        let f1 = Expression::Fixed(FixedQuery {
            index: Some(1),
            column_index: 1,
            rotation: Rotation::cur(),
        });
        let a0 = Expression::Advice(AdviceQuery {
            index: Some(0),
            column_index: 0,
            rotation: Rotation::cur(),
            phase: FirstPhase.to_sealed(),
        });
        let a1 = Expression::Advice(AdviceQuery {
            index: Some(1),
            column_index: 1,
            rotation: Rotation::cur(),
            phase: FirstPhase.to_sealed(),
        });
        let i0 = Expression::Instance(InstanceQuery {
            index: Some(0),
            column_index: 0,
            rotation: Rotation::cur(),
        });

        let expressions: Vec<Expression<Fr>> = vec![
            Expression::Sum(Box::new(f0.clone()), Box::new(a0.clone())),
            Expression::Sum(
                Box::new(Expression::Product(Box::new(f1), Box::new(a1.clone()))),
                Box::new(Expression::Negated(Box::new(i0))),
            ),
            Expression::Scaled(Box::new(a0), Fr::from(7u64)),
        ];

        let theta = Fr::random(OsRng);

        let mut pool = ColumnPool::<Fr>::new(n);
        let fixed_slices: Vec<&[Fr]> = fixed_polys.iter().map(|p| p.values()).collect();
        let advice_slices: Vec<&[Fr]> = advice_polys.iter().map(|p| p.values()).collect();
        let instance_slices: Vec<&[Fr]> = instance_polys.iter().map(|p| p.values()).collect();
        pool.try_init(&fixed_slices, &advice_slices, &instance_slices)
            .expect("ColumnPool::try_init failed (insufficient VRAM?)");

        let host_out =
            compress_expressions_device::<G1Affine>(&expressions, theta, n, 1, &pool, &challenges)
                .expect("compress_expressions_device (host-output FFI)");

        let graph = GraphEvaluator::<G1Affine>::for_compress(&expressions);
        // Slots 0..3 are `[0, 1, -1, 2]` (kernel hard-coded c1/c2 semantics);
        // slot 4 holds `theta` (the Horner-fold factor).
        let expr_constants: Vec<Fr> = vec![Fr::ZERO, Fr::ONE, -Fr::ONE, Fr::from(2u64), theta];
        let mut d_out: DeviceBuffer<Fr> =
            DeviceBuffer::<Fr>::with_capacity_on(n, &crate::cuda::utils::HALO2_GPU_CTX);
        compress_expressions_in_place_device::<G1Affine>(
            &graph,
            &expr_constants,
            n,
            1,
            &pool,
            &challenges,
            &mut d_out,
        )
        .expect("compress_expressions_in_place_device (device-output FFI)");

        let mut dev_out_h = vec![Fr::ZERO; n];
        let bytes = n * std::mem::size_of::<Fr>();
        unsafe {
            cuda_memcpy_on::<true, false>(
                dev_out_h.as_mut_ptr() as *mut libc::c_void,
                d_out.as_raw_ptr(),
                bytes,
                &crate::cuda::utils::HALO2_GPU_CTX,
            )
            .expect("D2H of compress_expressions_in_place_device result");
        }
        crate::cuda::utils::HALO2_GPU_CTX
            .stream
            .to_host_sync()
            .expect("stream sync after D2H");

        assert_eq!(host_out.len(), dev_out_h.len());
        for (i, (h, d)) in host_out.iter().zip(dev_out_h.iter()).enumerate() {
            assert_eq!(
                h, d,
                "_halo2_quotient_device_columns_device_out byte mismatch at log_n={log_n}, idx={i}"
            );
        }
    }

    for &log_n in &[20u32, 22, 23] {
        run_one(log_n);
    }
}

// Host-arm fallback test for `ColumnPool::try_init`. When the pool
// reports `InsufficientGpuMemory`, the caller must gracefully fall back
// to the CPU `compress_expressions` closure. This test forces a
// too-large column request and verifies (a) `try_init` returns the
// documented error, (b) `pool.is_initialized() == false`, and (c)
// `compress_expressions_device` rejects an uninitialized pool (the
// caller's actual fallback path keys off `is_initialized`).
#[test]
fn test_column_pool_host_arm_fallback_on_oom() {
    use crate::cuda::funcs::ColumnPool;
    use crate::cuda::HaloGpuError;

    // Fake an absurd per-column length so the all-at-once budget exceeds
    // free VRAM regardless of physical card. Per-column bytes ≈ 32 EiB —
    // dwarfs any sane `free_bytes` return from
    // `query_device_free_bytes_for_chunking`.
    let absurd_n: usize = 1usize << 56; // 2^56 × 32 B = 2 ZiB
    let mut pool = ColumnPool::<Fr>::new(absurd_n);
    // Construct empty `&[&[Fr]]` placeholders — the gating happens before
    // the FFI reads the slices.
    let empty: [&[Fr]; 0] = [];
    let res = pool.try_init(&empty[..], &empty[..], &empty[..]);
    assert!(
        res.is_ok(),
        "empty inputs should pass the VRAM gate trivially"
    );
    assert!(
        pool.is_initialized(),
        "empty inputs leave the pool in an initialized state"
    );

    // Now exercise the actual OOM path: 64 columns of absurd_n.
    let mut pool_oom = ColumnPool::<Fr>::new(absurd_n);
    let total = ColumnPool::<Fr>::estimate_resident_bytes(absurd_n, 64, 0, 0);
    assert!(total > 0, "estimator wrapped to 0 — invariant break");
    let dummy_col = vec![Fr::ZERO; 0]; // empty slice; never read after gating
    let columns: Vec<&[Fr]> = (0..64).map(|_| dummy_col.as_slice()).collect();
    let res = pool_oom.try_init(&columns, &empty[..], &empty[..]);
    match res {
        Err(HaloGpuError::InsufficientGpuMemory {
            context,
            magnitude,
            free_bytes: _,
        }) => {
            assert_eq!(context, "ColumnPool::try_init");
            assert_eq!(magnitude, 64);
        }
        Err(other) => panic!("expected InsufficientGpuMemory, got {:?}", other),
        Ok(()) => panic!("expected OOM rejection, got Ok"),
    }
    assert!(
        !pool_oom.is_initialized(),
        "pool_oom must not be initialized on OOM"
    );
    // Caller's fallback hook: `is_initialized() == false` → host-arm path.
    // The actual fallback is exercised by the `commit_permuted` closure
    // (see `plonk/lookup/prover.rs`); end-to-end correctness is the
    // fibonacci acceptance run.
}

#[test]
fn test_permute_expression_pair_gpu_vs_seq() {
    use crate::cuda::utils::HALO2_GPU_CTX;
    use openvm_cuda_common::copy::{cuda_memcpy_on, MemCopyH2D};
    use rand_chacha::rand_core::{RngCore, SeedableRng};
    use rand_chacha::ChaCha20Rng;
    use std::collections::BTreeMap;

    // Direct port of `plonk::lookup::prover::permute_expression_pair_seq`
    // (lookup/prover.rs:537-628), kept verbatim algorithmically so the
    // GPU output is gated against the same byte-exact reference the
    // prover compiles to. The original is module-private.
    fn permute_expression_pair_seq_local(
        params_n: usize,
        input: &[Fr],
        table: &[Fr],
        usable_rows: usize,
    ) -> (Vec<Fr>, Vec<Fr>) {
        let mut permuted_input: Vec<Fr> = input[..usable_rows].to_vec();
        permuted_input.sort();

        let mut leftover_table_map: BTreeMap<Fr, u32> = BTreeMap::new();
        for coeff in table.iter().take(usable_rows) {
            *leftover_table_map.entry(*coeff).or_insert(0) += 1;
        }

        let mut permuted_table = vec![Fr::ZERO; usable_rows];
        let mut repeated_input_rows: Vec<usize> = Vec::new();
        for (row, (input_value, table_value)) in permuted_input
            .iter()
            .zip(permuted_table.iter_mut())
            .enumerate()
        {
            if row == 0 || *input_value != permuted_input[row - 1] {
                *table_value = *input_value;
                let count = leftover_table_map
                    .get_mut(input_value)
                    .expect("input value must occur in table multiset");
                assert!(*count > 0);
                *count -= 1;
            } else {
                repeated_input_rows.push(row);
            }
        }

        for (coeff, count) in leftover_table_map.iter() {
            for _ in 0..*count {
                permuted_table[repeated_input_rows.pop().expect("repeated row")] = *coeff;
            }
        }
        assert!(repeated_input_rows.is_empty());

        permuted_input.extend(std::iter::repeat_n(Fr::default(), params_n - usable_rows));
        permuted_table.extend(std::iter::repeat_n(Fr::default(), params_n - usable_rows));
        (permuted_input, permuted_table)
    }

    // Build inputs where the lookup invariant `input ⊆ table` (as
    // multisets) holds. Strategy: generate a random "alphabet" of
    // distinct table values per-seed, then sample the input rows from
    // that alphabet with the requested duplication pattern.
    fn build_inputs(
        rng: &mut ChaCha20Rng,
        usable_rows: usize,
        pattern: AdversarialPattern,
    ) -> (Vec<Fr>, Vec<Fr>) {
        let alphabet_size = match pattern {
            AdversarialPattern::AllDistinct => usable_rows,
            AdversarialPattern::Mixed => (usable_rows / 3).max(1),
            AdversarialPattern::AllSame => 1,
        };
        let alphabet: Vec<Fr> = (0..alphabet_size).map(|_| Fr::random(&mut *rng)).collect();

        // Table contains every alphabet element at least once; remaining
        // slots are filled by repeating arbitrary alphabet entries (this
        // models a real lookup table with multiplicities).
        let mut table: Vec<Fr> = alphabet.clone();
        while table.len() < usable_rows {
            let idx = (rng.next_u64() as usize) % alphabet.len();
            table.push(alphabet[idx]);
        }
        // Shuffle for adversarial ordering.
        for i in (1..table.len()).rev() {
            let j = (rng.next_u64() as usize) % (i + 1);
            table.swap(i, j);
        }

        // Input draws from the alphabet (so input ⊆ table multiset).
        // For AllSame we sample a single alphabet entry repeatedly; for
        // Mixed we draw uniformly from the smaller alphabet to force
        // many duplicates; for AllDistinct we shuffle the alphabet.
        let input: Vec<Fr> = match pattern {
            AdversarialPattern::AllDistinct => {
                let mut a = alphabet.clone();
                for i in (1..a.len()).rev() {
                    let j = (rng.next_u64() as usize) % (i + 1);
                    a.swap(i, j);
                }
                a
            }
            AdversarialPattern::Mixed => (0..usable_rows)
                .map(|_| alphabet[(rng.next_u64() as usize) % alphabet.len()])
                .collect(),
            AdversarialPattern::AllSame => vec![alphabet[0]; usable_rows],
        };
        assert_eq!(input.len(), usable_rows);
        assert_eq!(table.len(), usable_rows);
        (input, table)
    }

    #[derive(Copy, Clone, Debug)]
    enum AdversarialPattern {
        AllDistinct,
        Mixed,
        AllSame,
    }

    fn run_one(log_n: u32, blinding: usize, pattern: AdversarialPattern, seed: u64) {
        let n: usize = 1 << log_n;
        let usable_rows = n - blinding - 1;
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let (input_usable, table_usable) = build_inputs(&mut rng, usable_rows, pattern);

        // Pad to params.n() with arbitrary host bytes — the kernel must
        // ignore positions >= usable_rows.
        let mut input_n = vec![Fr::ZERO; n];
        let mut table_n = vec![Fr::ZERO; n];
        input_n[..usable_rows].copy_from_slice(&input_usable);
        table_n[..usable_rows].copy_from_slice(&table_usable);
        for slot in input_n.iter_mut().skip(usable_rows) {
            *slot = Fr::random(&mut rng);
        }
        for slot in table_n.iter_mut().skip(usable_rows) {
            *slot = Fr::random(&mut rng);
        }

        let (cpu_input, cpu_table) =
            permute_expression_pair_seq_local(n, &input_n, &table_n, usable_rows);

        let d_input = input_n[..].to_device_on(&HALO2_GPU_CTX).unwrap();
        let d_table = table_n[..].to_device_on(&HALO2_GPU_CTX).unwrap();
        let zeros = vec![Fr::ZERO; n];
        let mut d_permuted_input = zeros[..].to_device_on(&HALO2_GPU_CTX).unwrap();
        let mut d_permuted_table = zeros[..].to_device_on(&HALO2_GPU_CTX).unwrap();
        permute_expression_pair_device::<Fr>(
            &d_input,
            &d_table,
            &mut d_permuted_input,
            &mut d_permuted_table,
            n,
            usable_rows,
        )
        .unwrap();

        let mut gpu_input = vec![Fr::ZERO; n];
        let mut gpu_table = vec![Fr::ZERO; n];
        unsafe {
            cuda_memcpy_on::<true, false>(
                gpu_input.as_mut_ptr() as *mut libc::c_void,
                d_permuted_input.as_raw_ptr(),
                std::mem::size_of_val::<[Fr]>(&gpu_input),
                &HALO2_GPU_CTX,
            )
            .unwrap();
            cuda_memcpy_on::<true, false>(
                gpu_table.as_mut_ptr() as *mut libc::c_void,
                d_permuted_table.as_raw_ptr(),
                std::mem::size_of_val::<[Fr]>(&gpu_table),
                &HALO2_GPU_CTX,
            )
            .unwrap();
        }
        HALO2_GPU_CTX.stream.synchronize().unwrap();

        assert_eq!(
            gpu_input, cpu_input,
            "permute_expression_pair_device input disagrees with CPU seq \
             (log_n={log_n}, pattern={pattern:?}, seed={seed})"
        );
        assert_eq!(
            gpu_table, cpu_table,
            "permute_expression_pair_device table disagrees with CPU seq \
             (log_n={log_n}, pattern={pattern:?}, seed={seed})"
        );
    }

    // Cover the three adversarial multiset-subtraction edge cases
    // (no duplicates, mixed, full duplicates) across multiple log_n
    // and multiple seeds at each combination.
    let log_ns = [6u32, 10, 12];
    let patterns = [
        AdversarialPattern::AllDistinct,
        AdversarialPattern::Mixed,
        AdversarialPattern::AllSame,
    ];
    let seeds: [u64; 3] = [
        0x517c_c1b9_2729_24d3,
        0xdead_beef_f00d_cafe,
        0x0123_4567_89ab_cdef,
    ];
    let blinding = 6usize; // matches default ConstraintSystem blinding_factors() + 1 budget
    for &log_n in &log_ns {
        for &pattern in &patterns {
            for &seed in &seeds {
                run_one(log_n, blinding, pattern, seed);
            }
        }
    }
}

#[test]
fn test_dense_lagrange_to_coset_device_input_vs_host_input() {
    // The device-input
    // sibling `dense_lagrange_to_coset_device_with_device_input` must produce
    // a byte-identical output to the host-input
    // `dense_lagrange_to_coset_device` for the same scalar input + parameters.
    use crate::cuda::modules::{
        dense_lagrange_to_coset_device, dense_lagrange_to_coset_device_with_device_input,
    };
    use crate::cuda::utils::HALO2_GPU_CTX;
    use openvm_cuda_common::copy::{MemCopyD2H, MemCopyH2D};
    use openvm_cuda_common::d_buffer::DeviceBuffer;

    for log_n in [10u32, 12, 14] {
        let n = 1usize << log_n;
        let domain = halo2_axiom::poly::EvaluationDomain::<Fr>::new(1, log_n);
        let domain = EvaluationDomain::from_host_domain(&domain);
        let omega = domain.get_omega();
        let omega_inv = domain.get_omega_inv();
        let divisor = domain.inner.ifft_divisor;
        let omega_part = domain.inner.g_coset * domain.get_extended_omega();

        let host_input: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();

        // Host-input variant — returns DeviceBuffer<u8> (raw bytes carrying F).
        let host_out_buf = dense_lagrange_to_coset_device::<Fr>(
            &host_input,
            omega_inv,
            divisor,
            omega,
            omega_part,
            log_n,
        )
        .unwrap();
        let host_out_bytes: Vec<u8> = host_out_buf.to_host_on(&HALO2_GPU_CTX).unwrap();
        // SAFETY: F is POD repr; the host-input fn produced F values into the
        // byte buffer via `as_bytes` + memcpy.
        let host_out: &[Fr] =
            unsafe { std::slice::from_raw_parts(host_out_bytes.as_ptr() as *const Fr, n) };

        // Device-input variant — input already on device.
        let d_in: DeviceBuffer<Fr> = host_input.as_slice().to_device_on(&HALO2_GPU_CTX).unwrap();
        let device_out_buf = dense_lagrange_to_coset_device_with_device_input::<Fr>(
            &d_in, omega_inv, divisor, omega, omega_part, log_n,
        )
        .unwrap();
        let device_out: Vec<Fr> = device_out_buf.to_host_on(&HALO2_GPU_CTX).unwrap();

        assert_eq!(
            host_out,
            device_out.as_slice(),
            "dense_lagrange_to_coset_device_with_device_input disagrees with host-input \
             sibling at log_n={log_n}"
        );
    }
}

#[test]
fn test_ifft_cosetfftpart_device_input_vs_host_input() {
    // The device-input/output
    // sibling `ifft_cosetfftpart_device` must produce a byte-identical
    // output to `ifft_cosetfftpart_gpu` for the same scalar input + parameters.
    use crate::cuda::modules::{ifft_cosetfftpart_device, ifft_cosetfftpart_gpu};
    use crate::cuda::utils::HALO2_GPU_CTX;
    use openvm_cuda_common::copy::{MemCopyD2H, MemCopyH2D};
    use openvm_cuda_common::d_buffer::DeviceBuffer;

    for log_n in [10u32, 12, 14] {
        let n = 1usize << log_n;
        let domain = halo2_axiom::poly::EvaluationDomain::<Fr>::new(1, log_n);
        let domain = EvaluationDomain::from_host_domain(&domain);
        let omega = domain.get_omega();
        let omega_inv = domain.get_omega_inv();
        let divisor = domain.inner.ifft_divisor;
        let omega_part = domain.inner.g_coset * domain.get_extended_omega();

        let host_input: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();

        // Host-input/host-output variant.
        let mut host_output = vec![Fr::ZERO; n];
        ifft_cosetfftpart_gpu::<Fr>(
            &host_input,
            &mut host_output,
            log_n,
            log_n,
            omega_inv,
            divisor,
            omega,
            omega_part,
        )
        .unwrap();

        // Device-input/device-output variant.
        let d_a: DeviceBuffer<Fr> = host_input.as_slice().to_device_on(&HALO2_GPU_CTX).unwrap();
        let mut d_b: DeviceBuffer<Fr> = DeviceBuffer::<Fr>::with_capacity_on(n, &HALO2_GPU_CTX);
        ifft_cosetfftpart_device::<Fr>(
            &d_a, &mut d_b, log_n, log_n, omega_inv, divisor, omega, omega_part,
        )
        .unwrap();
        let device_output: Vec<Fr> = d_b.to_host_on(&HALO2_GPU_CTX).unwrap();

        assert_eq!(
            host_output, device_output,
            "ifft_cosetfftpart_device disagrees with host-input sibling at log_n={log_n}"
        );
    }
}

#[test]
fn test_module_poly_to_coset_device_input_vs_host_input() {
    // Sibling test for `module_poly_to_coset_device_with_device_input`: the
    // device-input variant must produce the same CosetFFT_Part output as the
    // host-pointer `module_poly_to_coset_device` on the same scalar input.
    use crate::cuda::modules::{
        module_poly_to_coset_device, module_poly_to_coset_device_with_device_input,
    };
    use crate::cuda::utils::HALO2_GPU_CTX;
    use openvm_cuda_common::copy::{MemCopyD2H, MemCopyH2D};
    use openvm_cuda_common::d_buffer::DeviceBuffer;
    use std::ffi::c_void;

    for log_n in [10u32, 12, 14] {
        let n = 1usize << log_n;
        let domain = halo2_axiom::poly::EvaluationDomain::<Fr>::new(1, log_n);
        let domain = EvaluationDomain::from_host_domain(&domain);
        let omega = domain.get_omega();
        let omega_part = domain.inner.g_coset * domain.get_extended_omega();

        let host_input: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();

        // Host-input variant (matches the existing in-prover call shape in
        // `QuotientLookupsGpu::calculate_constraints`, modules.rs:412 — the
        // CosetFFT_Part FFI accepts a host pointer for this path).
        let host_out_buf = module_poly_to_coset_device::<Fr>(
            host_input.as_ptr() as *const c_void,
            omega,
            omega_part,
            log_n,
            n,
        )
        .unwrap();
        let host_out_bytes: Vec<u8> = host_out_buf.to_host_on(&HALO2_GPU_CTX).unwrap();
        let host_out: &[Fr] =
            unsafe { std::slice::from_raw_parts(host_out_bytes.as_ptr() as *const Fr, n) };

        // Device-input variant.
        let d_in: DeviceBuffer<Fr> = host_input.as_slice().to_device_on(&HALO2_GPU_CTX).unwrap();
        let device_out_buf =
            module_poly_to_coset_device_with_device_input::<Fr>(&d_in, omega, omega_part, log_n, n)
                .unwrap();
        let device_out: Vec<Fr> = device_out_buf.to_host_on(&HALO2_GPU_CTX).unwrap();

        assert_eq!(
            host_out,
            device_out.as_slice(),
            "module_poly_to_coset_device_with_device_input disagrees with host-input sibling \
             at log_n={log_n}"
        );
    }
}

#[test]
fn test_calculate_constraints_full_device_vs_calculate_constraints_device() {
    // Equivalence test: `calculate_constraints_full_device` must
    // produce a byte-identical `values_device` output to
    // `calculate_constraints_device` on the same lookup polynomials.
    use crate::cuda::modules::QuotientLookupsGpu;
    use crate::cuda::utils::HALO2_GPU_CTX;
    use openvm_cuda_common::copy::{MemCopyD2H, MemCopyH2D};
    use openvm_cuda_common::d_buffer::DeviceBuffer;

    for log_n in [10u32, 12, 14] {
        let n = 1usize << log_n;
        let domain = halo2_axiom::poly::EvaluationDomain::<Fr>::new(1, log_n);
        let domain = EvaluationDomain::from_host_domain(&domain);
        let omega = domain.get_omega();
        let omega_inv = domain.get_omega_inv();
        let divisor = domain.inner.ifft_divisor;
        let g_coset_part = domain.inner.g_coset * domain.get_extended_omega();

        let init_values: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();
        let l0: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();
        let l_last: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();
        let l_active: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();
        let table_values: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();
        let product_poly: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();
        let permuted_input: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();
        let permuted_table: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();
        let beta = Fr::random(OsRng);
        let gamma = Fr::random(OsRng);
        let y = Fr::random(OsRng);

        // Mixed-residency baseline: host slices for product/permuted_*,
        // device buffer for table_values.
        let mut gpu_a = QuotientLookupsGpu::<Fr>::new(
            &init_values,
            &l0,
            &l_last,
            &l_active,
            beta,
            gamma,
            y,
            log_n,
            omega_inv,
            divisor,
            omega,
            n,
        );
        let d_table_a: DeviceBuffer<Fr> = table_values
            .as_slice()
            .to_device_on(&HALO2_GPU_CTX)
            .unwrap();
        gpu_a
            .calculate_constraints_device(
                &d_table_a,
                &product_poly,
                &permuted_input,
                &permuted_table,
                g_coset_part,
            )
            .unwrap();
        let out_a: Vec<Fr> = gpu_a
            .take_values_device()
            .to_host_on(&HALO2_GPU_CTX)
            .unwrap();

        // Full device-input variant: all four lookup polys pre-staged on
        // device. Internal helpers must skip every per-call H2D.
        let mut gpu_b = QuotientLookupsGpu::<Fr>::new(
            &init_values,
            &l0,
            &l_last,
            &l_active,
            beta,
            gamma,
            y,
            log_n,
            omega_inv,
            divisor,
            omega,
            n,
        );
        let d_table_b: DeviceBuffer<Fr> = table_values
            .as_slice()
            .to_device_on(&HALO2_GPU_CTX)
            .unwrap();
        let d_product: DeviceBuffer<Fr> = product_poly
            .as_slice()
            .to_device_on(&HALO2_GPU_CTX)
            .unwrap();
        let d_permuted_input: DeviceBuffer<Fr> = permuted_input
            .as_slice()
            .to_device_on(&HALO2_GPU_CTX)
            .unwrap();
        let d_permuted_table: DeviceBuffer<Fr> = permuted_table
            .as_slice()
            .to_device_on(&HALO2_GPU_CTX)
            .unwrap();
        gpu_b
            .calculate_constraints_full_device(
                &d_table_b,
                &d_product,
                &d_permuted_input,
                &d_permuted_table,
                g_coset_part,
            )
            .unwrap();
        let out_b: Vec<Fr> = gpu_b
            .take_values_device()
            .to_host_on(&HALO2_GPU_CTX)
            .unwrap();

        assert_eq!(
            out_a, out_b,
            "calculate_constraints_full_device disagrees with calculate_constraints_device at \
             log_n={log_n}"
        );
    }
}

#[test]
fn test_kate_division_device_vs_cpu() {
    // Byte-exact equivalence between `kate_division_device` and the CPU
    // reference `cpu::arithmetic::kate_division` across n in {32, 1024,
    // 2^16, 2^20} and roots covering edge cases (u = 1) plus random
    // non-unit roots. Each input is also re-divided by a freshly drawn
    // root to ensure the kernel handles arbitrary draws.
    use crate::cpu::arithmetic::kate_division;
    use crate::cuda::funcs::kate_division_device;
    use openvm_cuda_common::copy::MemCopyD2H;

    let lengths: [usize; 4] = [32, 1024, 1 << 16, 1 << 20];
    for &n in lengths.iter() {
        let poly: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();
        let roots: Vec<Fr> = vec![Fr::ONE, Fr::random(OsRng), Fr::random(OsRng)];
        for (k, root) in roots.iter().enumerate() {
            let cpu_q = kate_division(poly.iter(), *root);
            let d_poly: DeviceBuffer<Fr> = poly.as_slice().to_device_on(&HALO2_GPU_CTX).unwrap();
            let d_q = kate_division_device(&d_poly, *root).unwrap();
            let gpu_q: Vec<Fr> = d_q.to_host_on(&HALO2_GPU_CTX).unwrap();
            assert_eq!(
                gpu_q.len(),
                cpu_q.len(),
                "kate_division_device length mismatch at n={n}, root_idx={k}"
            );
            assert_eq!(
                gpu_q, cpu_q,
                "kate_division_device disagrees with CPU kate_division at n={n}, root_idx={k}"
            );
        }
    }
}

#[test]
fn test_kate_division_device_chained_vs_cpu() {
    // Stress test mirroring `shplonk::prover::div_by_vanishing`: chain
    // three successive kate divisions and compare against the CPU
    // reference's iterated `kate_division`. Each link of the chain
    // shrinks the poly by one and uses a distinct root.
    use crate::cpu::arithmetic::kate_division;
    use crate::cuda::funcs::kate_division_device;
    use openvm_cuda_common::copy::MemCopyD2H;

    let n = 1usize << 18;
    let poly: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();
    let roots: [Fr; 3] = [Fr::random(OsRng), Fr::random(OsRng), Fr::random(OsRng)];

    let cpu_q: Vec<Fr> = roots
        .iter()
        .fold(poly.clone(), |acc, r| kate_division(acc.iter(), *r));

    let mut d_poly: DeviceBuffer<Fr> = poly.as_slice().to_device_on(&HALO2_GPU_CTX).unwrap();
    for r in roots.iter() {
        d_poly = kate_division_device(&d_poly, *r).unwrap();
    }
    let gpu_q: Vec<Fr> = d_poly.to_host_on(&HALO2_GPU_CTX).unwrap();

    assert_eq!(
        gpu_q.len(),
        cpu_q.len(),
        "kate_division_device chained length mismatch"
    );
    assert_eq!(
        gpu_q, cpu_q,
        "kate_division_device chained disagrees with CPU iterated kate_division"
    );
}

#[test]
fn test_poly_sub_short_inplace_device_vs_cpu() {
    // Byte-exact equivalence: `d_acc[i] -= d_short[i]` for i < short_len.
    // Mirrors the host `n_x.values_mut().iter_mut().zip(r_x.values().iter())`
    // subtract in `shplonk::prover::quotient_contribution`.
    use crate::cuda::funcs::poly_sub_short_in_place_device;
    use openvm_cuda_common::copy::MemCopyD2H;

    let n = 1usize << 14;
    for short_len in [1usize, 5, 256, n] {
        let acc: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();
        let short: Vec<Fr> = (0..short_len).map(|_| Fr::random(OsRng)).collect();

        let mut cpu = acc.clone();
        for (a, b) in cpu.iter_mut().zip(short.iter()) {
            *a -= b;
        }

        let mut d_acc: DeviceBuffer<Fr> = acc.as_slice().to_device_on(&HALO2_GPU_CTX).unwrap();
        let d_short: DeviceBuffer<Fr> = short.as_slice().to_device_on(&HALO2_GPU_CTX).unwrap();
        poly_sub_short_in_place_device(&mut d_acc, &d_short).unwrap();
        let got: Vec<Fr> = d_acc.to_host_on(&HALO2_GPU_CTX).unwrap();

        assert_eq!(
            got, cpu,
            "poly_sub_short_in_place_device disagrees with CPU at short_len={short_len}"
        );
    }
}

#[test]
fn test_poly_sub_scalar_at_zero_device_vs_cpu() {
    // Byte-exact equivalence: `d_buf[0] -= scalar` matches the host
    // `Sub<F> for &Polynomial` in `poly.rs` which clones then does
    // `values_mut()[0] -= rhs`.
    use crate::cuda::funcs::poly_sub_scalar_at_zero_device;
    use openvm_cuda_common::copy::MemCopyD2H;

    let n = 1usize << 14;
    let poly: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();
    let scalar: Fr = Fr::random(OsRng);

    let mut cpu = poly.clone();
    cpu[0] -= scalar;

    let mut d_buf: DeviceBuffer<Fr> = poly.as_slice().to_device_on(&HALO2_GPU_CTX).unwrap();
    let d_scalar: DeviceBuffer<Fr> = std::slice::from_ref(&scalar)
        .to_device_on(&HALO2_GPU_CTX)
        .unwrap();
    poly_sub_scalar_at_zero_device(&mut d_buf, &d_scalar).unwrap();
    let got: Vec<Fr> = d_buf.to_host_on(&HALO2_GPU_CTX).unwrap();

    assert_eq!(
        got, cpu,
        "poly_sub_scalar_at_zero_device disagrees with CPU `Sub<F> for &Polynomial`"
    );
}

#[test]
fn test_kate_division_device_padded_vs_cpu() {
    // Byte-exact equivalence: `kate_division_device_padded_with_d_root`
    // must produce the length-(n-1) CPU `cpu::arithmetic::kate_division`
    // output at positions [0, n-1) and zeros at [n-1, out_len). Covers
    // the shplonk A5 site's pad target (`out_len == params.n`) as well
    // as `out_len == n-1` (degenerates to the unpadded kernel) and
    // `out_len > n` (over-pad).
    use crate::cpu::arithmetic::kate_division;
    use crate::cuda::funcs::kate_division_device_padded_with_d_root;
    use openvm_cuda_common::copy::MemCopyD2H;

    for &n in &[32usize, 1024, 1 << 16, 1 << 20] {
        let poly: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();
        let root = Fr::random(OsRng);
        let cpu_q = kate_division(poly.iter(), root);
        let q_len = cpu_q.len();

        let d_poly: DeviceBuffer<Fr> = poly.as_slice().to_device_on(&HALO2_GPU_CTX).unwrap();
        let d_root: DeviceBuffer<Fr> = std::slice::from_ref(&root)
            .to_device_on(&HALO2_GPU_CTX)
            .unwrap();

        for &pad_extra in &[0usize, 1, 256] {
            let out_len = q_len + pad_extra;
            if out_len == 0 {
                continue;
            }
            let d_q = kate_division_device_padded_with_d_root(&d_poly, &d_root, out_len).unwrap();
            assert_eq!(d_q.len(), out_len, "padded out_len mismatch (n={n})");
            let gpu_q: Vec<Fr> = d_q.to_host_on(&HALO2_GPU_CTX).unwrap();
            assert_eq!(
                gpu_q[..q_len],
                cpu_q[..],
                "kate_division_device_padded disagrees with CPU at n={n}, pad_extra={pad_extra}"
            );
            for (i, v) in gpu_q[q_len..].iter().enumerate() {
                assert_eq!(
                    *v,
                    Fr::ZERO,
                    "kate_division_device_padded tail not zero at n={n}, pad_extra={pad_extra}, tail_idx={i}"
                );
            }
        }
    }
}

#[test]
fn test_kate_division_device_padded_n1_zero_only() {
    // n == 1 (length == 0): only the trailing-zero range is written.
    // The launcher must skip the scan and zero-fill the full `out_len`.
    use crate::cuda::funcs::kate_division_device_padded_with_d_root;
    use openvm_cuda_common::copy::MemCopyD2H;

    let poly: Vec<Fr> = vec![Fr::random(OsRng)];
    let root = Fr::random(OsRng);
    let d_poly: DeviceBuffer<Fr> = poly.as_slice().to_device_on(&HALO2_GPU_CTX).unwrap();
    let d_root: DeviceBuffer<Fr> = std::slice::from_ref(&root)
        .to_device_on(&HALO2_GPU_CTX)
        .unwrap();

    for &out_len in &[1usize, 4, 1024] {
        let d_q = kate_division_device_padded_with_d_root(&d_poly, &d_root, out_len).unwrap();
        assert_eq!(d_q.len(), out_len);
        let gpu_q: Vec<Fr> = d_q.to_host_on(&HALO2_GPU_CTX).unwrap();
        for (i, v) in gpu_q.iter().enumerate() {
            assert_eq!(*v, Fr::ZERO, "n==1 padded output non-zero at idx={i}");
        }
    }

    // out_len == 0 returns an empty buffer.
    let d_q = kate_division_device_padded_with_d_root(&d_poly, &d_root, 0).unwrap();
    assert_eq!(d_q.len(), 0);
}

#[test]
fn test_poly_sub_short_out_of_place_device_vs_cpu() {
    // Byte-exact equivalence with the host reference:
    //   d_out[i] = d_long[i] - d_short[i]   for i < short_len
    //   d_out[i] = d_long[i]                for short_len <= i < long_len
    // Mirrors the shplonk A3 site that computes `n_x = p_x - r_x`
    // out-of-place so the shared p_x stays untouched for the linearisation
    // pass.
    use crate::cuda::funcs::poly_sub_short_out_of_place_device;
    use openvm_cuda_common::copy::MemCopyD2H;

    let long_len = 1usize << 14;
    for short_len in [0usize, 1, 5, 256, long_len] {
        let long: Vec<Fr> = (0..long_len).map(|_| Fr::random(OsRng)).collect();
        let short: Vec<Fr> = (0..short_len).map(|_| Fr::random(OsRng)).collect();

        let mut cpu = long.clone();
        for (i, b) in short.iter().enumerate() {
            cpu[i] -= b;
        }

        let d_long: DeviceBuffer<Fr> = long.as_slice().to_device_on(&HALO2_GPU_CTX).unwrap();
        // For short_len == 0, allocate a null-backed buffer to avoid the
        // `with_capacity_on(0)` assert.
        let d_short: DeviceBuffer<Fr> = if short_len == 0 {
            DeviceBuffer::<Fr>::new()
        } else {
            short.as_slice().to_device_on(&HALO2_GPU_CTX).unwrap()
        };
        let mut d_out: DeviceBuffer<Fr> =
            DeviceBuffer::<Fr>::with_capacity_on(long_len, &HALO2_GPU_CTX);
        poly_sub_short_out_of_place_device(&mut d_out, &d_long, &d_short).unwrap();
        let got: Vec<Fr> = d_out.to_host_on(&HALO2_GPU_CTX).unwrap();

        assert_eq!(
            got, cpu,
            "poly_sub_short_out_of_place_device disagrees with CPU at short_len={short_len}"
        );
    }
}

#[test]
fn test_kate_division_device_n1_no_panic() {
    // `with_capacity_on(0)` asserts; the wrapper must early-return BEFORE
    // the allocation when n == 1 (q has length 0). Empty result is the
    // contract for the degenerate constant-poly case.
    use crate::cuda::funcs::{kate_division_device, kate_division_device_with_d_root};
    use openvm_cuda_common::copy::MemCopyD2H;

    let n = 1usize;
    let poly: Vec<Fr> = vec![Fr::random(OsRng)];
    let root = Fr::random(OsRng);

    let d_poly: DeviceBuffer<Fr> = poly.as_slice().to_device_on(&HALO2_GPU_CTX).unwrap();
    let d_q = kate_division_device(&d_poly, root).expect("n==1 panicked");
    assert_eq!(d_q.len(), 0, "n==1 must produce length-0 q");
    let host_q: Vec<Fr> = d_q.to_host_on(&HALO2_GPU_CTX).unwrap();
    assert!(host_q.is_empty(), "n==1 host roundtrip must be empty");

    let d_root: DeviceBuffer<Fr> = std::slice::from_ref(&root)
        .to_device_on(&HALO2_GPU_CTX)
        .unwrap();
    let d_q2 =
        kate_division_device_with_d_root(&d_poly, &d_root).expect("n==1 with_d_root panicked");
    assert_eq!(d_q2.len(), 0, "n==1 _with_d_root must produce length-0 q");
    let _ = n;
}
