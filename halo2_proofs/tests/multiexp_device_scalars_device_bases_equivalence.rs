//! Equivalence test for `multiexp_gpu_device_scalars_device_bases_chunked`
//! (device-scalars + device-bases MSM) vs `best_multiexp_cpu`.

use ff::Field;
use group::Curve;
use halo2_axiom_gpu::cpu::arithmetic::best_multiexp_cpu;
use halo2_axiom_gpu::cuda::funcs::multiexp_gpu_device_scalars_device_bases_chunked;
use halo2_axiom_gpu::cuda::utils::HALO2_GPU_CTX;
use halo2curves::bn256::{Fr, G1Affine, G1};
use openvm_cuda_common::copy::MemCopyH2D;
use rand_core::OsRng;

fn run_one(log_n: u32) {
    let n: usize = 1usize << log_n;
    let coeffs: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();
    let bases_host: Vec<G1Affine> = (0..n)
        .map(|_| (G1::generator() * Fr::random(OsRng)).to_affine())
        .collect();

    let cpu = best_multiexp_cpu::<G1Affine>(&coeffs, &bases_host);

    let d_coeffs = coeffs
        .as_slice()
        .to_device_on(&HALO2_GPU_CTX)
        .expect("H2D coeffs");
    let d_bases = bases_host
        .as_slice()
        .to_device_on(&HALO2_GPU_CTX)
        .expect("H2D bases");

    let gpu = multiexp_gpu_device_scalars_device_bases_chunked::<G1Affine>(&d_coeffs, &d_bases, n)
        .expect("device-scalars+device-bases MSM");

    assert_eq!(
        gpu.to_affine(),
        cpu.to_affine(),
        "multiexp_device_scalars_device_bases disagrees with best_multiexp_cpu at log_n={log_n}"
    );
}

#[test]
fn multiexp_device_scalars_device_bases_equivalence_log_n_20() {
    run_one(20);
}

#[test]
#[ignore = "large GPU allocation + slow CPU MSM reference"]
fn multiexp_device_scalars_device_bases_equivalence_log_n_22() {
    run_one(22);
}

#[test]
#[ignore = "large GPU allocation + slow CPU MSM reference"]
fn multiexp_device_scalars_device_bases_equivalence_log_n_23() {
    run_one(23);
}
