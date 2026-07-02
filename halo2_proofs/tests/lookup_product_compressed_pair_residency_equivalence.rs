//! Byte-identity gate for A2/A3 (INT-8020): the lookup grand-product
//! numerator/denominator kernel (`lookup_product_device`) must produce
//! identical output whether the compressed input/table pair is fed straight
//! from its device buffers (the C3 `MaybeDevice::Device` carry) or first
//! round-tripped device→host→device (the pre-C3 route).
//!
//! Before C3, `commit_permuted` (`plonk/lookup/prover.rs`) D2H'd the
//! device-produced compressed pair (`d_ci`/`d_ct` from
//! `run_compress_permute_device`) into host `Polynomial`s, and `commit_product`
//! re-uploaded them via `to_device_on` before calling `lookup_product_device`
//! (which already takes `&DeviceBuffer`). C3 carries the compressed pair as
//! `MaybeDevice::Device` and passes the device buffers straight through,
//! deleting the producer D2H and the consumer H2D. Because a D2H followed by an
//! H2D is value-preserving, both routes must feed byte-identical buffers to the
//! kernel and therefore produce byte-identical lookup products. This test pins
//! that invariant on production-sized shapes.

use ff::Field;
use halo2_axiom_gpu::cuda::funcs::lookup_product_device;
use halo2_axiom_gpu::cuda::utils::HALO2_GPU_CTX;
use halo2curves::bn256::Fr;
use openvm_cuda_common::copy::{MemCopyD2H, MemCopyH2D};
use openvm_cuda_common::d_buffer::DeviceBuffer;
use rand_core::OsRng;

/// Assert the lookup product is byte-identical when the compressed pair is fed
/// device-direct (C3) versus round-tripped device→host→device (pre-C3).
fn run_one(log_n: u32) {
    let n = 1usize << log_n;
    let beta = Fr::random(OsRng);
    let gamma = Fr::random(OsRng);

    // Random source values standing in for `run_compress_permute_device`'s
    // device outputs. The permuted pair is uploaded once and shared by both
    // routes — it is already device-resident in both pre- and post-C3 code, so
    // only the *compressed* pair's residency differs between the two routes.
    let ci: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();
    let ct: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();
    let pi: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();
    let pt: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();

    let d_pi: DeviceBuffer<Fr> = pi.as_slice().to_device_on(&HALO2_GPU_CTX).unwrap();
    let d_pt: DeviceBuffer<Fr> = pt.as_slice().to_device_on(&HALO2_GPU_CTX).unwrap();

    // Compressed pair as produced on device (the C3 carry source).
    let d_ci: DeviceBuffer<Fr> = ci.as_slice().to_device_on(&HALO2_GPU_CTX).unwrap();
    let d_ct: DeviceBuffer<Fr> = ct.as_slice().to_device_on(&HALO2_GPU_CTX).unwrap();

    // Route B (C3): compressed device buffers fed straight into the kernel.
    let prod_device = lookup_product_device(&d_pi, &d_pt, &d_ci, &d_ct, beta, gamma).unwrap();
    let prod_device_host: Vec<Fr> = prod_device.to_host_on(&HALO2_GPU_CTX).unwrap();

    // Route A (pre-C3): compressed pair round-tripped device→host→device before
    // being fed into the kernel (mirrors the deleted D2H + H2D).
    let ci_rt: Vec<Fr> = d_ci.to_host_on(&HALO2_GPU_CTX).unwrap();
    let ct_rt: Vec<Fr> = d_ct.to_host_on(&HALO2_GPU_CTX).unwrap();
    let d_ci_rt: DeviceBuffer<Fr> = ci_rt.as_slice().to_device_on(&HALO2_GPU_CTX).unwrap();
    let d_ct_rt: DeviceBuffer<Fr> = ct_rt.as_slice().to_device_on(&HALO2_GPU_CTX).unwrap();
    let prod_host = lookup_product_device(&d_pi, &d_pt, &d_ci_rt, &d_ct_rt, beta, gamma).unwrap();
    let prod_host_host: Vec<Fr> = prod_host.to_host_on(&HALO2_GPU_CTX).unwrap();

    assert_eq!(
        prod_device_host.len(),
        n,
        "length mismatch at log_n={log_n}"
    );
    assert_eq!(prod_host_host.len(), n, "length mismatch at log_n={log_n}");
    for (i, (b, a)) in prod_device_host
        .iter()
        .zip(prod_host_host.iter())
        .enumerate()
    {
        assert_eq!(
            b, a,
            "device-carried vs round-tripped compressed pair disagree at log_n={log_n}, idx={i}"
        );
    }
}

#[test]
fn lookup_product_compressed_pair_residency_equivalence_log_n_18() {
    run_one(18);
}

#[test]
fn lookup_product_compressed_pair_residency_equivalence_log_n_20() {
    run_one(20);
}

#[test]
#[ignore = "large GPU allocation"]
fn lookup_product_compressed_pair_residency_equivalence_log_n_22() {
    run_one(22);
}
