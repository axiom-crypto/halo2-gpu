//! Equivalence test for the device-side Horner eval kernel: the device
//! eval must produce byte-identical results to the CPU
//! `eval_polynomial` reference at multiple `log_n` sizes.
//!
//! The `_halo2_eval_polynomial` FFI reuses the existing
//! `eval_polynomial_batch` + `eval_polynomial_epilogue` CUDA kernels
//! on a caller-owned device-resident polynomial. This test exercises
//! that path against the CPU reference for correctness.

use ff::Field;
use halo2_axiom_gpu::arithmetic::eval_polynomial;
use halo2_axiom_gpu::cuda::funcs::eval_polynomial_device;
use halo2_axiom_gpu::cuda::utils::HALO2_GPU_CTX;
use halo2curves::bn256::Fr;
use openvm_cuda_common::copy::MemCopyH2D;
use rand_core::OsRng;

fn run_one(log_n: u32) {
    let n: usize = 1usize << log_n;
    let coeffs: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();
    let point = Fr::random(OsRng);

    let expected = eval_polynomial(&coeffs, point);

    let d_poly = coeffs.as_slice().to_device_on(&HALO2_GPU_CTX).expect("H2D for Horner eval test");
    let actual = eval_polynomial_device(&d_poly, point).expect("device Horner eval");

    assert_eq!(actual, expected, "device Horner eval mismatch at log_n={log_n}, n={n}");
}

#[test]
fn eval_polynomial_horner_device_equivalence_log_n_20() {
    run_one(20);
}

#[test]
fn eval_polynomial_horner_device_equivalence_log_n_22() {
    run_one(22);
}

#[test]
fn eval_polynomial_horner_device_equivalence_log_n_23() {
    run_one(23);
}
