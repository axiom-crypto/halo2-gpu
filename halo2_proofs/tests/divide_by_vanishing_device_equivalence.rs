//! Equivalence tests for device-backed vanishing polynomial division.
//!
//! Host and device execution should produce identical extended-Lagrange output.

use ff::Field;
use halo2_axiom_gpu::poly::{
    Device, DevicePolyExt, EvaluationDomain, ExtendedLagrangeCoeff, Polynomial,
};
use halo2curves::bn256::Fr;
use rand_core::OsRng;

fn run_one(log_n: u32) {
    let j: u32 = 4;
    let domain = EvaluationDomain::<Fr>::new(j, log_n);
    let n_ext = 1usize << domain.extended_k();

    // Random extended-Lagrange polynomial.
    let h_vec: Vec<Fr> = (0..n_ext).map(|_| Fr::random(OsRng)).collect();

    let host_in = Polynomial::<Fr, ExtendedLagrangeCoeff>::new(h_vec.clone());
    let host_out = domain
        .divide_by_vanishing_poly(host_in)
        .expect("host arm failed");

    let device_in: Polynomial<Fr, ExtendedLagrangeCoeff, Device> = {
        use openvm_cuda_common::copy::MemCopyH2D;
        let d_buf = h_vec
            .as_slice()
            .to_device_on(&halo2_axiom_gpu::cuda::utils::HALO2_GPU_CTX)
            .expect("H2D upload failed");
        Polynomial::<Fr, ExtendedLagrangeCoeff, Device>::from_device(d_buf)
    };
    let device_out = domain
        .divide_by_vanishing_poly_device(device_in)
        .expect("device arm failed");

    let device_out_host = device_out.to_host();
    let device_out_slice = device_out_host.values();
    let host_out_vec = host_out.values().to_vec();
    let host_out_slice = host_out_vec.as_slice();

    assert_eq!(
        device_out_slice.len(),
        host_out_slice.len(),
        "length mismatch at log_n={log_n}"
    );
    for (i, (h, d)) in host_out_slice
        .iter()
        .zip(device_out_slice.iter())
        .enumerate()
    {
        assert_eq!(
            h, d,
            "divide_by_vanishing host vs device disagree at log_n={log_n}, idx={i}"
        );
    }
}

#[test]
fn divide_by_vanishing_device_equivalence_log_n_20() {
    run_one(20);
}

#[test]
#[ignore = "large GPU allocation"]
fn divide_by_vanishing_device_equivalence_log_n_22() {
    run_one(22);
}

#[test]
#[ignore = "large GPU allocation"]
fn divide_by_vanishing_device_equivalence_log_n_23() {
    run_one(23);
}
