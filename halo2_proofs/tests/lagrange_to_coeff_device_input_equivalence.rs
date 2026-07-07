//! Byte-identity gate for A1 (INT-8020): the lookup permuted-poly opening-prep
//! iFFT must produce identical coefficients whether it is routed through the
//! device-input primitive (`lagrange_to_coeff_device_input`, the new `Device`
//! arm at `plonk/prover.rs`) or through the host-input primitive
//! (`lagrange_to_coeff_device`, the retained `Host` fallback).
//!
//! Before A1, phase-5 opening prep always did
//!   `permuted_expr.into_host_polynomial()` (D2H) → `lagrange_to_coeff_device()`
//!   (H2D + iFFT)
//! even when the permuted poly was already `MaybeDevice::Device`. A1 routes the
//! `Device` arm straight into `lagrange_to_coeff_device_input` (device-in →
//! device-out iFFT, no PCIe round-trip). This test proves the two iFFT routes
//! agree element-for-element on the same source Lagrange values, so the routing
//! change is byte-identical.

use ff::Field;
use halo2_axiom_gpu::poly::{Device, DevicePolyExt, EvaluationDomain, LagrangeCoeff, Polynomial};
use halo2curves::bn256::Fr;
use openvm_cuda_common::copy::MemCopyH2D;
use rand_core::OsRng;

/// Assert that both iFFT routes on the same source Lagrange values produce
/// byte-identical coefficient polynomials.
fn run_one(log_n: u32) {
    // `j` (blowup factor) only affects the extended domain; the base-n iFFT
    // exercised here is independent of it.
    let j: u32 = 4;
    let cpu_domain = halo2_axiom::poly::EvaluationDomain::<Fr>::new(j, log_n);
    let domain = EvaluationDomain::from_host_domain(&cpu_domain);
    let n = 1usize << log_n;

    // One source of Lagrange values feeds both routes.
    let vals: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();

    // Host route: mirrors the pre-A1 path (`into_host_polynomial()` D2H is
    // value-preserving, so a host `Polynomial` built from the same values is
    // the faithful oracle) → `lagrange_to_coeff_device` (H2D + iFFT).
    let host_out = domain
        .lagrange_to_coeff_device(Polynomial::<Fr, LagrangeCoeff>::new(vals.clone()))
        .expect("host-input iFFT route failed");
    let host_out_host = host_out.to_host();
    let host_slice = host_out_host.values();

    // Device route: the new A1 `Device` arm — upload once, iFFT device-in →
    // device-out, no D2H→H2D round-trip.
    let device_in: Polynomial<Fr, LagrangeCoeff, Device> = {
        let d_buf = vals
            .as_slice()
            .to_device_on(&halo2_axiom_gpu::cuda::utils::HALO2_GPU_CTX)
            .expect("H2D upload of Lagrange values failed");
        Polynomial::<Fr, LagrangeCoeff, Device>::from_device(d_buf)
    };
    let device_out = domain
        .lagrange_to_coeff_device_input(device_in)
        .expect("device-input iFFT route failed");
    let device_out_host = device_out.to_host();
    let device_slice = device_out_host.values();

    assert_eq!(
        host_slice.len(),
        device_slice.len(),
        "coeff length mismatch at log_n={log_n}"
    );
    for (i, (h, d)) in host_slice.iter().zip(device_slice.iter()).enumerate() {
        assert_eq!(
            h, d,
            "device-input vs host-input iFFT disagree at log_n={log_n}, idx={i}"
        );
    }
}

#[test]
fn lagrange_to_coeff_device_input_equivalence_log_n_18() {
    run_one(18);
}

#[test]
fn lagrange_to_coeff_device_input_equivalence_log_n_20() {
    run_one(20);
}

#[test]
#[ignore = "large GPU allocation"]
fn lagrange_to_coeff_device_input_equivalence_log_n_22() {
    run_one(22);
}
