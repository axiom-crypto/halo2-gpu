//! Equivalence tests for device-backed extended-domain assembly.
//!
//! Host and device execution should produce identical flattened output.

use ff::Field;
use halo2_axiom_gpu::poly::{
    Device, DevicePolyExt, EvaluationDomain, LagrangeCoeff, MaybeDevice, Polynomial,
};
use halo2curves::bn256::Fr;
use rand_core::OsRng;

fn run_one(log_n: u32) {
    let j: u32 = 4;
    let domain = halo2_axiom::poly::EvaluationDomain::<Fr>::new(j, log_n);
    let domain = EvaluationDomain::from_host_domain(&domain);
    let n: usize = 1usize << log_n;
    let num_parts: usize = domain.extended_len() >> log_n;

    let mut parts_host: Vec<Vec<Fr>> =
        (0..num_parts).map(|_| (0..n).map(|_| Fr::random(OsRng)).collect::<Vec<Fr>>()).collect();

    let host_polys: Vec<Polynomial<Fr, LagrangeCoeff>> =
        parts_host.iter().cloned().map(Polynomial::<Fr, LagrangeCoeff>::new).collect();
    let host_out = domain.extended_from_lagrange_vec(host_polys);
    let host_out_vec = host_out.values().to_vec();

    use openvm_cuda_common::copy::MemCopyH2D;
    let device_polys: Vec<MaybeDevice<Fr, LagrangeCoeff>> = parts_host
        .drain(..)
        .map(|v| {
            let d_buf = v
                .as_slice()
                .to_device_on(&halo2_axiom_gpu::cuda::utils::HALO2_GPU_CTX)
                .expect("H2D upload failed");
            MaybeDevice::Device(Polynomial::<Fr, LagrangeCoeff, Device>::from_device(d_buf))
        })
        .collect();
    let device_out =
        domain.extended_from_lagrange_vec_device(device_polys).expect("device arm failed");
    let device_out_device = match device_out {
        MaybeDevice::Device(p) => p,
        MaybeDevice::Host(_) => {
            panic!("expected Device variant from extended_from_lagrange_vec_device")
        }
    };
    let device_out_host = device_out_device.to_host();
    let device_out_slice = device_out_host.values();

    assert_eq!(host_out_vec.len(), device_out_slice.len(), "length mismatch at log_n={log_n}");
    for (i, (h, d)) in host_out_vec.iter().zip(device_out_slice.iter()).enumerate() {
        assert_eq!(
            h, d,
            "extended_from_lagrange_vec host vs device disagree at log_n={log_n}, idx={i}"
        );
    }
}

#[test]
fn extended_from_lagrange_vec_device_equivalence_log_n_20() {
    run_one(20);
}

#[test]
#[ignore = "large GPU allocation"]
fn extended_from_lagrange_vec_device_equivalence_log_n_22() {
    run_one(22);
}

#[test]
#[ignore = "large GPU allocation"]
fn extended_from_lagrange_vec_device_equivalence_log_n_23() {
    run_one(23);
}
