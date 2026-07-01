//! Equivalence test for `poly_multiply_add_device(d_acc, d_in, scalar)` vs
//! the CPU reference `acc[i] += scalar * in[i]`.

use ff::Field;
use halo2_axiom_gpu::cuda::funcs::poly_multiply_add_device;
use halo2_axiom_gpu::cuda::utils::HALO2_GPU_CTX;
use halo2curves::bn256::Fr;
use openvm_cuda_common::copy::{cuda_memcpy_on, MemCopyH2D};
use rand_core::OsRng;

fn run_one(log_n: u32) {
    let n: usize = 1usize << log_n;
    let acc_host: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();
    let in_host: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();
    let scalar = Fr::random(OsRng);

    let mut expected = acc_host.clone();
    for (e, b) in expected.iter_mut().zip(in_host.iter()) {
        *e += scalar * b;
    }

    let mut d_acc = acc_host
        .as_slice()
        .to_device_on(&HALO2_GPU_CTX)
        .expect("H2D d_acc for poly_multiply_add test");
    let d_in = in_host
        .as_slice()
        .to_device_on(&HALO2_GPU_CTX)
        .expect("H2D d_in for poly_multiply_add test");

    poly_multiply_add_device::<Fr>(&mut d_acc, &d_in, scalar)
        .expect("poly_multiply_add_device failed");

    let mut actual = vec![Fr::ZERO; n];
    let bytes = n * std::mem::size_of::<Fr>();
    unsafe {
        cuda_memcpy_on::<true, false>(
            actual.as_mut_ptr() as *mut libc::c_void,
            d_acc.as_raw_ptr(),
            bytes,
            &HALO2_GPU_CTX,
        )
        .expect("D2H for poly_multiply_add test");
    }
    HALO2_GPU_CTX.stream.to_host_sync().expect("stream sync after D2H");

    for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
        assert_eq!(e, a, "poly_multiply_add_device byte mismatch at log_n={log_n}, idx={i}");
    }
}

#[test]
fn poly_multiply_add_device_equivalence_log_n_20() {
    run_one(20);
}

#[test]
#[ignore = "large GPU allocation"]
fn poly_multiply_add_device_equivalence_log_n_22() {
    run_one(22);
}

#[test]
#[ignore = "large GPU allocation"]
fn poly_multiply_add_device_equivalence_log_n_23() {
    run_one(23);
}
