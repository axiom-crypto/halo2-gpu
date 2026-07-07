//! Device-out phase-5 batch eval equivalence.
//!
//! Oracle: the CPU `eval_polynomial` reference (the exact function the host
//! `PolyEvalAt` arm and the pre-M2 per-eval `batch_eval_polynomial_d2h`
//! consume). This test proves the new device-OUT batch eval
//! (`batch_eval_polynomial_device_out`, which computes every `(poly, point)`
//! evaluation into a `DeviceBuffer<F>` with NO per-eval `to_host_sync` — the
//! M2 change) reproduces it element-for-element, in the exact slot order the
//! `write_scalar` sequence consumes, on production-representative shapes and
//! the actual phase-5 eval points (cur, next, prev/inverse, -(blinding+1)).

use ff::Field;
use halo2_axiom_gpu::arithmetic::eval_polynomial;
use halo2_axiom_gpu::cuda::funcs::batch_eval_polynomial_device_out;
use halo2_axiom_gpu::cuda::utils::HALO2_GPU_CTX;
use halo2_axiom_gpu::poly::{EvaluationDomain, Rotation};
use halo2curves::bn256::Fr;
use openvm_cuda_common::copy::{cuda_memcpy_on, MemCopyH2D};
use openvm_cuda_common::d_buffer::DeviceBuffer;
use rand_core::OsRng;
use std::ffi::c_void;

/// Device-out batch eval, then ONE batched D2H (exactly how the rewired
/// phase-5 sites consume it). Asserts each slot equals the CPU reference.
fn check(polys: &[Vec<Fr>], points: &[Fr]) {
    assert_eq!(polys.len(), points.len());
    let n = polys.len();

    // Host oracle: the CPU eval_polynomial reference, one per (poly, point).
    let expected: Vec<Fr> = polys
        .iter()
        .zip(points.iter())
        .map(|(p, pt)| eval_polynomial(p, *pt))
        .collect();

    // Device-out: polys resident, evals land in a DeviceBuffer<F> (no sync).
    let d_polys_owned: Vec<DeviceBuffer<Fr>> = polys
        .iter()
        .map(|p| p.as_slice().to_device_on(&HALO2_GPU_CTX).expect("H2D poly"))
        .collect();
    let d_poly_refs: Vec<&DeviceBuffer<Fr>> = d_polys_owned.iter().collect();
    let d_out =
        batch_eval_polynomial_device_out(&d_poly_refs, points).expect("device-out batch eval");

    // ONE batched D2H of the whole result buffer.
    let mut actual = vec![Fr::ZERO; n];
    let bytes = n * std::mem::size_of::<Fr>();
    unsafe {
        cuda_memcpy_on::<true, false>(
            actual.as_mut_ptr() as *mut c_void,
            d_out.as_raw_ptr(),
            bytes,
            &HALO2_GPU_CTX,
        )
        .expect("batched D2H of eval buffer");
    }
    HALO2_GPU_CTX
        .stream
        .to_host_sync()
        .expect("stream sync after batched D2H");

    for i in 0..n {
        assert_eq!(
            actual[i], expected[i],
            "device-out eval mismatch at slot {i}/{n} (order must match write_scalar)"
        );
    }
}

fn rand_poly(len: usize) -> Vec<Fr> {
    (0..len).map(|_| Fr::random(OsRng)).collect()
}

/// The four phase-5 rotation flavours of a base challenge `x`.
fn phase5_points(k: u32, x: Fr) -> Vec<Fr> {
    let cpu_domain = halo2_axiom::poly::EvaluationDomain::<Fr>::new(1, k);
    let domain = EvaluationDomain::from_host_domain(&cpu_domain);
    let blinding = 5i32;
    vec![
        x,                                        // cur (instance/advice/fixed at Rotation::cur, product@x)
        domain.rotate_omega(x, Rotation::next()), // next (product_next)
        domain.rotate_omega(x, Rotation::prev()), // inverse (permuted_input_inv)
        domain.rotate_omega(x, Rotation(-(blinding + 1))), // last -(blinding+1) (permutation product_last)
    ]
}

#[test]
fn single_poly_single_point() {
    let k = 10;
    check(&[rand_poly(1 << k)], &[Fr::random(OsRng)]);
}

#[test]
fn many_distinct_polys_rotation_points() {
    // instance/advice/fixed-like: several distinct polys of the same size,
    // each evaluated at a (rotation-derived) point.
    let k = 12;
    let n = 8;
    let polys: Vec<Vec<Fr>> = (0..n).map(|_| rand_poly(1 << k)).collect();
    let x = Fr::random(OsRng);
    let rots = phase5_points(k, x);
    let points: Vec<Fr> = (0..n).map(|i| rots[i % rots.len()]).collect();
    check(&polys, &points);
}

#[test]
fn same_poly_multiple_points() {
    // permutation/lookup-like: ONE poly evaluated at cur/next/prev/last, so a
    // per-slot bug (wrong point or wrong destination slot) is caught.
    let k = 11;
    let poly = rand_poly(1 << k);
    let x = Fr::random(OsRng);
    let points = phase5_points(k, x);
    let polys: Vec<Vec<Fr>> = points.iter().map(|_| poly.clone()).collect();
    check(&polys, &points);
}

#[test]
fn mixed_poly_sizes_in_one_batch() {
    // Per-poly scratch sizing: different lengths in a single device-out call.
    let x = Fr::random(OsRng);
    let polys = vec![
        rand_poly(1 << 8),
        rand_poly(1 << 12),
        rand_poly(1 << 10),
        rand_poly(1 << 4),
    ];
    let cpu_domain_12 = halo2_axiom::poly::EvaluationDomain::<Fr>::new(1, 12);
    let domain_12 = EvaluationDomain::from_host_domain(&cpu_domain_12);
    let cpu_domain_10 = halo2_axiom::poly::EvaluationDomain::<Fr>::new(1, 10);
    let domain_10 = EvaluationDomain::from_host_domain(&cpu_domain_10);
    let points = vec![
        x,
        domain_12.rotate_omega(x, Rotation::next()),
        domain_10.rotate_omega(x, Rotation::prev()),
        Fr::random(OsRng),
    ];
    check(&polys, &points);
}

#[test]
fn ascending_mixed_poly_sizes_largest_last() {
    // Strictly ascending lengths with the LARGEST poly LAST. The single hoisted
    // scratch must be sized to the MAX workspace over all polys up front: a
    // buffer sized to the first (or any earlier) poly would be too small for
    // the later, larger one. Complements `mixed_poly_sizes_in_one_batch` (max
    // not last) and guards the max-len sizing + per-iteration `scratch_bytes`.
    let x = Fr::random(OsRng);
    let polys = vec![rand_poly(1 << 8), rand_poly(1 << 10), rand_poly(1 << 12)];
    let cpu_domain_10 = halo2_axiom::poly::EvaluationDomain::<Fr>::new(1, 10);
    let domain_10 = EvaluationDomain::from_host_domain(&cpu_domain_10);
    let cpu_domain_12 = halo2_axiom::poly::EvaluationDomain::<Fr>::new(1, 12);
    let domain_12 = EvaluationDomain::from_host_domain(&cpu_domain_12);
    let points = vec![
        x,
        domain_10.rotate_omega(x, Rotation::next()),
        domain_12.rotate_omega(x, Rotation::prev()),
    ];
    check(&polys, &points);
}

#[test]
fn production_size_batch() {
    // Production-representative "large" batch.
    let k = 16;
    let n = 6;
    let polys: Vec<Vec<Fr>> = (0..n).map(|_| rand_poly(1 << k)).collect();
    let x = Fr::random(OsRng);
    let rots = phase5_points(k, x);
    let points: Vec<Fr> = (0..n).map(|i| rots[i % rots.len()]).collect();
    check(&polys, &points);
}
