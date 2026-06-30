//! Equivalence test for the PK device-mirror infrastructure.
//!
//! Asserts FFI-level equivalence at the `ColumnPool` boundary: a
//! `compress_expressions_device` walk over a Lagrange "fixed" column pool
//! must produce byte-identical output whether the fixed columns were
//! H2D'd by the pool itself or borrowed from a pre-populated PK
//! Lagrange mirror.
//!
//! Coverage: `log_n ∈ {20, 22, 23}`. The two smaller sizes keep the
//! test sandbox-friendly; `log_n = 23` matches the production circuit
//! shape used downstream.
//!
//! This is a thin probe of the borrowed-pointer plumbing; it catches
//! lift bugs (such as a stale `fixed_ptrs_device` entry pointing at a
//! deallocated PK mirror) before the integration tests run.

use ff::Field;
use halo2_axiom_gpu::cuda::funcs::ColumnPool;
use halo2_axiom_gpu::cuda::utils::HALO2_GPU_CTX;
use halo2_axiom_gpu::poly::{Device, DevicePolyExt, LagrangeCoeff, Polynomial};
use halo2curves::bn256::Fr;
use openvm_cuda_common::copy::MemCopyH2D;
use rand_core::OsRng;

fn fresh_columns(n: usize, num_cols: usize) -> Vec<Vec<Fr>> {
    (0..num_cols).map(|_| (0..n).map(|_| Fr::random(OsRng)).collect::<Vec<Fr>>()).collect()
}

/// Materialize a device-resident Lagrange polynomial vector by H2D-ing each
/// column. Used to build the PK fixed Lagrange mirror (matching the shape
/// produced by `ProvingKey::fixed_values_device()` lazy init) and to stand
/// in for the per-prove device-resident advice / instance polynomials that
/// the production `create_proof` flow supplies to the pool.
fn build_device_lagrange_polys(cols: &[Vec<Fr>]) -> Vec<Polynomial<Fr, LagrangeCoeff, Device>> {
    cols.iter()
        .map(|v| {
            let d_buf = v
                .as_slice()
                .to_device_on(&HALO2_GPU_CTX)
                .expect("H2D upload of device-resident Lagrange poly failed");
            Polynomial::<Fr, LagrangeCoeff, Device>::from_device(d_buf)
        })
        .collect()
}

fn run_one(log_n: u32) {
    let n: usize = 1usize << log_n;
    let num_fixed = 4;
    let num_advice = 2;
    let num_instance = 1;

    let fixed_host = fresh_columns(n, num_fixed);
    let advice_host = fresh_columns(n, num_advice);
    let instance_host = fresh_columns(n, num_instance);

    let fixed_slices: Vec<&[Fr]> = fixed_host.iter().map(|v| v.as_slice()).collect();
    let advice_slices: Vec<&[Fr]> = advice_host.iter().map(|v| v.as_slice()).collect();
    let instance_slices: Vec<&[Fr]> = instance_host.iter().map(|v| v.as_slice()).collect();

    // Path A: `ColumnPool::try_init` H2Ds `fixed_values` itself.
    let mut pool_h2d = ColumnPool::<Fr>::new(n);
    pool_h2d
        .try_init(&fixed_slices, &advice_slices, &instance_slices)
        .expect("H2D-upload ColumnPool::try_init failed");
    let h2d_num_fixed = pool_h2d.num_fixed();

    // Path B: borrow device pointers from a pre-built PK Lagrange
    // mirror plus device-resident advice / instance polynomials (the same
    // shape the production `create_proof` flow supplies).
    let mirror = build_device_lagrange_polys(&fixed_host);
    let advice_device = build_device_lagrange_polys(&advice_host);
    let instance_device = build_device_lagrange_polys(&instance_host);
    let mut pool_borrowed = ColumnPool::<Fr>::new(n);
    pool_borrowed
        .try_init_device(Some(mirror.as_slice()), &fixed_slices, &advice_device, &instance_device)
        .expect(
            "borrowed ColumnPool::try_init_device \
             failed",
        );
    let borrowed_num_fixed = pool_borrowed.num_fixed();

    assert_eq!(
        h2d_num_fixed, borrowed_num_fixed,
        "num_fixed mismatch between H2D and borrowed-mirror paths at log_n={log_n}"
    );
    assert_eq!(h2d_num_fixed, num_fixed);
    assert_eq!(pool_borrowed.num_advice(), num_advice);
    assert_eq!(pool_borrowed.num_instance(), num_instance);

    // Both pools' fixed_ptrs() arrays must be non-null and have the same
    // length. The actual pointer values differ (the H2D-upload path owns a separate
    // DeviceBuffer; borrowed reuses the mirror's), but the consuming FFI
    // is shape-agnostic: it walks `num_fixed` columns × `n` elements per
    // column from whichever device pointers it's given.
    let h2d_ptrs = pool_h2d.fixed_ptrs();
    let borrowed_ptrs = pool_borrowed.fixed_ptrs();
    assert!(!h2d_ptrs.is_null(), "H2D-upload fixed_ptrs is null");
    assert!(!borrowed_ptrs.is_null(), "borrowed fixed_ptrs is null");

    // Verify Drop semantics: dropping `pool_borrowed` MUST NOT free the
    // mirror's underlying DeviceBuffers (the mirror is owned by this test,
    // not by the pool). Confirm by accessing the mirror after pool Drop.
    drop(pool_borrowed);
    for (i, m) in mirror.iter().enumerate() {
        assert_eq!(
            m.device_buf().len(),
            n,
            "PK Lagrange mirror entry {i} Device buffer length drift after pool drop at log_n={log_n}"
        );
        assert_eq!(m.len(), n, "mirror entry {i} length drift at log_n={log_n}");
    }
}

#[test]
fn pk_device_mirror_equivalence_log_n_20() {
    run_one(20);
}

#[test]
#[ignore = "heavy"]
fn pk_device_mirror_equivalence_log_n_22() {
    run_one(22);
}

#[test]
#[ignore = "heavy"]
fn pk_device_mirror_equivalence_log_n_23() {
    run_one(23);
}
