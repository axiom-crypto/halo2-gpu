//! This module provides common utilities, traits and structures for group,
//! field and polynomial arithmetic.

use crate::cuda::funcs::multiexp_gpu_many;
pub use ff::Field;
use ff::PrimeField;
use group::Group as _;
pub use halo2curves::{CurveAffine, CurveExt};

// GPU-neutral items re-exported from canonical halo2-axiom so host folds share
// one source of truth with downstream consumers.
pub use halo2_axiom::arithmetic::{
    best_fft, compute_inner_product, eval_polynomial, evaluate_vanishing_polynomial, g_to_lagrange,
    lagrange_interpolate, powers, FftGroup,
};

/// Mirrors `DENSE_POWER_DEGREE` in `halo2_proofs/cuda/include/kernel/omega.h`.
/// The GPU omega LUT layout assumes this value — changing one side without
/// the other produces silently wrong FFT twiddles.
pub const DENSE_POWER_DEGREE: u32 = 10;

// PLEASE UPDATE this whenever the max size inside the gpu kernel source changed.
// This constant is related to `n` in `cuda_prover_general.cu`
cfg_if::cfg_if!(
    if #[cfg(feature = "small")] {
        pub const GPU_MAX_MSM_SIZE: usize = 1 << 24;
        pub const GPU_MAX_MSM_LOG: usize = 24;
    } else {
        pub const GPU_MAX_MSM_SIZE: usize = 1 << 26;
        pub const GPU_MAX_MSM_LOG: usize = 26;
    }
);

/// Performs a small multi-exponentiation operation.
/// Uses the double-and-add algorithm with doublings shared across points.
pub fn small_multiexp<C: CurveAffine>(coeffs: &[C::Scalar], bases: &[C]) -> C::Curve {
    let coeffs: Vec<_> = coeffs.iter().map(|a| a.to_repr()).collect();
    let mut acc = C::Curve::identity();

    for byte_idx in (0..32).rev() {
        for bit_idx in (0..8).rev() {
            acc = acc.double();
            for coeff_idx in 0..coeffs.len() {
                let byte = coeffs[coeff_idx].as_ref()[byte_idx];
                if ((byte >> bit_idx) & 1) != 0 {
                    acc += bases[coeff_idx];
                }
            }
        }
    }

    acc
}

/// Performs a multi-exponentiation operation.
///
/// This function will panic if coeffs and bases have a different length.
///
/// This will use multithreading if beneficial.
pub fn best_multiexp<C: CurveAffine>(coeffs: &[C::Scalar], bases: &[C]) -> C::Curve {
    assert_eq!(coeffs.len(), bases.len());

    if bases.len() < (1 << 14) {
        return crate::cpu::arithmetic::best_multiexp_cpu(coeffs, bases);
    }

    multiexp_gpu_many(coeffs, bases).expect("multiexp_gpu_many failed in best_multiexp")
}

#[cfg(test)]
pub(crate) mod tests {
    use ff::BatchInvert;

    use super::*;
    use crate::cpu::arithmetic::parallelize;

    #[test]
    fn test_lagrange_interpolate() {
        use halo2curves::bn256::Fr as Fp;
        use rand_core::OsRng;

        let rng = OsRng;

        let points = (0..5).map(|_| Fp::random(rng)).collect::<Vec<_>>();
        let evals = (0..5).map(|_| Fp::random(rng)).collect::<Vec<_>>();

        for coeffs in 0..5 {
            let points = &points[0..coeffs];
            let evals = &evals[0..coeffs];

            let poly = lagrange_interpolate(points, evals);
            assert_eq!(poly.len(), points.len());

            for (point, eval) in points.iter().zip(evals) {
                assert_eq!(eval_polynomial(&poly, *point), *eval);
            }
        }
    }

    #[test]
    fn test_invert() {
        use std::time::Instant;

        use halo2curves::bn256::Fr;
        use rand_core::OsRng;

        let max_n = 26;
        let beta = Fr::random(OsRng);
        let gamma = Fr::random(OsRng);
        for n in 16..=max_n {
            println!("========= {} =========", n);
            let mut scalars: Vec<Fr> = (0..(1 << n))
                .map(|i| Fr::from((i + 1) as u64) * beta + gamma)
                .collect();
            let mut scalars_clone = scalars.clone();

            let batch_invert_time = Instant::now();
            scalars.batch_invert();
            let batch_invert_time = batch_invert_time.elapsed();
            println!("batch invert single thread took {:?}", batch_invert_time);

            let batch_invert_par_time = Instant::now();
            parallelize(&mut scalars_clone, |scalars, _| {
                scalars.batch_invert();
            });
            let batch_invert_par_time = batch_invert_par_time.elapsed();
            println!(
                "batch invert multiple thread took {:?}",
                batch_invert_par_time
            );

            println!(
                "speedup: {}",
                batch_invert_time.as_micros() as f64 / batch_invert_par_time.as_micros() as f64
            );
        }
    }
}
