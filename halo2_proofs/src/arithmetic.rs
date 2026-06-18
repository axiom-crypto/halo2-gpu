//! This module provides common utilities, traits and structures for group,
//! field and polynomial arithmetic.

use crate::cpu::arithmetic::parallelize;
use crate::{cuda::funcs::multiexp_gpu_many, fft::recursive::FFTData};
pub use ff::Field;
use ff::{BatchInvert, PrimeField};
use group::{prime::PrimeCurveAffine, Curve, Group as _, GroupOpsOwned, ScalarMulOwned};
pub use halo2curves::{CurveAffine, CurveExt};

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

/// This represents an element of a group with basic operations that can be
/// performed. This allows an FFT implementation (for example) to operate
/// generically over either a field or elliptic curve group.
pub trait FftGroup<Scalar: Field>:
    Copy + Send + Sync + 'static + GroupOpsOwned + ScalarMulOwned<Scalar>
{
}

impl<T, Scalar> FftGroup<Scalar> for T
where
    Scalar: Field,
    T: Copy + Send + Sync + 'static + GroupOpsOwned + ScalarMulOwned<Scalar>,
{
}

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

/// Dispatcher
pub fn best_fft<Scalar: Field, G: FftGroup<Scalar>>(
    a: &mut [G],
    omega: Scalar,
    log_n: u32,
    data: &FFTData<Scalar>,
    inverse: bool,
) {
    crate::fft::fft(a, omega, log_n, data, inverse);
}

/// Convert coefficient bases group elements to lagrange basis by inverse FFT.
pub fn g_to_lagrange<C: PrimeCurveAffine>(g_projective: Vec<C::Curve>, k: u32) -> Vec<C> {
    let n_inv = C::Scalar::TWO_INV.pow_vartime([k as u64, 0, 0, 0]);
    let omega = C::Scalar::ROOT_OF_UNITY;
    let mut omega_inv = C::Scalar::ROOT_OF_UNITY_INV;
    for _ in k..C::Scalar::S {
        omega_inv = omega_inv.square();
    }

    let mut g_lagrange_projective = g_projective;
    let n = g_lagrange_projective.len();
    let fft_data = FFTData::new(n, omega, omega_inv);

    best_fft(&mut g_lagrange_projective, omega_inv, k, &fft_data, true);
    parallelize(&mut g_lagrange_projective, |g, _| {
        for g in g.iter_mut() {
            *g *= n_inv;
        }
    });

    let mut g_lagrange = vec![C::identity(); 1 << k];
    parallelize(&mut g_lagrange, |g_lagrange, starts| {
        C::Curve::batch_normalize(
            &g_lagrange_projective[starts..(starts + g_lagrange.len())],
            g_lagrange,
        );
    });

    g_lagrange
}

/// This evaluates a provided polynomial (in coefficient form) at `point`.
pub fn eval_polynomial<F: Field>(poly: &[F], point: F) -> F {
    fn evaluate<F: Field>(poly: &[F], point: F) -> F {
        poly.iter()
            .rev()
            .fold(F::ZERO, |acc, coeff| acc * point + coeff)
    }
    let n = poly.len();
    let num_threads = rayon::current_num_threads();
    if n * 2 < num_threads {
        evaluate(poly, point)
    } else {
        let chunk_size = n.div_ceil(num_threads);
        let mut parts = vec![F::ZERO; num_threads];
        rayon::scope(|scope| {
            for (chunk_idx, (out, poly)) in
                parts.chunks_mut(1).zip(poly.chunks(chunk_size)).enumerate()
            {
                scope.spawn(move |_| {
                    let start = chunk_idx * chunk_size;
                    out[0] = evaluate(poly, point) * point.pow_vartime([start as u64, 0, 0, 0]);
                });
            }
        });
        parts.iter().fold(F::ZERO, |acc, coeff| acc + coeff)
    }
}

/// This computes the inner product of two vectors `a` and `b`.
///
/// This function will panic if the two vectors are not the same size.
pub fn compute_inner_product<F: Field>(a: &[F], b: &[F]) -> F {
    assert_eq!(a.len(), b.len());

    let mut acc = F::ZERO;
    for (a, b) in a.iter().zip(b.iter()) {
        acc += (*a) * (*b);
    }

    acc
}

/// Returns coefficients of an n - 1 degree polynomial given a set of n points
/// and their evaluations. This function will panic if two values in `points`
/// are the same.
pub fn lagrange_interpolate<F: Field>(points: &[F], evals: &[F]) -> Vec<F> {
    assert_eq!(points.len(), evals.len());
    if points.len() == 1 {
        // Constant polynomial
        vec![evals[0]]
    } else {
        let mut denoms = Vec::with_capacity(points.len());
        for (j, x_j) in points.iter().enumerate() {
            let mut denom = Vec::with_capacity(points.len() - 1);
            for x_k in points
                .iter()
                .enumerate()
                .filter(|&(k, _)| k != j)
                .map(|a| a.1)
            {
                denom.push(*x_j - x_k);
            }
            denoms.push(denom);
        }
        // Compute (x_j - x_k)^(-1) for each j != i
        denoms.iter_mut().flat_map(|v| v.iter_mut()).batch_invert();

        let mut final_poly = vec![F::ZERO; points.len()];
        for (j, (denoms, eval)) in denoms.into_iter().zip(evals.iter()).enumerate() {
            let mut tmp: Vec<F> = Vec::with_capacity(points.len());
            let mut product = Vec::with_capacity(points.len() - 1);
            tmp.push(F::ONE);
            for (x_k, denom) in points
                .iter()
                .enumerate()
                .filter(|&(k, _)| k != j)
                .map(|a| a.1)
                .zip(denoms)
            {
                product.resize(tmp.len() + 1, F::ZERO);
                for ((a, b), product) in tmp
                    .iter()
                    .chain(std::iter::once(&F::ZERO))
                    .zip(std::iter::once(&F::ZERO).chain(tmp.iter()))
                    .zip(product.iter_mut())
                {
                    *product = *a * (-denom * x_k) + *b * denom;
                }
                std::mem::swap(&mut tmp, &mut product);
            }
            assert_eq!(tmp.len(), points.len());
            assert_eq!(product.len(), points.len() - 1);
            for (final_coeff, interpolation_coeff) in final_poly.iter_mut().zip(tmp) {
                *final_coeff += interpolation_coeff * eval;
            }
        }
        final_poly
    }
}

pub(crate) fn evaluate_vanishing_polynomial<F: Field>(roots: &[F], z: F) -> F {
    fn evaluate<F: Field>(roots: &[F], z: F) -> F {
        roots.iter().fold(F::ONE, |acc, point| (z - point) * acc)
    }
    let n = roots.len();
    let num_threads = rayon::current_num_threads();
    if n * 2 < num_threads {
        evaluate(roots, z)
    } else {
        let chunk_size = n.div_ceil(num_threads);
        let mut parts = vec![F::ONE; num_threads];
        rayon::scope(|scope| {
            for (out, roots) in parts.chunks_mut(1).zip(roots.chunks(chunk_size)) {
                scope.spawn(move |_| out[0] = evaluate(roots, z));
            }
        });
        parts.iter().fold(F::ONE, |acc, part| acc * part)
    }
}

pub(crate) fn powers<F: Field>(base: F) -> impl Iterator<Item = F> {
    std::iter::successors(Some(F::ONE), move |power| Some(base * power))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

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
