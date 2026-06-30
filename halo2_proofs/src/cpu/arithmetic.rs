//! CPU counterparts of operations defined in `crate::arithmetic`.

use ff::{BatchInvert, Field, PrimeField};
use group::Group as _;
use halo2curves::CurveAffine;

use crate::multicore;

// GPU-neutral CPU helpers re-exported from canonical halo2-axiom so the GPU
// crate and downstream consumers share one source of truth.
pub use halo2_axiom::arithmetic::{bitreverse, kate_division, log2_floor, parallelize};

// ASSUMES C::Scalar::Repr is little endian
fn multiexp_serial<C: CurveAffine>(coeffs: &[C::Scalar], bases: &[C], acc: &mut C::Curve) {
    let coeffs: Vec<_> = coeffs.iter().map(|a| a.to_repr()).collect();

    let c = if bases.len() < 4 {
        1
    } else if bases.len() < 32 {
        3
    } else {
        (f64::from(bases.len() as u32)).ln().ceil() as usize
    };

    // Group `bytes` into bits and take the `segment`th chunk of `c` bits
    fn get_at<F: PrimeField>(segment: usize, c: usize, bytes: &F::Repr) -> usize {
        let skip_bits = segment * c;
        let skip_bytes = skip_bits / 8;

        if skip_bytes >= 32 {
            return 0;
        }

        let mut v = [0; 8];
        for (v, o) in v.iter_mut().zip(bytes.as_ref()[skip_bytes..].iter()) {
            *v = *o;
        }

        let mut tmp = u64::from_le_bytes(v);
        tmp >>= skip_bits - (skip_bytes * 8);
        tmp %= 1 << c;

        tmp as usize
    }

    let segments = (C::Scalar::NUM_BITS as usize).div_ceil(c);

    // this can be optimized
    let mut coeffs_in_segments = Vec::with_capacity(segments);
    // track what is the last segment where we actually have nonzero bits, so we completely skip buckets where the scalar bits for all coeffs are 0
    let mut max_nonzero_segment = None;
    for current_segment in 0..segments {
        let coeff_segments: Vec<_> = coeffs
            .iter()
            .map(|coeff| {
                let c_bits = get_at::<C::Scalar>(current_segment, c, coeff);
                if c_bits != 0 {
                    max_nonzero_segment = Some(current_segment);
                }
                c_bits
            })
            .collect();
        coeffs_in_segments.push(coeff_segments);
    }

    if max_nonzero_segment.is_none() {
        return;
    }
    for coeffs_seg in coeffs_in_segments.into_iter().take(max_nonzero_segment.unwrap() + 1).rev() {
        for _ in 0..c {
            *acc = acc.double();
        }

        #[derive(Clone, Copy)]
        enum Bucket<C: CurveAffine> {
            None,
            Affine(C),
            Projective(C::Curve),
        }

        impl<C: CurveAffine> Bucket<C> {
            fn add_assign(&mut self, other: &C) {
                *self = match *self {
                    Bucket::None => Bucket::Affine(*other),
                    Bucket::Affine(a) => Bucket::Projective(a + *other),
                    Bucket::Projective(mut a) => {
                        a += *other;
                        Bucket::Projective(a)
                    }
                }
            }

            fn add(self, mut other: C::Curve) -> C::Curve {
                match self {
                    Bucket::None => other,
                    Bucket::Affine(a) => {
                        other += a;
                        other
                    }
                    Bucket::Projective(a) => other + &a,
                }
            }
        }

        let mut buckets: Vec<Bucket<C>> = vec![Bucket::None; (1 << c) - 1];

        let mut max_bits = 0;
        for (coeff, base) in coeffs_seg.into_iter().zip(bases.iter()) {
            if coeff != 0 {
                max_bits = std::cmp::max(max_bits, coeff);
                buckets[coeff - 1].add_assign(base);
            }
        }

        // Summation by parts
        // e.g. 3a + 2b + 1c = a +
        //                    (a) + b +
        //                    ((a) + b) + c
        let mut running_sum = C::Curve::identity();
        for exp in buckets.into_iter().take(max_bits).rev() {
            running_sum = exp.add(running_sum);
            *acc += &running_sum;
        }
    }
}

pub fn best_multiexp_cpu<C: CurveAffine>(coeffs: &[C::Scalar], bases: &[C]) -> C::Curve {
    let num_threads = multicore::current_num_threads();
    if coeffs.len() > num_threads {
        let chunk = coeffs.len() / num_threads;
        let num_chunks = coeffs.chunks(chunk).len();
        let mut results = vec![C::Curve::identity(); num_chunks];
        multicore::scope(|scope| {
            let chunk = coeffs.len() / num_threads;

            for ((coeffs, bases), acc) in
                coeffs.chunks(chunk).zip(bases.chunks(chunk)).zip(results.iter_mut())
            {
                scope.spawn(move |_| {
                    multiexp_serial(coeffs, bases, acc);
                });
            }
        });
        results.iter().fold(C::Curve::identity(), |a, b| a + b)
    } else {
        let mut acc = C::Curve::identity();
        multiexp_serial(coeffs, bases, &mut acc);
        acc
    }
}

pub fn lookup_product_cpu<F: Field>(
    lookup_product: &mut [F],
    permuted_input: &[F],
    permuted_table: &[F],
    compressed_input: &[F],
    compressed_table: &[F],
    beta: F,
    gamma: F,
) {
    // Denominator uses the permuted input expression and permuted table expression
    parallelize(lookup_product, |lookup_product, start| {
        for ((lookup_product, permuted_input_value), permuted_table_value) in lookup_product
            .iter_mut()
            .zip(permuted_input[start..].iter())
            .zip(permuted_table[start..].iter())
        {
            *lookup_product = (beta + permuted_input_value) * &(gamma + permuted_table_value);
        }
    });

    // Batch invert to obtain the denominators for the lookup product
    // polynomials
    if lookup_product.len() >= (1 << 19) {
        parallelize(lookup_product, |lookup_product, _| {
            lookup_product.iter_mut().batch_invert();
        });
    } else {
        lookup_product.batch_invert();
    }

    // Finish the computation of the entire fraction by computing the numerators
    parallelize(lookup_product, |product, start| {
        for (i, product) in product.iter_mut().enumerate() {
            let i = i + start;

            *product *= &(compressed_input[i] + &beta);
            *product *= &(compressed_table[i] + &gamma);
        }
    });
}

pub fn generate_omega_lut_cpu<F: Field>(omega: F, log_n: u32, dense_degree: u32) -> Vec<F> {
    let mut low_degree_dense_lut = vec![F::ZERO; (1 << dense_degree) as usize];
    low_degree_dense_lut.iter_mut().enumerate().for_each(|(i, v)| {
        *v = omega.pow_vartime([i as u64]);
    });

    let high_degree_lut_len = 1 << (log_n - dense_degree);
    let sparse_omega_start = omega.pow_vartime([(1 << dense_degree) as u64, 0, 0, 0]);
    let mut high_degree_sparse_lut = vec![F::ZERO; high_degree_lut_len as usize];
    high_degree_sparse_lut.iter_mut().enumerate().for_each(|(i, v)| {
        *v = sparse_omega_start.pow_vartime([i as u64]);
    });

    low_degree_dense_lut.extend(high_degree_sparse_lut);
    low_degree_dense_lut
}

#[cfg(test)]
pub(crate) fn quotient_lookups_cpu<F: Field>(
    values: &mut [F],
    table_values: &[F],
    product_coset: &[F],
    permuted_input_coset: &[F],
    permuted_table_coset: &[F],
    l0: &[F],
    l_last: &[F],
    l_active_row: &[F],
    beta: F,
    gamma: F,
    y: F,
    isize: usize,
) {
    /// Return the index in the polynomial of size `isize` after rotation `rot`.
    fn get_rotation_idx(idx: usize, rot: i32, rot_scale: i32, isize: i32) -> usize {
        (((idx as i32) + (rot * rot_scale)).rem_euclid(isize)) as usize
    }

    let one = F::ONE;
    parallelize(values, |values, start| {
        for (i, value) in values.iter_mut().enumerate() {
            let idx = start + i;
            let r_next = get_rotation_idx(idx, 1, 1, isize as i32);
            let r_prev = get_rotation_idx(idx, -1, 1, isize as i32);

            let a_minus_s = permuted_input_coset[idx] - permuted_table_coset[idx];
            // l_0(X) * (1 - z(X)) = 0
            *value = *value * y + ((one - product_coset[idx]) * l0[idx]);
            // l_last(X) * (z(X)^2 - z(X)) = 0
            *value = *value * y
                + ((product_coset[idx] * product_coset[idx] - product_coset[idx]) * l_last[idx]);
            // (1 - (l_last(X) + l_blind(X))) * (
            //   z(\omega X) (a'(X) + \beta) (s'(X) + \gamma)
            //   - z(X) (\theta^{m-1} a_0(X) + ... + a_{m-1}(X) + \beta)
            //          (\theta^{m-1} s_0(X) + ... + s_{m-1}(X) + \gamma)
            // ) = 0
            *value = *value * y
                + ((product_coset[r_next]
                    * (permuted_input_coset[idx] + beta)
                    * (permuted_table_coset[idx] + gamma)
                    - product_coset[idx] * table_values[idx])
                    * l_active_row[idx]);
            // Check that the first values in the permuted input expression and permuted
            // fixed expression are the same.
            // l_0(X) * (a'(X) - s'(X)) = 0
            *value = *value * y + (a_minus_s * l0[idx]);
            // Check that each value in the permuted lookup input expression is either
            // equal to the value above it, or the value at the same index in the
            // permuted table expression.
            // (1 - (l_last + l_blind)) * (a′(X) − s′(X))⋅(a′(X) − a′(\omega^{-1} X)) = 0
            *value = *value * y
                + (a_minus_s
                    * (permuted_input_coset[idx] - permuted_input_coset[r_prev])
                    * l_active_row[idx]);
        }
    });
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    // Test-only CPU baseline used by the kernel-level micro-tests in `cuda::tests`.
    pub(crate) fn permutation_product_cpu<F: Field>(
        modified_values: &mut [F],
        values: &[Vec<F>],
        permutations: &[Vec<F>],
        omega: F,
        beta: F,
        gamma: F,
        delta: F,
        deltaomega: F,
    ) {
        // denominator
        for (values, permuted_values) in values.iter().zip(permutations.iter()) {
            parallelize(modified_values, |modified_values, start| {
                for ((modified_values, value), permuted_value) in modified_values
                    .iter_mut()
                    .zip(values[start..].iter())
                    .zip(permuted_values[start..].iter())
                {
                    *modified_values *= &(beta * permuted_value + gamma + value);
                }
            });
        }
        // invert
        parallelize(modified_values, |modified_values, _| {
            modified_values.batch_invert();
        });
        // numerator
        let mut deltaomega = deltaomega;
        for values in values.iter() {
            parallelize(modified_values, |modified_values, start| {
                let mut _deltaomega = deltaomega * &omega.pow_vartime([start as u64, 0, 0, 0]);
                for (modified_values, value) in
                    modified_values.iter_mut().zip(values[start..].iter())
                {
                    *modified_values *= &(_deltaomega * beta + gamma + value);
                    _deltaomega *= &omega;
                }
            });
            deltaomega *= &delta;
        }
    }
}
