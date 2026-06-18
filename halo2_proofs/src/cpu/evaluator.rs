use ff::{Field, WithSmallOrderMulGroup};

use crate::cuda::HaloGpuError;
use crate::multicore;

use super::arithmetic::quotient_lookups_cpu;
use crate::cpu::arithmetic::parallelize;
use crate::plonk::evaluation::{get_rotation_idx, Evaluator, EvaluatorVkView};
use crate::plonk::{lookup, permutation, Any};
use crate::{
    arithmetic::{best_fft, CurveAffine},
    poly::{
        Coeff, DevicePolyExt, EvaluationDomain, ExtendedLagrangeCoeff, LagrangeCoeff, Polynomial,
        Rotation,
    },
};

/// CPU equivalent of [`EvaluationDomain::coeff_to_extended_part`].
///
/// For input coefficient-form polynomial `a` (length `n = 1 << k`), this
/// computes `FFT(a(g_coset * extended_omega_factor * X), n)` — i.e. one part
/// of the extended evaluation domain. The algebra mirrors the CUDA
/// `CosetFFT_Part` kernel (`distribute_powers(coset_shift)` then forward FFT
/// with `omega`); see the equivalence test in
/// `cuda::tests::test_fft_normal_to_device_coset_part_vs_cpu`.
fn coeff_to_extended_part_cpu<F: WithSmallOrderMulGroup<3>>(
    domain: &EvaluationDomain<F>,
    input: &Polynomial<F, Coeff>,
    extended_omega_factor: F,
) -> Polynomial<F, LagrangeCoeff> {
    let log_n = domain.k();
    let n = 1usize << log_n;
    let omega = domain.get_omega();
    let coset_shift = domain.g_coset * extended_omega_factor;
    let fft_data = domain.get_fft_data(n);

    let mut a = input.values().to_vec();
    let mut c_power = F::ONE;
    for v in a.iter_mut() {
        *v *= c_power;
        c_power *= coset_shift;
    }
    best_fft(&mut a, omega, log_n, fft_data, false);
    Polynomial::new(a)
}

/// Batched CPU equivalent of [`EvaluationDomain::coeff_to_extended_part_many_device`].
/// See [`coeff_to_extended_part_cpu`] for the per-poly semantics.
fn coeff_to_extended_part_many_cpu<F: WithSmallOrderMulGroup<3>>(
    domain: &EvaluationDomain<F>,
    inputs: Vec<&Polynomial<F, Coeff>>,
    extended_omega_factor: F,
) -> Vec<Polynomial<F, LagrangeCoeff>> {
    inputs
        .into_iter()
        .map(|poly| coeff_to_extended_part_cpu(domain, poly, extended_omega_factor))
        .collect()
}

/// Add the permutation-argument quotient contribution to a single contiguous
/// chunk of `values` over the extended evaluation domain.
#[allow(clippy::too_many_arguments)]
pub fn permutation_quotient_cpu_chunk<F: WithSmallOrderMulGroup<3>>(
    values: &mut [F],
    start: usize,
    rot_scale: i32,
    isize_: i32,
    last_rotation: i32,
    chunk_len: usize,
    column_values: &[&[F]],
    permutation_product_cosets: &[&[F]],
    permutation_cosets: &[&[F]],
    l0: &[F],
    l_last: &[F],
    l_active_row: &[F],
    y: F,
    beta: F,
    gamma: F,
    delta_start: F,
    current_extended_omega: F,
    omega: F,
) {
    debug_assert!(!permutation_product_cosets.is_empty());
    debug_assert_eq!(column_values.len(), permutation_cosets.len());
    // The last set may be partial when `column_values.len() % chunk_len != 0`
    // — the CPU loop body handles this naturally via `chunks(chunk_len)`. The
    // bounds say: every set has 1..=chunk_len columns, all sets together
    // cover exactly `column_values.len()` columns.
    debug_assert!(column_values.len() <= permutation_product_cosets.len() * chunk_len);
    debug_assert!(column_values.len() + chunk_len > permutation_product_cosets.len() * chunk_len);

    let one = F::ONE;
    let first_set_permutation_product_coset = permutation_product_cosets.first().unwrap();
    let last_set_permutation_product_coset = permutation_product_cosets.last().unwrap();

    let mut beta_term = current_extended_omega * omega.pow_vartime([start as u64, 0, 0, 0]);
    for (i, value) in values.iter_mut().enumerate() {
        let idx = start + i;
        let r_next = get_rotation_idx(idx, 1, rot_scale, isize_);
        let r_last = get_rotation_idx(idx, last_rotation, rot_scale, isize_);

        // Enforce only for the first set.
        // l_0(X) * (1 - z_0(X)) = 0
        *value = *value * y + ((one - first_set_permutation_product_coset[idx]) * l0[idx]);
        // Enforce only for the last set.
        // l_last(X) * (z_l(X)^2 - z_l(X)) = 0
        *value = *value * y
            + ((last_set_permutation_product_coset[idx] * last_set_permutation_product_coset[idx]
                - last_set_permutation_product_coset[idx])
                * l_last[idx]);
        // Except for the first set, enforce.
        // l_0(X) * (z_i(X) - z_{i-1}(\omega^(last) X)) = 0
        for (set_idx, permutation_product_coset) in permutation_product_cosets.iter().enumerate() {
            if set_idx != 0 {
                *value = *value * y
                    + ((permutation_product_coset[idx]
                        - permutation_product_cosets[set_idx - 1][r_last])
                        * l0[idx]);
            }
        }
        // And for all the sets we enforce:
        // (1 - (l_last(X) + l_blind(X))) * (
        //   z_i(\omega X) \prod_j (p(X) + \beta s_j(X) + \gamma)
        // - z_i(X) \prod_j (p(X) + \delta^j \beta X + \gamma)
        // )
        let mut current_delta = delta_start * beta_term;
        for ((cols_chunk, permutation_product_coset), permutation_coset_chunk) in column_values
            .chunks(chunk_len)
            .zip(permutation_product_cosets.iter())
            .zip(permutation_cosets.chunks(chunk_len))
        {
            let mut left = permutation_product_coset[r_next];
            for (col_vals, permutation) in cols_chunk.iter().zip(permutation_coset_chunk.iter()) {
                left *= col_vals[idx] + beta * permutation[idx] + gamma;
            }

            let mut right = permutation_product_coset[idx];
            for col_vals in cols_chunk.iter() {
                right *= col_vals[idx] + current_delta + gamma;
                current_delta *= &F::DELTA;
            }

            *value = *value * y + ((left - right) * l_active_row[idx]);
        }
        beta_term *= &omega;
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_h_inner<C: CurveAffine>(
    evaluator: &Evaluator<C>,
    view: &EvaluatorVkView<'_, C::ScalarExt>,
    pk_l0: &Polynomial<C::ScalarExt, Coeff>,
    pk_l_last: &Polynomial<C::ScalarExt, Coeff>,
    pk_l_active_row: &Polynomial<C::ScalarExt, Coeff>,
    _pk_fixed_values: &[Polynomial<C::ScalarExt, LagrangeCoeff>],
    pk_permutation_polys: &[Polynomial<C::ScalarExt, Coeff>],
    pk_fixed_polys: &[Polynomial<C::ScalarExt, Coeff>],
    advice_polys: &[&[Polynomial<C::ScalarExt, Coeff>]],
    instance_polys: &[&[Polynomial<C::ScalarExt, Coeff>]],
    challenges: &[C::ScalarExt],
    y: C::ScalarExt,
    beta: C::ScalarExt,
    gamma: C::ScalarExt,
    theta: C::ScalarExt,
    lookups: &[Vec<lookup::prover::Committed<C>>],
    permutations: &[permutation::prover::Committed<C>],
) -> Result<crate::poly::Polynomial<C::ScalarExt, ExtendedLagrangeCoeff>, crate::plonk::Error>
where
    C::ScalarExt: WithSmallOrderMulGroup<3>,
{
    crate::perf_section!("evaluate_h");
    let domain = view.domain;
    let size: usize = 1 << domain.k() as usize;
    let rot_scale = 1;
    let extended_omega = domain.get_extended_omega();
    let omega = domain.get_omega();
    let isize = size as i32;
    let one = C::ScalarExt::ONE;
    let p = view.permutation_argument;
    let num_parts = domain.extended_len() >> domain.k();

    // Calculate the quotient polynomial for each part
    let mut current_extended_omega = one;
    let value_parts: Vec<crate::poly::Polynomial<C::ScalarExt, LagrangeCoeff>> = (0..num_parts)
        .map(
            |_i| -> Result<
                crate::poly::Polynomial<C::ScalarExt, LagrangeCoeff>,
                crate::plonk::Error,
            > {
                // Pure-CPU per-part cosetFFT, one logical group at a time.
                // Outputs are host-resident so every downstream consumer
                // (custom gates, permutation chunk, lookups) just borrows.
                let fixed: &[Polynomial<C::ScalarExt, LagrangeCoeff>] =
                    &coeff_to_extended_part_many_cpu(
                        domain,
                        pk_fixed_polys.iter().collect(),
                        current_extended_omega,
                    );
                let l0 = coeff_to_extended_part_cpu(domain, pk_l0, current_extended_omega);
                let l_last = coeff_to_extended_part_cpu(domain, pk_l_last, current_extended_omega);
                let l_active =
                    coeff_to_extended_part_cpu(domain, pk_l_active_row, current_extended_omega);

                let advice_parts: Vec<Vec<Polynomial<C::ScalarExt, LagrangeCoeff>>> = advice_polys
                    .iter()
                    .map(|polys| {
                        coeff_to_extended_part_many_cpu(
                            domain,
                            polys.iter().collect(),
                            current_extended_omega,
                        )
                    })
                    .collect();
                let instance_parts: Vec<Vec<Polynomial<C::ScalarExt, LagrangeCoeff>>> =
                    instance_polys
                        .iter()
                        .map(|polys| {
                            coeff_to_extended_part_many_cpu(
                                domain,
                                polys.iter().collect(),
                                current_extended_omega,
                            )
                        })
                        .collect();

                let mut values = domain.empty_lagrange();

                // Core expression evaluations
                let num_threads = multicore::current_num_threads();
                for (((advice, instance), lookups), permutation) in advice_parts
                    .iter()
                    .zip(instance_parts.iter())
                    .zip(lookups.iter())
                    .zip(permutations.iter())
                {
                    let advice: &[Polynomial<C::Scalar, LagrangeCoeff>] = advice;
                    let instance: &[Polynomial<C::Scalar, LagrangeCoeff>] = instance;

                    // Custom gates
                    multicore::scope(|scope| -> Result<(), HaloGpuError> {
                        let chunk_size = size.div_ceil(num_threads);
                        for (thread_idx, values) in values.chunks_mut(chunk_size).enumerate() {
                            let start = thread_idx * chunk_size;
                            scope.spawn(move |_| {
                                let mut eval_data = evaluator.custom_gates.instance();
                                for (i, value) in values.iter_mut().enumerate() {
                                    let idx = start + i;
                                    *value = evaluator.custom_gates.evaluate(
                                        &mut eval_data,
                                        fixed,
                                        advice,
                                        instance,
                                        challenges,
                                        &beta,
                                        &gamma,
                                        &theta,
                                        &y,
                                        value,
                                        idx,
                                        rot_scale,
                                        isize,
                                    );
                                }
                            });
                        }
                        Ok(())
                    })?;

                    // Permutations
                    let sets = &permutation.sets;
                    if !sets.is_empty() {
                        let blinding_factors = view.blinding_factors;
                        let last_rotation = Rotation(-((blinding_factors + 1) as i32));
                        let chunk_len = view.cs_degree - 2;
                        let delta_start = beta * &C::Scalar::ZETA;

                        // permutation_product_poly is device-resident on the
                        // input; the D2H here is unavoidable for the CPU path.
                        let permutation_product_polys_host: Vec<Polynomial<C::Scalar, Coeff>> =
                            sets.iter()
                                .map(|set| set.permutation_product_poly.to_host())
                                .collect();

                        let column_values_cold: Vec<&[C::Scalar]> = p
                            .columns
                            .iter()
                            .map(|column| match column.column_type() {
                                Any::Advice(_) => advice[column.index()].values(),
                                Any::Fixed => fixed[column.index()].values(),
                                Any::Instance => instance[column.index()].values(),
                            })
                            .collect();

                        let permutation_product_cosets = coeff_to_extended_part_many_cpu(
                            domain,
                            permutation_product_polys_host.iter().collect(),
                            current_extended_omega,
                        );
                        let permutation_cosets = coeff_to_extended_part_many_cpu(
                            domain,
                            pk_permutation_polys.iter().collect(),
                            current_extended_omega,
                        );

                        let permutation_product_coset_slices: Vec<&[C::Scalar]> =
                            permutation_product_cosets
                                .iter()
                                .map(|p| p.values())
                                .collect();
                        let permutation_coset_slices: Vec<&[C::Scalar]> =
                            permutation_cosets.iter().map(|p| p.values()).collect();

                        parallelize(&mut values, |values, start| {
                            permutation_quotient_cpu_chunk(
                                values,
                                start,
                                rot_scale,
                                isize,
                                last_rotation.0,
                                chunk_len,
                                &column_values_cold,
                                &permutation_product_coset_slices,
                                &permutation_coset_slices,
                                l0.values(),
                                l_last.values(),
                                l_active.values(),
                                y,
                                beta,
                                gamma,
                                delta_start,
                                current_extended_omega,
                                omega,
                            );
                        });
                    }

                    // Lookups
                    let mut table_values = vec![C::ScalarExt::ZERO; size];
                    for (n, lookup) in lookups.iter().enumerate() {
                        parallelize(&mut table_values, |table_values, start| {
                            let lookup_evaluator = &evaluator.lookups[n];
                            let mut eval_data = lookup_evaluator.instance();
                            for (i, table_value) in table_values.iter_mut().enumerate() {
                                let idx = start + i;

                                *table_value = lookup_evaluator.evaluate(
                                    &mut eval_data,
                                    fixed,
                                    advice,
                                    instance,
                                    challenges,
                                    &beta,
                                    &gamma,
                                    &theta,
                                    &y,
                                    &C::ScalarExt::ZERO,
                                    idx,
                                    rot_scale,
                                    isize,
                                );
                            }
                        });

                        let permuted_input_host =
                            lookup.permuted_input_expression.to_host_polynomial();
                        let permuted_table_host =
                            lookup.permuted_table_expression.to_host_polynomial();
                        let permuted_input_coset = domain.lagrange_to_extend_part(
                            &permuted_input_host,
                            current_extended_omega,
                        )?;
                        let permuted_table_coset = domain.lagrange_to_extend_part(
                            &permuted_table_host,
                            current_extended_omega,
                        )?;
                        // product_poly is device-resident on the input;
                        // D2H once, then CPU coset transform.
                        let product_poly_host = lookup.product_poly.to_host();
                        let product_coset = coeff_to_extended_part_cpu(
                            domain,
                            &product_poly_host,
                            current_extended_omega,
                        );

                        quotient_lookups_cpu(
                            &mut values,
                            &table_values,
                            &product_coset,
                            &permuted_input_coset,
                            &permuted_table_coset,
                            l0.values(),
                            l_last.values(),
                            l_active.values(),
                            beta,
                            gamma,
                            y,
                            isize as usize,
                        );
                    }

                    current_extended_omega *= extended_omega;
                }

                Ok(values)
            },
        )
        .collect::<Result<Vec<_>, _>>()?;

    Ok(domain.extended_from_lagrange_vec(value_parts))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::utils::HALO2_GPU_CTX;
    use halo2curves::bn256::G1Affine;
    use halo2curves::CurveAffine;
    use openvm_cuda_common::copy::MemCopyD2H;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    type Fr = <G1Affine as CurveAffine>::ScalarExt;

    #[test]
    fn coeff_to_extended_part_many_cpu_matches_gpu() {
        let k = 8u32;
        let j = 4u32;
        let n = 1usize << k;
        let n_polys = 5;

        let domain = EvaluationDomain::<Fr>::new(j, k);
        let mut rng = ChaCha20Rng::seed_from_u64(0xC0FFEE);

        let host_polys: Vec<Polynomial<Fr, Coeff>> = (0..n_polys)
            .map(|_| Polynomial::new((0..n).map(|_| Fr::random(&mut rng)).collect()))
            .collect();

        // Sweep over the part indices that `evaluate_h` actually visits:
        // `extended_omega^i` for i in 0..num_parts.
        let extended_omega = domain.get_extended_omega();
        let num_parts = domain.extended_len() >> domain.k();
        let mut factor = Fr::ONE;
        for part_idx in 0..num_parts {
            let cpu_out =
                coeff_to_extended_part_many_cpu(&domain, host_polys.iter().collect(), factor);
            let gpu_out = domain
                .coeff_to_extended_part_many_device(host_polys.iter().collect::<Vec<_>>(), factor)
                .expect("coeff_to_extended_part_many_device");

            assert_eq!(cpu_out.len(), gpu_out.len(), "part {part_idx}");
            for (i, (cpu, gpu_buf)) in cpu_out.iter().zip(gpu_out.iter()).enumerate() {
                let gpu_host: Vec<Fr> = gpu_buf.to_host_on(&HALO2_GPU_CTX).unwrap();
                assert_eq!(
                    cpu.values(),
                    gpu_host.as_slice(),
                    "part {part_idx} poly {i}"
                );
            }
            factor *= extended_omega;
        }
    }
}

#[cfg(test)]
mod test_eval {
    use crate::plonk::evaluation::{self, Evaluator, EvaluatorVkView};
    use crate::plonk::sealed::SealedPhase;
    use halo2curves::bn256::G1Affine;
    use halo2curves::CurveAffine;

    use ff::Field;

    use openvm_cuda_common::copy::MemCopyH2D;

    use crate::cuda::utils::HALO2_GPU_CTX;
    use crate::plonk::{
        lookup, permutation, AdviceQuery, Any, Column, ConstraintSystem, Expression, FirstPhase,
        FixedQuery, Gate, InstanceQuery,
    };
    use crate::poly::{
        Coeff, Device, DevicePolyExt, EvaluationDomain, ExtendedLagrangeCoeff, HostPolyExt,
        LagrangeCoeff, MaybeDevice, Polynomial, Rotation,
    };

    use rand::Rng;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    type C = G1Affine;
    type F = <G1Affine as CurveAffine>::ScalarExt;

    struct EvaluateHData {
        pk_l0: Polynomial<F, Coeff>,                        // [n]
        pk_l_last: Polynomial<F, Coeff>,                    // [n]
        pk_l_active_row: Polynomial<F, Coeff>,              // [n]
        pk_fixed_values: Vec<Polynomial<F, LagrangeCoeff>>, // [nfixed, n]
        pk_permutation_polys: Vec<Polynomial<F, Coeff>>,    // [nperm, n]
        pk_fixed_polys: Vec<Polynomial<F, Coeff>>,          // [nfixed, n]
        advice_polys: Vec<Vec<Polynomial<F, Coeff>>>,       // [ncirc, advice_polys, n]
        instance_polys: Vec<Vec<Polynomial<F, Coeff>>>,     // [ncirc, inst_polys, n]
        challenges: Vec<F>,                                 // [challenges]
        y: F,
        beta: F,
        gamma: F,
        theta: F,
        lookups: Vec<Vec<lookup::prover::Committed<C>>>, // [ncirc, nlookups]
        permutations: Vec<permutation::prover::Committed<C>>, // [ncirc]
    }

    impl EvaluateHData {
        #[allow(clippy::too_many_arguments)]
        fn new(
            view: &EvaluatorVkView<'_, F>,
            n: usize,
            nfixed: usize,
            nperm: usize,
            n_advice: usize,
            n_instance: usize,
            nlookups: usize,
            num_challenges: usize,
            seed: u64,
        ) -> Self {
            let mut rng = ChaCha20Rng::seed_from_u64(seed);

            let rand_vec = |rng: &mut ChaCha20Rng| -> Vec<F> {
                (0..n).map(|_| F::random(&mut *rng)).collect()
            };
            let rand_coeff =
                |rng: &mut ChaCha20Rng| -> Polynomial<F, Coeff> { Polynomial::new(rand_vec(rng)) };
            let rand_lagrange = |rng: &mut ChaCha20Rng| -> Polynomial<F, LagrangeCoeff> {
                Polynomial::new(rand_vec(rng))
            };
            let rand_coeff_device = |rng: &mut ChaCha20Rng| -> Polynomial<F, Coeff, Device> {
                let v = rand_vec(rng);
                let buf = v
                    .as_slice()
                    .to_device_on(&HALO2_GPU_CTX)
                    .expect("upload random poly to device");
                Polynomial::from_device(buf)
            };

            let pk_l0 = rand_coeff(&mut rng);
            let pk_l_last = rand_coeff(&mut rng);
            let pk_l_active_row = rand_coeff(&mut rng);
            let pk_fixed_values: Vec<_> = (0..nfixed).map(|_| rand_lagrange(&mut rng)).collect();
            let pk_permutation_polys: Vec<_> = (0..nperm).map(|_| rand_coeff(&mut rng)).collect();
            let pk_fixed_polys: Vec<_> = (0..nfixed).map(|_| rand_coeff(&mut rng)).collect();

            let advice_polys: Vec<Vec<_>> =
                vec![(0..n_advice).map(|_| rand_coeff(&mut rng)).collect()];
            let instance_polys: Vec<Vec<_>> =
                vec![(0..n_instance).map(|_| rand_coeff(&mut rng)).collect()];

            let challenges: Vec<F> = (0..num_challenges).map(|_| F::random(&mut rng)).collect();
            let y = F::random(&mut rng);
            let beta = F::random(&mut rng);
            let gamma = F::random(&mut rng);
            let theta = F::random(&mut rng);

            let lookups: Vec<Vec<lookup::prover::Committed<C>>> = vec![(0..nlookups)
                .map(|_| lookup::prover::Committed {
                    permuted_input_expression: MaybeDevice::Host(rand_lagrange(&mut rng)),
                    permuted_table_expression: MaybeDevice::Host(rand_lagrange(&mut rng)),
                    product_poly: rand_coeff_device(&mut rng),
                })
                .collect()];

            // `n_sets = ceil(nperm / chunk_len)` matches the chunking that
            // `permutation::Argument::commit` would have produced, satisfying
            // both `n_perm_cols <= n_sets * chunk_len` and
            // `n_perm_cols + chunk_len > n_sets * chunk_len` in `evaluate_h_inner`.
            let chunk_len = view.cs_degree.saturating_sub(2).max(1);
            let n_sets = if nperm == 0 {
                0
            } else {
                nperm.div_ceil(chunk_len)
            };
            let permutations: Vec<permutation::prover::Committed<C>> = vec![{
                let sets: Vec<permutation::prover::CommittedSet<C>> = (0..n_sets)
                    .map(|_| permutation::prover::CommittedSet {
                        permutation_product_poly: rand_coeff_device(&mut rng),
                    })
                    .collect();
                permutation::prover::Committed { sets }
            }];

            Self {
                pk_l0,
                pk_l_last,
                pk_l_active_row,
                pk_fixed_values,
                pk_permutation_polys,
                pk_fixed_polys,
                advice_polys,
                instance_polys,
                challenges,
                y,
                beta,
                gamma,
                theta,
                lookups,
                permutations,
            }
        }

        fn eval(
            &self,
            view: &EvaluatorVkView<'_, F>,
            f: impl Fn(
                &EvaluatorVkView<'_, F>,
                &Polynomial<F, Coeff>,
                &Polynomial<F, Coeff>,
                &Polynomial<F, Coeff>,
                &[Polynomial<F, LagrangeCoeff>],
                &[Polynomial<F, Coeff>],
                &[Polynomial<F, Coeff>],
                &[&[Polynomial<F, Coeff>]],
                &[&[Polynomial<F, Coeff>]],
                &[F],
                F,
                F,
                F,
                F,
                &[Vec<lookup::prover::Committed<C>>],
                &[permutation::prover::Committed<C>],
            ) -> Polynomial<<C as CurveAffine>::ScalarExt, ExtendedLagrangeCoeff>,
        ) -> Polynomial<<C as CurveAffine>::ScalarExt, ExtendedLagrangeCoeff> {
            let advice_slices: Vec<&[Polynomial<F, Coeff>]> =
                self.advice_polys.iter().map(|v| v.as_slice()).collect();
            let instance_slices: Vec<&[Polynomial<F, Coeff>]> =
                self.instance_polys.iter().map(|v| v.as_slice()).collect();

            f(
                view,
                &self.pk_l0,
                &self.pk_l_last,
                &self.pk_l_active_row,
                &self.pk_fixed_values,
                &self.pk_permutation_polys,
                &self.pk_fixed_polys,
                &advice_slices,
                &instance_slices,
                &self.challenges,
                self.y,
                self.beta,
                self.gamma,
                self.theta,
                &self.lookups,
                &self.permutations,
            )
        }
    }

    fn make_gate(exprs: Vec<Expression<F>>) -> Gate<F> {
        Gate {
            name: "".to_string(),
            constraint_names: vec!["".to_string(); exprs.len()],
            polys: exprs,
            queried_selectors: vec![],
            queried_cells: vec![],
        }
    }

    fn fixed(column_index: usize, rotation: i32) -> Expression<F> {
        Expression::Fixed(FixedQuery {
            index: None,
            column_index,
            rotation: Rotation(rotation),
        })
    }

    fn advice(column_index: usize, rotation: i32) -> Expression<F> {
        Expression::Advice(AdviceQuery {
            index: None,
            column_index,
            rotation: Rotation(rotation),
            phase: FirstPhase.to_sealed(),
        })
    }

    fn instance(column_index: usize, rotation: i32) -> Expression<F> {
        Expression::Instance(InstanceQuery {
            index: None,
            column_index,
            rotation: Rotation(rotation),
        })
    }

    fn constant(value: u64) -> Expression<F> {
        Expression::Constant(F::from(value))
    }

    fn challenges(count: usize) -> Vec<Expression<F>> {
        let mut cs = ConstraintSystem::<F>::default();
        cs.advice_column();
        (0..count)
            .map(|_| cs.challenge_usable_after(FirstPhase).expr())
            .collect()
    }

    fn lookup_argument(
        name: &str,
        input_expressions: Vec<Expression<F>>,
        table_expressions: Vec<Expression<F>>,
    ) -> lookup::Argument<F> {
        lookup::Argument {
            name: name.to_string(),
            input_expressions,
            table_expressions,
        }
    }

    fn random_leaf(
        rng: &mut ChaCha20Rng,
        n_fixed: usize,
        n_advice: usize,
        n_instance: usize,
        challenge_exprs: &[Expression<F>],
    ) -> Expression<F> {
        let rotation = rng.gen_range(-2..=2);
        match rng.gen_range(0..4) {
            0 => fixed(rng.gen_range(0..n_fixed), rotation),
            1 => advice(rng.gen_range(0..n_advice), rotation),
            2 => instance(rng.gen_range(0..n_instance), rotation),
            _ if !challenge_exprs.is_empty() => {
                challenge_exprs[rng.gen_range(0..challenge_exprs.len())].clone()
            }
            _ => constant(rng.gen_range(0..32)),
        }
    }

    fn random_expression(
        rng: &mut ChaCha20Rng,
        depth: usize,
        n_fixed: usize,
        n_advice: usize,
        n_instance: usize,
        challenge_exprs: &[Expression<F>],
    ) -> Expression<F> {
        if depth == 0 {
            return random_leaf(rng, n_fixed, n_advice, n_instance, challenge_exprs);
        }

        match rng.gen_range(0..7) {
            0 => {
                random_expression(
                    rng,
                    depth - 1,
                    n_fixed,
                    n_advice,
                    n_instance,
                    challenge_exprs,
                ) + random_expression(
                    rng,
                    depth - 1,
                    n_fixed,
                    n_advice,
                    n_instance,
                    challenge_exprs,
                )
            }
            1 => {
                random_expression(
                    rng,
                    depth - 1,
                    n_fixed,
                    n_advice,
                    n_instance,
                    challenge_exprs,
                ) - random_expression(
                    rng,
                    depth - 1,
                    n_fixed,
                    n_advice,
                    n_instance,
                    challenge_exprs,
                )
            }
            2 => {
                random_expression(
                    rng,
                    depth - 1,
                    n_fixed,
                    n_advice,
                    n_instance,
                    challenge_exprs,
                ) * random_expression(
                    rng,
                    depth - 1,
                    n_fixed,
                    n_advice,
                    n_instance,
                    challenge_exprs,
                )
            }
            3 => -random_expression(
                rng,
                depth - 1,
                n_fixed,
                n_advice,
                n_instance,
                challenge_exprs,
            ),
            4 => {
                let expr = random_expression(
                    rng,
                    depth - 1,
                    n_fixed,
                    n_advice,
                    n_instance,
                    challenge_exprs,
                );
                expr * F::from(rng.gen_range(2..31))
            }
            5 => {
                let scalar = constant(rng.gen_range(0..64));
                scalar
                    * random_expression(
                        rng,
                        depth - 1,
                        n_fixed,
                        n_advice,
                        n_instance,
                        challenge_exprs,
                    )
            }
            _ => random_leaf(rng, n_fixed, n_advice, n_instance, challenge_exprs),
        }
    }

    fn random_expressions(
        rng: &mut ChaCha20Rng,
        count: usize,
        depth: usize,
        n_fixed: usize,
        n_advice: usize,
        n_instance: usize,
        challenge_exprs: &[Expression<F>],
    ) -> Vec<Expression<F>> {
        (0..count)
            .map(|_| random_expression(rng, depth, n_fixed, n_advice, n_instance, challenge_exprs))
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn assert_close_cpu_gpu(
        perm_arg: permutation::Argument,
        cs_gates: Vec<Gate<F>>,
        lookups: Vec<lookup::Argument<F>>,
        k: u32,
        expansion: u32,
        n_fixed: usize,
        n_perm: usize,
        n_advice: usize,
        n_instance: usize,
        n_lookup: usize,
        n_challenges: usize,
        seed: u64,
    ) {
        assert_close_cpu_gpu_with_data(
            perm_arg,
            cs_gates,
            lookups,
            k,
            expansion,
            n_fixed,
            n_perm,
            n_advice,
            n_instance,
            n_lookup,
            n_challenges,
            seed,
            |_| {},
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn assert_close_cpu_gpu_with_data(
        perm_arg: permutation::Argument,
        cs_gates: Vec<Gate<F>>,
        lookups: Vec<lookup::Argument<F>>,
        k: u32,
        expansion: u32,
        n_fixed: usize,
        n_perm: usize,
        n_advice: usize,
        n_instance: usize,
        n_lookup: usize,
        n_challenges: usize,
        seed: u64,
        mutate_data: impl FnOnce(&mut EvaluateHData),
    ) {
        let n = 1 << k;

        let domain = EvaluationDomain::<F>::new(expansion, k);
        let view = EvaluatorVkView {
            blinding_factors: 2,
            cs_degree: 3,
            permutation_argument: &perm_arg,
            domain: &domain,
        };
        let mut data = EvaluateHData::new(
            &view,
            n,
            n_fixed,
            n_perm,
            n_advice,
            n_instance,
            n_lookup,
            n_challenges,
            seed,
        );
        mutate_data(&mut data);

        let eval_insts = Evaluator::<C>::new_inner(&cs_gates, &lookups);

        let poly_cpu = data.eval(
            &view,
            |view,
             l0,
             l_last,
             l_active_row,
             fixed_values,
             perm_polys,
             fixed_polys,
             advice,
             instance,
             challenges,
             y,
             beta,
             gamma,
             theta,
             lookups,
             perms| {
                super::evaluate_h_inner(
                    &eval_insts,
                    view,
                    l0,
                    l_last,
                    l_active_row,
                    fixed_values,
                    perm_polys,
                    fixed_polys,
                    advice,
                    instance,
                    challenges,
                    y,
                    beta,
                    gamma,
                    theta,
                    lookups,
                    perms,
                )
                .unwrap()
            },
        );

        let poly_gpu = data.eval(
            &view,
            |view,
             l0,
             l_last,
             l_active_row,
             fixed_values,
             perm_polys,
             fixed_polys,
             advice,
             instance,
             challenges,
             y,
             beta,
             gamma,
             theta,
             lookups,
             perms| {
                let l0_d = l0.to_device_on(&HALO2_GPU_CTX).unwrap();
                let l_last_d = l_last.to_device_on(&HALO2_GPU_CTX).unwrap();
                let l_active_row_d = l_active_row.to_device_on(&HALO2_GPU_CTX).unwrap();
                let fixed_values_d: Vec<_> = fixed_values
                    .iter()
                    .map(|v| v.to_device_on(&HALO2_GPU_CTX).unwrap())
                    .collect();
                let fixed_polys_d: Vec<_> = fixed_polys
                    .iter()
                    .map(|v| v.to_device_on(&HALO2_GPU_CTX).unwrap())
                    .collect();
                let perm_polys_d: Vec<_> = perm_polys
                    .iter()
                    .map(|v| v.to_device_on(&HALO2_GPU_CTX).unwrap())
                    .collect();
                let advice_d = advice
                    .iter()
                    .map(|polys| {
                        polys
                            .iter()
                            .map(|p| p.to_device_on(&HALO2_GPU_CTX).unwrap())
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                let instance_d = instance
                    .iter()
                    .map(|polys| {
                        polys
                            .iter()
                            .map(|p| p.to_device_on(&HALO2_GPU_CTX).unwrap())
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();

                evaluation::evaluate_h_inner(
                    &eval_insts,
                    view,
                    &l0_d,
                    &l_last_d,
                    &l_active_row_d,
                    &fixed_values_d,
                    &perm_polys_d,
                    &fixed_polys_d,
                    &advice_d.iter().map(|v| v.as_slice()).collect::<Vec<_>>(),
                    &instance_d.iter().map(|v| v.as_slice()).collect::<Vec<_>>(),
                    challenges,
                    y,
                    beta,
                    gamma,
                    theta,
                    lookups,
                    perms,
                )
                .unwrap()
                .into_host_polynomial()
            },
        );

        assert_eq!(poly_gpu.len(), poly_cpu.len());
        for (idx, (gpu, cpu)) in poly_gpu.iter().zip(poly_cpu.iter()).enumerate() {
            assert_eq!(gpu, cpu, "row {idx}");
        }
    }

    #[test]
    fn build_evaluator_1() {
        // Baseline case: one small custom gate, one lookup, and one
        // permutation touching fixed, advice, and instance columns.
        let mut perm_arg = permutation::Argument::new();
        perm_arg.add_column(Column::new(0, Any::advice()));
        perm_arg.add_column(Column::new(1, Any::Fixed));
        perm_arg.add_column(Column::new(2, Any::Instance));

        let cs_gates = vec![make_gate(vec![
            Expression::Sum(
                Box::new(Expression::Constant(F::ONE)),
                Box::new(Expression::Fixed(FixedQuery {
                    index: None,
                    column_index: 1,
                    rotation: Rotation(1),
                })),
            ),
            Expression::Product(
                Box::new(Expression::Advice(AdviceQuery {
                    index: None,
                    column_index: 2,
                    rotation: Rotation(1),
                    phase: FirstPhase.to_sealed(),
                })),
                Box::new(Expression::Fixed(FixedQuery {
                    index: None,
                    column_index: 0,
                    rotation: Rotation(1),
                })),
            ),
        ])];

        let lookup_gates = vec![
            Expression::Sum(
                Box::new(Expression::Constant(F::ONE)),
                Box::new(Expression::Fixed(FixedQuery {
                    index: None,
                    column_index: 1,
                    rotation: Rotation(1),
                })),
            ),
            Expression::Product(
                Box::new(Expression::Advice(AdviceQuery {
                    index: None,
                    column_index: 2,
                    rotation: Rotation(1),
                    phase: FirstPhase.to_sealed(),
                })),
                Box::new(Expression::Fixed(FixedQuery {
                    index: None,
                    column_index: 0,
                    rotation: Rotation(1),
                })),
            ),
        ];

        let lookups = vec![lookup::Argument {
            name: "".to_string(),
            input_expressions: lookup_gates.clone(),
            table_expressions: lookup_gates.clone(),
        }];

        assert_close_cpu_gpu(perm_arg, cs_gates, lookups, 10, 4, 3, 3, 3, 3, 1, 0, 0);
    }

    #[test]
    fn evaluate_h_mixed_custom_gate_expressions() {
        // Custom-gate case: covers subtraction, negation, square, double,
        // scaled expressions, challenges, and positive/negative rotations.
        let challenge = challenges(2);
        let sum = advice(0, 0) + fixed(0, 0);

        let cs_gates = vec![make_gate(vec![
            sum.clone() * sum,
            (constant(0) * advice(1, -1))
                + (constant(1) * fixed(1, 1))
                + (constant(2) * advice(2, 2)),
            -((advice(1, -1) - fixed(2, -1)) * (instance(1, 1) + challenge[0].clone()))
                * F::from(7),
            (advice(0, 0) * advice(0, 0)) + (fixed(0, -2) * F::from(5)),
            (instance(0, -1) - constant(9)) * (fixed(2, 2) + challenge[1].clone()),
        ])];

        assert_close_cpu_gpu(
            permutation::Argument::new(),
            cs_gates,
            vec![],
            6,
            4,
            3,
            0,
            3,
            2,
            0,
            2,
            1,
        );
    }

    #[test]
    fn evaluate_h_varied_lookup_expressions() {
        // Lookup case: two lookup arguments whose input/table expressions mix
        // challenges, scaled terms, squares, and rotated column queries.
        let challenge = challenges(2);
        let cs_gates = vec![make_gate(vec![
            advice(0, 0) + fixed(0, 0),
            (instance(0, 1) + constant(3)) * fixed(1, -1),
        ])];

        let lookups = vec![
            lookup_argument(
                "mixed_lookup",
                vec![
                    (advice(0, 0) + fixed(0, 1) - instance(0, -1)) * F::from(9),
                    (advice(1, -1) * advice(1, -1)) + challenge[0].clone(),
                    -fixed(1, 0) + (constant(3) * instance(1, 1)),
                ],
                vec![
                    fixed(2, 0) + challenge[1].clone(),
                    (advice(2, 1) * fixed(3, -1)) - constant(5),
                    (instance(2, 0) + fixed(4, 2)) * (advice(0, -2) + constant(7)),
                ],
            ),
            lookup_argument(
                "second_lookup",
                vec![advice(2, 0), fixed(4, -2) * F::from(11)],
                vec![
                    fixed(0, 0) - advice(0, 1),
                    instance(2, -1) + challenge[0].clone(),
                ],
            ),
        ];

        assert_close_cpu_gpu(
            permutation::Argument::new(),
            cs_gates,
            lookups,
            6,
            4,
            5,
            0,
            3,
            3,
            2,
            2,
            2,
        );
    }

    #[test]
    fn evaluate_h_multiple_permutation_columns() {
        // Permutation case: a longer permutation argument spanning multiple
        // fixed/advice columns plus an instance column.
        let mut perm_arg = permutation::Argument::new();
        perm_arg.add_column(Column::new(0, Any::advice()));
        perm_arg.add_column(Column::new(2, Any::advice()));
        perm_arg.add_column(Column::new(1, Any::Fixed));
        perm_arg.add_column(Column::new(0, Any::Instance));
        perm_arg.add_column(Column::new(3, Any::Fixed));

        let cs_gates = vec![make_gate(vec![
            (advice(0, 0) + advice(2, -1)) * (fixed(1, 1) - instance(0, 0)),
            (fixed(3, -2) * F::from(13)) - (advice(1, 2) * fixed(0, 0)),
        ])];

        assert_close_cpu_gpu(perm_arg, cs_gates, vec![], 6, 4, 4, 5, 3, 1, 0, 0, 3);
    }

    #[test]
    fn evaluate_h_degenerate_zero_columns_and_arguments() {
        // when there are 0 columns
        let perm_arg = permutation::Argument::new();

        let cs_gates = vec![];
        let lookups = vec![];

        assert_close_cpu_gpu_with_data(
            perm_arg,
            cs_gates,
            lookups,
            6,
            4,
            0,
            0,
            1,
            0,
            0,
            2,
            4,
            |_| {},
        );
    }

    #[test]
    fn evaluate_h_large_expression_lists_from_rng() {
        // Scale case: generates about 100 custom-gate expressions and 100
        // lookup input/table expressions from a seeded RNG; expression depth is
        // a parameter so deeper recursive trees can be tested without changing
        // the generator.
        let expression_count = 100;
        let expression_depth = 3;
        let n_fixed = 4;
        let n_advice = 4;
        let n_instance = 3;
        let challenge = challenges(3);
        let mut rng = ChaCha20Rng::seed_from_u64(0x5eed);

        let cs_gates = vec![make_gate(random_expressions(
            &mut rng,
            expression_count,
            expression_depth,
            n_fixed,
            n_advice,
            n_instance,
            &challenge,
        ))];
        let lookup_inputs = random_expressions(
            &mut rng,
            expression_count,
            expression_depth,
            n_fixed,
            n_advice,
            n_instance,
            &challenge,
        );
        let lookup_tables = random_expressions(
            &mut rng,
            expression_count,
            expression_depth,
            n_fixed,
            n_advice,
            n_instance,
            &challenge,
        );
        let lookups = vec![lookup_argument(
            "rng_large_lookup",
            lookup_inputs,
            lookup_tables,
        )];

        let mut perm_arg = permutation::Argument::new();
        perm_arg.add_column(Column::new(0, Any::advice()));
        perm_arg.add_column(Column::new(1, Any::Fixed));
        perm_arg.add_column(Column::new(2, Any::Instance));
        perm_arg.add_column(Column::new(3, Any::advice()));

        assert_close_cpu_gpu(
            perm_arg,
            cs_gates,
            lookups,
            5,
            4,
            n_fixed,
            4,
            n_advice,
            n_instance,
            1,
            challenge.len(),
            5,
        );
    }

    // Appends `extra` additional independent random circuits to `data`, so a
    // batch of `1 + extra` circuits is fed to `evaluate_h_inner`. The same `data`
    // drives both the CPU and GPU runs in `assert_close_cpu_gpu_with_data`, so
    // both paths observe identical inputs; any divergence is due to the prover
    // paths, not the inputs. Mirrors the per-circuit construction in
    // `EvaluateHData::new`.
    #[allow(clippy::too_many_arguments)]
    fn append_random_circuits(
        data: &mut EvaluateHData,
        extra: usize,
        n: usize,
        n_advice: usize,
        n_instance: usize,
        nlookups: usize,
        n_sets: usize,
        seed: u64,
    ) {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let rand_vec =
            |rng: &mut ChaCha20Rng| -> Vec<F> { (0..n).map(|_| F::random(&mut *rng)).collect() };
        let rand_coeff =
            |rng: &mut ChaCha20Rng| -> Polynomial<F, Coeff> { Polynomial::new(rand_vec(rng)) };
        let rand_lagrange = |rng: &mut ChaCha20Rng| -> Polynomial<F, LagrangeCoeff> {
            Polynomial::new(rand_vec(rng))
        };
        let rand_coeff_device = |rng: &mut ChaCha20Rng| -> Polynomial<F, Coeff, Device> {
            let v = rand_vec(rng);
            let buf = v
                .as_slice()
                .to_device_on(&HALO2_GPU_CTX)
                .expect("upload random poly to device");
            Polynomial::from_device(buf)
        };

        for _ in 0..extra {
            data.advice_polys
                .push((0..n_advice).map(|_| rand_coeff(&mut rng)).collect());
            data.instance_polys
                .push((0..n_instance).map(|_| rand_coeff(&mut rng)).collect());
            data.lookups.push(
                (0..nlookups)
                    .map(|_| lookup::prover::Committed {
                        permuted_input_expression: MaybeDevice::Host(rand_lagrange(&mut rng)),
                        permuted_table_expression: MaybeDevice::Host(rand_lagrange(&mut rng)),
                        product_poly: rand_coeff_device(&mut rng),
                    })
                    .collect(),
            );
            data.permutations.push({
                let sets = (0..n_sets)
                    .map(|_| permutation::prover::CommittedSet {
                        permutation_product_poly: rand_coeff_device(&mut rng),
                    })
                    .collect();
                permutation::prover::Committed { sets }
            });
        }
    }

    // Drives the CPU and GPU `evaluate_h_inner` over a batch of `ncirc`
    // independent random circuits sharing one gate set (custom gates, one lookup,
    // and a permutation over fixed/advice/instance columns) and asserts the
    // quotient polynomials match row-by-row. This covers the cross-circuit
    // accumulation: both paths fold every circuit into one polynomial via the
    // custom-gates value-part Horner in `y`, then permutation, then lookups.
    fn assert_multi_circuit_quotient_equivalence(ncirc: usize) {
        let k = 4u32;
        let n = 1usize << k;
        let n_fixed = 3usize;
        let n_perm = 3usize;
        let n_advice = 3usize;
        let n_instance = 3usize;
        let n_lookup = 1usize;
        // Matches `view.cs_degree = 3` set inside `assert_close_cpu_gpu_with_data`
        // and the `n_sets` derivation in `EvaluateHData::new`.
        let chunk_len = 3usize.saturating_sub(2).max(1);
        let n_sets = if n_perm == 0 {
            0
        } else {
            n_perm.div_ceil(chunk_len)
        };

        let mut perm_arg = permutation::Argument::new();
        perm_arg.add_column(Column::new(0, Any::advice()));
        perm_arg.add_column(Column::new(1, Any::Fixed));
        perm_arg.add_column(Column::new(2, Any::Instance));

        let cs_gates = vec![make_gate(vec![
            constant(1) + fixed(1, 1),
            advice(2, 1) * fixed(0, 1),
        ])];
        let lookup_exprs = vec![constant(1) + fixed(1, 1), advice(2, 1) * fixed(0, 1)];
        let lookups = vec![lookup_argument(
            "multi_circuit",
            lookup_exprs.clone(),
            lookup_exprs,
        )];

        assert_close_cpu_gpu_with_data(
            perm_arg,
            cs_gates,
            lookups,
            k,
            4,
            n_fixed,
            n_perm,
            n_advice,
            n_instance,
            n_lookup,
            0,
            0x00C0_FFEE,
            |data| {
                append_random_circuits(
                    data,
                    ncirc - 1,
                    n,
                    n_advice,
                    n_instance,
                    n_lookup,
                    n_sets,
                    0x0000_BEEF,
                );
            },
        );
    }

    #[test]
    fn evaluate_h_multi_circuit_batch_equivalence() {
        assert_multi_circuit_quotient_equivalence(2);
        assert_multi_circuit_quotient_equivalence(3);
    }
}
