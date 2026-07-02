#![allow(clippy::type_complexity)]
use ff::{Field, FromUniformBytes, WithSmallOrderMulGroup};
use group::Curve;
use log::info;
use rand_core::RngCore;
use rayon::prelude::*;

use std::hash::Hash;
use std::marker::PhantomData;
use std::ops::RangeTo;
use std::{collections::HashMap, iter};

use super::{
    circuit::sealed::{self},
    lookup, permutation, vanishing, ChallengeBeta, ChallengeGamma, ChallengeTheta, ChallengeX,
    ChallengeY, GpuError, GpuProvingKey, ProvingKey,
};
use crate::{
    arithmetic::CurveAffine,
    circuit::Value,
    cuda::funcs::batch_eval_polynomial_d2h,
    plonk::{
        evaluation, Advice, Any, Assigned, Assignment, Challenge, Circuit, Column,
        ConstraintSystem, Fixed, FloorPlanner, GpuAssigned, Instance, Selector,
    },
    poly::{
        commitment::{Blind, CommitmentScheme, Params, Prover},
        Coeff, EvaluationDomain, LagrangeCoeff, Polynomial, ProverQuery,
    },
};
#[cfg(feature = "profile")]
use crate::{end_timer, start_timer};

use crate::{
    poly::{batch_invert_assigned_device, Device, DevicePolyExt, HostPolyExt},
    transcript::{EncodedChallenge, TranscriptWrite},
};

use crate::cuda::funcs::ColumnPool;
use crate::cuda::utils::HALO2_GPU_CTX;
use crate::plonk::lookup::prover::CommittedUnpacked;
use openvm_cuda_common::copy::MemCopyH2D;
use openvm_cuda_common::d_buffer::DeviceBuffer;

/// Creates a proof for the provided `circuit` given the public parameters
/// `params` and a [`ProvingKey`] generated for the same circuit. The
/// provided `instances` are zero-padded internally.
pub fn create_proof<
    'params: 'a,
    'a,
    Scheme: CommitmentScheme,
    P: Prover<'params, Scheme>,
    E: EncodedChallenge<Scheme::Curve>,
    R: RngCore + Send + 'a,
    T: TranscriptWrite<Scheme::Curve, E>,
    ConcreteCircuit: Circuit<Scheme::Scalar>,
>(
    params: &'params Scheme::ParamsProver,
    pk: &ProvingKey<Scheme::Curve>,
    circuits: &[ConcreteCircuit],
    instances: &[&'a [&'a [Scheme::Scalar]]],
    mut rng: R,
    mut transcript: &'a mut T,
) -> Result<(), GpuError>
where
    // `FromUniformBytes<64>` is needed to build the GPU proving-key view via
    // `GpuProvingKey::from_host_ref` → `ProvingKey::get_vk`.
    Scheme::Scalar: Hash + WithSmallOrderMulGroup<3> + FromUniformBytes<64>,
    // The prover spawns a scoped thread that borrows `params`, so it needs Sync
    // (the `ParamsProver` trait itself does not, to match CPU halo2's API).
    Scheme::ParamsProver: Sync,
{
    // Resets the GPU memory peak so the reported peak is per-proof. Must run
    // before any early-return path.
    crate::perf_section_root!("create_proof");

    assert_eq!(circuits.len(), instances.len());
    assert!(!circuits.is_empty());

    // Build the GPU proving-key view by borrowing the canonical key (no host-poly
    // clone; device mirrors stay lazy).
    let gpu_pk = GpuProvingKey::from_host_ref(pk);
    let pk = &gpu_pk;

    let num_instance = pk.cs.num_instance_columns;
    for instance in instances.iter() {
        if instance.len() != num_instance {
            return Err(GpuError::Canonical(
                halo2_axiom::plonk::Error::InvalidInstances,
            ));
        }
    }
    pk.hash_into(transcript)?;

    let domain = &pk.domain;
    #[cfg(feature = "profile")]
    info!("extended_k: {}", domain.extended_k());
    let mut meta = ConstraintSystem::default();
    #[cfg(feature = "circuit-params")]
    let config = ConcreteCircuit::configure_with_params(&mut meta, circuits[0].params());
    #[cfg(not(feature = "circuit-params"))]
    let config = ConcreteCircuit::configure(&mut meta);

    // Capture the constant columns from the freshly-configured canonical cs
    // (which `FloorPlanner::synthesize` needs as `Vec<Column<Fixed>>`) before
    // shadowing `meta` with `pk.cs`. Selector compression never alters the
    // constant columns, so they match by construction.
    let constants = meta.constants().clone();
    let meta = &pk.cs;

    #[derive(Default)]
    struct InstanceSingle<C: CurveAffine> {
        pub instance_values: Vec<Polynomial<C::Scalar, LagrangeCoeff, Device>>,
        pub instance_polys: Vec<Polynomial<C::Scalar, Coeff, Device>>,
    }

    struct AdviceSingle<C: CurveAffine> {
        pub advice_values: Vec<Polynomial<C::Scalar, LagrangeCoeff, Device>>,
        pub advice_polys: Vec<Polynomial<C::Scalar, Coeff, Device>>,
    }

    struct AdviceSingleOption<C: CurveAffine> {
        pub advice_values: Vec<Option<Polynomial<C::Scalar, LagrangeCoeff, Device>>>,
        pub advice_polys: Vec<Option<Polynomial<C::Scalar, Coeff, Device>>>,
    }

    struct WitnessCollection<'params, 'a, 'b, Scheme, P, C, E, R, T>
    where
        Scheme: CommitmentScheme<Curve = C, Scalar = C::ScalarExt>,
        P: Prover<'params, Scheme>,
        C: CurveAffine,
        E: EncodedChallenge<C>,
        R: RngCore + 'a,
        T: TranscriptWrite<C, E>,
    {
        params: &'params Scheme::ParamsProver,
        params_n: usize,
        domain: &'b EvaluationDomain<Scheme::Scalar>,
        current_phase: sealed::Phase,
        #[allow(dead_code)]
        num_instance_columns: usize,
        #[allow(dead_code)]
        num_advice_columns: usize,
        // Advice cells in device-repr `GpuAssigned` so the per-phase
        // `batch_invert_assigned_device` upload borrows each column directly. The
        // canonical `Assigned` from `assign_advice` is reinterpreted from these
        // bytes (layouts checked by `assert_canonical_assigned_matches_gpu_layout`).
        pub advice: Vec<GpuAssigned<C::Scalar>>,
        challenges: &'b mut HashMap<usize, C::Scalar>,
        instances: &'a [&'a [C::Scalar]],
        usable_rows: RangeTo<usize>,
        advice_single: AdviceSingleOption<C>,
        instance_single: InstanceSingle<C>,
        rng: &'b mut R,
        transcript: &'b mut &'a mut T,
        column_indices: [Vec<usize>; 3],
        challenge_indices: [Vec<usize>; 3],
        unusable_rows_start: usize,
        _marker: PhantomData<(P, E)>,
    }

    impl<'params: 'b, 'a, 'b, F, Scheme, P, C, E, R, T> Assignment<F>
        for WitnessCollection<'params, 'a, 'b, Scheme, P, C, E, R, T>
    where
        F: WithSmallOrderMulGroup<3>,
        Scheme: CommitmentScheme<Curve = C, Scalar = C::ScalarExt>,
        P: Prover<'params, Scheme>,
        C: CurveAffine<ScalarExt = F>,
        E: EncodedChallenge<C>,
        R: RngCore + Send + 'a,
        T: TranscriptWrite<C, E>,
    {
        fn enter_region<NR, N>(&mut self, _: N)
        where
            NR: Into<String>,
            N: FnOnce() -> NR,
        {
        }

        fn exit_region(&mut self) {}

        fn enable_selector<A, AR>(
            &mut self,
            _: A,
            _: &Selector,
            _: usize,
        ) -> Result<(), halo2_axiom::plonk::Error>
        where
            A: FnOnce() -> AR,
            AR: Into<String>,
        {
            Ok(())
        }

        fn annotate_column<A, AR>(&mut self, _annotation: A, _column: Column<Any>)
        where
            A: FnOnce() -> AR,
            AR: Into<String>,
        {
        }

        fn query_instance(
            &self,
            column: Column<Instance>,
            row: usize,
        ) -> Result<Value<F>, halo2_axiom::plonk::Error> {
            if !self.usable_rows.contains(&row) {
                // Build the canonical error variant directly (its constructor is `pub(crate)`).
                return Err(halo2_axiom::plonk::Error::NotEnoughRowsAvailable {
                    current_k: self.params.k(),
                });
            }

            self.instances
                .get(column.index())
                .and_then(|column| column.get(row))
                .map(|v| Value::known(*v))
                .ok_or(halo2_axiom::plonk::Error::BoundsFailure)
        }

        fn assign_advice<'v>(
            &mut self,
            column: Column<Advice>,
            row: usize,
            to: Value<Assigned<F>>,
        ) -> Value<&'v Assigned<F>> {
            debug_assert!(
                self.usable_rows.contains(&row),
                "{:?}",
                GpuError::not_enough_rows_available(self.params.k())
            );

            // 3-4% witness-gen speedup vs `get_mut` + bounds checks.
            let advice_get_mut = unsafe {
                self.advice
                    .get_unchecked_mut(column.index() * self.params_n + row)
            };
            // `Value::assign()` is `pub(crate)`; extract the known `Assigned` via
            // the public `map`, preserving the panic-on-unknown contract.
            let mut assigned = None;
            to.map(|v| assigned = Some(v));
            *advice_get_mut = GpuAssigned::from(
                assigned.expect("No Value::unknown() in advice column allowed during create_proof"),
            );
            // The contract returns `Value<&Assigned<F>>`. The stored cell is
            // reinterpreted in place (layouts coincide per
            // `assert_canonical_assigned_matches_gpu_layout`); the reference stays
            // valid because `advice` is pre-sized and never reallocated.
            let immutable_raw_ptr = advice_get_mut as *const GpuAssigned<F> as *const Assigned<F>;
            Value::known(unsafe { &*immutable_raw_ptr })
        }

        fn assign_fixed(&mut self, _: Column<Fixed>, _: usize, _: Assigned<F>) {}

        fn copy(&mut self, _: Column<Any>, _: usize, _: Column<Any>, _: usize) {}

        fn fill_from_row(
            &mut self,
            _: Column<Fixed>,
            _: usize,
            _: Value<Assigned<F>>,
        ) -> Result<(), halo2_axiom::plonk::Error> {
            Ok(())
        }

        fn get_challenge(&self, challenge: Challenge) -> Value<F> {
            self.challenges
                .get(&challenge.index())
                .cloned()
                .map(Value::known)
                .unwrap_or_else(Value::unknown)
        }

        fn push_namespace<NR, N>(&mut self, _: N)
        where
            NR: Into<String>,
            N: FnOnce() -> NR,
        {
        }

        fn pop_namespace(&mut self, _: Option<String>) {}

        fn next_phase(&mut self) {
            crate::perf_section!("witness.next_phase");
            let phase = self.current_phase.to_u8() as usize;
            let mut instance_values = vec![];
            if phase == 0 {
                #[cfg(feature = "profile")]
                let timer = start_timer!(|| "absorb instances into transcript");
                if !P::QUERY_INSTANCE {
                    for values in self.instances.iter() {
                        for value in values.iter() {
                            self.transcript
                                .common_scalar(*value)
                                .expect("Absorb instance value failed");
                        }
                    }
                }
                instance_values = self
                    .instances
                    .iter()
                    .map(|values| {
                        let mut poly = self.domain.empty_lagrange();
                        debug_assert_eq!(poly.len(), self.params.n() as usize);
                        debug_assert!(
                            values.len() <= self.unusable_rows_start,
                            "GpuError: InstanceTooLarge"
                        );
                        poly.values_mut()
                            .par_iter_mut()
                            .zip(values.par_iter())
                            .for_each(|(poly, value)| {
                                *poly = *value;
                            });
                        poly
                    })
                    .collect::<Vec<_>>();
                #[cfg(feature = "profile")]
                end_timer!(timer);
            }

            #[cfg(feature = "profile")]
            let bf_time = start_timer!(|| "add blinding factors");
            // Write blinding-factor cells into each phase column's tail before the
            // device batch_invert. `Trivial(F::random)` round-trips through
            // batch_invert (denominator None), so the device cells get the value
            // directly.
            let column_indices_for_phase = self
                .column_indices
                .get(phase)
                .expect("The API only supports 3 phases right now")
                .clone();
            for column_index in column_indices_for_phase.iter() {
                let col_start = *column_index * self.params_n;
                let blind_start = col_start + self.unusable_rows_start;
                let col_end = col_start + self.params_n;
                for cell in &mut self.advice[blind_start..col_end] {
                    *cell = GpuAssigned::Trivial(F::random(&mut self.rng));
                }
            }
            #[cfg(feature = "profile")]
            end_timer!(bf_time);

            #[cfg(feature = "profile")]
            let batch_invert_time = start_timer!(|| "batch invert witness assignment");
            // Advice cells are already device-repr `GpuAssigned`, handed to the upload
            // kernel by borrow.
            let advice_values = batch_invert_assigned_device(
                column_indices_for_phase
                    .iter()
                    .map(|column_index| {
                        &self.advice
                            [*column_index * self.params_n..(*column_index + 1) * self.params_n]
                    })
                    .collect::<Vec<&[GpuAssigned<C::Scalar>]>>(),
            )
            .expect("batch_invert_assigned_device (CUDA) failed inside Assignment::next_phase");
            #[cfg(feature = "profile")]
            end_timer!(batch_invert_time);

            #[cfg(feature = "profile")]
            let timer = start_timer!(|| "ifft & MSM on instance/advice columns (GPU)");
            let (instance_single, advice_polys, commitments) = new_gpu_thread::<Scheme, C>(
                self.params,
                self.domain,
                instance_values,
                &advice_values,
                P::QUERY_INSTANCE,
            )
            .expect("new_gpu_thread (CUDA FFT/MSM) failed inside Assignment::next_phase");
            #[cfg(feature = "profile")]
            end_timer!(timer);
            #[cfg(feature = "profile")]
            let timer = start_timer!(|| "transcript write / squeeze");

            if phase == 0 {
                self.instance_single = instance_single;
            }

            let mut commitments = commitments.into_iter();
            if phase == 0 && P::QUERY_INSTANCE {
                let num_instance_commitments = self.instance_single.instance_polys.len();
                for _ in 0..num_instance_commitments {
                    self.transcript
                        .common_point(
                            commitments
                                .next()
                                .expect("Did not commit to instance polynomials"),
                        )
                        .unwrap();
                }
            }

            for commitment in commitments {
                self.transcript
                    .write_point(commitment)
                    .expect("absorb commitment point");
            }
            let column_indices = self.column_indices[phase].iter().copied();
            let advice_values = advice_values.into_iter();
            let advice_polys = advice_polys.into_iter();
            for (column_index, (advice_value, advice_poly)) in
                column_indices.zip(advice_values.zip(advice_polys))
            {
                self.advice_single.advice_values[column_index] = Some(advice_value);
                self.advice_single.advice_polys[column_index] = Some(advice_poly);
            }

            for challenge_index in self.challenge_indices[phase].iter() {
                let existing = self.challenges.insert(
                    *challenge_index,
                    *self.transcript.squeeze_challenge_scalar::<()>(),
                );
                assert!(existing.is_none());
            }
            self.current_phase = self.current_phase.next();
            #[cfg(feature = "profile")]
            end_timer!(timer);
        }
    }

    fn new_gpu_thread<Scheme, C>(
        params: &Scheme::ParamsProver,
        domain: &EvaluationDomain<C::Scalar>,
        instance_values: Vec<Polynomial<C::Scalar, LagrangeCoeff>>,
        advice_values: &[Polynomial<C::Scalar, LagrangeCoeff, Device>],
        query_instance: bool,
    ) -> Result<
        (
            InstanceSingle<C>,
            Vec<Polynomial<C::Scalar, Coeff, Device>>,
            Vec<C>,
        ),
        GpuError,
    >
    where
        Scheme: CommitmentScheme<Curve = C, Scalar = C::ScalarExt>,
        C: CurveAffine,
    {
        crate::perf_section!("new_gpu_thread");
        let num_instance = instance_values.len();
        let num_advice = advice_values.len();
        let mut instance_polys = Vec::with_capacity(num_instance);

        let mut commitments =
            Vec::with_capacity(num_advice + (query_instance as usize) * num_instance);

        if query_instance {
            #[cfg(feature = "profile")]
            let msm_time = start_timer!(|| format!("{} MSMs", instance_values.len()));
            for v in instance_values.iter() {
                let commit = params.commit_lagrange(v, Blind::default());
                commitments.push(commit);
            }
            #[cfg(feature = "profile")]
            end_timer!(msm_time);
        }

        #[cfg(feature = "profile")]
        let msm_time = start_timer!(|| format!("{} MSMs", advice_values.len()));
        for v in advice_values.iter() {
            let commit = params.commit_lagrange_device(v, Blind::default());
            commitments.push(commit);
        }
        #[cfg(feature = "profile")]
        end_timer!(msm_time);

        if !instance_values.is_empty() {
            #[cfg(feature = "profile")]
            let ifft_time = start_timer!(|| format!("{} ifft time", instance_values.len()));
            let batch_polys = domain.lagrange_to_coeff_many(
                &instance_values
                    .iter()
                    .map(|p| p.to_device_on(&HALO2_GPU_CTX).unwrap())
                    .collect::<Vec<_>>(),
            )?;
            #[cfg(feature = "profile")]
            end_timer!(ifft_time);
            instance_polys.extend(batch_polys);
        }

        // Device-resident advice iFFT: `advice_values` → `advice_polys`.
        #[cfg(feature = "profile")]
        let advice_ifft_time =
            start_timer!(|| format!("{} advice ifft (device)", advice_values.len()));
        let advice_polys: Vec<Polynomial<C::Scalar, Coeff, Device>> = {
            crate::perf_section!("advice_ifft");
            domain.lagrange_to_coeff_many_device_inputs(advice_values)?
        };
        #[cfg(feature = "profile")]
        end_timer!(advice_ifft_time);

        let instance_values_device: Vec<Polynomial<C::Scalar, LagrangeCoeff, Device>> = {
            crate::perf_section!("new_gpu_thread.instance_to_device");
            instance_values
                .iter()
                .map(|p| -> Result<_, GpuError> {
                    let d_buf = p
                        .values()
                        .to_device_on(&HALO2_GPU_CTX)
                        .map_err(crate::cuda::HaloGpuError::from)?;
                    Ok(Polynomial::<C::Scalar, LagrangeCoeff, Device>::from_device(
                        d_buf,
                    ))
                })
                .collect::<Result<_, _>>()?
        };

        let instance_single = InstanceSingle {
            instance_values: instance_values_device,
            instance_polys,
        };
        #[cfg(feature = "profile")]
        let batch_time = start_timer!(|| "batch normalize projective points");
        let commitments_projective = commitments;
        let mut commitments = vec![Scheme::Curve::identity(); commitments_projective.len()];
        C::CurveExt::batch_normalize(&commitments_projective, &mut commitments);
        #[cfg(feature = "profile")]
        end_timer!(batch_time);

        Ok((instance_single, advice_polys, commitments))
    }

    let (instance, advice, challenges, theta) = {
        crate::perf_section!("phase1");

        let mut column_indices = [(); 3].map(|_| vec![]);
        for (index, phase) in meta.advice_column_phase.iter().enumerate() {
            column_indices[phase.to_u8() as usize].push(index);
        }
        let mut challenge_indices = [(); 3].map(|_| vec![]);
        for (index, phase) in meta.challenge_phase.iter().enumerate() {
            challenge_indices[phase.to_u8() as usize].push(index);
        }

        let (instance, advice, challenges) = {
            let mut advice = Vec::with_capacity(instances.len());
            let mut instance = Vec::with_capacity(instances.len());

            let unusable_rows_start = params.n() as usize - (meta.blinding_factors() + 1);
            let phases = pk.cs.phases().collect::<Vec<_>>();
            let num_phases = phases.len();
            // WARNING: this will currently not work if `circuits` has more than 1 circuit
            // because the original API squeezes the challenges for a phase after running all circuits
            // once in that phase.
            if num_phases > 1 {
                assert_eq!(
                    circuits.len(),
                    1,
                    "New challenge API doesn't work with multiple circuits yet"
                );
            }

            let mut challenges =
                HashMap::<usize, Scheme::Scalar>::with_capacity(meta.num_challenges);

            // Soundness guard for the `assign_advice` reinterpret of `GpuAssigned`
            // as canonical `Assigned`; verifies the layouts coincide.
            crate::plonk::assert_canonical_assigned_matches_gpu_layout::<Scheme::Scalar>();

            for (circuit, instances) in circuits.iter().zip(instances) {
                #[cfg(feature = "profile")]
                let start = std::time::Instant::now();
                // `usable_rows` excludes blinding-factor rows and the permutation-argument row.
                let mut witness: WitnessCollection<Scheme, P, _, E, _, _> = WitnessCollection {
                    params,
                    params_n: params.n() as usize,
                    domain,
                    current_phase: phases[0],
                    num_instance_columns: num_instance,
                    num_advice_columns: meta.num_advice_columns,
                    advice: vec![GpuAssigned::Zero; params.n() as usize * meta.num_advice_columns],
                    instances,
                    challenges: &mut challenges,
                    usable_rows: ..unusable_rows_start,
                    advice_single: AdviceSingleOption::<Scheme::Curve> {
                        advice_values: (0..meta.num_advice_columns).map(|_| None).collect(),
                        advice_polys: (0..meta.num_advice_columns).map(|_| None).collect(),
                    },
                    instance_single: InstanceSingle::<Scheme::Curve>::default(),
                    rng: &mut rng,
                    transcript: &mut transcript,
                    column_indices: column_indices.clone(),
                    challenge_indices: challenge_indices.clone(),
                    unusable_rows_start,
                    _marker: PhantomData,
                };
                #[cfg(feature = "profile")]
                log::debug!(
                    "time to create empty WitnessCollection struct; initialize columns with zero: {:?}",
                    start.elapsed()
                );

                #[cfg(feature = "profile")]
                let witness_assign_time = start_timer!(|| "synthesize + next phase calls");
                // Loop covers legacy circuits that don't use `next_phase`; new
                // circuits run synthesize once.
                while witness.current_phase.to_u8() < num_phases as u8 {
                    ConcreteCircuit::FloorPlanner::synthesize(
                        &mut witness,
                        circuit,
                        config.clone(),
                        constants.clone(),
                    )
                    .expect("synthesize failed");
                    if witness.current_phase.to_u8() < num_phases as u8 {
                        witness.next_phase();
                    }
                }
                #[cfg(feature = "profile")]
                end_timer!(witness_assign_time);

                let advice_values = witness
                    .advice_single
                    .advice_values
                    .into_iter()
                    .map(|c| c.unwrap())
                    .collect();
                let advice_polys = witness
                    .advice_single
                    .advice_polys
                    .into_iter()
                    .map(|c| c.unwrap())
                    .collect();
                advice.push(AdviceSingle::<Scheme::Curve> {
                    advice_values,
                    advice_polys,
                });
                instance.push(witness.instance_single);
            }

            assert_eq!(challenges.len(), meta.num_challenges);
            let challenges = (0..meta.num_challenges)
                .map(|index| challenges.remove(&index).unwrap())
                .collect::<Vec<_>>();
            (instance, advice, challenges)
        };

        // theta keeps lookup columns linearly independent.
        let theta: ChallengeTheta<_> = transcript.squeeze_challenge_scalar();
        (instance, advice, challenges, theta)
    };

    let (lookups, beta, gamma) = {
        crate::perf_section!("phase2");

        info!("{} lookups (A', S') ifft/msm", pk.cs.lookups.len());
        // Build a per-prove `ColumnPool` once per (instance, advice) tuple and
        // reuse it across the N lookups. The pool uploads fixed/advice/instance
        // columns to device memory once and hands device pointers to
        // `compress_expressions_device`. VRAM gating happens lazily in `try_init`;
        // on failure the caller falls back to the CPU `compress_expressions`.
        let lookups: Vec<Vec<lookup::prover::Permuted<Scheme::Curve>>> = instance
            .iter()
            .zip(advice.iter())
            .map(|(instance, advice)| -> Result<_, GpuError> {
                let mut pool = ColumnPool::<Scheme::Scalar>::new(params.n() as usize);
                if !pk.cs.lookups.is_empty() {
                    let fixed_slices: Vec<&[Scheme::Scalar]> =
                        pk.inner.fixed_values().iter().map(|p| p.values()).collect();
                    let pk_fixed_mirror = pk.fixed_values_device();
                    if let Err(e) = pool.try_init_device(
                        pk_fixed_mirror,
                        &fixed_slices,
                        &advice.advice_values,
                        &instance.instance_values,
                    ) {
                        log::warn!("ColumnPool::try_init failed ({:?}); host-arm fallback", e);
                    }
                }
                let pool_ref = pool.is_initialized().then_some(&pool);
                pk.cs
                    .lookups
                    .iter()
                    .map(
                        |lookup| -> Result<lookup::prover::Permuted<Scheme::Curve>, GpuError> {
                            let (permuted, input_c, table_c) = lookup.commit_permuted(
                                pk,
                                params,
                                domain,
                                theta,
                                &advice.advice_values,
                                pk.inner.fixed_values(),
                                &instance.instance_values,
                                &challenges,
                                pool_ref,
                            )?;
                            transcript.write_point(input_c)?;
                            transcript.write_point(table_c)?;
                            Ok(permuted)
                        },
                    )
                    .collect::<Result<Vec<_>, GpuError>>()
            })
            .collect::<Result<Vec<_>, GpuError>>()?;

        let beta: ChallengeBeta<_> = transcript.squeeze_challenge_scalar();
        let gamma: ChallengeGamma<_> = transcript.squeeze_challenge_scalar();

        (lookups, beta, gamma)
    };

    let (permutations, lookups) = {
        crate::perf_section!("phase3");

        let permutations: Vec<permutation::prover::Committed<Scheme::Curve>> = {
            crate::perf_section!("phase3a");
            instance
                .iter()
                .zip(advice.iter())
                .map(|(instance, advice)| -> Result<_, GpuError> {
                    let fixed_values_device = pk.fixed_values_device().ok_or(GpuError::HaloGpu(
                        crate::cuda::HaloGpuError::InsufficientGpuMemory {
                            context: "plonk::prover: pk.fixed_values_device() unavailable",
                            magnitude: pk.inner.fixed_values().len() as u64,
                            free_bytes: 0,
                        },
                    ))?;
                    let (committed, commitments) = pk.cs.permutation.commit(
                        params,
                        pk,
                        &advice.advice_values,
                        fixed_values_device,
                        &instance.instance_values,
                        beta,
                        gamma,
                        &mut rng,
                    )?;
                    for commitment in commitments {
                        transcript.write_point(commitment)?;
                    }
                    Ok(committed)
                })
                .collect::<Result<Vec<_>, _>>()?
        };

        let lookups: Vec<Vec<lookup::prover::Committed<Scheme::Curve>>> = {
            crate::perf_section!("phase3b");
            lookups
                .into_iter()
                .map(|lookups| -> Result<Vec<_>, GpuError> {
                    lookups
                        .into_iter()
                        .map(
                            |lookup| -> Result<lookup::prover::Committed<Scheme::Curve>, GpuError> {
                                let (committed, commitment) =
                                    lookup.commit_product(pk, params, beta, gamma, &mut rng)?;
                                transcript.write_point(commitment)?;
                                Ok(committed)
                            },
                        )
                        .collect()
                })
                .collect::<Result<Vec<_>, _>>()?
        };

        (permutations, lookups)
    };

    #[cfg(feature = "profile")]
    let vanishing_time = start_timer!(|| "Commit to vanishing argument's random poly");
    // Random polynomial blinds h(x_3).
    let vanishing = vanishing::Argument::commit(params, domain, &mut rng, transcript).unwrap();
    #[cfg(feature = "profile")]
    end_timer!(vanishing_time);

    // y keeps all gates linearly independent.
    let y: ChallengeY<_> = transcript.squeeze_challenge_scalar();

    info!("num_advice: {}", advice[0].advice_polys.len());
    info!("instance: {}", instance[0].instance_polys.len());
    info!("fixed: {}", pk.inner.fixed_polys().len());
    info!("lookup: {}", lookups[0].len());
    info!("permutation: {}", permutations[0].sets.len());
    info!("cals: {:?}", pk.ev.custom_gates.calculations.len());
    info!(
        "num_of_gates: {}",
        pk.cs
            .gates
            .iter()
            .map(|gate| gate.polynomials().len())
            .sum::<usize>()
    );
    info!("rotations: {:?}", pk.ev.custom_gates.rotations.len());

    // cosetfft at logn=28 inside evaluate_h takes ~16GB of GPU memory; the
    // following ifft at logn=28 (in vanishing.construct) needs the same headroom.
    let h_poly = {
        crate::perf_section!("phase4a");
        evaluation::evaluate_h_device(
            &pk.ev,
            pk,
            &advice
                .iter()
                .map(|a| a.advice_polys.as_slice())
                .collect::<Vec<_>>(),
            &instance
                .iter()
                .map(|i| i.instance_polys.as_slice())
                .collect::<Vec<_>>(),
            &challenges,
            *y,
            *beta,
            *gamma,
            *theta,
            &lookups,
            &permutations,
        )?
    };

    let (vanishing, x, xn) = {
        crate::perf_section!("phase4b");
        #[cfg(feature = "profile")]
        let timer = start_timer!(|| "Commit to vanishing argument's h(X) commitments");
        let vanishing = vanishing.construct(params, domain, h_poly, &mut rng, transcript)?;

        let x: ChallengeX<_> = transcript.squeeze_challenge_scalar();
        let xn = x.pow([params.n(), 0, 0, 0]);

        #[cfg(feature = "profile")]
        end_timer!(timer);
        (vanishing, x, xn)
    };

    {
        crate::perf_section!("phase5");

        #[cfg(feature = "profile")]
        let eval_polys_timer = start_timer!(|| "evaluate polys");

        {
            fn materialize_polys_for_batch_eval<'a, C: CurveAffine>(
                domain: &EvaluationDomain<C::ScalarExt>,
                x: &C::ScalarExt,
                polys: &'a [Polynomial<C::ScalarExt, Coeff, Device>],
                queries_indexed: &[(usize, crate::poly::Rotation)],
            ) -> (Vec<&'a DeviceBuffer<C::ScalarExt>>, Vec<C::ScalarExt>) {
                let mut ffi_objs: Vec<_> = Vec::with_capacity(queries_indexed.len());
                let mut eval_points: Vec<C::ScalarExt> = Vec::with_capacity(queries_indexed.len());
                for (col_idx, at) in queries_indexed.iter() {
                    ffi_objs.push(polys[*col_idx].device_buf());
                    eval_points.push(domain.rotate_omega(*x, *at));
                }
                (ffi_objs, eval_points)
            }

            if P::QUERY_INSTANCE {
                info!("instance: {}", instance.len());
                for instance in instance.iter() {
                    let batch_size = meta.instance_queries.len();
                    info!("    batch_size: {}", batch_size);
                    info!(
                        "    instance.instance_polys.len: {}",
                        instance.instance_polys.len()
                    );
                    if batch_size == 0 || instance.instance_polys.is_empty() {
                        continue;
                    }
                    let queries_idx: Vec<_> = meta
                        .instance_queries
                        .iter()
                        .map(|(c, at)| (c.index(), *at))
                        .collect();
                    let (poly_in_many_ori, eval_points) =
                        materialize_polys_for_batch_eval::<Scheme::Curve>(
                            domain,
                            &x,
                            &instance.instance_polys,
                            &queries_idx,
                        );
                    let mut instance_evals = vec![Scheme::Scalar::ZERO; batch_size];
                    batch_eval_polynomial_d2h(
                        &poly_in_many_ori,
                        &eval_points,
                        &mut instance_evals,
                    )?;
                    for eval in instance_evals.iter() {
                        transcript.write_scalar(*eval)?;
                    }
                }
            }

            info!("advice: {}", advice.len());
            for advice in advice.iter() {
                let batch_size = meta.advice_queries.len();
                info!("    batch_size: {}", batch_size);
                info!("    advice.advice_polys.len: {}", advice.advice_polys.len());
                if advice.advice_polys.is_empty() {
                    continue;
                }
                let queries_idx: Vec<_> = meta
                    .advice_queries
                    .iter()
                    .map(|(c, at)| (c.index(), *at))
                    .collect();
                // Per-query DeviceBuffer references borrowed from `advice_polys`.
                let d_polys: Vec<&DeviceBuffer<Scheme::Scalar>> = queries_idx
                    .iter()
                    .map(|(col_idx, _)| advice.advice_polys[*col_idx].device_buf())
                    .collect();
                let eval_points: Vec<Scheme::Scalar> = queries_idx
                    .iter()
                    .map(|(_, at)| domain.rotate_omega(*x, *at))
                    .collect();
                let mut advice_evals = vec![Scheme::Scalar::ZERO; batch_size];
                batch_eval_polynomial_d2h(&d_polys, &eval_points, &mut advice_evals)?;
                for eval in advice_evals.iter() {
                    transcript.write_scalar(*eval)?;
                }
            }

            let batch_size = meta.fixed_queries.len();
            info!("fixed batch size: {}", batch_size);
            info!("    pk.fixed_polys.len: {}", pk.inner.fixed_polys().len());
            if batch_size > 0 && !pk.inner.fixed_polys().is_empty() {
                let queries_idx: Vec<_> = meta
                    .fixed_queries
                    .iter()
                    .map(|(c, at)| (c.index(), *at))
                    .collect();
                let (poly_in_many_ori, eval_points) =
                    materialize_polys_for_batch_eval::<Scheme::Curve>(
                        domain,
                        &x,
                        pk.fixed_polys_device().unwrap(),
                        &queries_idx,
                    );
                let mut fixed_evals = vec![Scheme::Scalar::ZERO; batch_size];
                batch_eval_polynomial_d2h(&poly_in_many_ori, &eval_points, &mut fixed_evals)?;
                info!("fixed_evals: {}", fixed_evals.len());
                for eval in fixed_evals.iter() {
                    transcript.write_scalar(*eval)?;
                }
            }
        }

        let vanishing = vanishing.evaluate(x, xn, domain, transcript)?;

        pk.evaluate_permutation(x, transcript)?;

        let permutations: Vec<permutation::prover::Evaluated<Scheme::Curve>> = permutations
            .into_iter()
            .map(|permutation| permutation.construct().evaluate(pk, x, transcript).unwrap())
            .collect();

        // Opening prep: per-poly iFFT on GPU, one dispatch per permuted poly per
        // `Committed`.
        #[cfg(feature = "profile")]
        let unpack_timer = start_timer!(|| "lagrange_to_coeff_timer");
        let lookups = lookups
            .into_iter()
            .map(
                |lookups| -> Result<Vec<lookup::prover::CommittedUnpacked<Scheme::Curve>>, _> {
                    lookups
                        .into_iter()
                        .map(|p| {
                            // Device-output iFFT for the lookup permuted polys,
                            // which then flow into multiopen via `ProverQuery`.
                            // Route on residency: the device-fused lookup path
                            // yields `MaybeDevice::Device`, so feed its device
                            // buffer straight into the device-input iFFT
                            // (device-in → device-out, no PCIe round-trip). The
                            // `Host` arm (VRAM fallback) keeps the H2D+iFFT path
                            // and doubles as the byte-identity oracle.
                            let permuted_input_poly = match p.permuted_input_expression {
                                crate::poly::MaybeDevice::Device(dp) => {
                                    domain.lagrange_to_coeff_device_input(dp)?
                                }
                                crate::poly::MaybeDevice::Host(hp) => {
                                    domain.lagrange_to_coeff_device(hp)?
                                }
                            };
                            let permuted_table_poly = match p.permuted_table_expression {
                                crate::poly::MaybeDevice::Device(dp) => {
                                    domain.lagrange_to_coeff_device_input(dp)?
                                }
                                crate::poly::MaybeDevice::Host(hp) => {
                                    domain.lagrange_to_coeff_device(hp)?
                                }
                            };
                            Ok(CommittedUnpacked {
                                permuted_input_poly,
                                permuted_table_poly,
                                product_poly: p.product_poly,
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()
                },
            )
            .collect::<Result<Vec<_>, GpuError>>()?;
        #[cfg(feature = "profile")]
        end_timer!(unpack_timer);

        let lookups: Vec<Vec<lookup::prover::Evaluated<Scheme::Curve>>> = lookups
            .into_iter()
            .map(|lookups| -> Vec<_> {
                lookups
                    .into_iter()
                    .map(|p| p.evaluate(pk, x, transcript).unwrap())
                    .collect()
            })
            .collect();
        #[cfg(feature = "profile")]
        end_timer!(eval_polys_timer);

        // Use `PolyRef::Device` when the PK Coeff device mirrors are populated,
        // else fall back to `PolyRef::Host` over the host slices.
        let fixed_polys_device_opt = pk.fixed_polys_device();
        if fixed_polys_device_opt.is_none() {
            tracing::warn!(
                target: "halo2_vram_fallback",
                site = "pk.fixed_polys_device.none",
                "VRAM fallback fired: PK fixed_polys device mirror absent; ProverQuery routes through host slice"
            );
        }
        let permutation_polys_device_opt = pk.permutation_polys_device();
        if permutation_polys_device_opt.is_none() {
            tracing::warn!(
                target: "halo2_vram_fallback",
                site = "pk.permutation_polys_device.none",
                "VRAM fallback fired: PK permutation_polys device mirror absent; ProverQuery routes through host slice"
            );
        }
        let permutation_host_polys = pk.inner.permutation().polys();
        let instances = instance
            .iter()
            .zip(advice.iter())
            .zip(permutations.iter())
            .zip(lookups.iter())
            .flat_map(|(((instance, advice), permutation), lookups)| {
                iter::empty()
                    .chain(
                        P::QUERY_INSTANCE
                            .then_some(pk.cs.instance_queries.iter().map(move |&(column, at)| {
                                ProverQuery {
                                    point: domain.rotate_omega(*x, at),
                                    poly: (&instance.instance_polys[column.index()]).into(),
                                }
                            }))
                            .into_iter()
                            .flatten(),
                    )
                    .chain(
                        pk.cs
                            .advice_queries
                            .iter()
                            .map(move |&(column, at)| ProverQuery {
                                point: domain.rotate_omega(*x, at),
                                poly: (&advice.advice_polys[column.index()]).into(),
                            }),
                    )
                    .chain(permutation.open(pk, x))
                    .chain(lookups.iter().flat_map(move |p| p.open(pk, x)))
            })
            .chain(pk.cs.fixed_queries.iter().map(|&(column, at)| {
                let poly = match fixed_polys_device_opt {
                    Some(d) => crate::poly::PolyRef::Device(&d[column.index()]),
                    None => crate::poly::PolyRef::Host(&pk.inner.fixed_polys()[column.index()]),
                };
                ProverQuery {
                    point: domain.rotate_omega(*x, at),
                    poly,
                }
            }))
            .chain((0..permutation_host_polys.len()).map(|idx| {
                let poly = match permutation_polys_device_opt {
                    Some(d) => crate::poly::PolyRef::Device(&d[idx]),
                    None => crate::poly::PolyRef::Host(&permutation_host_polys[idx]),
                };
                ProverQuery { point: *x, poly }
            }))
            .chain(vanishing.open(x));

        let prover = P::new(params);
        #[cfg(feature = "profile")]
        let multiopen_timer = start_timer!(|| "phase5 multiopen");
        let multiopen_res = prover
            .create_proof(rng, transcript, instances)
            .map_err(|_| GpuError::Canonical(halo2_axiom::plonk::Error::ConstraintSystemFailure));
        #[cfg(feature = "profile")]
        end_timer!(multiopen_timer);

        multiopen_res
    }
}
