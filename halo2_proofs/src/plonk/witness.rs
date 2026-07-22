use ff::{Field, FromUniformBytes, WithSmallOrderMulGroup};
use group::Curve;
use itertools::Itertools;
use rand_core::RngCore;
use rayon::prelude::*;

use std::collections::HashMap;
use std::hash::Hash;
use std::marker::PhantomData;
use std::ops::RangeTo;

use super::{
    circuit::sealed::{self},
    GpuError, GpuProvingKey, ProvingKey,
};
use crate::plonk::GpuConstraintSystem;
use crate::{
    arithmetic::CurveAffine,
    circuit::Value,
    plonk::{
        Advice, Any, Assigned, Assignment, Challenge, Circuit, Column, ConstraintSystem, Fixed,
        FloorPlanner, GpuAssigned, Instance, Selector,
    },
    poly::{
        commitment::{Blind, CommitmentScheme, Params, Prover},
        Coeff, EvaluationDomain, LagrangeCoeff, Polynomial,
    },
};
use crate::{
    poly::{batch_invert_assigned_device, Device, DevicePolyExt, HostPolyExt},
    transcript::{EncodedChallenge, TranscriptWrite},
};
use halo2_axiom::poly;
use tracing::info_span;

use crate::cuda::utils::HALO2_GPU_CTX;
use openvm_cuda_common::copy::MemCopyH2D;
use openvm_cuda_common::d_buffer::DeviceBuffer;

#[derive(Default, Debug)]
pub struct InstanceSingle<C: CurveAffine> {
    pub instance_values: Vec<Polynomial<C::Scalar, LagrangeCoeff, Device>>,
    pub instance_polys: Vec<Polynomial<C::Scalar, Coeff, Device>>,
}

#[derive(Default, Debug)]
pub struct AdviceSingle<C: CurveAffine> {
    pub advice_values: Vec<Polynomial<C::Scalar, LagrangeCoeff, Device>>,
    pub advice_polys: Vec<Polynomial<C::Scalar, Coeff, Device>>,
}

/// The shape of advice columns accepted by [`super::create_proof_from_advice`]:
/// one device-resident `DeviceBuffer<F>` per physical advice column, each of
/// length `params.n()`.
pub type AdviceColumns<F> = Vec<DeviceBuffer<F>>;

pub(super) struct AdviceSingleOption<C: CurveAffine> {
    pub advice_values: Vec<Option<Polynomial<C::Scalar, LagrangeCoeff, Device>>>,
    pub advice_polys: Vec<Option<Polynomial<C::Scalar, Coeff, Device>>>,
}

/// Bucket advice columns and challenges by their phase index (0/1/2).
pub(super) fn column_and_challenge_indices<F: Field>(
    meta: &GpuConstraintSystem<F>,
) -> ([Vec<usize>; 3], [Vec<usize>; 3]) {
    let mut column_indices = [(); 3].map(|_| vec![]);
    for (index, phase) in meta.advice_column_phase().iter().enumerate() {
        column_indices[*phase as usize].push(index);
    }
    let mut challenge_indices = [(); 3].map(|_| vec![]);
    for (index, phase) in meta.challenge_phase().iter().enumerate() {
        challenge_indices[*phase as usize].push(index);
    }
    (column_indices, challenge_indices)
}

pub(super) struct WitnessCollection<'params, 'a, 'b, Scheme, P, C, E, R, T>
where
    Scheme: CommitmentScheme<Curve = C, Scalar = C::ScalarExt>,
    P: Prover<'params, Scheme>,
    C: CurveAffine,
    E: EncodedChallenge<C>,
    R: RngCore + 'a,
    T: TranscriptWrite<C, E>,
{
    pub params: &'params Scheme::ParamsProver,
    pub params_n: usize,
    pub domain: &'b EvaluationDomain<'b, Scheme::Scalar>,
    pub current_phase: sealed::Phase,
    #[allow(dead_code)]
    pub num_instance_columns: usize,
    #[allow(dead_code)]
    pub num_advice_columns: usize,
    // Advice cells in device-repr `GpuAssigned` so the per-phase
    // `batch_invert_assigned_device` upload borrows each column directly. The
    // canonical `Assigned` from `assign_advice` is reinterpreted from these
    // bytes (layouts checked by `assert_canonical_assigned_matches_gpu_layout`).
    pub advice: Vec<GpuAssigned<C::Scalar>>,
    pub challenges: HashMap<usize, C::Scalar>,
    pub instances: &'a [&'a [C::Scalar]],
    pub usable_rows: RangeTo<usize>,
    pub advice_single: AdviceSingleOption<C>,
    pub instance_single: InstanceSingle<C>,
    pub rng: &'b mut R,
    pub transcript: &'b mut &'a mut T,
    pub column_indices: [Vec<usize>; 3],
    pub challenge_indices: [Vec<usize>; 3],
    pub unusable_rows_start: usize,
    pub _marker: PhantomData<(P, E)>,
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
            let timer =
                info_span!("halo2_section", phase = "absorb instances into transcript").entered();
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
            timer.exit();
        }

        let bf_span = info_span!("halo2_section", phase = "add blinding factors").entered();
        // `Trivial(F::random)` round-trips through batch_invert unchanged
        // (denominator None), so the device cells get the blinder directly.
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
        bf_span.exit();

        let batch_invert_span =
            info_span!("halo2_section", phase = "batch invert witness assignment").entered();
        // Advice cells are already device-repr `GpuAssigned`, handed to the upload
        // kernel by borrow.
        let advice_values = batch_invert_assigned_device(
            column_indices_for_phase
                .iter()
                .map(|column_index| {
                    &self.advice[*column_index * self.params_n..(*column_index + 1) * self.params_n]
                })
                .collect::<Vec<&[GpuAssigned<C::Scalar>]>>(),
        )
        .expect("batch_invert_assigned_device (CUDA) failed inside Assignment::next_phase");
        batch_invert_span.exit();

        let timer = info_span!(
            "halo2_section",
            phase = "ifft & MSM on instance/advice columns (GPU)"
        )
        .entered();
        let (instance_single, advice_polys, commitments) = new_gpu_thread::<Scheme, C>(
            self.params,
            self.domain,
            instance_values,
            &advice_values,
            P::QUERY_INSTANCE,
        )
        .expect("new_gpu_thread (CUDA FFT/MSM) failed inside Assignment::next_phase");
        timer.exit();
        let timer = info_span!("halo2_section", phase = "transcript write / squeeze").entered();

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
        timer.exit();
    }
}

/// Builds a fresh [`WitnessCollection`] for a single circuit's phase-1
/// synthesis. The returned collection borrows `challenges`, `rng`, and
/// `transcript`; the caller drives `Circuit::synthesize` and any subsequent
/// `next_phase` calls, then drops the collection to release those borrows.
///
/// Shared by [`synthesize_advices_and_instances`] and [`synthesize_witness`].
#[allow(clippy::too_many_arguments)]
pub(super) fn make_witness_collection<
    'params: 'b,
    'a: 'b,
    'b,
    Scheme,
    P,
    E,
    R,
    T,
    ConcreteCircuit: Circuit<Scheme::Scalar>,
>(
    params: &'params Scheme::ParamsProver,
    pk: &'b GpuProvingKey<'_, Scheme::Curve>,
    instances: &'a [&'a [Scheme::Scalar]],
    rng: &'b mut R,
    transcript: &'b mut &'a mut T,
    circuit: &ConcreteCircuit,
) -> WitnessCollection<'params, 'a, 'b, Scheme, P, Scheme::Curve, E, R, T>
where
    Scheme: CommitmentScheme,
    Scheme::Scalar: WithSmallOrderMulGroup<3>,
    P: Prover<'params, Scheme>,
    E: EncodedChallenge<Scheme::Curve>,
    R: RngCore + 'a + Send,
    T: TranscriptWrite<Scheme::Curve, E>,
{
    let meta = &pk.cs;
    let params_n = params.n() as usize;
    let unusable_rows_start = params_n - (meta.blinding_factors() + 1);
    let (column_indices, challenge_indices) = column_and_challenge_indices(meta);
    let phases = meta.phases().collect::<Vec<_>>();
    let challenges = HashMap::<usize, Scheme::Scalar>::with_capacity(pk.cs.num_challenges);

    // Layout guard for the `GpuAssigned` → `Assigned` reinterpret in `assign_advice`.
    crate::plonk::assert_canonical_assigned_matches_gpu_layout::<Scheme::Scalar>();

    let mut cs = ConstraintSystem::default();
    #[cfg(feature = "circuit-params")]
    let config = ConcreteCircuit::configure_with_params(&mut cs, circuit.params());
    #[cfg(not(feature = "circuit-params"))]
    let config = ConcreteCircuit::configure(&mut cs);
    let constants = cs.constants().clone();
    let mut witness = WitnessCollection {
        params,
        params_n,
        domain: &pk.domain,
        current_phase: phases[0],
        num_instance_columns: meta.num_instance_columns,
        num_advice_columns: meta.num_advice_columns,
        advice: vec![GpuAssigned::Zero; params_n * meta.num_advice_columns],
        instances,
        challenges,
        usable_rows: ..unusable_rows_start,
        advice_single: AdviceSingleOption {
            advice_values: (0..meta.num_advice_columns).map(|_| None).collect(),
            advice_polys: (0..meta.num_advice_columns).map(|_| None).collect(),
        },
        instance_single: InstanceSingle::default(),
        rng,
        transcript,
        column_indices,
        challenge_indices,
        unusable_rows_start,
        _marker: PhantomData,
    };
    ConcreteCircuit::FloorPlanner::synthesize(&mut witness, circuit, config, constants)
        .expect("synthesize failed");
    assert_eq!(witness.challenges.len(), meta.num_challenges);
    witness
}

/// Runs `Circuit::synthesize` and returns device-resident advice columns in
/// the shape [`super::create_proof_from_advice`] expects, along with the
/// single-circuit instance slice.
///
/// Restricted to single-circuit, single-phase circuits: the returned advice
/// still needs [`convert_raw_advice`] (invoked inside
/// `create_proof_from_advice_with_pk`) to blind, commit, and squeeze the
/// phase-0 challenges, so multi-phase would need additional round trips this
/// entry point does not support.
pub fn synthesize_advices_and_instances<
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
    pk: &GpuProvingKey<'_, Scheme::Curve>,
    circuits: &[ConcreteCircuit],
    instances: &[&'a [&'a [Scheme::Scalar]]],
    rng: &mut R,
    transcript: &mut &'a mut T,
) -> Result<(AdviceColumns<Scheme::Scalar>, &'a [&'a [Scheme::Scalar]]), GpuError>
where
    Scheme::Scalar: Hash + WithSmallOrderMulGroup<3>,
    Scheme::ParamsProver: Sync,
{
    assert_eq!(circuits.len(), instances.len());
    assert_eq!(
        circuits.len(),
        1,
        "synthesize_advices_and_instances supports exactly one circuit"
    );
    let phases = pk.cs.phases().collect::<Vec<_>>();
    assert_eq!(
        phases.len(),
        1,
        "synthesize_advices_and_instances supports single-phase circuits only: multi-phase \
         challenge squeezes interleave with synthesis and cannot go \
         through create_proof_from_advice"
    );

    let num_instance = pk.cs.num_instance_columns;
    for instance in instances.iter() {
        if instance.len() != num_instance {
            return Err(GpuError::Canonical(
                halo2_axiom::plonk::Error::InvalidInstances,
            ));
        }
    }

    let params_n = params.n() as usize;
    let num_advice_columns = pk.cs.num_advice_columns;

    let witness_assign_span = info_span!("halo2_section", phase = "synthesize").entered();
    let advice_raw = {
        let witness = make_witness_collection::<Scheme, P, E, R, T, ConcreteCircuit>(
            params,
            pk,
            instances[0],
            rng,
            transcript,
            &circuits[0],
        );
        // Move the advice buffer out; `witness` (with its outstanding borrows) drops here.
        witness.advice
    };
    witness_assign_span.exit();

    // Blinding rows stay `Zero` here; `convert_raw_advice` draws blinders from
    // `rng` on device in the legacy column-major order — proof bytes unchanged.
    let batch_invert_span =
        info_span!("halo2_section", phase = "batch invert witness assignment").entered();
    let advice = batch_invert_assigned_device(
        (0..num_advice_columns)
            .map(|column_index| &advice_raw[column_index * params_n..(column_index + 1) * params_n])
            .collect::<Vec<&[GpuAssigned<Scheme::Scalar>]>>(),
    )
    .map_err(GpuError::HaloGpu)?;
    batch_invert_span.exit();

    let advice: AdviceColumns<Scheme::Scalar> = advice
        .into_iter()
        .map(DevicePolyExt::into_device_buf)
        .collect();

    Ok((advice, instances[0]))
}

/// Diagnostic: runs [`Circuit::synthesize`] and returns the phase-1
/// instance/advice singles without proving, for comparison against externally
/// generated witnesses.
pub fn synthesize_witness<
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
) -> Result<
    (
        Vec<InstanceSingle<Scheme::Curve>>,
        Vec<AdviceSingle<Scheme::Curve>>,
        Vec<Scheme::Scalar>,
    ),
    GpuError,
>
where
    Scheme::Scalar: Hash + WithSmallOrderMulGroup<3> + FromUniformBytes<64>,
    Scheme::ParamsProver: Sync,
{
    assert_eq!(circuits.len(), instances.len());
    assert_eq!(
        circuits.len(),
        1,
        "synthesize_witness supports exactly one circuit"
    );

    let gpu_pk = GpuProvingKey::from_host(pk);
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

    let mut witness = make_witness_collection::<Scheme, P, E, R, T, ConcreteCircuit>(
        params,
        pk,
        instances[0],
        &mut rng,
        &mut transcript,
        &circuits[0],
    );

    witness.next_phase();

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
    let advice = AdviceSingle::<Scheme::Curve> {
        advice_values,
        advice_polys,
    };
    let instance = witness.instance_single;
    let mut challenges = witness.challenges;

    let challenges = (0..pk.cs.num_challenges)
        .map(|index| challenges.remove(&index).unwrap())
        .collect::<Vec<_>>();

    Ok((vec![instance], vec![advice], challenges))
}

/// Mirrors [`WitnessCollection::next_phase`] for advice that already lives on
/// device: absorbs the instances, blinds and commits the advice columns, and
/// squeezes this phase's challenges in the same transcript/rng order.
pub(super) fn convert_raw_advice<Scheme: CommitmentScheme, R: RngCore, T, E>(
    domain: &poly::EvaluationDomain<Scheme::Scalar>,
    params: &Scheme::ParamsProver,
    meta: &GpuConstraintSystem<Scheme::Scalar>,
    instances: &[&[Scheme::Scalar]],
    mut advice_values: Vec<DeviceBuffer<Scheme::Scalar>>,
    transcript: &mut T,
    mut rng: &mut R,
    phase: usize,
    prover_query_instance: bool,
) -> Result<
    (
        InstanceSingle<Scheme::Curve>,
        AdviceSingle<Scheme::Curve>,
        Vec<Scheme::Scalar>, // challenge
    ),
    GpuError,
>
where
    T: TranscriptWrite<Scheme::Curve, E>,
    E: EncodedChallenge<Scheme::Curve>,
    Scheme::Scalar: WithSmallOrderMulGroup<3>,
{
    crate::perf_section!("convert_raw_advice");

    let param_n = params.n() as usize;
    let mut challenge_indices = [(); 3].map(|_| vec![]);
    for (index, phase) in meta.challenge_phase().iter().enumerate() {
        challenge_indices[*phase as usize].push(index);
    }

    let mut column_indices = [(); 3].map(|_| vec![]);
    for (index, phase) in meta.advice_column_phase().iter().enumerate() {
        column_indices[*phase as usize].push(index);
    }

    if !prover_query_instance {
        for values in instances.iter() {
            for value in values.iter() {
                transcript
                    .common_scalar(*value)
                    .expect("Absorb instance value failed");
            }
        }
    }

    let unusable_rows_start = param_n - (meta.blinding_factors() + 1);
    let instance_values = instances
        .iter()
        .map(|values| {
            let mut poly = domain.empty_lagrange();
            debug_assert_eq!(poly.len(), param_n);
            debug_assert!(
                values.len() <= unusable_rows_start,
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

    let bf_span = info_span!("halo2_section", phase = "add blinding factors").entered();
    // Draw all blinders on host in the legacy rng order, then H2D each column's
    // tail on the crate stream.
    let column_indices_for_phase = column_indices
        .get(phase)
        .expect("The API only supports 3 phases right now")
        .clone();
    let n_blind = param_n - unusable_rows_start;
    let blinders: Vec<Vec<Scheme::Scalar>> = column_indices_for_phase
        .iter()
        .map(|_| {
            (0..n_blind)
                .map(|_| Scheme::Scalar::random(&mut rng))
                .collect()
        })
        .collect();
    for (column_index, col_blinders) in column_indices_for_phase.iter().zip(blinders.iter()) {
        advice_values[*column_index]
            .mut_slice(unusable_rows_start..param_n)
            .copy_from_host(col_blinders, &HALO2_GPU_CTX)
            .expect("blinding factor H2D copy failed");
    }
    bf_span.exit();

    let advice_values = advice_values
        .into_iter()
        .map(Polynomial::from_device)
        .collect_vec();

    let timer = info_span!(
        "halo2_section",
        phase = "ifft & MSM on instance/advice columns (GPU)"
    )
    .entered();
    let (instance_single, advice_polys, commitments) = new_gpu_thread::<Scheme, _>(
        params,
        &EvaluationDomain::from_host_domain(domain),
        instance_values,
        &advice_values,
        prover_query_instance,
    )
    .expect("new_gpu_thread (CUDA FFT/MSM) failed inside convert_raw_advice");
    timer.exit();

    let timer = info_span!("halo2_section", phase = "transcript write / squeeze").entered();

    let mut commitments = commitments.into_iter();
    if phase == 0 && prover_query_instance {
        let num_instance_commitments = instance_single.instance_polys.len();
        for _ in 0..num_instance_commitments {
            transcript
                .common_point(
                    commitments
                        .next()
                        .expect("Did not commit to instance polynomials"),
                )
                .unwrap();
        }
    }

    for commitment in commitments {
        transcript
            .write_point(commitment)
            .expect("absorb commitment point");
    }

    let mut challenges = vec![];
    for _ in challenge_indices[phase].iter() {
        challenges.push(*transcript.squeeze_challenge_scalar::<()>());
    }
    timer.exit();

    Ok((
        instance_single,
        AdviceSingle {
            advice_polys,
            advice_values,
        },
        challenges,
    ))
}

pub(super) fn new_gpu_thread<Scheme, C>(
    params: &Scheme::ParamsProver,
    domain: &EvaluationDomain<'_, C::Scalar>,
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

    let mut commitments = Vec::with_capacity(num_advice + (query_instance as usize) * num_instance);

    if query_instance {
        let msm_span =
            info_span!("halo2_section", phase = %format!("{} MSMs", instance_values.len()))
                .entered();
        for v in instance_values.iter() {
            let commit = params.commit_lagrange(v, Blind::default());
            commitments.push(commit);
        }
        msm_span.exit();
    }

    let msm_span =
        info_span!("halo2_section", phase = %format!("{} MSMs", advice_values.len())).entered();
    for v in advice_values.iter() {
        let commit = params.commit_lagrange_device(v, Blind::default());
        commitments.push(commit);
    }
    msm_span.exit();

    if !instance_values.is_empty() {
        let ifft_span =
            info_span!("halo2_section", phase = %format!("{} ifft time", instance_values.len()))
                .entered();
        let batch_polys = domain.lagrange_to_coeff_many(
            &instance_values
                .iter()
                .map(|p| p.to_device_on(&HALO2_GPU_CTX).unwrap())
                .collect::<Vec<_>>(),
        )?;
        ifft_span.exit();
        instance_polys.extend(batch_polys);
    }

    // Device-resident advice iFFT: `advice_values` → `advice_polys`.
    let advice_ifft_span = info_span!(
        "halo2_section",
        phase = %format!("{} advice ifft (device)", advice_values.len())
    )
    .entered();
    let advice_polys: Vec<Polynomial<C::Scalar, Coeff, Device>> = {
        crate::perf_section!("advice_ifft");
        domain.lagrange_to_coeff_many_device_inputs(advice_values)?
    };
    advice_ifft_span.exit();

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
    let batch_span =
        info_span!("halo2_section", phase = "batch normalize projective points").entered();
    let commitments_projective = commitments;
    let mut commitments = vec![C::identity(); commitments_projective.len()];
    C::CurveExt::batch_normalize(&commitments_projective, &mut commitments);
    batch_span.exit();

    Ok((instance_single, advice_polys, commitments))
}
