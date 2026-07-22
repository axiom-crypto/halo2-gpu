#![allow(clippy::type_complexity)]
use ff::{Field, FromUniformBytes, WithSmallOrderMulGroup};
use log::info;
use rand_core::RngCore;

use std::hash::Hash;
use std::iter;

use super::witness::{convert_raw_advice, AdviceColumns};
use super::{
    lookup, permutation, vanishing, witness, ChallengeBeta, ChallengeGamma, ChallengeTheta,
    ChallengeX, ChallengeY, GpuError, GpuProvingKey, ProvingKey,
};
use crate::{
    arithmetic::CurveAffine,
    cuda::funcs::batch_eval_polynomial_d2h,
    plonk::{evaluation, Circuit},
    poly::{
        commitment::{CommitmentScheme, Params, Prover},
        Coeff, EvaluationDomain, Polynomial, ProverQuery,
    },
};
use crate::{
    poly::{Device, DevicePolyExt},
    transcript::{EncodedChallenge, TranscriptWrite},
};
use tracing::info_span;

use crate::cuda::funcs::ColumnPool;
use crate::plonk::lookup::prover::CommittedUnpacked;
use openvm_cuda_common::d_buffer::DeviceBuffer;

/// Warms the witness-independent device caches from a background thread.
/// Byte-neutral: the set-once `OnceCell`s get exactly what the lazy paths
/// would upload.
///
/// PK mirror and params SRS uploads are all enqueued on `HALO2_COMM_STREAM`
/// so they overlap with main-stream compute; the warming calls never sync
/// that stream, and consumers are fenced by `CommWrapper`'s first-deref sync.
fn warm_pk_device_caches<'params, C, P>(params: &P, pk: &GpuProvingKey<'_, C>)
where
    C: CurveAffine,
    P: Params<'params, C>,
{
    // Fresh thread: bind to the ctx device before any H2D. On bind failure,
    // skip warming and let the caches init lazily as before.
    if crate::cuda::utils::ensure_current_device_matches_ctx().is_err() {
        return;
    }
    params.warm_device_caches();
    pk.warm_device_mirrors();
}

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
    // `GpuProvingKey::from_host` → `ProvingKey::get_vk`.
    Scheme::Scalar: Hash + WithSmallOrderMulGroup<3> + FromUniformBytes<64>,
    // The prover spawns a scoped thread that borrows `params`, so it needs Sync
    // (the `ParamsProver` trait itself does not, to match CPU halo2's API).
    Scheme::ParamsProver: Sync,
{
    // Resets the GPU memory peak so the reported peak is per-proof. Must run
    // before any early-return path.
    crate::perf_section_root!("create_proof");

    let setup_span = info_span!("setup").entered();
    // The same view is handed to `create_proof_from_advice_with_pk` below, so mirrors
    // warmed during synthesis are the ones the later phases read.
    let gpu_pk = GpuProvingKey::from_host(pk);
    setup_span.exit();

    let (advice, instance_refs) = witness::synthesize_advices_and_instances::<
        Scheme,
        P,
        E,
        R,
        T,
        ConcreteCircuit,
    >(params, &gpu_pk, circuits, instances, &mut rng, &mut transcript)?;

    create_proof_from_advice_with_pk::<Scheme, P, E, R, T>(
        params,
        &gpu_pk,
        instance_refs,
        advice,
        rng,
        transcript,
    )
}

/// Creates a proof from pre-synthesized advice columns.
pub fn create_proof_from_advice<
    'params: 'a,
    'a,
    Scheme: CommitmentScheme,
    P: Prover<'params, Scheme>,
    E: EncodedChallenge<Scheme::Curve>,
    R: RngCore + Send + 'a,
    T: TranscriptWrite<Scheme::Curve, E>,
>(
    params: &'params Scheme::ParamsProver,
    pk: &ProvingKey<Scheme::Curve>,
    instances: &'a [&'a [Scheme::Scalar]],
    advice: AdviceColumns<Scheme::Scalar>,
    rng: R,
    transcript: &'a mut T,
) -> Result<(), GpuError>
where
    Scheme::Scalar: Hash + WithSmallOrderMulGroup<3> + FromUniformBytes<64>,
    Scheme::ParamsProver: Sync,
{
    crate::perf_section_root!("create_proof");

    let pk_span = info_span!("setup_gpu_pk").entered();
    let gpu_pk = GpuProvingKey::from_host(pk);
    pk_span.exit();

    create_proof_from_advice_with_pk::<Scheme, P, E, R, T>(
        params, &gpu_pk, instances, advice, rng, transcript,
    )
}

/// [`create_proof_from_advice`] over an already-built [`GpuProvingKey`] view, so
/// callers that constructed (and possibly warmed) the view reuse its device
/// mirrors instead of re-uploading them.
fn create_proof_from_advice_with_pk<
    'params: 'a,
    'a,
    Scheme: CommitmentScheme,
    P: Prover<'params, Scheme>,
    E: EncodedChallenge<Scheme::Curve>,
    R: RngCore + Send + 'a,
    T: TranscriptWrite<Scheme::Curve, E>,
>(
    params: &'params Scheme::ParamsProver,
    pk: &GpuProvingKey<'_, Scheme::Curve>,
    instances: &'a [&'a [Scheme::Scalar]],
    advice: AdviceColumns<Scheme::Scalar>,
    mut rng: R,
    transcript: &'a mut T,
) -> Result<(), GpuError>
where
    Scheme::Scalar: Hash + WithSmallOrderMulGroup<3>,
    Scheme::ParamsProver: Sync,
{
    if instances.len() != pk.cs.num_instance_columns {
        return Err(GpuError::Canonical(halo2_axiom::plonk::Error::InvalidInstances));
    }
    pk.hash_into(transcript)?;

    let domain = &pk.domain;
    let meta = &pk.cs;

    std::thread::scope(|s| {
        // Warm the pk mirrors and params SRS on the comm stream while phase 1
        // runs. Each mirror's `CommWrapper` fences its uploads on first deref,
        // so consumers below never race the warming thread's copies.
        let handle = s.spawn(move || warm_pk_device_caches(params, pk));
        let (instance, advice, challenges, theta) = {
            let (instance, advice, challenges) = convert_raw_advice::<Scheme, _, _, E>(
                domain.inner,
                params,
                meta,
                instances,
                advice,
                transcript,
                &mut rng,
                0,
                P::QUERY_INSTANCE,
            )?;

            let theta: ChallengeTheta<_> = transcript.squeeze_challenge_scalar();
            (vec![instance], vec![advice], challenges, theta)
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
                        let fixed_values_device = pk.fixed_values_device().ok_or(
                            GpuError::HaloGpu(crate::cuda::HaloGpuError::InsufficientGpuMemory {
                                context: "plonk::prover: pk.fixed_values_device() unavailable",
                                magnitude: pk.inner.fixed_values().len() as u64,
                                free_bytes: 0,
                            }),
                        )?;
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

        let vanishing_span =
            info_span!("halo2_section", phase = "Commit to vanishing argument's random poly")
                .entered();
        // Random polynomial blinds h(x_3).
        let vanishing = vanishing::Argument::commit(params, domain, &mut rng, transcript).unwrap();
        vanishing_span.exit();

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
            pk.cs.gates.iter().map(|gate| gate.polynomials().len()).sum::<usize>()
        );
        info!("rotations: {:?}", pk.ev.custom_gates.rotations.len());

        // cosetfft at logn=28 inside evaluate_h takes ~16GB of GPU memory; the
        // following ifft at logn=28 (in vanishing.construct) needs the same headroom.
        let h_poly = {
            crate::perf_section!("phase4a");
            evaluation::evaluate_h_device(
                &pk.ev,
                pk,
                &advice.iter().map(|a| a.advice_polys.as_slice()).collect::<Vec<_>>(),
                &instance.iter().map(|i| i.instance_polys.as_slice()).collect::<Vec<_>>(),
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
            let timer = info_span!(
                "halo2_section",
                phase = "Commit to vanishing argument's h(X) commitments"
            )
            .entered();
            let vanishing = vanishing.construct(params, domain, h_poly, &mut rng, transcript)?;

            let x: ChallengeX<_> = transcript.squeeze_challenge_scalar();
            let xn = x.pow([params.n(), 0, 0, 0]);

            timer.exit();
            (vanishing, x, xn)
        };

        handle.join().expect("handle shouldn't fail");
        {
            crate::perf_section!("phase5");

            let eval_polys_span = info_span!("halo2_section", phase = "evaluate polys").entered();

            {
                fn materialize_polys_for_batch_eval<'a, C: CurveAffine>(
                    domain: &EvaluationDomain<'_, C::ScalarExt>,
                    x: &C::ScalarExt,
                    polys: &'a [Polynomial<C::ScalarExt, Coeff, Device>],
                    queries_indexed: &[(usize, crate::poly::Rotation)],
                ) -> (Vec<&'a DeviceBuffer<C::ScalarExt>>, Vec<C::ScalarExt>) {
                    let mut ffi_objs: Vec<_> = Vec::with_capacity(queries_indexed.len());
                    let mut eval_points: Vec<C::ScalarExt> =
                        Vec::with_capacity(queries_indexed.len());
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
                        info!("    instance.instance_polys.len: {}", instance.instance_polys.len());
                        if batch_size == 0 || instance.instance_polys.is_empty() {
                            continue;
                        }
                        let queries_idx: Vec<_> =
                            meta.instance_queries.iter().map(|(c, at)| (c.index(), *at)).collect();
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
                    let queries_idx: Vec<_> =
                        meta.advice_queries.iter().map(|(c, at)| (c.index(), *at)).collect();
                    // Per-query DeviceBuffer references borrowed from `advice_polys`.
                    let d_polys: Vec<&DeviceBuffer<Scheme::Scalar>> = queries_idx
                        .iter()
                        .map(|(col_idx, _)| advice.advice_polys[*col_idx].device_buf())
                        .collect();
                    let eval_points: Vec<Scheme::Scalar> =
                        queries_idx.iter().map(|(_, at)| domain.rotate_omega(*x, *at)).collect();
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
                    let queries_idx: Vec<_> =
                        meta.fixed_queries.iter().map(|(c, at)| (c.index(), *at)).collect();
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
            let unpack_span =
                info_span!("halo2_section", phase = "lagrange_to_coeff_timer").entered();
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
            unpack_span.exit();

            let lookups: Vec<Vec<lookup::prover::Evaluated<Scheme::Curve>>> = lookups
                .into_iter()
                .map(|lookups| -> Vec<_> {
                    lookups.into_iter().map(|p| p.evaluate(pk, x, transcript).unwrap()).collect()
                })
                .collect();
            eval_polys_span.exit();

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
                                .then_some(pk.cs.instance_queries.iter().map(
                                    move |&(column, at)| ProverQuery {
                                        point: domain.rotate_omega(*x, at),
                                        poly: (&instance.instance_polys[column.index()]).into(),
                                    },
                                ))
                                .into_iter()
                                .flatten(),
                        )
                        .chain(pk.cs.advice_queries.iter().map(move |&(column, at)| ProverQuery {
                            point: domain.rotate_omega(*x, at),
                            poly: (&advice.advice_polys[column.index()]).into(),
                        }))
                        .chain(permutation.open(pk, x))
                        .chain(lookups.iter().flat_map(move |p| p.open(pk, x)))
                })
                .chain(pk.cs.fixed_queries.iter().map(|&(column, at)| {
                    let poly = match fixed_polys_device_opt {
                        Some(d) => crate::poly::PolyRef::Device(&d[column.index()]),
                        None => crate::poly::PolyRef::Host(&pk.inner.fixed_polys()[column.index()]),
                    };
                    ProverQuery { point: domain.rotate_omega(*x, at), poly }
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
            let multiopen_span = info_span!("halo2_section", phase = "phase5 multiopen").entered();
            let multiopen_res = prover.create_proof(rng, transcript, instances).map_err(|_| {
                GpuError::Canonical(halo2_axiom::plonk::Error::ConstraintSystemFailure)
            });
            multiopen_span.exit();

            multiopen_res
        }
    })
}
