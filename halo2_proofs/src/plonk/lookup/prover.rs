use super::super::{
    circuit::GpuExpression, ChallengeBeta, ChallengeGamma, ChallengeTheta, ChallengeX, GpuError,
    GpuProvingKey,
};
use super::Argument;
use crate::cuda::funcs::{
    grand_product_device, lookup_product_device, permute_expression_pair_device, ColumnPool,
};
use crate::cuda::utils::HALO2_GPU_CTX;
use crate::cuda::HaloGpuError;
use crate::plonk::evaluation::{
    compress_expressions_device, compress_expressions_in_place_device, evaluate, GraphEvaluator,
};
use crate::{
    arithmetic::CurveAffine,
    poly::{
        commitment::{Blind, Params},
        Coeff, Device, DevicePolyExt, EvaluationDomain, LagrangeCoeff, MaybeDevice, PolyEvalAt,
        Polynomial, ProverQuery, Rotation,
    },
    transcript::{EncodedChallenge, TranscriptWrite},
};
use openvm_cuda_common::copy::{cuda_memcpy_on, MemCopyH2D};
use openvm_cuda_common::d_buffer::DeviceBuffer;

#[cfg(feature = "profile")]
use ark_std::{end_timer, start_timer};

use ff::WithSmallOrderMulGroup;

use group::ff::Field;
use group::Curve;

use rand_core::RngCore;
use rayon::prelude::*;
#[cfg(any(not(feature = "permute_par"), test))]
use std::collections::BTreeMap;
use std::{
    collections::HashMap,
    hash::Hash,
    iter,
    ops::{Mul, MulAssign},
};

#[derive(Debug)]
pub(in crate::plonk) struct Permuted<C: CurveAffine> {
    pub(in crate::plonk) compressed_input_expression: Polynomial<C::Scalar, LagrangeCoeff>,
    pub(in crate::plonk) permuted_input_expression: MaybeDevice<C::Scalar, LagrangeCoeff>,
    pub(in crate::plonk) compressed_table_expression: Polynomial<C::Scalar, LagrangeCoeff>,
    pub(in crate::plonk) permuted_table_expression: MaybeDevice<C::Scalar, LagrangeCoeff>,
}

#[derive(Debug)]
pub(crate) struct Committed<C: CurveAffine> {
    pub(crate) permuted_input_expression: MaybeDevice<C::Scalar, LagrangeCoeff>,
    pub(crate) permuted_table_expression: MaybeDevice<C::Scalar, LagrangeCoeff>,
    pub(crate) product_poly: Polynomial<C::Scalar, Coeff, Device>,
}

#[derive(Debug)]
pub(in crate::plonk) struct CommittedUnpacked<C: CurveAffine> {
    pub(in crate::plonk) permuted_input_poly: Polynomial<C::Scalar, Coeff, Device>,
    pub(in crate::plonk) permuted_table_poly: Polynomial<C::Scalar, Coeff, Device>,
    pub(in crate::plonk) product_poly: Polynomial<C::Scalar, Coeff, Device>,
}

pub(in crate::plonk) struct Evaluated<C: CurveAffine> {
    constructed: CommittedUnpacked<C>,
}

impl<F: WithSmallOrderMulGroup<3>> Argument<F> {
    /// Given a Lookup with input expressions [A_0, A_1, ..., A_{m-1}] and table expressions
    /// [S_0, S_1, ..., S_{m-1}], this method
    /// - constructs A_compressed = \theta^{m-1} A_0 + theta^{m-2} A_1 + ... + \theta A_{m-2} + A_{m-1}
    ///   and S_compressed = \theta^{m-1} S_0 + theta^{m-2} S_1 + ... + \theta S_{m-2} + S_{m-1},
    /// - permutes A_compressed and S_compressed using permute_expression_pair() helper,
    ///   obtaining A' and S', and
    /// - constructs Permuted<C> struct using permuted_input_value = A', and
    ///   permuted_table_expression = S'.
    ///
    /// The Permuted<C> struct is used to update the Lookup, and is then returned.
    pub(in crate::plonk) fn commit_permuted<'a, 'params: 'a, C, P: Params<'params, C>>(
        &self,
        pk: &GpuProvingKey<'_, C>,
        params: &P,
        domain: &EvaluationDomain<C::Scalar>,
        theta: ChallengeTheta<C>,
        advice_values_device: &'a [Polynomial<C::Scalar, LagrangeCoeff, Device>],
        fixed_values: &'a [Polynomial<C::Scalar, LagrangeCoeff>],
        instance_values_device: &'a [Polynomial<C::Scalar, LagrangeCoeff, Device>],
        challenges: &'a [C::Scalar],
        column_pool: Option<&ColumnPool<C::Scalar>>,
    ) -> Result<(Permuted<C>, C, C), GpuError>
    where
        F: Hash,
        C: CurveAffine<ScalarExt = F>,
        C::Curve: Mul<F, Output = C::Curve> + MulAssign<F>,
    {
        crate::perf_section!("lookup_commit_permuted");
        // Host-arm closure (fallback when the column pool is not present
        // or its lazy upload failed VRAM gating). Materialises the
        // device-resident advice / instance columns on the host on demand;
        // the device-arm common path does not enter this closure.
        let compress_expressions_host = |expressions: &[GpuExpression<C::Scalar>]| {
            let advice_values_host: Vec<Polynomial<C::Scalar, LagrangeCoeff>> =
                advice_values_device.iter().map(|p| p.to_host()).collect();
            let instance_values_host: Vec<Polynomial<C::Scalar, LagrangeCoeff>> =
                instance_values_device.iter().map(|p| p.to_host()).collect();
            let compressed_expression = expressions
                .iter()
                .map(|expression| {
                    pk.domain.lagrange_from_vec(evaluate(
                        expression,
                        params.n() as usize,
                        1,
                        fixed_values,
                        &advice_values_host,
                        &instance_values_host,
                        challenges,
                    ))
                })
                .fold(domain.empty_lagrange(), |acc, expression| {
                    acc * *theta + &expression
                });
            compressed_expression
        };

        // Device-side `compress_expressions` when a pre-built column
        // pool is available. Falls back to the CPU closure on pool miss
        // (the caller may pass `None` when VRAM gating failed at pool
        // init).
        let compress_expressions = |expressions: &[GpuExpression<C::Scalar>]| {
            if let Some(pool) = column_pool {
                if pool.is_initialized() {
                    match compress_expressions_device::<C>(
                        expressions,
                        *theta,
                        params.n() as usize,
                        1,
                        pool,
                        challenges,
                    ) {
                        Ok(values) => return pk.domain.lagrange_from_vec(values),
                        Err(e) => {
                            log::warn!(
                                "compress_expressions_device failed ({:?}); host-arm fallback",
                                e
                            );
                        }
                    }
                }
            }
            compress_expressions_host(expressions)
        };

        // Device-fused compress + permute. Runs both expression lists through
        // `compress_expressions_in_place_device` into device buffers, then
        // permutes both pairs in-place on the device via
        // `permute_expression_pair_device`. The four resulting `DeviceBuffer<F>`
        // are materialised to host `Polynomial<F, LagrangeCoeff>` once at the
        // end so `commit_product` and `params.commit_lagrange` consume the
        // same host slices they do today.
        let n = params.n() as usize;
        let usable_rows = n - (pk.cs.blinding_factors() + 1);
        let fused_device = column_pool
            .filter(|pool| pool.is_initialized())
            .and_then(|pool| {
                #[cfg(feature = "profile")]
                let device_fused_time = start_timer!(|| "compress + permute (device-fused)");
                let res = run_compress_permute_device::<C>(
                    pool,
                    *theta,
                    &self.input_expressions,
                    &self.table_expressions,
                    n,
                    usable_rows,
                    challenges,
                );
                #[cfg(feature = "profile")]
                end_timer!(device_fused_time);
                match res {
                    Ok(out) => Some(out),
                    Err(e) => {
                        log::warn!(
                            "compress+permute device-fused path failed ({:?}); host-arm fallback",
                            e
                        );
                        None
                    }
                }
            });

        let (
            compressed_input_expression,
            compressed_table_expression,
            permuted_input_expression,
            permuted_table_expression,
        ) = if let Some((d_ci, d_ct, d_pi, d_pt)) = fused_device {
            // `lookup_product_gpu` (called in `commit_product`) takes host
            // slices for the compressed pair; D2H here at the producer
            // boundary keeps that consumer signature untouched while the
            // permuted pair stays device-resident. Both copies queue on
            // the canonical stream back-to-back and share a single sync.
            let mut compressed_input_host: Vec<C::Scalar> = Vec::with_capacity(d_ci.len());
            let mut compressed_table_host: Vec<C::Scalar> = Vec::with_capacity(d_ct.len());
            unsafe {
                compressed_input_host.set_len(d_ci.len());
                compressed_table_host.set_len(d_ct.len());
                let bytes_ci = std::mem::size_of::<C::Scalar>() * d_ci.len();
                let bytes_ct = std::mem::size_of::<C::Scalar>() * d_ct.len();
                cuda_memcpy_on::<true, false>(
                    compressed_input_host.as_mut_ptr() as *mut libc::c_void,
                    d_ci.as_raw_ptr(),
                    bytes_ci,
                    &HALO2_GPU_CTX,
                )
                .map_err(HaloGpuError::from)?;
                cuda_memcpy_on::<true, false>(
                    compressed_table_host.as_mut_ptr() as *mut libc::c_void,
                    d_ct.as_raw_ptr(),
                    bytes_ct,
                    &HALO2_GPU_CTX,
                )
                .map_err(HaloGpuError::from)?;
            }
            HALO2_GPU_CTX
                .stream
                .to_host_sync()
                .map_err(HaloGpuError::from)?;
            (
                pk.domain.lagrange_from_vec(compressed_input_host),
                pk.domain.lagrange_from_vec(compressed_table_host),
                MaybeDevice::Device(Polynomial::<C::Scalar, LagrangeCoeff, Device>::from_device(
                    d_pi,
                )),
                MaybeDevice::Device(Polynomial::<C::Scalar, LagrangeCoeff, Device>::from_device(
                    d_pt,
                )),
            )
        } else {
            #[cfg(feature = "profile")]
            let compress_input_expr_time = start_timer!(|| "compress input_expr");
            let compressed_input_expression = compress_expressions(&self.input_expressions);
            #[cfg(feature = "profile")]
            end_timer!(compress_input_expr_time);

            #[cfg(feature = "profile")]
            let compress_table_expr_time = start_timer!(|| "compress table_expr");
            let compressed_table_expression = compress_expressions(&self.table_expressions);
            #[cfg(feature = "profile")]
            end_timer!(compress_table_expr_time);

            #[cfg(feature = "profile")]
            let permute_time = start_timer!(|| "get (A', S')");
            let (permuted_input_expression, permuted_table_expression) = permute_expression_pair(
                pk,
                params,
                domain,
                &compressed_input_expression,
                &compressed_table_expression,
            )?;
            #[cfg(feature = "profile")]
            end_timer!(permute_time);

            (
                compressed_input_expression,
                compressed_table_expression,
                MaybeDevice::Host(permuted_input_expression),
                MaybeDevice::Host(permuted_table_expression),
            )
        };

        // MSM-commit A' and S'. Single-stream GPU prover: no channel hop, no
        // worker thread. Blind is intentionally default — KZG verifier does
        // not check blind randomness; see plonk/prover.rs for the ZK-blind
        // note tracked as a follow-up ticket. The device arm of the
        // `MaybeDevice` carrier dispatches `commit_lagrange_device`, skipping
        // the host-arm scalar H2D.
        #[cfg(feature = "profile")]
        let msm_time = start_timer!(|| "MSM commit A' and S'");
        let permuted_input_commitment = match &permuted_input_expression {
            MaybeDevice::Host(p) => params.commit_lagrange(p, Blind::default()),
            MaybeDevice::Device(p) => params.commit_lagrange_device(p, Blind::default()),
        }
        .to_affine();
        let permuted_table_commitment = match &permuted_table_expression {
            MaybeDevice::Host(p) => params.commit_lagrange(p, Blind::default()),
            MaybeDevice::Device(p) => params.commit_lagrange_device(p, Blind::default()),
        }
        .to_affine();
        #[cfg(feature = "profile")]
        end_timer!(msm_time);

        Ok((
            Permuted {
                compressed_input_expression,
                permuted_input_expression,
                compressed_table_expression,
                permuted_table_expression,
            },
            permuted_input_commitment,
            permuted_table_commitment,
        ))
    }
}

impl<C: CurveAffine> Permuted<C> {
    /// Given a Lookup with input expressions, table expressions, and the permuted
    /// input expression and permuted table expression, this method constructs the
    /// grand product polynomial over the lookup. The grand product polynomial
    /// is used to populate the Product<C> struct. The Product<C> struct is
    /// added to the Lookup and finally returned by the method.
    pub(in crate::plonk) fn commit_product<'params, P: Params<'params, C>, R: RngCore>(
        self,
        pk: &GpuProvingKey<'_, C>,
        params: &P,
        beta: ChallengeBeta<C>,
        gamma: ChallengeGamma<C>,
        mut rng: R,
    ) -> Result<(Committed<C>, C), GpuError> {
        crate::perf_section!("lookup_commit_product");
        let blinding_factors = pk.cs.blinding_factors();
        // Goal is to compute the products of fractions
        //
        // Numerator: (\theta^{m-1} a_0(\omega^i) + \theta^{m-2} a_1(\omega^i) + ... + \theta a_{m-2}(\omega^i) + a_{m-1}(\omega^i) + \beta)
        //            * (\theta^{m-1} s_0(\omega^i) + \theta^{m-2} s_1(\omega^i) + ... + \theta s_{m-2}(\omega^i) + s_{m-1}(\omega^i) + \gamma)
        // Denominator: (a'(\omega^i) + \beta) (s'(\omega^i) + \gamma)
        //
        // where a_j(X) is the jth input expression in this lookup,
        // where a'(X) is the compression of the permuted input expressions,
        // s_j(X) is the jth table expression in this lookup,
        // s'(X) is the compression of the permuted table expressions,
        // and i is the ith row of the expression.

        #[cfg(feature = "profile")]
        let lookup_commit_product_time = start_timer!(|| "lookup Z(X) commit product");

        let n = params.n() as usize;
        let scalar_bytes = std::mem::size_of::<C::Scalar>();
        let acc_len = n - blinding_factors;

        // Device-resident running product Z(X). Layout:
        //   z[0]            = 1 (lookup running product starts at 1; no
        //                        cross-set roll-in unlike permutation)
        //   z[1..acc_len]   = scan of lookup_product[0..acc_len-1]
        //   z[acc_len..n]   = blinding factors (RNG-generated on host)
        let d_lookup_product = {
            // Materialise the permuted pair as device buffers. The Device
            // arm (common path under the device-fused commit_permuted)
            // borrows the existing `DeviceBuffer<F>` with zero copy; the
            // Host fallback H2D's the host slice into a fresh
            // `DeviceBuffer<F>` bound to this scope, kept alive until the
            // launcher below enqueues its kernels (stream-ordered
            // `cudaFreeAsync` on Drop guarantees no use-after-free). The
            // buffers free at the end of this scope, ahead of the scan and
            // commit work that follows.
            let permuted_input_owned: DeviceBuffer<C::Scalar>;
            let d_permuted_input: &DeviceBuffer<C::Scalar> = match &self.permuted_input_expression {
                MaybeDevice::Device(p) => p.device_buf(),
                MaybeDevice::Host(p) => {
                    permuted_input_owned = p
                        .values()
                        .to_device_on(&HALO2_GPU_CTX)
                        .map_err(HaloGpuError::from)?;
                    &permuted_input_owned
                }
            };
            let permuted_table_owned: DeviceBuffer<C::Scalar>;
            let d_permuted_table: &DeviceBuffer<C::Scalar> = match &self.permuted_table_expression {
                MaybeDevice::Device(p) => p.device_buf(),
                MaybeDevice::Host(p) => {
                    permuted_table_owned = p
                        .values()
                        .to_device_on(&HALO2_GPU_CTX)
                        .map_err(HaloGpuError::from)?;
                    &permuted_table_owned
                }
            };

            // compressed_*_expression is host-resident in the `Permuted`
            // carrier, so each compressed side is uploaded with one H2D
            // before the device-pointer launcher is invoked.
            let d_compressed_input = self
                .compressed_input_expression
                .values()
                .to_device_on(&HALO2_GPU_CTX)
                .map_err(HaloGpuError::from)?;
            let d_compressed_table = self
                .compressed_table_expression
                .values()
                .to_device_on(&HALO2_GPU_CTX)
                .map_err(HaloGpuError::from)?;

            lookup_product_device(
                d_permuted_input,
                d_permuted_table,
                &d_compressed_input,
                &d_compressed_table,
                *beta,
                *gamma,
            )?
        };

        // In-place device scan; `d_scanned[0..acc_len-1]` carries
        // `∏_{j=0..=i} lookup_product[j]` (prefix = 1).
        let d_scanned = grand_product_device(d_lookup_product, acc_len - 1, C::Scalar::ONE)?;

        let d_z = DeviceBuffer::<C::Scalar>::with_capacity_on(n, &HALO2_GPU_CTX);
        let one = C::Scalar::ONE;
        unsafe {
            cuda_memcpy_on::<false, true>(
                d_z.as_mut_raw_ptr(),
                std::slice::from_ref(&one).as_ptr() as *const libc::c_void,
                scalar_bytes,
                &HALO2_GPU_CTX,
            )
            .map_err(HaloGpuError::from)?;
            cuda_memcpy_on::<true, true>(
                (d_z.as_mut_raw_ptr() as *mut u8).add(scalar_bytes) as *mut libc::c_void,
                d_scanned.as_raw_ptr(),
                (acc_len - 1) * scalar_bytes,
                &HALO2_GPU_CTX,
            )
            .map_err(HaloGpuError::from)?;
        }
        drop(d_scanned);

        // Blinding factors are host-RNG-generated and uploaded with a
        // single tiny H2D into the tail of the device buffer — the
        // accumulator stays device-resident.
        let host_blind: Vec<C::Scalar> = (0..blinding_factors)
            .map(|_| C::Scalar::random(&mut rng))
            .collect();
        unsafe {
            cuda_memcpy_on::<false, true>(
                (d_z.as_mut_raw_ptr() as *mut u8).add(acc_len * scalar_bytes) as *mut libc::c_void,
                host_blind.as_ptr() as *const libc::c_void,
                blinding_factors * scalar_bytes,
                &HALO2_GPU_CTX,
            )
            .map_err(HaloGpuError::from)?;
        }

        let z = Polynomial::<C::Scalar, LagrangeCoeff, Device>::from_device(d_z);

        #[cfg(feature = "profile")]
        end_timer!(lookup_commit_product_time);

        // Single-stream GPU prover: commit Z(X) via device-scalars MSM, then
        // device-input iFFT to coeff form. No PCIe traffic on Z(X).
        let product_commitment = params
            .commit_lagrange_device(&z, Blind::default())
            .to_affine();
        let product_poly = pk.domain.lagrange_to_coeff_device_input(z)?;

        Ok((
            Committed::<C> {
                permuted_input_expression: self.permuted_input_expression,
                permuted_table_expression: self.permuted_table_expression,
                product_poly,
            },
            product_commitment,
        ))
    }
}

impl<C: CurveAffine> CommittedUnpacked<C> {
    pub(in crate::plonk) fn evaluate<E: EncodedChallenge<C>, T: TranscriptWrite<C, E>>(
        self,
        pk: &GpuProvingKey<'_, C>,
        x: ChallengeX<C>,
        transcript: &mut T,
    ) -> Result<Evaluated<C>, GpuError> {
        let domain = &pk.domain;
        let x_inv = domain.rotate_omega(*x, Rotation::prev());
        let x_next = domain.rotate_omega(*x, Rotation::next());

        // Storage-agnostic eval via `Polynomial::eval_at`. The host
        // arm runs rayon CPU Horner; the device arm dispatches to the
        // device-input Horner FFI when the polynomial is
        // device-resident.
        let (
            product_eval,
            product_next_eval,
            permuted_input_eval,
            permuted_input_inv_eval,
            permuted_table_eval,
        ) = {
            crate::perf_section!("lookup.evaluate.eval_at_block");
            let product_eval = self.product_poly.eval_at(*x);
            let product_next_eval = self.product_poly.eval_at(x_next);
            let permuted_input_eval = self.permuted_input_poly.eval_at(*x);
            let permuted_input_inv_eval = self.permuted_input_poly.eval_at(x_inv);
            let permuted_table_eval = self.permuted_table_poly.eval_at(*x);
            (
                product_eval,
                product_next_eval,
                permuted_input_eval,
                permuted_input_inv_eval,
                permuted_table_eval,
            )
        };

        // Hash each advice evaluation
        for eval in iter::empty()
            .chain(Some(product_eval))
            .chain(Some(product_next_eval))
            .chain(Some(permuted_input_eval))
            .chain(Some(permuted_input_inv_eval))
            .chain(Some(permuted_table_eval))
        {
            transcript.write_scalar(eval)?;
        }

        Ok(Evaluated {
            constructed: CommittedUnpacked::<C> {
                permuted_input_poly: self.permuted_input_poly,
                permuted_table_poly: self.permuted_table_poly,
                product_poly: self.product_poly,
            },
        })
    }
}

impl<C: CurveAffine> Evaluated<C> {
    pub(in crate::plonk) fn open<'a>(
        &'a self,
        pk: &'a GpuProvingKey<'_, C>,
        x: ChallengeX<C>,
    ) -> impl Iterator<Item = ProverQuery<'a, C>> + Clone {
        let x_inv = pk.domain.rotate_omega(*x, Rotation::prev());
        let x_next = pk.domain.rotate_omega(*x, Rotation::next());

        iter::empty()
            // Open lookup product commitments at x
            .chain(Some(ProverQuery {
                point: *x,
                poly: (&self.constructed.product_poly).into(),
            }))
            // Open lookup input commitments at x
            .chain(Some(ProverQuery {
                point: *x,
                poly: (&self.constructed.permuted_input_poly).into(),
            }))
            // Open lookup table commitments at x
            .chain(Some(ProverQuery {
                point: *x,
                poly: (&self.constructed.permuted_table_poly).into(),
            }))
            // Open lookup input commitments at x_inv
            .chain(Some(ProverQuery {
                point: x_inv,
                poly: (&self.constructed.permuted_input_poly).into(),
            }))
            // Open lookup product commitments at x_next
            .chain(Some(ProverQuery {
                point: x_next,
                poly: (&self.constructed.product_poly).into(),
            }))
    }
}

/// Device-fused `compress + permute` for a single lookup. Both expression
/// lists are compressed via the metadata-evaluator kernel directly into
/// device buffers, then permuted in-place on the device via the sort
/// kernel. All four output buffers stay on the device; the caller wires
/// permuted_* directly into MaybeDevice carriers and materializes
/// compressed_* on the host only at the lookup_product_gpu consumer site.
#[allow(clippy::type_complexity)]
fn run_compress_permute_device<C: CurveAffine>(
    pool: &ColumnPool<C::Scalar>,
    theta: C::Scalar,
    input_expressions: &[GpuExpression<C::Scalar>],
    table_expressions: &[GpuExpression<C::Scalar>],
    n: usize,
    usable_rows: usize,
    challenges: &[C::Scalar],
) -> Result<
    (
        DeviceBuffer<C::Scalar>,
        DeviceBuffer<C::Scalar>,
        DeviceBuffer<C::Scalar>,
        DeviceBuffer<C::Scalar>,
    ),
    HaloGpuError,
>
where
    C::Scalar: WithSmallOrderMulGroup<3>,
{
    // Slot 4 carries the Horner factor (`theta`); slots 0..3 are the
    // kernel's hard-coded `[0, 1, -1, 2]` for `c1`/`c2` combine decoding.
    let expr_constants = vec![
        C::Scalar::ZERO,
        C::Scalar::ONE,
        -C::Scalar::ONE,
        C::Scalar::from(2),
        theta,
    ];

    let graph_input = GraphEvaluator::<C>::for_compress(input_expressions);
    let graph_table = GraphEvaluator::<C>::for_compress(table_expressions);

    let mut d_compressed_input_device =
        DeviceBuffer::<C::Scalar>::with_capacity_on(n, &HALO2_GPU_CTX);
    let mut d_compressed_table_device =
        DeviceBuffer::<C::Scalar>::with_capacity_on(n, &HALO2_GPU_CTX);
    compress_expressions_in_place_device::<C>(
        &graph_input,
        &expr_constants,
        n,
        1,
        pool,
        challenges,
        &mut d_compressed_input_device,
    )?;
    compress_expressions_in_place_device::<C>(
        &graph_table,
        &expr_constants,
        n,
        1,
        pool,
        challenges,
        &mut d_compressed_table_device,
    )?;

    let mut d_permuted_input_device =
        DeviceBuffer::<C::Scalar>::with_capacity_on(n, &HALO2_GPU_CTX);
    let mut d_permuted_table_device =
        DeviceBuffer::<C::Scalar>::with_capacity_on(n, &HALO2_GPU_CTX);
    permute_expression_pair_device::<C::Scalar>(
        &d_compressed_input_device,
        &d_compressed_table_device,
        &mut d_permuted_input_device,
        &mut d_permuted_table_device,
        n,
        usable_rows,
    )?;

    Ok((
        d_compressed_input_device,
        d_compressed_table_device,
        d_permuted_input_device,
        d_permuted_table_device,
    ))
}

type ExpressionPair<F> = (Polynomial<F, LagrangeCoeff>, Polynomial<F, LagrangeCoeff>);

/// Given a vector of input values A and a vector of table values S,
/// this method permutes A and S to produce A' and S', such that:
/// - like values in A' are vertically adjacent to each other; and
/// - the first row in a sequence of like values in A' is the row
///   that has the corresponding value in S'.
///
/// This method returns (A', S') if no errors are encountered.
fn permute_expression_pair<'params, C: CurveAffine, P: Params<'params, C>>(
    pk: &GpuProvingKey<'_, C>,
    params: &P,
    domain: &EvaluationDomain<C::Scalar>,
    input_expression: &Polynomial<C::Scalar, LagrangeCoeff>,
    table_expression: &Polynomial<C::Scalar, LagrangeCoeff>,
) -> Result<ExpressionPair<C::Scalar>, GpuError>
where
    C::Scalar: Hash,
{
    let usable_rows = params.n() as usize - (pk.cs.blinding_factors() + 1);

    #[cfg(feature = "permute_par")]
    let (permuted_input_expression, permuted_table_coeffs) = permute_expression_pair_par::<C>(
        params.n() as usize,
        input_expression,
        table_expression,
        usable_rows,
    );

    #[cfg(not(feature = "permute_par"))]
    let (permuted_input_expression, permuted_table_coeffs) = permute_expression_pair_seq::<C>(
        params.n() as usize,
        input_expression,
        table_expression,
        usable_rows,
    );

    Ok((
        domain.lagrange_from_vec(permuted_input_expression),
        domain.lagrange_from_vec(permuted_table_coeffs),
    ))
}

fn permute_expression_pair_par<C: CurveAffine>(
    params_n: usize,
    input_expression: &Polynomial<C::Scalar, LagrangeCoeff>,
    table_expression: &Polynomial<C::Scalar, LagrangeCoeff>,
    usable_rows: usize,
) -> (Vec<C::Scalar>, Vec<C::Scalar>)
where
    C::Scalar: Hash,
{
    crate::perf_section!("lookup.permute_par_cpu");
    let num_threads = rayon::current_num_threads();
    let input_expression = &input_expression[0..usable_rows];

    // Sort input lookup expression values
    #[cfg(feature = "profile")]
    let input_time = start_timer!(|| "permute_par input hashmap (cpu par)");
    // count input_expression unique values using a HashMap, using rayon parallel fold+reduce
    let capacity = usable_rows / num_threads + 1;
    let input_uniques: HashMap<C::Scalar, usize, ahash::RandomState> = input_expression
        .par_iter()
        .fold(HashMap::default, |mut acc, coeff| {
            *acc.entry(*coeff).or_insert(0) += 1;
            acc
        })
        .reduce_with(|mut m1, m2| {
            m2.into_iter().for_each(|(k, v)| {
                *m1.entry(k).or_insert(0) += v;
            });
            m1
        })
        .unwrap();
    #[cfg(feature = "profile")]
    end_timer!(input_time);

    #[cfg(feature = "profile")]
    let timer = start_timer!(|| "permute_par input unique ranges (cpu par)");

    let input_unique_ranges = input_uniques
        .par_iter()
        .fold(
            || Vec::with_capacity(capacity),
            |mut input_ranges, (&coeff, &count)| {
                if input_ranges.is_empty() {
                    input_ranges.push((coeff, 0..count));
                } else {
                    let prev_end = input_ranges.last().unwrap().1.end;
                    input_ranges.push((coeff, prev_end..prev_end + count));
                }
                input_ranges
            },
        )
        .reduce_with(|r1, mut r2| {
            let r1_end = r1.last().unwrap().1.end;
            r2.par_iter_mut().for_each(|r2| {
                r2.1.start += r1_end;
                r2.1.end += r1_end;
            });
            [r1, r2].concat()
        })
        .unwrap();
    #[cfg(feature = "profile")]
    end_timer!(timer);

    #[cfg(feature = "profile")]
    let to_vec_time = start_timer!(|| "to_vec");
    let mut sorted_table_coeffs = table_expression[0..usable_rows].to_vec();
    #[cfg(feature = "profile")]
    end_timer!(to_vec_time);
    #[cfg(feature = "profile")]
    let sort_table_time = start_timer!(|| "permute_par sort table");
    sorted_table_coeffs.par_sort();
    #[cfg(feature = "profile")]
    end_timer!(sort_table_time);

    #[cfg(feature = "profile")]
    let timer = start_timer!(|| "leftover table coeffs (cpu par)");

    let leftover_table_coeffs: Vec<C::Scalar> = sorted_table_coeffs
        .as_slice()
        .into_par_iter()
        .enumerate()
        .filter_map(|(i, coeff)| {
            ((i != 0 && coeff == &sorted_table_coeffs[i - 1]) || !input_uniques.contains_key(coeff))
                .then_some(*coeff)
        })
        .collect();
    #[cfg(feature = "profile")]
    end_timer!(timer);

    let (permuted_input_expression, permuted_table_coeffs): (Vec<_>, Vec<_>) = input_unique_ranges
        .into_par_iter()
        .enumerate()
        .flat_map(|(i, (coeff, range))| {
            // subtract off the number of rows in table rows that correspond to input uniques
            let leftover_range_start = range.start - i;
            let leftover_range_end = range.end - i - 1;
            [(coeff, coeff)].into_par_iter().chain(
                leftover_table_coeffs[leftover_range_start..leftover_range_end]
                    .par_iter()
                    .map(move |leftover_table_coeff| (coeff, *leftover_table_coeff)),
            )
        })
        .chain(
            (usable_rows..params_n)
                .into_par_iter()
                .map(|_| (C::Scalar::default(), C::Scalar::default())),
        )
        .unzip();

    assert_eq!(permuted_input_expression.len(), params_n);
    assert_eq!(permuted_table_coeffs.len(), params_n);

    (permuted_input_expression, permuted_table_coeffs)
}

#[cfg(any(not(feature = "permute_par"), test))]
fn permute_expression_pair_seq<C: CurveAffine>(
    params_n: usize,
    input_expression: &Polynomial<C::Scalar, LagrangeCoeff>,
    table_expression: &Polynomial<C::Scalar, LagrangeCoeff>,
    usable_rows: usize,
) -> (Vec<C::Scalar>, Vec<C::Scalar>) {
    let mut permuted_input_expression: Vec<C::Scalar> = input_expression.to_vec();
    permuted_input_expression.truncate(usable_rows);
    // Sort input lookup expression values
    #[cfg(feature = "profile")]
    let sort_time = start_timer!(|| "permute_seq sort input");
    permuted_input_expression.par_sort();
    #[cfg(feature = "profile")]
    end_timer!(sort_time);

    // A BTreeMap of each unique element in the table expression and its count
    #[cfg(feature = "profile")]
    let leftover_map_time = start_timer!(|| "permute_seq construct leftover map");
    let mut leftover_table_map: BTreeMap<C::Scalar, u32> = table_expression
        .iter()
        .take(usable_rows)
        .fold(BTreeMap::new(), |mut acc, coeff| {
            *acc.entry(*coeff).or_insert(0) += 1;
            acc
        });
    #[cfg(feature = "profile")]
    end_timer!(leftover_map_time);

    let mut permuted_table_coeffs = vec![C::Scalar::ZERO; usable_rows];

    #[cfg(feature = "profile")]
    let repeated_row_time = start_timer!(|| "permute_seq get repeated rows");
    let mut repeated_input_rows = permuted_input_expression
        .iter()
        .zip(permuted_table_coeffs.iter_mut())
        .enumerate()
        .filter_map(|(row, (input_value, table_value))| {
            // If this is the first occurrence of `input_value` in the input expression
            if row == 0 || *input_value != permuted_input_expression[row - 1] {
                *table_value = *input_value;
                // Remove one instance of input_value from leftover_table_map
                if let Some(count) = leftover_table_map.get_mut(input_value) {
                    assert!(*count > 0);
                    *count -= 1;
                    None
                } else {
                    // Return error if input_value not found
                    panic!("{:?}", GpuError::ConstraintSystemFailure);
                }
                // If input value is repeated
            } else {
                Some(row)
            }
        })
        .collect::<Vec<_>>();
    #[cfg(feature = "profile")]
    end_timer!(repeated_row_time);

    // Populate permuted table at unfilled rows with leftover table elements
    #[cfg(feature = "profile")]
    let populate_time = start_timer!(|| "populate permuted table");
    for (coeff, count) in leftover_table_map.iter() {
        for _ in 0..*count {
            permuted_table_coeffs[repeated_input_rows.pop().unwrap()] = *coeff;
        }
    }
    assert!(repeated_input_rows.is_empty());
    #[cfg(feature = "profile")]
    end_timer!(populate_time);

    permuted_input_expression.extend((usable_rows..params_n).map(|_| C::Scalar::default()));
    permuted_table_coeffs.extend((usable_rows..params_n).map(|_| C::Scalar::default()));
    assert_eq!(permuted_input_expression.len(), params_n);
    assert_eq!(permuted_table_coeffs.len(), params_n);

    #[cfg(feature = "sanity-checks")]
    {
        let mut last = None;
        for (a, b) in permuted_input_expression
            .iter()
            .zip(permuted_table_coeffs.iter())
            .take(usable_rows)
        {
            if *a != *b {
                assert_eq!(*a, last.unwrap());
            }
            last = Some(*a);
        }
    }

    (permuted_input_expression, permuted_table_coeffs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::ScopedJoinHandle;

    #[allow(clippy::uninit_vec)]
    fn permute_expression_pair_par_scroll<C: CurveAffine>(
        permuted_input_expression: &mut [C::Scalar],
        table_expression: &Polynomial<C::Scalar, LagrangeCoeff>,
    ) -> Result<Vec<C::Scalar>, GpuError> {
        let usable_rows = permuted_input_expression.len();

        // Sort input lookup expression values
        #[cfg(feature = "profile")]
        let sort_input_time = start_timer!(|| "permute_par sort input");
        permuted_input_expression.par_sort();
        #[cfg(feature = "profile")]
        end_timer!(sort_input_time);

        #[cfg(feature = "profile")]
        let to_vec_time = start_timer!(|| "to_vec");
        let mut permuted_table_expression = table_expression.to_vec();
        permuted_table_expression.truncate(usable_rows);
        #[cfg(feature = "profile")]
        end_timer!(to_vec_time);

        #[cfg(feature = "profile")]
        let sort_table_time = start_timer!(|| "permute_par sort table");
        permuted_table_expression.par_sort();
        #[cfg(feature = "profile")]
        end_timer!(sort_table_time);

        let table_expression = permuted_table_expression;

        // A BTreeMap of each unique element in the table expression and its count
        #[cfg(feature = "profile")]
        let leftover_table_par_time = start_timer!(|| "permute_par construct leftover tables");
        let mut leftover_table_maps = std::thread::scope(|s| {
            let num_threads = rayon::current_num_threads();
            let chunk_size = table_expression.len().div_ceil(num_threads);
            let handles = table_expression
                .chunks(chunk_size)
                .map(
                    |table_expr| -> ScopedJoinHandle<BTreeMap<C::Scalar, usize>> {
                        s.spawn(move || -> BTreeMap<C::Scalar, usize> {
                            table_expr.iter().fold(BTreeMap::new(), |mut acc, coeff| {
                                *acc.entry(*coeff).or_insert(0) += 1;
                                acc
                            })
                        })
                    },
                )
                .collect::<Vec<_>>();
            // Note: Don't delete the `.collect::<Vec<_>>();` above if clippy error
            //       it will change the type of `handles` to Vec of closure, which is not what we want
            // Added this log output to bypass the `clippy::needless-collect` check
            log::debug!("spawned {} threads", handles.len());
            handles
                .into_iter()
                .map(|handle| handle.join())
                .collect::<Result<Vec<_>, _>>()
                .unwrap()
        });
        #[cfg(feature = "profile")]
        end_timer!(leftover_table_par_time);

        let mut permuted_table_coeffs = Vec::with_capacity(usable_rows);
        unsafe {
            permuted_table_coeffs.set_len(usable_rows);
        }

        #[cfg(feature = "profile")]
        let repeat_input_time = start_timer!(|| "permute_par construct repeated input rows");
        let mut last_leftover_table_idx = 0;
        let mut repeated_input_rows = permuted_input_expression
            .iter()
            .zip(permuted_table_coeffs.iter_mut())
            .enumerate()
            .filter_map(|(row, (input_value, table_value))| {
                // If this is the first occurrence of `input_value` in the input expression
                if row == 0 || *input_value != permuted_input_expression[row - 1] {
                    *table_value = *input_value;
                    // Remove one instance of input_value from leftover_table_map
                    let mut not_found = true;
                    for leftover_table_map in
                        leftover_table_maps.iter_mut().skip(last_leftover_table_idx)
                    {
                        if let Some(count) = leftover_table_map.get_mut(input_value) {
                            assert!(*count > 0);
                            *count -= 1;
                            not_found = false;
                            break;
                        } else {
                            log::trace!("left over idx increase: row = {}", row);
                            last_leftover_table_idx += 1;
                        }
                    }
                    if not_found {
                        // Return error if input_value not found
                        Some(Err(GpuError::ConstraintSystemFailure))
                    } else {
                        None
                    }
                    // If input value is repeated
                } else {
                    Some(Ok(row))
                }
            })
            // propagate the lookup error (some input is not included in the table) back
            .collect::<Result<Vec<_>, _>>()?;
        #[cfg(feature = "profile")]
        end_timer!(repeat_input_time);

        // Populate permuted table at unfilled rows with leftover table elements
        #[cfg(feature = "profile")]
        let populate_time = start_timer!(|| "permute_par populate rows");
        for leftover_table_map in leftover_table_maps.iter() {
            for (coeff, count) in leftover_table_map.iter() {
                for _ in 0..*count {
                    permuted_table_coeffs[repeated_input_rows.pop().unwrap()] = *coeff;
                }
            }
        }
        #[cfg(feature = "profile")]
        end_timer!(populate_time);

        assert!(repeated_input_rows.is_empty());

        #[cfg(feature = "sanity-checks")]
        {
            let mut last = None;
            for (a, b) in permuted_input_expression
                .iter()
                .zip(permuted_table_coeffs.iter())
                .take(usable_rows)
            {
                if *a != *b {
                    assert_eq!(*a, last.unwrap());
                }
                last = Some(*a);
            }
        }

        Ok(permuted_table_coeffs)
    }

    #[test]
    #[ignore = "expensive"]
    fn test_permute() {
        use ark_std::{end_timer, start_timer};
        use group::prime::PrimeCurveAffine;
        use halo2curves::bn256::G1Affine;
        use rand::rngs::StdRng;
        use rand_core::{OsRng, SeedableRng};
        type Fr = <G1Affine as PrimeCurveAffine>::Scalar;

        #[allow(non_snake_case)]
        fn gen_test(log_n: usize, log_k: usize) {
            println!("Lookup table size 2^log_n, log_n: {log_n}, with nonzero entries 2^log_m, log_m: {log_k}");
            let N = 1 << log_n;
            let M = 1 << log_k;

            let (beta, gamma) = if log_n == log_k {
                (Fr::random(OsRng), Fr::random(OsRng))
            } else {
                (Fr::ONE, Fr::ONE)
            };

            let mut input = (0..M)
                .into_par_iter()
                .map(|i| Fr::from(i as u64) * beta + gamma)
                .collect::<Vec<Fr>>();
            while input.len() < N {
                input.append(&mut input.clone());
            }
            println!("generated input data");

            let mut table = (0..M)
                .into_par_iter()
                .map(|i| Fr::from(i as u64) * beta + gamma)
                .collect::<Vec<Fr>>();

            for _ in table.len()..N {
                table.push(Fr::ZERO);
            }
            println!("generated table data");

            let table_poly = Polynomial::<Fr, LagrangeCoeff>::new(table);
            let input_seq = Polynomial::<Fr, LagrangeCoeff>::new(input.clone());
            let input_par = Polynomial::<Fr, LagrangeCoeff>::new(input.clone());
            let _rng = StdRng::from_seed([0u8; 32]);
            let seq_time = start_timer!(|| "permute_seq");
            let _table_seq =
                permute_expression_pair_seq::<G1Affine>(N, &input_seq, &table_poly, N - 9);
            end_timer!(seq_time);

            let par_time = start_timer!(|| "permute_par_axiom");
            let _table_par =
                permute_expression_pair_par::<G1Affine>(N, &input_par, &table_poly, N - 9);
            end_timer!(par_time);

            let par_time = start_timer!(|| "permute_par_scroll");
            let mut input_par = input_par;
            let _table_par =
                permute_expression_pair_par_scroll::<G1Affine>(&mut input_par, &table_poly)
                    .expect("permute par should succeed");
            end_timer!(par_time);
        }

        assert!(Fr::ONE > Fr::ZERO);

        let _ = env_logger::try_init();
        // sparse tests
        println!("======= sparse tests ===========");
        gen_test(23, 19);
        gen_test(23, 20);
        gen_test(23, 22);

        // dense tests
        let dense_tests = vec![18, 19, 20, 21, 22, 23];
        for test_n in dense_tests {
            println!("======= dense tests {} ===========", test_n);
            gen_test(test_n, test_n);
        }
    }
}
