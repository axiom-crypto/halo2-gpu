#[cfg(feature = "profile")]
use crate::{end_timer, start_timer};
use ff::{Field, PrimeField};
use group::Curve;

use log::info;
use openvm_cuda_common::copy::{cuda_memcpy_on, MemCopyH2D};
use openvm_cuda_common::d_buffer::DeviceBuffer;
use rand_core::RngCore;
use std::iter::{self, ExactSizeIterator};

use super::super::{circuit::Any, ChallengeBeta, ChallengeGamma, ChallengeX};
use super::Argument;
use crate::{
    arithmetic::CurveAffine,
    cuda::funcs::{grand_product_device, permutation_product_device},
    cuda::utils::{FFITraitObject, HALO2_GPU_CTX},
    cuda::HaloGpuError,
    plonk::{self, Error},
    poly::{
        commitment::{Blind, Params},
        Coeff, Device, DevicePolyExt, LagrangeCoeff, PolyEvalAt, Polynomial, ProverQuery, Rotation,
    },
    transcript::{EncodedChallenge, TranscriptWrite},
};

pub(crate) struct CommittedSet<C: CurveAffine> {
    pub(crate) permutation_product_poly: Polynomial<C::Scalar, Coeff, Device>,
}

pub(crate) struct Committed<C: CurveAffine> {
    pub(crate) sets: Vec<CommittedSet<C>>,
}

pub struct ConstructedSet<C: CurveAffine> {
    permutation_product_poly: Polynomial<C::Scalar, Coeff, Device>,
}

pub(crate) struct Constructed<C: CurveAffine> {
    sets: Vec<ConstructedSet<C>>,
}

pub(crate) struct Evaluated<C: CurveAffine> {
    constructed: Constructed<C>,
}

/// Returns the device-resident Lagrange buffer backing the column at
/// `index` within the family selected by `column_type`.
fn device_buf_for_column<'a, F: Field>(
    advice: &'a [Polynomial<F, LagrangeCoeff, Device>],
    fixed: &'a [Polynomial<F, LagrangeCoeff, Device>],
    instance: &'a [Polynomial<F, LagrangeCoeff, Device>],
    column_type: &Any,
    index: usize,
) -> &'a DeviceBuffer<F> {
    let columns = match column_type {
        Any::Advice(_) => advice,
        Any::Fixed => fixed,
        Any::Instance => instance,
    };
    columns[index].device_buf()
}

impl Argument {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::plonk) fn commit<'params, C: CurveAffine, P: Params<'params, C>, R: RngCore>(
        &self,
        params: &P,
        pk: &plonk::GpuProvingKey<C>,
        advice_device: &[Polynomial<C::Scalar, LagrangeCoeff, Device>],
        fixed_device: &[Polynomial<C::Scalar, LagrangeCoeff, Device>],
        instance_device: &[Polynomial<C::Scalar, LagrangeCoeff, Device>],
        beta: ChallengeBeta<C>,
        gamma: ChallengeGamma<C>,
        mut rng: R,
    ) -> Result<(Committed<C>, Vec<C>), Error> {
        crate::perf_section!("permutation_commit");
        let domain = &pk.domain;

        // Lagrange σ-columns staged once into a device-resident PK mirror (the
        // distinct Lagrange mirror, moved onto `GpuProvingKey`). The host-arm
        // permutation σ vector (`inner.permutation().permutations()`) backs it.
        let permutations_device = pk.permutation_lagrange_device().ok_or(Error::HaloGpu(
            HaloGpuError::InsufficientGpuMemory {
                context: "permutation::Argument::commit: permutations_device unavailable",
                magnitude: pk.inner.permutation().permutations().len() as u64,
                free_bytes: 0,
            },
        ))?;

        // How many columns can be included in a single permutation polynomial?
        // We need to multiply by z(X) and (1 - (l_last(X) + l_blind(X))). This
        // will never underflow because of the requirement of at least a degree
        // 3 circuit for the permutation argument.
        assert!(pk.cs_degree >= 3);
        let chunk_len = pk.cs_degree - 2;
        let blinding_factors = pk.cs.blinding_factors();

        // Each column gets its own delta power.
        let mut deltaomega = C::Scalar::ONE;

        // Track the "last" value from the previous column set
        let mut last_z = C::Scalar::ONE;

        let mut sets = vec![];
        let mut commitments = vec![];

        info!("domain.k() = {}", domain.k());
        info!("domain.extended_k() = {}", domain.extended_k());
        info!("columns.len() = {}", self.columns.len());
        info!(
            "pkey.permutations.len() = {}",
            pk.inner.permutation().permutations().len()
        );
        info!("chunk_len = {}", chunk_len);

        let n = params.n() as usize;
        let scalar_bytes = std::mem::size_of::<C::Scalar>();

        // Caller-owned device-resident ONE-fill template. Each chunk-loop
        // iteration allocates a fresh `d_modified_values` and initialises it
        // via a single D2D copy from this template.
        let d_ones_template = vec![C::Scalar::ONE; n]
            .as_slice()
            .to_device_on(&HALO2_GPU_CTX)
            .map_err(HaloGpuError::from)?;
        let modified_values_bytes = n * scalar_bytes;
        let acc_len = n - blinding_factors;

        for (columns, permutations_chunk) in self
            .columns
            .chunks(chunk_len)
            .zip(permutations_device.chunks(chunk_len))
        {
            // Goal is to compute the products of fractions
            //
            // (p_j(\omega^i) + \delta^j \omega^i \beta + \gamma) /
            // (p_j(\omega^i) + \beta s_j(\omega^i) + \gamma)
            //
            // where p_j(X) is the jth column in this permutation,
            // and i is the ith row of the column.

            // Fresh per-chunk accumulator: the device-input grand_product
            // wrapper consumes the buffer by value (returns the scan-bearing
            // buffer back to the caller), so a fresh allocation per chunk
            // restores the layout. ONE-init via D2D from the hoisted
            // template.
            let mut d_modified_values =
                DeviceBuffer::<C::Scalar>::with_capacity_on(n, &HALO2_GPU_CTX);
            unsafe {
                cuda_memcpy_on::<true, true>(
                    d_modified_values.as_mut_raw_ptr(),
                    d_ones_template.as_raw_ptr(),
                    modified_values_bytes,
                    &HALO2_GPU_CTX,
                )
                .map_err(HaloGpuError::from)?;
            }

            {
                #[cfg(feature = "profile")]
                let gpu_time = start_timer!(|| "Z_i(X) permutation_product_gpu");

                let mut permutations_poly_ffi: Vec<FFITraitObject> =
                    Vec::with_capacity(permutations_chunk.len());
                let mut values_poly_ffi: Vec<FFITraitObject> = Vec::with_capacity(columns.len());
                for (&column, permuted_column_values) in
                    columns.iter().zip(permutations_chunk.iter())
                {
                    permutations_poly_ffi.push(FFITraitObject::new(
                        permuted_column_values.device_buf().as_raw_ptr() as usize,
                    ));
                    values_poly_ffi.push(FFITraitObject::new(
                        device_buf_for_column(
                            advice_device,
                            fixed_device,
                            instance_device,
                            column.column_type(),
                            column.index(),
                        )
                        .as_raw_ptr() as usize,
                    ));
                }
                permutation_product_device(
                    &mut d_modified_values,
                    &permutations_poly_ffi,
                    &values_poly_ffi,
                    *beta,
                    *gamma,
                    C::Scalar::DELTA,
                    domain.get_omega(),
                    deltaomega,
                )?;
                for _ in 0..columns.len() {
                    deltaomega *= &C::Scalar::DELTA;
                }

                #[cfg(feature = "profile")]
                end_timer!(gpu_time);
            }

            // The modified_values vector is a vector of products of fractions
            // of the form
            //
            // (p_j(\omega^i) + \delta^j \omega^i \beta + \gamma) /
            // (p_j(\omega^i) + \beta s_j(\omega^i) + \gamma)
            //
            // where i is the index into modified_values, for the jth column in
            // the permutation

            #[cfg(feature = "profile")]
            let z_grand_product_time = start_timer!(|| "Z_i(X) grand product");
            // Device-resident running product Z_i(X). Layout:
            //   z[0]              = last_z (cross-chunk roll-in)
            //   z[1..acc_len]     = scan of modified_values[0..acc_len-1]
            //                       (last_z · ∏_{j=0..i} modified_values[j])
            //   z[acc_len..n]     = RNG blinding factors
            // The scan FFI is in-place on its input buffer, so its output
            // sits at d_scanned[0..acc_len-1]; a single D2D shifts it into
            // d_z[1..acc_len] — the shift stays in HBM and avoids a
            // full-buffer PCIe copy.
            let d_scanned = {
                crate::perf_section!("grand_product_scan");
                grand_product_device(d_modified_values, acc_len - 1, last_z)?
            };

            #[cfg(feature = "profile")]
            end_timer!(z_grand_product_time);

            let d_z = DeviceBuffer::<C::Scalar>::with_capacity_on(n, &HALO2_GPU_CTX);
            unsafe {
                cuda_memcpy_on::<false, true>(
                    d_z.as_mut_raw_ptr(),
                    std::slice::from_ref(&last_z).as_ptr() as *const libc::c_void,
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
            // single `blinding_factors`-element tail H2D into the device
            // buffer — the n-element accumulator stays device-resident.
            let host_blind: Vec<C::Scalar> = (0..blinding_factors)
                .map(|_| C::Scalar::random(&mut rng))
                .collect();
            unsafe {
                cuda_memcpy_on::<false, true>(
                    (d_z.as_mut_raw_ptr() as *mut u8).add(acc_len * scalar_bytes)
                        as *mut libc::c_void,
                    host_blind.as_ptr() as *const libc::c_void,
                    blinding_factors * scalar_bytes,
                    &HALO2_GPU_CTX,
                )
                .map_err(HaloGpuError::from)?;
            }

            // Roll the cross-chunk last_z dependency: matches the host arm
            // `last_z = z[n - (blinding_factors + 1)] = z[acc_len - 1]`,
            // i.e. the last element of the scan region. Single 32-byte D2H.
            unsafe {
                cuda_memcpy_on::<true, false>(
                    &mut last_z as *mut C::Scalar as *mut libc::c_void,
                    (d_z.as_raw_ptr() as *const u8).add((acc_len - 1) * scalar_bytes)
                        as *const libc::c_void,
                    scalar_bytes,
                    &HALO2_GPU_CTX,
                )
                .map_err(HaloGpuError::from)?;
            }
            HALO2_GPU_CTX
                .stream
                .to_host_sync()
                .map_err(HaloGpuError::from)?;

            let z = Polynomial::<C::Scalar, LagrangeCoeff, Device>::from_device(d_z);

            // Single-stream GPU prover: commit Z_i(X) via device-scalars MSM,
            // then device-input iFFT to coeff form. No PCIe traffic on Z_i.
            let commitment = params
                .commit_lagrange_device(&z, Blind::default())
                .to_affine();
            commitments.push(commitment);
            let permutation_product_poly = domain.lagrange_to_coeff_device_input(z)?;

            sets.push(CommittedSet {
                permutation_product_poly,
            });
        }
        drop(d_ones_template);

        Ok((Committed { sets }, commitments))
    }
}

impl<C: CurveAffine> Committed<C> {
    pub(in crate::plonk) fn construct(self) -> Constructed<C> {
        Constructed {
            sets: self
                .sets
                .into_iter()
                .map(|set| ConstructedSet {
                    permutation_product_poly: set.permutation_product_poly,
                })
                .collect(),
        }
    }
}

// NOTE: permutation σ-poly evaluation (previously `permutation::ProvingKey::evaluate`)
// now lives on `GpuProvingKey::evaluate_permutation`, since the gpu
// `permutation::ProvingKey` was deleted in favor of the canonical halo2-axiom pk.

impl<C: CurveAffine> Constructed<C> {
    pub(in crate::plonk) fn evaluate<E: EncodedChallenge<C>, T: TranscriptWrite<C, E>>(
        self,
        pk: &plonk::GpuProvingKey<C>,
        x: ChallengeX<C>,
        transcript: &mut T,
    ) -> Result<Evaluated<C>, Error> {
        let domain = &pk.domain;
        let blinding_factors = pk.cs.blinding_factors();

        {
            let mut sets = self.sets.iter();

            while let Some(set) = sets.next() {
                crate::perf_section!("permutation.evaluate.eval_at_loop");
                // `eval_at` dispatches on the polynomial's storage
                // variant (Host or Device).
                let permutation_product_eval = set.permutation_product_poly.eval_at(*x);

                let permutation_product_next_eval = set
                    .permutation_product_poly
                    .eval_at(domain.rotate_omega(*x, Rotation::next()));

                // Hash permutation product evals
                for eval in iter::empty()
                    .chain(Some(&permutation_product_eval))
                    .chain(Some(&permutation_product_next_eval))
                {
                    transcript.write_scalar(*eval)?;
                }

                // If we have any remaining sets to process, evaluate this set at omega^u
                // so we can constrain the last value of its running product to equal the
                // first value of the next set's running product, chaining them together.
                if sets.len() > 0 {
                    let permutation_product_last_eval = set.permutation_product_poly.eval_at(
                        domain.rotate_omega(*x, Rotation(-((blinding_factors + 1) as i32))),
                    );

                    transcript.write_scalar(permutation_product_last_eval)?;
                }
            }
        }

        Ok(Evaluated { constructed: self })
    }
}

impl<C: CurveAffine> Evaluated<C> {
    pub(in crate::plonk) fn open<'a>(
        &'a self,
        pk: &'a plonk::GpuProvingKey<C>,
        x: ChallengeX<C>,
    ) -> impl Iterator<Item = ProverQuery<'a, C>> + Clone {
        let blinding_factors = pk.cs.blinding_factors();
        let x_next = pk.domain.rotate_omega(*x, Rotation::next());
        let x_last = pk
            .domain
            .rotate_omega(*x, Rotation(-((blinding_factors + 1) as i32)));

        iter::empty()
            .chain(self.constructed.sets.iter().flat_map(move |set| {
                iter::empty()
                    // Open permutation product commitments at x and \omega x
                    .chain(Some(ProverQuery {
                        point: *x,
                        poly: (&set.permutation_product_poly).into(),
                    }))
                    .chain(Some(ProverQuery {
                        point: x_next,
                        poly: (&set.permutation_product_poly).into(),
                    }))
            }))
            // Open it at \omega^{last} x for all but the last set. This rotation is only
            // sensical for the first row, but we only use this rotation in a constraint
            // that is gated on l_0.
            .chain(
                self.constructed
                    .sets
                    .iter()
                    .rev()
                    .skip(1)
                    .flat_map(move |set| {
                        Some(ProverQuery {
                            point: x_last,
                            poly: (&set.permutation_product_poly).into(),
                        })
                    }),
            )
    }
}
