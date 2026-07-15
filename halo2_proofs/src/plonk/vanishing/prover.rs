use std::iter;
#[cfg(feature = "profile")]
use std::time::Instant;

use ff::{Field, WithSmallOrderMulGroup};
use group::Curve;

use openvm_cuda_common::copy::{cuda_memcpy_on, MemCopyH2D};
use openvm_cuda_common::d_buffer::DeviceBuffer;
use rand_core::RngCore;

use super::Argument;
use crate::cuda::funcs::{
    poly_multiply_add_device_with_d_scalar, poly_scale_device_with_d_s_minus_one,
};
use crate::cuda::utils::HALO2_GPU_CTX;
use crate::cuda::HaloGpuError;
use crate::{
    arithmetic::CurveAffine,
    plonk::{ChallengeX, GpuError},
    poly::{
        commitment::{Blind, ParamsProver},
        Coeff, Device, DeviceChunks, DevicePolyExt, EvaluationDomain, ExtendedLagrangeCoeff, Host,
        Polynomial, ProverQuery,
    },
    transcript::{EncodedChallenge, TranscriptWrite},
};

pub(in crate::plonk) struct Committed<C: CurveAffine> {
    random_poly: Polynomial<C::Scalar, Coeff, Device>,
}

/// Quotient pieces produced by `Constructed::construct`. Each variant is a
/// homogeneous-residency `Vec<Polynomial<F, Coeff, _>>`; the per-prove
/// residency choice happens at construction time and stays consistent through
/// the rest of the vanishing pipeline.
pub(in crate::plonk) enum HPieces<F> {
    Host(Vec<Polynomial<F, Coeff, Host>>),
    Device(Vec<Polynomial<F, Coeff, Device>>),
}

impl<F> HPieces<F> {
    fn len(&self) -> usize {
        match self {
            HPieces::Host(v) => v.len(),
            HPieces::Device(v) => v.len(),
        }
    }
}

pub(in crate::plonk) struct Constructed<C: CurveAffine> {
    h_pieces: HPieces<C::Scalar>,
    h_blinds: Vec<Blind<C::Scalar>>,
    committed: Committed<C>,
}

pub(in crate::plonk) struct Evaluated<C: CurveAffine> {
    h_poly: crate::poly::MaybeDevice<C::Scalar, Coeff>,
    #[allow(dead_code)]
    h_blind: Blind<C::Scalar>,
    committed: Committed<C>,
}

impl<C: CurveAffine> Argument<C> {
    /// This commitment scheme commits to a _zero polynomial_,
    /// that means our commitment scheme is binding but not hidding.
    /// This is fine for schemes that does not require zero-knowledge.
    pub(in crate::plonk) fn commit<
        'params,
        P: ParamsProver<'params, C>,
        E: EncodedChallenge<C>,
        R: RngCore,
        T: TranscriptWrite<C, E>,
    >(
        params: &P,
        domain: &EvaluationDomain<'_, C::Scalar>,
        _: R,
        transcript: &mut T,
    ) -> Result<Committed<C>, GpuError> {
        crate::perf_section!("vanishing.commit");
        // zk is disabled (see below), so this commits to the constant poly
        // `[ONE, 0, .., 0]`: zero the n coeffs on device, then set coeff[0] = ONE.
        let n = domain.get_n() as usize;
        let d_random_poly: DeviceBuffer<C::Scalar> =
            DeviceBuffer::with_capacity_on(n, &HALO2_GPU_CTX);
        d_random_poly.fill_zero_on(&HALO2_GPU_CTX)?;
        let d_one: DeviceBuffer<C::Scalar> = std::slice::from_ref(&C::Scalar::ONE)
            .to_device_on(&HALO2_GPU_CTX)
            .expect("H2D of ONE failed in vanishing::commit");
        unsafe {
            cuda_memcpy_on::<true, true>(
                d_random_poly.as_mut_raw_ptr(),
                d_one.as_raw_ptr(),
                std::mem::size_of::<C::Scalar>(),
                &HALO2_GPU_CTX,
            )
            .map_err(HaloGpuError::from)?;
        }
        let random_poly = Polynomial::<C::Scalar, Coeff, Device>::from_device(d_random_poly);
        /*
        for coeff in random_poly.iter_mut() {
            *coeff = C::Scalar::random(&mut rng);
        }
        // Sample a random blinding factor
        let random_blind = Blind(C::Scalar::random(rng));

        // Commit
        let c = params.commit(&random_poly, random_blind).to_affine();*/
        let c = params.get_g()[0];
        transcript.write_point(c).unwrap();

        Ok(Committed { random_poly })
    }
}

impl<C: CurveAffine> Committed<C> {
    pub(in crate::plonk) fn construct<
        'params,
        P: ParamsProver<'params, C>,
        E: EncodedChallenge<C>,
        R: RngCore,
        T: TranscriptWrite<C, E>,
    >(
        self,
        params: &P,
        domain: &EvaluationDomain<'_, C::Scalar>,
        h_poly: crate::poly::MaybeDevice<C::Scalar, ExtendedLagrangeCoeff>,
        mut rng: R,
        transcript: &mut T,
    ) -> Result<Constructed<C>, GpuError>
    where
        C::Scalar: WithSmallOrderMulGroup<3>,
    {
        crate::perf_section!("vanishing.construct");
        let h_pieces = match h_poly {
            crate::poly::MaybeDevice::Device(p) => {
                let p = domain.divide_by_vanishing_poly_device(p)?;
                #[cfg(feature = "profile")]
                let h_poly_coset_ifft_time = Instant::now();
                let p = domain.extended_to_coeff_device(p)?;
                #[cfg(feature = "profile")]
                log::info!(
                    "h_poly coset ifft took {:?}",
                    h_poly_coset_ifft_time.elapsed()
                );
                HPieces::Device(p.chunks_device(params.n() as usize))
            }
            crate::poly::MaybeDevice::Host(p) => {
                let p = domain.divide_by_vanishing_poly(p)?;
                #[cfg(feature = "profile")]
                let h_poly_coset_ifft_time = Instant::now();
                let p = domain.extended_to_coeff(p)?;
                #[cfg(feature = "profile")]
                log::info!(
                    "h_poly coset ifft took {:?}",
                    h_poly_coset_ifft_time.elapsed()
                );
                let pieces: Vec<_> = p
                    .values()
                    .chunks_exact(params.n() as usize)
                    .map(|v| domain.coeff_from_vec(v.to_vec()))
                    .collect();
                HPieces::Host(pieces)
            }
        };
        let h_blinds: Vec<_> = (0..h_pieces.len())
            .map(|_| Blind(C::Scalar::random(&mut rng)))
            .collect();

        #[cfg(feature = "profile")]
        let h_commitments_projective_time = Instant::now();
        let h_commitments_projective: Vec<_> = match &h_pieces {
            HPieces::Device(pieces) => pieces
                .iter()
                .zip(h_blinds.iter())
                .map(|(piece, blind)| params.commit_device(piece, *blind))
                .collect(),
            HPieces::Host(pieces) => pieces
                .iter()
                .zip(h_blinds.iter())
                .map(|(piece, blind)| params.commit(piece, *blind))
                .collect(),
        };
        let mut h_commitments = vec![C::identity(); h_commitments_projective.len()];
        C::Curve::batch_normalize(&h_commitments_projective, &mut h_commitments);
        let h_commitments = h_commitments;
        #[cfg(feature = "profile")]
        log::info!(
            "h_commitments_projective msm [{}] took {:?}",
            h_pieces.len(),
            h_commitments_projective_time.elapsed()
        );

        for c in h_commitments.iter() {
            transcript.write_point(*c)?;
        }

        Ok(Constructed {
            h_pieces,
            h_blinds,
            committed: self,
        })
    }
}

impl<C: CurveAffine> Constructed<C> {
    pub(in crate::plonk) fn evaluate<E: EncodedChallenge<C>, T: TranscriptWrite<C, E>>(
        self,
        // Unused: random_poly is constant, so its evaluation ignores the point.
        _x: ChallengeX<C>,
        xn: C::Scalar,
        domain: &EvaluationDomain<'_, C::Scalar>,
        transcript: &mut T,
    ) -> Result<Evaluated<C>, GpuError> {
        crate::perf_section!("vanishing.evaluate");
        // Device-resident h_pieces fold via an explicit accumulator loop
        // (`poly_scale_device` + `poly_multiply_add_device`); host-resident
        // pieces fold through the CPU `Mul<F>` / `Add` impls.
        let h_poly = match self.h_pieces {
            HPieces::Device(pieces) => {
                let n = domain.get_n() as usize;
                // Zero the fold accumulator on-device (memset) instead of
                // uploading a host zero-vec (~256 MiB alloc + pageable H2D at
                // k=23). Byte-identical on BN254: all-bits-zero == `Fr::ZERO`,
                // and same-stream ordering fills before the fold reads.
                let mut d_acc: DeviceBuffer<C::Scalar> =
                    DeviceBuffer::with_capacity_on(n, &HALO2_GPU_CTX);
                d_acc.fill_zero_on(&HALO2_GPU_CTX)?;
                // Hoist the per-iteration scalar H2Ds out of the fold loop:
                // `poly_scale_device` computes `acc += (xn - 1) * acc`
                // (≡ `acc *= xn`); `poly_multiply_add_device` runs
                // `acc += ONE * eval`. Both scalars are loop-invariant, so
                // upload each once.
                let xn_minus_one = xn - C::Scalar::ONE;
                let d_xn_minus_one: DeviceBuffer<C::Scalar> = std::slice::from_ref(&xn_minus_one)
                    .to_device_on(&HALO2_GPU_CTX)
                    .expect("H2D of xn-1 failed in Constructed::evaluate");
                let d_one: DeviceBuffer<C::Scalar> = std::slice::from_ref(&C::Scalar::ONE)
                    .to_device_on(&HALO2_GPU_CTX)
                    .expect("H2D of ONE failed in Constructed::evaluate");
                {
                    crate::perf_section!("device_fold");
                    for piece in pieces.iter().rev() {
                        let d_eval = piece.device_buf();
                        poly_scale_device_with_d_s_minus_one(&mut d_acc, &d_xn_minus_one)
                            .map_err(GpuError::HaloGpu)?;
                        poly_multiply_add_device_with_d_scalar(&mut d_acc, d_eval, &d_one)
                            .map_err(GpuError::HaloGpu)?;
                    }
                }
                crate::poly::MaybeDevice::Device(
                    Polynomial::<C::Scalar, Coeff, Device>::from_device(d_acc),
                )
            }
            HPieces::Host(pieces) => crate::poly::MaybeDevice::Host(
                pieces
                    .iter()
                    .rev()
                    .fold(domain.empty_coeff(), |acc, eval| acc * xn + eval),
            ),
        };

        let h_blind = self
            .h_blinds
            .iter()
            .rev()
            .fold(Blind(C::Scalar::ZERO), |acc, eval| acc * Blind(xn) + *eval);

        // random_poly is the constant `[ONE, 0, .., 0]`, so its eval is ONE.
        let random_eval = C::Scalar::ONE;
        transcript.write_scalar(random_eval)?;

        Ok(Evaluated {
            h_poly,
            h_blind,
            committed: self.committed,
        })
    }
}

impl<C: CurveAffine> Evaluated<C> {
    pub(in crate::plonk) fn open(
        &self,
        x: ChallengeX<C>,
    ) -> impl Iterator<Item = ProverQuery<'_, C>> + Clone {
        let h_poly_ref = match &self.h_poly {
            crate::poly::MaybeDevice::Host(p) => crate::poly::PolyRef::Host(p),
            crate::poly::MaybeDevice::Device(p) => crate::poly::PolyRef::Device(p),
        };
        iter::empty()
            .chain(Some(ProverQuery {
                point: *x,
                poly: h_poly_ref,
            }))
            .chain(Some(ProverQuery {
                point: *x,
                poly: crate::poly::PolyRef::Device(&self.committed.random_poly),
            }))
    }
}
