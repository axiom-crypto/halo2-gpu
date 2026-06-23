use super::{construct_intermediate_sets, ChallengeV, Query};
use crate::arithmetic::powers;
use crate::cpu::arithmetic::kate_division;
use crate::cuda::funcs::multiopen_poly_calculation_gpu;
use crate::cuda::utils::FFITraitObject;
use crate::poly::commitment::ParamsProver;
use crate::poly::commitment::Prover;
use crate::poly::kzg::commitment::{KZGCommitmentScheme, ParamsKZG};
use crate::poly::query::ProverQuery;
use crate::poly::Coeff;
use crate::poly::{commitment::Blind, PolyRef, Polynomial};
use crate::transcript::{EncodedChallenge, TranscriptWrite};
use crate::SerdeCurveAffine;

#[cfg(feature = "profile")]
use ark_std::{end_timer, start_timer};
use ff::Field;
use group::prime::PrimeCurveAffine;
use group::Curve;
use pairing::Engine;
use rand_core::RngCore;
use std::fmt::Debug;
use std::io;

/// Concrete KZG prover with GWC variant
#[derive(Debug)]
pub struct ProverGWC<'params, E: Engine> {
    params: &'params ParamsKZG<E>,
}

/// Create a multi-opening proof
impl<'params, E: Engine + Debug> Prover<'params, KZGCommitmentScheme<E>> for ProverGWC<'params, E>
where
    E::G1Affine: SerdeCurveAffine<ScalarExt = E::Fr, CurveExt = E::G1>,
    E::G2Affine: SerdeCurveAffine,
{
    const QUERY_INSTANCE: bool = false;

    fn new(params: &'params ParamsKZG<E>) -> Self {
        Self { params }
    }

    /// Create a multi-opening proof
    fn create_proof<
        'com,
        Ch: EncodedChallenge<E::G1Affine>,
        T: TranscriptWrite<E::G1Affine, Ch>,
        R,
        I,
    >(
        &self,
        _: R,
        transcript: &mut T,
        queries: I,
    ) -> io::Result<()>
    where
        I: IntoIterator<Item = ProverQuery<'com, E::G1Affine>> + Clone,
        R: RngCore,
    {
        #[cfg(feature = "profile")]
        let multiopen_time = start_timer!(|| "phase5 multiopen using kzg gwc");
        let v: ChallengeV<_> = transcript.squeeze_challenge_scalar();
        #[cfg(feature = "profile")]
        let intermediate_set_time = start_timer!(|| "construct intermediate set");
        let commitment_data = construct_intermediate_sets(queries);
        #[cfg(feature = "profile")]
        end_timer!(intermediate_set_time);

        {
            // Single-stream GPU prover: run the whole batch on gpu 0.
            let zero = || -> Polynomial<E::Fr, Coeff> {
                Polynomial::new(vec![E::Fr::ZERO; self.params.n as usize])
            };
            let mut w_result_many =
                vec![<E::G1Affine as PrimeCurveAffine>::identity(); commitment_data.len()];
            for (i, commitment_at_a_point) in commitment_data.iter().enumerate() {
                let poly_length = self.params.n as usize;
                let batch_size = commitment_at_a_point.queries.len();
                let z = commitment_at_a_point.point;
                #[cfg(feature = "profile")]
                let commitment_at_point_time = start_timer!(|| format!(
                    "gpu[{}] iter[{}] commitment_at_a_point, size {}",
                    0, i, batch_size
                ));
                let mut poly_acc = zero();
                let mut poly_in_many: Vec<FFITraitObject> = Vec::with_capacity(batch_size);
                let mut evaluate_point_many: Vec<E::Fr> = Vec::with_capacity(batch_size);
                let mut evaluate_result_many = vec![E::Fr::ZERO; batch_size];
                for query in commitment_at_a_point.queries.iter() {
                    assert_eq!(query.get_point(), z);
                    let host_slice: &[E::Fr] = match query.get_commitment().poly {
                        PolyRef::Host(p) => p.values(),
                        PolyRef::Device(_) => panic!(
                            "gwc::create_proof does not support Device polynomials; \
                             caller must hold a Host-resident poly before reaching gwc multiopen"
                        ),
                    };
                    poly_in_many.push(FFITraitObject::from_ref(&host_slice[0]));
                    evaluate_point_many.push(query.get_point());
                }
                let challenge_point = (0..batch_size)
                    .zip(powers(*v))
                    .map(|(_, power_of_v)| power_of_v)
                    .collect::<Vec<_>>();
                multiopen_poly_calculation_gpu(
                    poly_in_many,
                    challenge_point,
                    poly_acc.values_mut(), // multiply_add
                    evaluate_point_many,
                    &mut evaluate_result_many, // evaluation
                    poly_length,
                )
                .expect("multiopen_poly_calculation_gpu failed in gwc::create_proof");
                let acc_eval = evaluate_result_many
                    .iter()
                    .fold(E::Fr::ZERO, |acc, eval| acc * (*v) + eval);

                let poly_batch = &poly_acc - acc_eval;
                let witness_poly = Polynomial::new(kate_division(poly_batch.values(), z));
                w_result_many[i] = self
                    .params
                    .commit_with_gpu(&witness_poly, 0, Blind::default())
                    .to_affine();
                #[cfg(feature = "profile")]
                end_timer!(commitment_at_point_time);
            }

            for w in w_result_many.iter() {
                transcript.write_point(*w)?;
            }
        }
        #[cfg(feature = "profile")]
        end_timer!(multiopen_time);
        Ok(())
    }
}
