//! Fiat-Shamir transcript — re-export of the canonical halo2-axiom transcript.
//!
//! The transcript traits/types are the **canonical** `halo2_axiom::transcript`
//! ones rather than a GPU fork. This is required so that the canonical
//! [`VerifyingKey::hash_into`](crate::plonk::VerifyingKey) (and any other
//! canonical API bounded on `halo2_axiom::transcript::Transcript`) accepts the
//! transcript types the GPU prover/verifier and downstream consumers
//! (`snark-verifier`) construct.
//!
//! The two batch-read helpers below are `pub(crate)` in halo2-axiom, so a glob
//! `pub use` cannot re-export them; they are re-defined here as thin wrappers
//! over the re-exported [`TranscriptRead`] trait. They are crate-internal to the
//! GPU verifier and cross no external boundary.

pub use halo2_axiom::transcript::*;

use halo2curves::CurveAffine;
use std::io;

/// Read a vector of `n` points from the transcript.
pub(crate) fn read_n_points<C: CurveAffine, E: EncodedChallenge<C>, T: TranscriptRead<C, E>>(
    transcript: &mut T,
    n: usize,
) -> io::Result<Vec<C>> {
    (0..n).map(|_| transcript.read_point()).collect()
}

/// Read a vector of `n` scalars from the transcript.
pub(crate) fn read_n_scalars<C: CurveAffine, E: EncodedChallenge<C>, T: TranscriptRead<C, E>>(
    transcript: &mut T,
    n: usize,
) -> io::Result<Vec<C::Scalar>> {
    (0..n).map(|_| transcript.read_scalar()).collect()
}
