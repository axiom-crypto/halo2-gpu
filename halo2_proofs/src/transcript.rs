//! Fiat-Shamir transcript — the canonical halo2-axiom transcript, re-exported so
//! that canonical APIs bounded on `halo2_axiom::transcript::Transcript` (e.g.
//! `VerifyingKey::hash_into`) accept the types the GPU prover/verifier and
//! downstream consumers (`snark-verifier`) construct. The glob also carries the
//! `read_n_points`/`read_n_scalars` batch-read helpers.

pub use halo2_axiom::transcript::*;
