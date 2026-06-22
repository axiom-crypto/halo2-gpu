//! Serde helpers re-exported from the canonical halo2-axiom crate. Keeping one
//! source-of-truth trait family means the GPU key serialization (which delegates
//! to the canonical `ProvingKey`/`VerifyingKey`) and downstream consumers share
//! identical `Serde*` bounds and byte formats.

pub use halo2_axiom::{CurveRead, SerdeCurveAffine, SerdeFormat, SerdePrimeField};
