//! Key generation re-exports.
//!
//! The GPU crate no longer defines its own keygen. `VerifyingKey`/`ProvingKey`
//! are the canonical halo2-axiom types (the serde source-of-truth), produced by
//! halo2-axiom's keygen on a `halo2_axiom::plonk::Circuit`. The resulting host
//! `ProvingKey` is wrapped for GPU proving via [`crate::plonk::GpuProvingKey::from_host`].
pub use halo2_axiom::plonk::{keygen_pk, keygen_pk2, keygen_vk, keygen_vk_custom};
