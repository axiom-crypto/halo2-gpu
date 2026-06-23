//! # halo2-axiom-gpu

#![cfg_attr(docsrs, feature(doc_cfg))]
// Lints with too many false positives or stylistic disagreements for this
// crate's heavily-generic field-arithmetic code. Each is intentional; drop
// one only after auditing every site that fires.
#![allow(
    clippy::op_ref,             // field arithmetic uses `&Self` operators idiomatically
    clippy::assign_op_pattern,  // `a = a + b` patterns in math code remain readable
    clippy::too_many_arguments, // prover/keygen entry points have wide generic surfaces
    clippy::upper_case_acronyms // FFT/KZG/NTT enum-variant naming
)]
// FFI wrappers around CUDA kernels: `# Safety` docs are on the `extern "C"`
// declarations; `set_len` after capacity reserve is deliberate because the FFI
// ABI requires uninitialized buffers passed by pointer.
#![allow(clippy::missing_safety_doc, clippy::uninit_vec)]
#![deny(rustdoc::broken_intra_doc_links)]
#![deny(missing_debug_implementations)]
// #![deny(unsafe_code)]

pub mod arithmetic;
/// Frontend circuit API, re-exported from the canonical `halo2-axiom` crate.
pub use halo2_axiom::circuit;
pub use halo2curves;
/// Test-oracle and runtime-fallback CPU implementations corresponding to GPU
/// primitives elsewhere in the crate. This module is `pub` only to let
/// integration tests reach the equivalence oracles; it is **not part of the
/// public API** and is **semver-exempt**. Production consumers should never
/// import from this module directly.
#[doc(hidden)]
pub mod cpu;
pub mod cuda;
/// Circuit development tooling (MockProver etc.), re-exported from canonical `halo2-axiom`.
pub use halo2_axiom::dev;
pub mod fft;
mod multicore;
pub mod plonk;
pub mod poly;
pub use halo2_axiom::transcript;
pub use halo2_axiom::SerdeFormat;
pub(crate) use halo2_axiom::{CurveRead, SerdeCurveAffine, SerdePrimeField};
