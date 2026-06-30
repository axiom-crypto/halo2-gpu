//! Re-exports the canonical recursive FFT machinery from halo2-axiom.
//! halo2-axiom-gpu used to ship a verbatim fork of these items, which produced
//! two nominally distinct `FFTData<F>` types and a type-mismatch at the
//! `EvaluationDomain` wrapper boundary. The recursive internals reach into
//! `FFTData`'s private fields, so the only viable consolidation is to drop the
//! local copies entirely.
pub use halo2_axiom::fft::recursive::{fft, FFTData, FFTStage};
