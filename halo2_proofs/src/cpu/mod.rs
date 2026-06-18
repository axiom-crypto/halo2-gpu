//! Consolidated home for CPU implementations of operations that have a GPU
//! production equivalent: pure test oracles, small-input fallbacks,
//! VRAM-bounded fallbacks, and reference implementations. The folder mirrors
//! the production module structure so the CPU counterpart of a kernel sits at
//! the corresponding location under `cpu/`.
//!
//! Production callsites import the small-input and VRAM-bounded fallbacks
//! from here; pure test oracles inside this folder retain `#[cfg(test)]`.

pub mod arithmetic;
#[cfg(test)]
pub(crate) mod evaluator;
pub(crate) mod poly;
