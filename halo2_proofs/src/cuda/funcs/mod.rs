//! Rust-side wrappers around the halo2-gpu CUDA kernel launchers
//! (`extern "C"` block lives in `cuda::culib`).
//!
//! Wrapper contract: all work enqueues on `HALO2_GPU_CTX.stream`;
//! `&[F]` / `&mut [F]` args are H2D/D2H'd internally; `&DeviceBuffer<F>`
//! args stay device-resident. Scalar-result wrappers sync; in-place
//! device-buffer wrappers do not. Every wrapper calls
//! `ensure_current_device_matches_ctx()` before the FFI — the C-side
//! launchers do not call `cudaSetDevice`.

pub mod batch_invert;
pub mod column_pool;
pub mod grand_product;
pub mod lookup;
pub mod multiexp;
pub mod multiopen;
pub mod ntt;
pub mod omega;
pub mod permutation;
pub mod polynomial_ops;

pub use batch_invert::*;
pub use column_pool::*;
pub use grand_product::*;
pub use lookup::*;
pub use multiexp::*;
pub use multiopen::*;
pub use ntt::*;
pub use omega::*;
pub use permutation::*;
pub use polynomial_ops::*;
