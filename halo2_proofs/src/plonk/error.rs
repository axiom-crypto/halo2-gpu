use std::error;
use std::fmt;
use std::io;

use openvm_cuda_common::error::CudaError;

use crate::cuda::HaloGpuError;

/// This is an error that could occur during proving or circuit synthesis.
#[derive(Debug)]
pub enum GpuError {
    /// An error from the canonical halo2-axiom proving/synthesis stack
    /// (synthesis, instance and constraint-system failures, transcript I/O,
    /// lookup tables, ...). The GPU keygen and prover drive the canonical
    /// `Circuit`/`Assignment` frontend, so these already arrive typed as
    /// [`halo2_axiom::plonk::Error`]. Wrapping rather than re-enumerating keeps
    /// the variants and their `Display` in lockstep with upstream.
    Canonical(halo2_axiom::plonk::Error),
    /// A CUDA FFI call returned a non-zero status. Surfaces the typed
    /// `openvm_cuda_common::error::CudaError`.
    Cuda(CudaError),
    /// A halo2-gpu-internal error: synthetic preflight rejection or wrapped
    /// CUDA error from a fn that can produce both. Distinct from `Cuda`
    /// because the variants distinguish synthetic-vs-driver origin at the
    /// type level.
    HaloGpu(HaloGpuError),
}

impl From<io::Error> for GpuError {
    fn from(error: io::Error) -> Self {
        // The only place we can get io::Error from is the transcript; route it
        // through the canonical error so it Displays identically.
        GpuError::Canonical(error.into())
    }
}

impl From<CudaError> for GpuError {
    fn from(err: CudaError) -> Self {
        GpuError::Cuda(err)
    }
}

impl From<HaloGpuError> for GpuError {
    fn from(err: HaloGpuError) -> Self {
        GpuError::HaloGpu(err)
    }
}

/// Wraps a canonical halo2-axiom error. The GPU keygen and prover drive the
/// canonical `Circuit::synthesize` (whose `Assignment` methods return
/// `halo2_axiom::plonk::Error`), so this bridge lets them propagate via `?`.
impl From<halo2_axiom::plonk::Error> for GpuError {
    fn from(e: halo2_axiom::plonk::Error) -> Self {
        GpuError::Canonical(e)
    }
}

impl GpuError {
    /// Constructs the canonical `NotEnoughRowsAvailable` error.
    pub(crate) fn not_enough_rows_available(current_k: u32) -> Self {
        GpuError::Canonical(halo2_axiom::plonk::Error::NotEnoughRowsAvailable { current_k })
    }
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuError::Canonical(e) => write!(f, "{}", e),
            GpuError::Cuda(e) => write!(f, "CUDA error: {}", e),
            GpuError::HaloGpu(e) => write!(f, "halo2-gpu error: {}", e),
        }
    }
}

impl error::Error for GpuError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            // Transparent over the canonical error: preserves the prior
            // observable source (the transcript `io::Error`, otherwise none).
            GpuError::Canonical(e) => e.source(),
            GpuError::Cuda(e) => Some(e),
            GpuError::HaloGpu(e) => Some(e),
        }
    }
}
