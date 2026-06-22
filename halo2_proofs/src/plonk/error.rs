use std::error;
use std::fmt;
use std::io;

use openvm_cuda_common::error::CudaError;

use crate::cuda::HaloGpuError;

/// An error that could occur during proving or circuit synthesis.
#[derive(Debug)]
pub enum GpuError {
    /// An error from the canonical halo2-axiom proving/synthesis stack, already
    /// typed as [`halo2_axiom::plonk::Error`]. Wrapping keeps the variants and
    /// their `Display` in lockstep with upstream.
    Canonical(halo2_axiom::plonk::Error),
    /// A CUDA FFI call returned a non-zero status.
    Cuda(CudaError),
    /// A halo2-gpu-internal error: synthetic preflight rejection or wrapped CUDA
    /// error, kept distinct from `Cuda` to mark synthetic-vs-driver origin.
    HaloGpu(HaloGpuError),
}

impl From<io::Error> for GpuError {
    fn from(error: io::Error) -> Self {
        // io::Error only ever comes from the transcript; route it through the
        // canonical error so it Displays identically.
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

/// Bridges canonical `Assignment`/synthesis errors so they propagate via `?`.
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
            // Transparent over the canonical error (the transcript `io::Error`,
            // otherwise none).
            GpuError::Canonical(e) => e.source(),
            GpuError::Cuda(e) => Some(e),
            GpuError::HaloGpu(e) => Some(e),
        }
    }
}
