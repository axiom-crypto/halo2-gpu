//! Halo2-gpu-specific Result error type.
//!
//! Wraps `cuda_common::CudaError` for real CUDA driver errors and adds
//! typed variants for halo2-gpu-internal synthetic errors — host-side
//! OOM detected by `query_device_free_bytes_for_chunking`-style preflight
//! chunking, and shape-validation rejections in pure-host advisory FFIs.
//! The synthetic variants are kept distinct from `Cuda(CudaError)` at the
//! type level, so callers can pattern-match on origin without inspecting
//! integer codes.

use openvm_cuda_common::error::{CudaError, MemCopyError};
use std::fmt;

use crate::cuda::error::CudaStatus;

/// Halo2-gpu Result error type.
///
/// Marked `#[non_exhaustive]` so future synthetic variants (e.g. for
/// new host-side preflight FFIs) can be added without breaking
/// downstream pattern matches.
#[derive(Debug)]
#[non_exhaustive]
pub enum HaloGpuError {
    /// Host-side preflight detected insufficient GPU memory for the
    /// requested operation. `context` names the caller (e.g.
    /// `"get_fft_split_radix_gpu"`); `magnitude` is the shape parameter
    /// (`log_n` for FFT-shaped paths, raw `length` for unpack-shaped
    /// paths) and `free_bytes` is the result of the host-side free-memory
    /// query at the moment the rejection was emitted.
    InsufficientGpuMemory { context: &'static str, magnitude: u64, free_bytes: u64 },
    /// Caller passed an unsupported parameter combination — reachable in
    /// principle but indicates a shape-validation gap upstream. Typically
    /// signaled by a host-side advisory FFI returning a sentinel value.
    InvalidParameter { context: &'static str, magnitude: u64 },
    /// A real CUDA driver error returned by the FFI layer. Source of
    /// truth for `code` / `name` / `message`.
    Cuda(CudaError),
}

impl From<CudaError> for HaloGpuError {
    fn from(e: CudaError) -> Self {
        Self::Cuda(e)
    }
}

impl From<CudaStatus> for HaloGpuError {
    fn from(status: CudaStatus) -> Self {
        Self::Cuda(status.into())
    }
}

impl From<MemCopyError> for HaloGpuError {
    fn from(e: MemCopyError) -> Self {
        match e {
            MemCopyError::Cuda(c) => Self::Cuda(c),
            MemCopyError::SizeMismatch { .. } => unreachable!(
                "raw cuda_memcpy_on cannot return MemCopyError::SizeMismatch (only the slice-shaped MemCopyH2D helpers do)"
            ),
        }
    }
}

impl fmt::Display for HaloGpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InsufficientGpuMemory {
                context,
                magnitude,
                free_bytes,
            } => write!(
                f,
                "{context}: insufficient GPU memory (magnitude = {magnitude}, free_bytes = {free_bytes})"
            ),
            Self::InvalidParameter { context, magnitude } => {
                write!(f, "{context}: invalid parameter (magnitude = {magnitude})")
            }
            Self::Cuda(e) => write!(f, "CUDA error: {e}"),
        }
    }
}

impl std::error::Error for HaloGpuError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Cuda(e) => Some(e),
            _ => None,
        }
    }
}
