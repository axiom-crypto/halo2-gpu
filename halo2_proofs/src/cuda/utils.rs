//! Halo2-gpu runtime context, FFI shim, and the single sanctioned
//! `cudaMemGetInfo` site (`query_device_free_bytes_for_chunking`).

use git_version::git_version;
use once_cell::sync::Lazy;
use openvm_cuda_common::copy::MemCopyH2D;
use openvm_cuda_common::d_buffer::DeviceBuffer;
use openvm_cuda_common::error::MemCopyError;
use openvm_cuda_common::stream::GpuDeviceCtx;

pub const GIT_VERSION: &str = git_version!(fallback = "unknown");

/// Empty-slice-tolerant variant of `[T]::to_device_on(&HALO2_GPU_CTX)`.
///
/// `DeviceBuffer::with_capacity_on` asserts `len != 0`; returning a
/// null-backed `DeviceBuffer::new()` for empty inputs preserves the
/// no-op semantics callers expect (the kernel sees `ptr=null, len=0`
/// and never dereferences).
#[allow(dead_code)]
pub(crate) fn to_device_on_safe<T>(slice: &[T]) -> Result<DeviceBuffer<T>, MemCopyError> {
    if slice.is_empty() {
        Ok(DeviceBuffer::<T>::new())
    } else {
        slice.to_device_on(&HALO2_GPU_CTX)
    }
}

/// Crate-level single-device context: one non-blocking CUDA stream backing
/// all Rust-owned allocations and memcpys. C++ kernels still launch on
/// their own internal streams, so every alloc / copy is synchronized
/// against this stream before returning — see `src/cuda/modules.rs` and
/// `src/cuda/funcs.rs` call sites for where the sync fences are placed.
pub static HALO2_GPU_CTX: Lazy<GpuDeviceCtx> = Lazy::new(|| {
    GpuDeviceCtx::for_current_device().expect("failed to create halo2-gpu GpuDeviceCtx")
});

/// Opaque handle used by kernel ABIs that take an array of polynomial
/// pointers (e.g. `_halo2_fft_many`, `_halo2_multiopen_poly_calculation`).
/// The underlying value is a raw device pointer packed into a `usize`.
#[derive(Debug, Clone)]
#[repr(C)]
pub struct FFITraitObject {
    ptr: usize,
}

// Compile-time FFI safety assertions.
// `FFITraitObject` is passed to CUDA as an array of raw device pointers,
// so its layout must be exactly one machine word on the target platform.
const _: () = assert!(
    std::mem::size_of::<FFITraitObject>() == std::mem::size_of::<usize>(),
    "FFITraitObject must be a single machine word to match CUDA void*[] ABI"
);
const _: () = assert!(
    std::mem::align_of::<FFITraitObject>() == std::mem::align_of::<usize>(),
    "FFITraitObject alignment must match usize to match CUDA void*[] ABI"
);

impl FFITraitObject {
    pub fn new(ptr: usize) -> Self {
        Self { ptr }
    }

    /// Build an `FFITraitObject` wrapping the host address of `value`.
    /// Kernels that take a `*const FFITraitObject` for a single buffer or
    /// scalar expect a 1-element `{ ptr: usize }` word; take `&obj as
    /// *const FFITraitObject` at the call site to produce that pointer.
    ///
    /// `FFITraitObject` records the host address of `T` for the C++ kernel
    /// to read or write through. Rust does not enforce the kernel's actual
    /// access pattern — the kernel's documented contract does. `from_ref`
    /// accepts `&T` (and via auto-coerce, `&mut T`) because the conversion
    /// only extracts an address.
    pub fn from_ref<T>(value: &T) -> Self {
        Self::new(value as *const T as usize)
    }

    /// Build an `FFITraitObject` from the host address of `slice[0]`. Used
    /// by kernels that take `*const FFITraitObject` for an array of buffers
    /// (e.g. `_halo2_fft_many`, `_halo2_multiopen_*`).
    ///
    /// # Panics
    /// Panics if `slice` is empty. Matches the panicking behavior of the
    /// `transmute(&slice[0])` pattern this replaces.
    pub fn from_slice<T>(slice: &[T]) -> Self {
        assert!(
            !slice.is_empty(),
            "FFITraitObject::from_slice requires a non-empty slice"
        );
        Self::new(slice.as_ptr() as usize)
    }
}

/// Reinterpret a typed host slice as its raw byte representation. Used by
/// the `MemCopy*` traits, which operate on `&[u8]` → `DeviceBuffer<u8>`.
///
/// # Safety
/// `T: Copy` rules out drop glue but does **not** rule out padding bytes,
/// uninitialised bytes, or layout instability. The function is therefore
/// `unsafe`. Callers must guarantee:
///
/// 1. `T`'s in-memory representation has no padding / uninitialised
///    bytes — equivalently, `T` is plain-old-data (POD).
/// 2. `T`'s byte image is exactly what the consuming kernel expects.
///
/// All current call sites pass field scalars (`Fr`, `Fp`) and curve
/// coordinates whose `repr(C)` POD layout is what the halo2 GPU FFI was
/// implemented against.
pub(crate) unsafe fn as_bytes<T: Copy>(slice: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, std::mem::size_of_val(slice)) }
}

/// Free GPU memory minus a 256 MiB reservation (matches the legacy C++
/// helper) for chunk-sizing decisions. Single sanctioned
/// `cudaMemGetInfo` site; advisory FFIs take the value as a trailing
/// `free_bytes` parameter. Panics on device-bind failure — a wrong
/// device would produce silently incorrect counters, and downstream
/// `step_by(chunk_size)` panics on 0 anyway.
pub(crate) fn query_device_free_bytes_for_chunking() -> usize {
    // `cudaMemGetInfo` reads against the *current* device; worker
    // threads that never initialized `HALO2_GPU_CTX` default to
    // logical 0 — pin to the singleton's device first.
    ensure_current_device_matches_ctx().unwrap_or_else(|err| {
        panic!(
            "query_device_free_bytes_for_chunking: device-bind to \
             HALO2_GPU_CTX.device_id={} failed (err = {}); subsequent \
             CUDA ops would target the wrong device — aborting before \
             any silent wrong-device decisions are made",
            HALO2_GPU_CTX.device_id, err
        )
    });

    let mut free: usize = 0;
    let mut total: usize = 0;
    unsafe {
        cudaMemGetInfo(&mut free, &mut total);
    }
    const RESERVED: usize = 256 << 20;
    free.saturating_sub(RESERVED)
}

/// Pins the calling thread to `HALO2_GPU_CTX`'s device. Required
/// before any CUDA FFI: `Lazy` only sets the device on its initializing
/// thread, so worker threads otherwise inherit the runtime default.
pub(crate) fn ensure_current_device_matches_ctx() -> Result<(), crate::cuda::HaloGpuError> {
    openvm_cuda_common::common::set_device_by_id(HALO2_GPU_CTX.device_id as i32)
        .map_err(crate::cuda::HaloGpuError::from)
}

#[link(name = "cudart")]
extern "C" {
    fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> i32;
}

// ---------------------------------------------------------------------------
// Performance-instrumentation macros (perf/metrics branch).
//
// These are pure observability and intentionally unconditional: enabling /
// disabling them is the subscriber's job (`RUST_LOG` for tracing, presence
// of a metrics layer for counters). Their only runtime cost is one mutex
// read of `MEMORY_MANAGER` per `perf_section!` (`MemTracker::start`) and a
// single tracing event per `perf_h2d!` / `perf_d2h!`.
// ---------------------------------------------------------------------------

// Re-export the dependencies used by the four `#[macro_export]` macros below
// so callers don't need to depend on `tracing` or `openvm-cuda-common`
// themselves. Macro expansions resolve `$crate::cuda::utils::__perf_reexports::*`
// in the caller's namespace, which always points back at this crate.
#[doc(hidden)]
pub mod __perf_reexports {
    pub use ::openvm_cuda_common::memory_manager::MemTracker;
    pub use ::tracing::{info, info_span};
}

/// Open a perf scope: enters a tracing `info_span!` *and* arms a
/// `MemTracker` for the same label. Both close on scope drop, so a single
/// `perf_section!("foo")` line at the top of a function gives wall-clock
/// (via ForestLayer) and GPU-mem delta/peak (via MemTracker's drop info!)
/// for that function. The `section`/`label` value matches between the two
/// emitters so report extraction can join them on label.
///
/// # Caveats
///
/// - **Sync only.** Holds an `EnteredSpan` guard which is `!Send` — do
///   not call across an `.await`. Async callers should use
///   `info_span!(...).in_scope(|| async { ... }.instrument(span))`
///   patterns directly, not this macro.
#[macro_export]
macro_rules! perf_section {
    ($label:literal) => {
        let _perf_span =
            $crate::cuda::utils::__perf_reexports::info_span!("halo2_section", phase = $label)
                .entered();
        let _perf_mem = $crate::cuda::utils::__perf_reexports::MemTracker::start($label);
    };
}

/// Open a perf scope with a fresh GPU memory peak baseline. Use at the
/// outermost prover entry point so the reported peak is per-proof rather
/// than process-lifetime.
///
/// # Caveats
///
/// - **Single-proof scope only.** `start_and_reset_peak` zeroes a
///   process-global counter in `openvm_cuda_common::memory_manager`. Two
///   concurrent proofs in the same process would stomp each other's
///   baseline. halo2-gpu's prover loop is sync-sequential and the
///   `HALO2_GPU_CTX` singleton enforces single-stream, so this is fine
///   today — but new callers must preserve that invariant.
///
/// - **Outermost entry only.** Use at `plonk::prover::create_proof` (or
///   any future top-level prover entry); never nest. Inner sections want
///   `perf_section!`, which leaves the peak counter alone.
///
/// - **Sync only.** Holds an `EnteredSpan` guard which is `!Send`; do
///   not call across an `.await`. (Same shape as `tracing::span!.entered()`.)
#[macro_export]
macro_rules! perf_section_root {
    ($label:literal) => {
        let _perf_span =
            $crate::cuda::utils::__perf_reexports::info_span!("halo2_section", phase = $label)
                .entered();
        let _perf_mem =
            $crate::cuda::utils::__perf_reexports::MemTracker::start_and_reset_peak($label);
    };
}

/// Record an H2D transfer: one `tracing::info!` line on `target =
/// "halo2_perf"` carrying the byte count and a stable label. The report
/// generator parses these lines to total transfer volume per section.
/// `$bytes` is the raw byte count (typed-slice callers should pass
/// `len * size_of::<T>()`).
#[macro_export]
macro_rules! perf_h2d {
    ($label:literal, $bytes:expr) => {
        $crate::cuda::utils::__perf_reexports::info!(
            target: "halo2_perf",
            kind = "h2d",
            label = $label,
            bytes = $bytes as u64,
        );
    };
}

/// Record a D2H transfer; complement of `perf_h2d!`.
#[macro_export]
macro_rules! perf_d2h {
    ($label:literal, $bytes:expr) => {
        $crate::cuda::utils::__perf_reexports::info!(
            target: "halo2_perf",
            kind = "d2h",
            label = $label,
            bytes = $bytes as u64,
        );
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ffi_trait_object_from_slice_round_trip() {
        let v: [u64; 4] = [42, 43, 44, 45];
        let obj = FFITraitObject::from_slice(&v);
        assert_eq!(obj.ptr, v.as_ptr() as usize);
    }

    #[test]
    fn ffi_trait_object_from_slice_matches_from_ref_on_single_element() {
        let v: [u64; 1] = [42];
        assert_eq!(
            FFITraitObject::from_slice(&v).ptr,
            FFITraitObject::from_ref(&v[0]).ptr,
        );
    }

    #[test]
    #[should_panic(expected = "non-empty")]
    fn ffi_trait_object_from_slice_panics_on_empty() {
        let v: [u64; 0] = [];
        let _ = FFITraitObject::from_slice(&v);
    }
}
