//! Halo2-gpu extensions to `openvm_cuda_common::d_buffer::DeviceBuffer<T>`:
//! range-based mutable slicing and host-to-slice copy.
//!
//! [`DeviceBufferMutSlice`] borrows the parent buffer mutably and remembers
//! the `[lo, hi)` element range it covers.
//! [`DeviceBufferMutSlice::copy_from_host`] copies an equally-sized host
//! slice into the range with a single `cudaMemcpyAsync`, enqueued on the
//! provided `GpuDeviceCtx`'s stream (it does not touch the crate-global
//! `HALO2_GPU_CTX`).
//!
//! This lives in halo2-gpu because `openvm_cuda_common` only exposes
//! zero-fill (`fill_zero_on` / `fill_zero_suffix_on`, both `cudaMemsetAsync`)
//! and whole-buffer H2D copy (`MemCopyH2D::copy_to_on`), with no slice type.

use std::ffi::c_void;
use std::ops::{Bound, RangeBounds};

use openvm_cuda_common::copy::cuda_memcpy_on;
use openvm_cuda_common::d_buffer::DeviceBuffer;
use openvm_cuda_common::stream::GpuDeviceCtx;

use crate::cuda::HaloGpuError;

/// A mutable device-side range `[lo, hi)` inside a [`DeviceBuffer<T>`].
///
/// Obtained via [`DeviceBufferExt::mut_slice`]. Holds a `&mut DeviceBuffer<T>`
/// so the borrow checker prevents overlapping mutable views of the same
/// buffer for the lifetime of the slice.
#[derive(Debug)]
pub struct DeviceBufferMutSlice<'a, T> {
    buf: &'a mut DeviceBuffer<T>,
    pub lo: usize,
    pub hi: usize,
}

impl<'a, T> DeviceBufferMutSlice<'a, T> {
    /// Number of elements in the slice.
    pub fn len(&self) -> usize {
        self.hi - self.lo
    }

    pub fn is_empty(&self) -> bool {
        self.hi == self.lo
    }

    /// Copies `src` (host memory) into the slice on `device_ctx`'s stream via
    /// `cudaMemcpyAsync`.
    ///
    /// Errors with [`HaloGpuError::InvalidParameter`] if `src.len() != self.len()`.
    /// This is a straight H2D transfer of the same byte pattern as
    /// [`openvm_cuda_common::copy::MemCopyH2D::copy_to_on`] but targeting the
    /// sub-range `[lo, hi)` rather than the whole buffer.
    pub fn copy_from_host(
        &mut self,
        src: &[T],
        device_ctx: &GpuDeviceCtx,
    ) -> Result<(), HaloGpuError> {
        crate::perf_section!("d_buffer_mut_slice_copy_from_host");
        let count = self.hi - self.lo;
        if src.len() != count {
            return Err(HaloGpuError::InvalidParameter {
                context: "DeviceBufferMutSlice::copy_from_host: src.len() != slice.len()",
                magnitude: src.len() as u64,
            });
        }
        if count == 0 {
            return Ok(());
        }
        openvm_cuda_common::common::set_device_by_id(device_ctx.device_id as i32)
            .map_err(HaloGpuError::from)?;
        let dst_ptr = unsafe { self.buf.as_mut_ptr().add(self.lo) as *mut c_void };
        let src_ptr = src.as_ptr() as *const c_void;
        let size_bytes = std::mem::size_of::<T>() * count;
        unsafe {
            cuda_memcpy_on::<false, true>(dst_ptr, src_ptr, size_bytes, device_ctx)
                .map_err(HaloGpuError::from)?;
        }
        Ok(())
    }
}

/// Extension trait adding range-based mutable slicing to
/// [`openvm_cuda_common::d_buffer::DeviceBuffer<T>`]. `openvm-cuda-common`
/// itself only offers whole-buffer accessors and zero-fill; the
/// halo2-gpu-side implementation lives here rather than upstream.
pub trait DeviceBufferExt<T> {
    /// Returns a mutable slice over the elements indexed by `range`.
    ///
    /// Range semantics follow Rust standard slicing: an inclusive start / an
    /// exclusive end. Unbounded start defaults to `0`; unbounded end defaults
    /// to `self.len()`. Panics if the resulting `[lo, hi)` is out of bounds
    /// or has `lo > hi`.
    fn mut_slice<R>(&mut self, range: R) -> DeviceBufferMutSlice<'_, T>
    where
        R: RangeBounds<usize>;
}

impl<T> DeviceBufferExt<T> for DeviceBuffer<T> {
    fn mut_slice<R>(&mut self, range: R) -> DeviceBufferMutSlice<'_, T>
    where
        R: RangeBounds<usize>,
    {
        let len = self.len();
        let lo = match range.start_bound() {
            Bound::Included(&i) => i,
            Bound::Excluded(&i) => i.checked_add(1).expect("mut_slice: range start overflow"),
            Bound::Unbounded => 0,
        };
        let hi = match range.end_bound() {
            Bound::Included(&i) => i.checked_add(1).expect("mut_slice: range end overflow"),
            Bound::Excluded(&i) => i,
            Bound::Unbounded => len,
        };
        assert!(lo <= hi, "mut_slice: lo ({}) > hi ({})", lo, hi);
        assert!(hi <= len, "mut_slice: hi ({}) > len ({})", hi, len);
        DeviceBufferMutSlice { buf: self, lo, hi }
    }
}
