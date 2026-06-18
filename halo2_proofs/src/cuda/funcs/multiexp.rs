use crate::cpu::arithmetic::best_multiexp_cpu;
use crate::cuda::culib::{
    _halo2_msm_max_length, _halo2_multiexp, _halo2_multiexp_device_bases,
    _halo2_multiexp_device_bases_workspace_size, _halo2_multiexp_device_scalars_device_bases,
    _halo2_multiexp_device_scalars_device_bases_workspace_size, _halo2_multiexp_workspace_size,
};
use crate::cuda::utils::{
    ensure_current_device_matches_ctx, query_device_free_bytes_for_chunking, FFITraitObject,
    HALO2_GPU_CTX,
};
use crate::cuda::HaloGpuError;
use group::Group as _;
use halo2curves::CurveAffine;
use openvm_cuda_common::d_buffer::DeviceBuffer;
use std::ffi::c_void;
use std::mem;

/// Smallest MSM length the C++ Pippenger setup tolerates. Below this
/// `CudaMsmInfo::set_params` divides by zero when computing
/// `win_num_` (truncated `log2(length)/2` yields a `win_bit_` of 0).
const MSM_MIN_KERNEL_LEN: usize = 4;

/// MSM size below which `multiexp_gpu` and its siblings fall back to
/// the CPU implementation. Calibrated against the launch overhead of
/// the Pippenger kernel suite — below this threshold the dispatch
/// cost dominates the work and the CPU is faster.
pub const GPU_MSM_THRESHOLD: usize = 1 << 14;

/// Pippenger MSM with host inputs (scalars + bases). Falls back to
/// `best_multiexp_cpu` for input lengths below `GPU_MSM_THRESHOLD`.
/// Returns the curve-point result host-side. Follows the module-wide
/// wrapper contract.
pub fn multiexp_gpu<C: CurveAffine>(
    coeffs: &[C::Scalar],
    bases: &[C],
) -> Result<C::Curve, HaloGpuError> {
    if bases.len() < GPU_MSM_THRESHOLD {
        return Ok(best_multiexp_cpu(coeffs, bases));
    }
    // Section enters AFTER the CPU early-return so the `cuda.multiexp`
    // label accurately scopes only the GPU dispatch path.
    crate::perf_section!("multiexp");
    ensure_current_device_matches_ctx()?;

    let mut num_chunks = 1;
    let mut chunk_size = bases.len();
    let max_len = unsafe { _halo2_msm_max_length(query_device_free_bytes_for_chunking() as u64) };
    if max_len < chunk_size {
        chunk_size = max_len;
        num_chunks = ((bases.len() as f64) / (chunk_size as f64)).ceil() as usize;
        log::debug!(
            "msm_length[{}] > max_length [{}], split into [{}] chunks",
            bases.len(),
            max_len,
            num_chunks
        );
    }
    if chunk_size < MSM_MIN_KERNEL_LEN {
        // `_halo2_msm_max_length` came back below the kernel's safe
        // floor — `CudaMsmInfo::set_params` would divide by zero in
        // `win_num_` if we sent a chunk this small. Surface a clean
        // error instead of crashing the C++ side.
        return Err(HaloGpuError::InsufficientGpuMemory {
            context: "multiexp_gpu",
            magnitude: bases.len() as u64,
            free_bytes: query_device_free_bytes_for_chunking() as u64,
        });
    }

    let mut results = vec![C::Curve::identity(); num_chunks];
    for ((coeff, base), res) in coeffs
        .chunks(chunk_size)
        .zip(bases.chunks(chunk_size))
        .zip(results.iter_mut())
    {
        let coeffs_obj = FFITraitObject::from_ref(&coeff[0]);
        let bases_obj = FFITraitObject::from_ref(&base[0]);
        let out_obj = FFITraitObject::from_ref(res);
        let scratch_bytes = unsafe { _halo2_multiexp_workspace_size(base.len() as u64) } as usize;
        let scratch = DeviceBuffer::<u8>::with_capacity_on(scratch_bytes, &HALO2_GPU_CTX);
        let status = unsafe {
            _halo2_multiexp(
                &coeffs_obj,
                &bases_obj,
                &out_obj,
                base.len(),
                scratch.as_mut_raw_ptr(),
                scratch_bytes as u64,
                HALO2_GPU_CTX.stream.as_raw(),
            )
        };

        if status.code != 0 {
            return Err(status.into());
        }
    }

    Ok(results.iter().fold(C::Curve::identity(), |a, b| a + b))
}

pub fn multiexp_gpu_many<C: CurveAffine>(
    coeffs: &[C::Scalar],
    bases: &[C],
) -> Result<C::Curve, HaloGpuError> {
    // Single-stream GPU prover: run the whole batch on gpu 0.
    multiexp_gpu(coeffs, bases)
}

/// Device-bases variant of [`multiexp_gpu`].
///
/// `bases` is a device-resident buffer of at least `coeffs.len()` affine
/// points (the caller typically caches a KZG SRS device mirror across
/// many MSM calls). Scalars are uploaded host→device each call; the bases
/// are read directly from the caller's device buffer. Workspace sizing
/// excludes the point slot via `_halo2_multiexp_device_bases_workspace_size`.
///
/// Chunks the MSM at `_halo2_msm_max_length` (matching `multiexp_gpu`'s
/// chunking under tight GPU memory). Per-chunk scratch allocations are
/// independent; their per-chunk results are folded into a single Jacobian
/// sum.
pub(crate) fn multiexp_gpu_device_bases<C: CurveAffine>(
    coeffs: &[C::Scalar],
    bases: &DeviceBuffer<C>,
) -> Result<C::Curve, HaloGpuError> {
    let max_len = unsafe { _halo2_msm_max_length(query_device_free_bytes_for_chunking() as u64) };
    multiexp_gpu_device_bases_chunked(coeffs, bases, max_len)
}

/// Chunking-controllable variant of [`multiexp_gpu_device_bases`].
/// Production callers go through the wrapper above which derives
/// `max_chunk_len` from runtime free GPU memory; tests exercise the
/// chunking loop with smaller `max_chunk_len` values to validate the
/// fold logic without provoking real OOM pressure.
pub(crate) fn multiexp_gpu_device_bases_chunked<C: CurveAffine>(
    coeffs: &[C::Scalar],
    bases: &DeviceBuffer<C>,
    max_chunk_len: usize,
) -> Result<C::Curve, HaloGpuError> {
    let length = coeffs.len();
    if length == 0 {
        return Ok(C::Curve::identity());
    }
    assert!(
        bases.len() >= length,
        "multiexp_gpu_device_bases: bases.len() = {} < coeffs.len() = {}",
        bases.len(),
        length
    );
    let chunk_size = max_chunk_len.min(length);
    if chunk_size < MSM_MIN_KERNEL_LEN {
        // Either `max_chunk_len` came back below the kernel's safe
        // floor (GPU memory pressure → `_halo2_msm_max_length` returned
        // a tiny value) or the caller asked for an MSM smaller than
        // the kernel can handle. The Pippenger setup crashes on
        // `length < 4` via a divide-by-zero in `win_num_`; surface a
        // clean error instead.
        return Err(HaloGpuError::InsufficientGpuMemory {
            context: "multiexp_gpu_device_bases",
            magnitude: length as u64,
            free_bytes: query_device_free_bytes_for_chunking() as u64,
        });
    }
    crate::perf_section!("multiexp_device_bases");
    ensure_current_device_matches_ctx()?;

    let num_chunks = length.div_ceil(chunk_size);

    let bases_base = bases.as_raw_ptr() as *const u8;
    let elem_bytes = mem::size_of::<C>();

    let mut results: Vec<C::Curve> = vec![C::Curve::identity(); num_chunks];
    for (idx, (coeffs_chunk, res)) in coeffs
        .chunks(chunk_size)
        .zip(results.iter_mut())
        .enumerate()
    {
        let chunk_len = coeffs_chunk.len();
        debug_assert!(idx * chunk_size + chunk_len <= length);
        let coeffs_obj = FFITraitObject::from_ref(&coeffs_chunk[0]);
        let out_obj = FFITraitObject::from_ref(res);
        let scratch_bytes =
            unsafe { _halo2_multiexp_device_bases_workspace_size(chunk_len as u64) } as usize;
        let scratch = DeviceBuffer::<u8>::with_capacity_on(scratch_bytes, &HALO2_GPU_CTX);
        let d_bases_chunk =
            unsafe { bases_base.add(idx * chunk_size * elem_bytes) } as *const c_void;
        let status = unsafe {
            _halo2_multiexp_device_bases(
                &coeffs_obj,
                d_bases_chunk,
                &out_obj,
                chunk_len,
                scratch.as_mut_raw_ptr(),
                scratch_bytes as u64,
                HALO2_GPU_CTX.stream.as_raw(),
            )
        };
        if status.code != 0 {
            return Err(status.into());
        }
    }

    Ok(results.iter().fold(C::Curve::identity(), |a, b| a + b))
}

/// Device-scalars + device-bases variant of [`multiexp_gpu_device_bases`].
///
/// Both `coeffs` and `bases` are caller-owned device pointers; the FFI reads
/// the scalars directly from the caller's `DeviceBuffer<C::Scalar>`. Output
/// is host-resident (a single Jacobian point per chunk, summed on host).
///
/// Chunking matches [`multiexp_gpu_device_bases_chunked`]: per-chunk
/// scratch is sized via
/// `_halo2_multiexp_device_scalars_device_bases_workspace_size`. Sub-views
/// into both the scalars and bases buffers are obtained via raw pointer
/// arithmetic on the underlying device buffers.
pub(crate) fn multiexp_gpu_device_scalars_device_bases<C: CurveAffine>(
    coeffs: &DeviceBuffer<C::Scalar>,
    bases: &DeviceBuffer<C>,
) -> Result<C::Curve, HaloGpuError> {
    let max_len = unsafe { _halo2_msm_max_length(query_device_free_bytes_for_chunking() as u64) };
    multiexp_gpu_device_scalars_device_bases_chunked(coeffs, bases, max_len)
}

/// Chunking-controllable variant of [`multiexp_gpu_device_scalars_device_bases`].
/// Production callers go through the wrapper above which derives
/// `max_chunk_len` from runtime free GPU memory; tests exercise the
/// chunking loop with smaller `max_chunk_len` values to validate the
/// fold logic without provoking real OOM pressure.
/// Chunked variant of the device-scalars + device-bases MSM. Splits the
/// input into chunks of `max_chunk_len` (or smaller for the tail),
/// accumulates partial results, and folds them into the final MSM
/// result. Used by `ParamsKZG::commit_device` to keep the per-chunk
/// scratch budget under `query_device_free_bytes_for_chunking`.
///
/// # Contract
/// - **Ownership**: `coeffs` and `bases` are caller-owned `DeviceBuffer`
///   handles; the per-chunk scratch is allocated + dropped within each
///   loop iteration.
/// - **Sync**: kernel launches enqueue on `HALO2_GPU_CTX.stream`. Each
///   chunk's partial result is copied to host before the next chunk's
///   launch (single-stream serialization). Returns only after the final
///   copy completes; the returned `C::Curve` is host-resident.
/// - **Errors**: `Err(HaloGpuError::InsufficientGpuMemory)` if
///   `max_chunk_len < MSM_MIN_KERNEL_LEN`; `Err(HaloGpuError::Cuda)` on
///   any kernel launch failure.
pub fn multiexp_gpu_device_scalars_device_bases_chunked<C: CurveAffine>(
    coeffs: &DeviceBuffer<C::Scalar>,
    bases: &DeviceBuffer<C>,
    max_chunk_len: usize,
) -> Result<C::Curve, HaloGpuError> {
    let length = coeffs.len();
    if length == 0 {
        return Ok(C::Curve::identity());
    }
    assert!(
        bases.len() >= length,
        "multiexp_gpu_device_scalars_device_bases: bases.len() = {} < coeffs.len() = {}",
        bases.len(),
        length
    );
    let chunk_size = max_chunk_len.min(length);
    if chunk_size < MSM_MIN_KERNEL_LEN {
        return Err(HaloGpuError::InsufficientGpuMemory {
            context: "multiexp_gpu_device_scalars_device_bases",
            magnitude: length as u64,
            free_bytes: query_device_free_bytes_for_chunking() as u64,
        });
    }
    crate::perf_section!("multiexp_device_scalars_device_bases");
    ensure_current_device_matches_ctx()?;

    let num_chunks = length.div_ceil(chunk_size);

    let coeffs_base = coeffs.as_raw_ptr() as *const u8;
    let bases_base = bases.as_raw_ptr() as *const u8;
    let scalar_bytes = mem::size_of::<C::Scalar>();
    let base_bytes = mem::size_of::<C>();

    let mut results: Vec<C::Curve> = vec![C::Curve::identity(); num_chunks];
    for (idx, res) in results.iter_mut().enumerate().take(num_chunks) {
        let chunk_start = idx * chunk_size;
        let chunk_len = (length - chunk_start).min(chunk_size);
        debug_assert!(chunk_start + chunk_len <= length);
        let out_obj = FFITraitObject::from_ref(res);
        let scratch_bytes =
            unsafe { _halo2_multiexp_device_scalars_device_bases_workspace_size(chunk_len as u64) }
                as usize;
        let scratch = DeviceBuffer::<u8>::with_capacity_on(scratch_bytes, &HALO2_GPU_CTX);
        let d_scalar_chunk =
            unsafe { coeffs_base.add(chunk_start * scalar_bytes) } as *const c_void;
        let d_bases_chunk = unsafe { bases_base.add(chunk_start * base_bytes) } as *const c_void;
        let status = unsafe {
            _halo2_multiexp_device_scalars_device_bases(
                d_scalar_chunk,
                d_bases_chunk,
                &out_obj,
                chunk_len,
                scratch.as_mut_raw_ptr(),
                scratch_bytes as u64,
                HALO2_GPU_CTX.stream.as_raw(),
            )
        };
        if status.code != 0 {
            return Err(status.into());
        }
    }

    Ok(results.iter().fold(C::Curve::identity(), |a, b| a + b))
}
