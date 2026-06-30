use crate::cuda::culib::{
    _halo2_decode_assigned, _halo2_eval_poly_batch_max_len, _halo2_eval_polynomial,
    _halo2_eval_polynomial_batch, _halo2_eval_polynomial_batch_workspace_size,
    _halo2_eval_polynomial_workspace_size, _halo2_kate_division_device,
    _halo2_kate_division_device_padded, _halo2_kate_division_workspace_size,
    _halo2_poly_multiply_add, _halo2_poly_sub_scalar_at_zero, _halo2_poly_sub_short_inplace,
    _halo2_poly_sub_short_out_of_place, AssignedLayout,
};
use crate::cuda::utils::{
    ensure_current_device_matches_ctx, query_device_free_bytes_for_chunking, FFITraitObject,
    HALO2_GPU_CTX,
};
use crate::cuda::HaloGpuError;
use crate::plonk::{
    assert_assigned_kernel_field_is_bn256_fr, assigned_layout_offsets, verify_assigned_layout,
    GpuAssigned,
};
use ff::Field;
use openvm_cuda_common::copy::{cuda_memcpy_on, MemCopyH2D};
use openvm_cuda_common::d_buffer::DeviceBuffer;
use std::mem;

/// Evaluates a host-resident polynomial at `point`: stages the polynomial
/// into a device buffer and dispatches through `eval_polynomial_device`.
pub fn eval_polynomial_gpu<F: Field>(poly: &[F], point: F) -> Result<F, HaloGpuError> {
    crate::perf_section!("eval_polynomial");
    let d_poly = poly.to_device_on(&HALO2_GPU_CTX)?;
    eval_polynomial_device(&d_poly, point)
}

// batch poly evaluation
pub fn basic_batch_eval_polynomial_gpu<F: Field>(
    poly_in_many_ori: &[FFITraitObject],
    eval_points: &[F],
    eval_result: &mut [F],
    poly_offset: usize,
    poly_length: usize,
) -> Result<(), HaloGpuError> {
    crate::perf_section!("basic_batch_eval_polynomial");
    ensure_current_device_matches_ctx()?;
    let batch_size = poly_in_many_ori.len();
    assert_eq!(batch_size, eval_points.len());
    let eval_points_obj = FFITraitObject::from_ref(&eval_points[0]);
    let eval_result_obj = FFITraitObject::from_ref(&eval_result[0]);
    let scratch_bytes = unsafe {
        _halo2_eval_polynomial_batch_workspace_size(poly_length as u64, batch_size as u64)
    } as usize;
    let scratch = DeviceBuffer::<u8>::with_capacity_on(scratch_bytes, &HALO2_GPU_CTX);
    let status = unsafe {
        _halo2_eval_polynomial_batch(
            poly_in_many_ori.as_ptr(),
            &eval_points_obj,
            &eval_result_obj,
            poly_offset,
            poly_length,
            batch_size,
            scratch.as_mut_raw_ptr(),
            scratch_bytes as u64,
            HALO2_GPU_CTX.stream.as_raw(),
        )
    };

    if status.code != 0 {
        return Err(status.into());
    }
    Ok(())
}

// batch poly evaluation : memory-aware + basic impl
pub fn batch_eval_polynomial_gpu<F: Field>(
    poly_in_many_ori: &[FFITraitObject],
    eval_points: &[F],
    eval_result: &mut [F],
    poly_length: usize,
) -> Result<(), HaloGpuError> {
    crate::perf_section!("batch_eval_polynomial");
    let batch_size = poly_in_many_ori.len();
    let max_length = unsafe {
        _halo2_eval_poly_batch_max_len(
            poly_length,
            batch_size,
            query_device_free_bytes_for_chunking() as u64,
        )
    };
    if poly_length <= max_length {
        basic_batch_eval_polynomial_gpu(poly_in_many_ori, eval_points, eval_result, 0, poly_length)
    } else {
        // max_length < poly_length
        // split ploy data into into chunks ( batch_size remains unchanged )
        let num_chunks = ((poly_length as f64) / (max_length as f64)).ceil() as usize;
        let chunk_size = max_length;
        log::debug!("poly_length: {} > max_length: {}", poly_length, max_length);
        log::debug!("num_chunks: {}, chunk_size: {}", num_chunks, chunk_size);
        let mut multi_eval_result: Vec<Vec<F>> = Vec::with_capacity(num_chunks);
        for chunk_idx in 0..num_chunks {
            log::debug!("chunk_idx: {}", chunk_idx);
            let mut temp_result = vec![F::ZERO; batch_size];
            let _offset = chunk_idx * chunk_size;
            let _lenght = if chunk_idx == num_chunks - 1 {
                poly_length - _offset
            } else {
                chunk_size
            };
            basic_batch_eval_polynomial_gpu(
                poly_in_many_ori,
                eval_points,
                &mut temp_result,
                _offset,
                _lenght,
            )?;
            temp_result.iter_mut().enumerate().for_each(|(i, result)| {
                *result = (*result) * eval_points[i].pow_vartime([_offset as u64, 0, 0, 0])
            });
            multi_eval_result.push(temp_result);
        }
        eval_result.iter_mut().enumerate().for_each(|(i, result)| {
            *result = multi_eval_result
                .iter()
                .fold(F::ZERO, |acc, res| acc + res[i]);
        });
        Ok(())
    }
}

/// Device-input batch evaluation. For each `(d_poly, point)` pair,
/// evaluates `d_poly` at `point` via [`eval_polynomial_device`] and writes
/// the scalar into `eval_result[i]`. No H2D of polynomial coefficients —
/// the device buffers stay caller-owned and resident.
///
/// Each iteration enqueues one Horner-tree kernel + a 32-byte D2H of the
/// result on `HALO2_GPU_CTX.stream` and synchronizes. This is the
/// equivalent of the host-input [`batch_eval_polynomial_gpu`] for callers
/// whose polys are already on device; it trades the batched kernel's
/// double-buffered H2D-with-compute overlap for zero coefficient transfer.
pub fn batch_eval_polynomial_d2h<F: Field>(
    d_polys: &[&DeviceBuffer<F>],
    eval_points: &[F],
    eval_result: &mut [F],
) -> Result<(), HaloGpuError> {
    crate::perf_section!("batch_eval_polynomial_device");
    assert_eq!(d_polys.len(), eval_points.len());
    assert_eq!(d_polys.len(), eval_result.len());
    for (i, (d_poly, point)) in d_polys.iter().zip(eval_points.iter()).enumerate() {
        eval_result[i] = eval_polynomial_device(d_poly, *point)?;
    }
    Ok(())
}

/// `poly_acc[i] += scalar * poly_in[i]`. Host-slice wrapper around the
/// device-only `_halo2_poly_multiply_add` FFI; handles H2D / D2H here.
///
/// Production hot paths use the device-resident variants
/// ([`poly_multiply_add_device`], [`poly_multiply_add_device_at_lut_offset`])
/// to avoid the per-call H2D + D2H round-trips this wrapper performs.
pub fn poly_multiply_add_single_gpu<F: Field>(
    poly_acc: &mut [F],
    poly_in: &[F],
    scalar: F,
) -> Result<(), HaloGpuError> {
    crate::perf_section!("poly_multiply_add");
    assert_eq!(poly_acc.len(), poly_in.len());

    let poly_len = poly_acc.len();
    let acc_bytes = std::mem::size_of_val(poly_acc);

    crate::perf_h2d!("cuda.poly_multiply_add.acc", acc_bytes);
    let acc_device = poly_acc.to_device_on(&HALO2_GPU_CTX).unwrap();
    crate::perf_h2d!("cuda.poly_multiply_add.poly_in", acc_bytes);
    let poly_in_device = poly_in.to_device_on(&HALO2_GPU_CTX).unwrap();
    crate::perf_h2d!("cuda.poly_multiply_add.scalar", std::mem::size_of::<F>());
    let scalar_device = std::slice::from_ref(&scalar)
        .to_device_on(&HALO2_GPU_CTX)
        .unwrap();

    let status = unsafe {
        _halo2_poly_multiply_add(
            acc_device.as_mut_raw_ptr(),
            poly_in_device.as_raw_ptr(),
            scalar_device.as_raw_ptr(),
            poly_len,
            HALO2_GPU_CTX.stream.as_raw(),
        )
    };
    if status.code != 0 {
        return Err(status.into());
    }

    unsafe {
        crate::perf_d2h!("cuda.poly_multiply_add.acc_back", acc_bytes);
        cuda_memcpy_on::<true, false>(
            poly_acc.as_mut_ptr() as *mut libc::c_void,
            acc_device.as_raw_ptr(),
            acc_bytes,
            &HALO2_GPU_CTX,
        )?;
    }
    HALO2_GPU_CTX.stream.to_host_sync().unwrap();
    Ok(())
}

/// `d_acc[i] += scalar * d_in[i]` on device. Pure-device variant of
/// [`poly_multiply_add_single_gpu`]: no H2D/D2H, both buffers caller-owned,
/// `scalar` is the only host-side input (H→D'd into a 32-byte throwaway).
///
/// Aliasing `d_acc == d_in` is permitted by the underlying kernel
/// (each thread reads one element then writes one element at the same
/// index, no inter-thread interference). This is used by
/// [`poly_scale_device_with_d_s_minus_one`] to implement in-place scalar multiply.
///
/// # Contract
/// - **Ownership**: `d_acc` and `d_in` are caller-owned `DeviceBuffer<F>`
///   handles; this function does not allocate them. The 32-byte
///   throwaway scalar buffer is allocated + dropped within this call.
/// - **Sync**: All work is enqueued on `HALO2_GPU_CTX.stream`. The
///   function returns BEFORE the kernel completes — callers that need
///   the result host-readable must stream-sync. Same-stream subsequent
///   kernel reads see the result.
/// - **Errors**: returns `Err(HaloGpuError::Cuda)` if either the scalar
///   H2D or the kernel launch fails. `debug_assert!` panics in debug
///   builds on `d_acc.len() == 0`; release builds quietly no-op the
///   underlying kernel.
pub fn poly_multiply_add_device<F: Field>(
    d_acc: &mut DeviceBuffer<F>,
    d_in: &DeviceBuffer<F>,
    scalar: F,
) -> Result<(), HaloGpuError> {
    crate::perf_section!("poly_multiply_add_device");
    assert_eq!(d_acc.len(), d_in.len());
    debug_assert!(
        !d_acc.is_empty(),
        "poly_multiply_add_device: zero-length buffer (no-op kernel launch)"
    );
    ensure_current_device_matches_ctx()?;
    let d_scalar = std::slice::from_ref(&scalar)
        .to_device_on(&HALO2_GPU_CTX)
        .map_err(HaloGpuError::from)?;
    poly_multiply_add_device_with_d_scalar(d_acc, d_in, &d_scalar)
}

/// Variant of [`poly_multiply_add_device`] that takes a caller-owned
/// device-resident scalar. Lets the caller hoist a constant scalar's
/// H2D out of a loop, avoiding the per-iteration 32-byte allocation +
/// upload that the `scalar: F` form does internally.
pub(crate) fn poly_multiply_add_device_with_d_scalar<F: Field>(
    d_acc: &mut DeviceBuffer<F>,
    d_in: &DeviceBuffer<F>,
    d_scalar: &DeviceBuffer<F>,
) -> Result<(), HaloGpuError> {
    assert_eq!(d_acc.len(), d_in.len());
    assert_eq!(d_scalar.len(), 1);
    ensure_current_device_matches_ctx()?;
    let status = unsafe {
        _halo2_poly_multiply_add(
            d_acc.as_mut_raw_ptr(),
            d_in.as_raw_ptr(),
            d_scalar.as_raw_ptr(),
            d_acc.len(),
            HALO2_GPU_CTX.stream.as_raw(),
        )
    };
    if status.code != 0 {
        return Err(status.into());
    }
    Ok(())
}

/// `d_acc[i] += d_lut[lut_offset] * d_in[i]`. Indexes a device-resident
/// scalar-power LUT to avoid the 32-byte per-iteration H2D that
/// [`poly_multiply_add_device`] performs when called with a host scalar.
pub fn poly_multiply_add_device_at_lut_offset<F: Field>(
    d_acc: &mut DeviceBuffer<F>,
    d_in: &DeviceBuffer<F>,
    d_lut: &DeviceBuffer<F>,
    lut_offset: usize,
) -> Result<(), HaloGpuError> {
    assert_eq!(d_acc.len(), d_in.len());
    assert!(
        lut_offset < d_lut.len(),
        "poly_multiply_add_device_at_lut_offset: offset out of range"
    );
    debug_assert!(
        !d_acc.is_empty(),
        "poly_multiply_add_device_at_lut_offset: zero-length acc"
    );
    ensure_current_device_matches_ctx()?;
    let elt_bytes = mem::size_of::<F>();
    let d_scalar_ptr = unsafe { (d_lut.as_raw_ptr() as *const u8).add(lut_offset * elt_bytes) }
        as *const libc::c_void;
    let status = unsafe {
        _halo2_poly_multiply_add(
            d_acc.as_mut_raw_ptr(),
            d_in.as_raw_ptr(),
            d_scalar_ptr,
            d_acc.len(),
            HALO2_GPU_CTX.stream.as_raw(),
        )
    };
    if status.code != 0 {
        return Err(status.into());
    }
    Ok(())
}

/// Device-input variant of `eval_polynomial`.
///
/// `d_poly` is caller-owned device-resident. Dispatches to the
/// `_halo2_eval_polynomial` FFI launcher, which reuses the
/// existing `eval_polynomial_batch` + `eval_polynomial_epilogue`
/// CUDA kernels on the caller's device buffer (no poly H2D). Only
/// the 32-byte eval point is H2D'd; only the 32-byte result is D→H'd.
///
/// Used by `ProverQuery::get_eval` when the underlying polynomial is
/// device-resident.
pub fn eval_polynomial_device<F: Field>(
    d_poly: &DeviceBuffer<F>,
    point: F,
) -> Result<F, HaloGpuError> {
    crate::perf_section!("eval_polynomial_device");
    debug_assert!(
        !d_poly.is_empty(),
        "eval_polynomial_device: zero-length poly (Horner kernel undefined)"
    );
    ensure_current_device_matches_ctx()?;

    let n = d_poly.len();
    let point_obj = FFITraitObject::from_ref(&point);
    let scratch_bytes = unsafe { _halo2_eval_polynomial_workspace_size(n as u64) } as usize;
    let scratch = DeviceBuffer::<u8>::with_capacity_on(scratch_bytes, &HALO2_GPU_CTX);

    let mut d_result_ptr: *mut libc::c_void = std::ptr::null_mut();
    let status = unsafe {
        _halo2_eval_polynomial(
            d_poly.as_raw_ptr(),
            &point_obj,
            &mut d_result_ptr,
            n,
            scratch.as_mut_raw_ptr(),
            scratch_bytes as u64,
            HALO2_GPU_CTX.stream.as_raw(),
        )
    };
    if status.code != 0 {
        return Err(status.into());
    }

    // Async D2H of the 32-byte result on the explicit stream, then
    // `to_host_sync()` from the Rust side to fence; the FFI launcher
    // itself does not synchronize. The host-bound boundary here is the
    // 32-byte eval result.
    let mut acc = F::ZERO;
    crate::perf_d2h!(
        "cuda.eval_polynomial_device.result",
        mem::size_of::<F>() as u64
    );
    unsafe {
        cuda_memcpy_on::<true, false>(
            &mut acc as *mut F as *mut libc::c_void,
            d_result_ptr,
            mem::size_of::<F>(),
            &HALO2_GPU_CTX,
        )?;
    }
    HALO2_GPU_CTX.stream.to_host_sync()?;
    Ok(acc)
}

/// `d_buf[i] *= s`, where `d_s_minus_one` is a caller-owned device-resident
/// scalar holding the pre-computed `s - 1`. Reuses [`poly_multiply_add_device`]
/// with `d_in == d_buf` and scalar `s - 1`, so the kernel computes
/// `acc[i] += (s - 1) * acc[i]` ≡ `acc[i] *= s`. Self-aliasing is safe
/// because the underlying kernel is per-element (no inter-thread interference).
pub(crate) fn poly_scale_device_with_d_s_minus_one<F: Field>(
    d_buf: &mut DeviceBuffer<F>,
    d_s_minus_one: &DeviceBuffer<F>,
) -> Result<(), HaloGpuError> {
    crate::perf_section!("poly_scale_device_with_d_s_minus_one");
    assert_eq!(d_s_minus_one.len(), 1);
    debug_assert!(
        !d_buf.is_empty(),
        "poly_scale_device_with_d_s_minus_one: zero-length buffer (no-op kernel launch)"
    );
    ensure_current_device_matches_ctx()?;
    let status = unsafe {
        _halo2_poly_multiply_add(
            d_buf.as_mut_raw_ptr(),
            d_buf.as_raw_ptr(),
            d_s_minus_one.as_raw_ptr(),
            d_buf.len(),
            HALO2_GPU_CTX.stream.as_raw(),
        )
    };
    if status.code != 0 {
        return Err(status.into());
    }
    Ok(())
}

/// Device-resident `kate_division`: `q(X) = (a(X) - a(u)) / (X - u)`.
///
/// `d_a` is a caller-owned length-n device polynomial; `root` is the
/// linear-factor root `u`. The returned [`DeviceBuffer`] holds the
/// length-(n-1) quotient. The kernel implements the recurrence
/// `q[j] = a[j+1] + u * q[j+1]` as an affine-pair Brent-Kung prefix
/// scan in field arithmetic. Byte-exact against `cpu::arithmetic::kate_division`.
///
/// # Contract
/// - **Ownership**: `d_a` is borrowed; the returned `DeviceBuffer` is
///   newly allocated on `HALO2_GPU_CTX`. The internal scan scratch is
///   freed when this function returns.
/// - **Sync**: All work enqueues on `HALO2_GPU_CTX.stream`; no
///   `cudaStreamSynchronize` here.
/// - **Edge case**: `n == 1` returns an empty `DeviceBuffer` (q has
///   length 0); `n == 0` panics in debug builds.
pub fn kate_division_device<F: Field>(
    d_a: &DeviceBuffer<F>,
    root: F,
) -> Result<DeviceBuffer<F>, HaloGpuError> {
    crate::perf_section!("kate_division_device");
    let n = d_a.len();
    debug_assert!(n >= 1, "kate_division_device: input length must be >= 1");
    ensure_current_device_matches_ctx()?;

    let out_len = n.saturating_sub(1);
    // `with_capacity_on(0)` asserts; emit a null-backed empty buffer for
    // the degenerate n==1 case (q has length 0).
    if out_len == 0 {
        return Ok(DeviceBuffer::<F>::new());
    }
    let d_q = DeviceBuffer::<F>::with_capacity_on(out_len, &HALO2_GPU_CTX);

    let d_root = std::slice::from_ref(&root)
        .to_device_on(&HALO2_GPU_CTX)
        .map_err(HaloGpuError::from)?;

    let scratch_bytes = unsafe { _halo2_kate_division_workspace_size(n as u64) } as usize;
    let scratch = DeviceBuffer::<u8>::with_capacity_on(scratch_bytes, &HALO2_GPU_CTX);

    let status = unsafe {
        _halo2_kate_division_device(
            d_a.as_raw_ptr(),
            d_q.as_mut_raw_ptr(),
            d_root.as_raw_ptr(),
            n as u64,
            scratch.as_mut_raw_ptr(),
            scratch_bytes as u64,
            HALO2_GPU_CTX.stream.as_raw(),
        )
    };
    if status.code != 0 {
        return Err(status.into());
    }
    Ok(d_q)
}

/// Variant of [`kate_division_device`] that consumes a device-resident
/// root scalar. Lets callers hoist the per-root H2D out of a loop when
/// the same `d_root` is used across multiple invocations.
pub fn kate_division_device_with_d_root<F: Field>(
    d_a: &DeviceBuffer<F>,
    d_root: &DeviceBuffer<F>,
) -> Result<DeviceBuffer<F>, HaloGpuError> {
    crate::perf_section!("kate_division_device_with_d_root");
    let n = d_a.len();
    debug_assert!(
        n >= 1,
        "kate_division_device_with_d_root: input length must be >= 1"
    );
    assert_eq!(d_root.len(), 1);
    ensure_current_device_matches_ctx()?;

    let out_len = n.saturating_sub(1);
    // `with_capacity_on(0)` asserts; emit a null-backed empty buffer for
    // the degenerate n==1 case (q has length 0).
    if out_len == 0 {
        return Ok(DeviceBuffer::<F>::new());
    }
    let d_q = DeviceBuffer::<F>::with_capacity_on(out_len, &HALO2_GPU_CTX);

    let scratch_bytes = unsafe { _halo2_kate_division_workspace_size(n as u64) } as usize;
    let scratch = DeviceBuffer::<u8>::with_capacity_on(scratch_bytes, &HALO2_GPU_CTX);

    let status = unsafe {
        _halo2_kate_division_device(
            d_a.as_raw_ptr(),
            d_q.as_mut_raw_ptr(),
            d_root.as_raw_ptr(),
            n as u64,
            scratch.as_mut_raw_ptr(),
            scratch_bytes as u64,
            HALO2_GPU_CTX.stream.as_raw(),
        )
    };
    if status.code != 0 {
        return Err(status.into());
    }
    Ok(d_q)
}

/// Padded variant of [`kate_division_device_with_d_root`]: returns a
/// length-`out_len` device buffer holding the length-(n-1) quotient at
/// positions [0, n-1) and zeros at [n-1, out_len). The quotient
/// reverse-write and the trailing zero-pad share a single kernel
/// launch (the underlying `_halo2_kate_division_device_padded` writes
/// both regions in one pass).
///
/// # Contract
/// - **Ownership**: `d_a` and `d_root` are borrowed; the returned buffer
///   is freshly allocated on `HALO2_GPU_CTX`. `out_len` must be `>=
///   d_a.len().saturating_sub(1)`; the caller enforces the upper bound
///   (typically `params.n`).
/// - **Sync**: All work enqueues on `HALO2_GPU_CTX.stream`.
/// - **Edge case**: `out_len == 0` returns a null-backed empty buffer
///   (no kernel launches).
pub fn kate_division_device_padded_with_d_root<F: Field>(
    d_a: &DeviceBuffer<F>,
    d_root: &DeviceBuffer<F>,
    out_len: usize,
) -> Result<DeviceBuffer<F>, HaloGpuError> {
    crate::perf_section!("kate_division_device_padded");
    let n = d_a.len();
    debug_assert!(
        n >= 1,
        "kate_division_device_padded_with_d_root: input length must be >= 1"
    );
    assert_eq!(d_root.len(), 1);
    assert!(
        out_len >= n.saturating_sub(1),
        "kate_division_device_padded_with_d_root: out_len {} < n-1 {}",
        out_len,
        n.saturating_sub(1)
    );
    ensure_current_device_matches_ctx()?;

    if out_len == 0 {
        return Ok(DeviceBuffer::<F>::new());
    }
    let d_q = DeviceBuffer::<F>::with_capacity_on(out_len, &HALO2_GPU_CTX);

    let scratch_bytes = unsafe { _halo2_kate_division_workspace_size(n as u64) } as usize;
    let scratch = if scratch_bytes == 0 {
        DeviceBuffer::<u8>::new()
    } else {
        DeviceBuffer::<u8>::with_capacity_on(scratch_bytes, &HALO2_GPU_CTX)
    };

    let status = unsafe {
        _halo2_kate_division_device_padded(
            d_a.as_raw_ptr(),
            d_q.as_mut_raw_ptr(),
            d_root.as_raw_ptr(),
            n as u64,
            out_len as u64,
            scratch.as_mut_raw_ptr(),
            scratch_bytes as u64,
            HALO2_GPU_CTX.stream.as_raw(),
        )
    };
    if status.code != 0 {
        return Err(status.into());
    }
    Ok(d_q)
}

/// `d_acc[i] -= d_short[i]` for i in [0, d_short.len()). The longer
/// accumulator is touched only on its short prefix; trailing elements
/// are untouched. Both buffers caller-owned device-resident.
pub fn poly_sub_short_in_place_device<F: Field>(
    d_acc: &mut DeviceBuffer<F>,
    d_short: &DeviceBuffer<F>,
) -> Result<(), HaloGpuError> {
    crate::perf_section!("poly_sub_short_in_place_device");
    let short_len = d_short.len();
    if short_len == 0 {
        return Ok(());
    }
    assert!(
        d_acc.len() >= short_len,
        "poly_sub_short_in_place_device: short_len exceeds acc len"
    );
    ensure_current_device_matches_ctx()?;
    let status = unsafe {
        _halo2_poly_sub_short_inplace(
            d_acc.as_mut_raw_ptr(),
            d_short.as_raw_ptr(),
            short_len as u64,
            HALO2_GPU_CTX.stream.as_raw(),
        )
    };
    if status.code != 0 {
        return Err(status.into());
    }
    Ok(())
}

/// Out-of-place sibling of [`poly_sub_short_in_place_device`]:
///   `d_out[i] = d_long[i] - d_short[i]`  for i in [0, d_short.len())
///   `d_out[i] = d_long[i]`                for i in [d_short.len(), d_long.len())
///
/// Lets a caller materialise a fresh `d_out` from `d_long` without an
/// intervening D2D clone, preserving `d_long` for a second consumer.
/// All buffers caller-owned device-resident; `d_out` must already have
/// `d_long.len()` capacity.
pub fn poly_sub_short_out_of_place_device<F: Field>(
    d_out: &mut DeviceBuffer<F>,
    d_long: &DeviceBuffer<F>,
    d_short: &DeviceBuffer<F>,
) -> Result<(), HaloGpuError> {
    crate::perf_section!("poly_sub_short_out_of_place_device");
    let long_len = d_long.len();
    let short_len = d_short.len();
    assert_eq!(
        d_out.len(),
        long_len,
        "poly_sub_short_out_of_place_device: d_out len {} must equal d_long len {}",
        d_out.len(),
        long_len
    );
    assert!(
        short_len <= long_len,
        "poly_sub_short_out_of_place_device: short_len {} exceeds long_len {}",
        short_len,
        long_len
    );
    if long_len == 0 {
        return Ok(());
    }
    ensure_current_device_matches_ctx()?;
    let status = unsafe {
        _halo2_poly_sub_short_out_of_place(
            d_out.as_mut_raw_ptr(),
            d_long.as_raw_ptr(),
            d_short.as_raw_ptr(),
            short_len as u64,
            long_len as u64,
            HALO2_GPU_CTX.stream.as_raw(),
        )
    };
    if status.code != 0 {
        return Err(status.into());
    }
    Ok(())
}

/// `d_buf[0] -= *d_scalar`. Both pointers caller-owned device-resident.
/// Implements the host `Polynomial - scalar` (poly.rs) op that
/// subtracts a scalar from the index-0 coefficient.
pub fn poly_sub_scalar_at_zero_device<F: Field>(
    d_buf: &mut DeviceBuffer<F>,
    d_scalar: &DeviceBuffer<F>,
) -> Result<(), HaloGpuError> {
    crate::perf_section!("poly_sub_scalar_at_zero_device");
    debug_assert!(
        !d_buf.is_empty(),
        "poly_sub_scalar_at_zero_device: zero-length buffer"
    );
    assert_eq!(d_scalar.len(), 1);
    ensure_current_device_matches_ctx()?;
    let status = unsafe {
        _halo2_poly_sub_scalar_at_zero(
            d_buf.as_mut_raw_ptr(),
            d_scalar.as_raw_ptr(),
            HALO2_GPU_CTX.stream.as_raw(),
        )
    };
    if status.code != 0 {
        return Err(status.into());
    }
    Ok(())
}

/// Decode a host `&[GpuAssigned<F>]` into two fresh device buffers — the
/// per-element numerators (Zero→0, Trivial(x)→x, Rational(n,_)→n) and
/// denominators (Zero/Trivial→1, Rational(_,d)→d) — via a single GPU
/// kernel.
///
/// The host `&[GpuAssigned<F>]` is uploaded with a single bytewise H2D
/// (`to_device_on` — no enum-decode iteration); the kernel reads each
/// element's discriminant + payload bytes at the offsets pinned by
/// `#[repr(C, u8)]` on `GpuAssigned<F>` and emits the SoA numerator /
/// denominator pair, avoiding the host enum-decode passes
/// (`par_iter().map(|v| v.numerator())` / `denominator()`) that would
/// otherwise dominate `witness.next_phase`.
///
/// # Contract
/// - **Field**: the CUDA decoder is hardwired to `bn256::Fr`; non-Fr
///   `F` panics before the FFI launch
///   (`assert_assigned_kernel_field_is_bn256_fr`).
/// - **Layout**: relies on `#[repr(C, u8)]` on `GpuAssigned<F>`; a runtime
///   `verify_assigned_layout::<F>()` probe self-check runs once per call.
/// - **Sync**: H2D + kernel enqueue on `HALO2_GPU_CTX.stream`; function
///   returns before the kernel completes. Same-stream subsequent device
///   ops see the result; host reads require an explicit `to_host_sync()`.
/// - **Edge case**: an empty input returns two empty `DeviceBuffer<F>`s
///   without launching the kernel.
pub fn decode_assigned_to_num_denom_device<F: Field>(
    column: &[GpuAssigned<F>],
) -> Result<(DeviceBuffer<F>, DeviceBuffer<F>), HaloGpuError> {
    crate::perf_section!("decode_assigned_to_num_denom_device");
    let n = column.len();
    if n == 0 {
        return Ok((DeviceBuffer::<F>::new(), DeviceBuffer::<F>::new()));
    }
    assert_assigned_kernel_field_is_bn256_fr::<F>();
    ensure_current_device_matches_ctx()?;
    verify_assigned_layout::<F>();

    let (stride_bytes, num_offset, denom_offset) = assigned_layout_offsets::<F>();
    let layout = AssignedLayout {
        stride_bytes,
        num_offset,
        denom_offset,
    };
    let d_raw = column.to_device_on(&HALO2_GPU_CTX)?;
    let d_nums = DeviceBuffer::<F>::with_capacity_on(n, &HALO2_GPU_CTX);
    let d_denoms = DeviceBuffer::<F>::with_capacity_on(n, &HALO2_GPU_CTX);

    let status = unsafe {
        _halo2_decode_assigned(
            d_nums.as_mut_raw_ptr(),
            d_denoms.as_mut_raw_ptr(),
            d_raw.as_raw_ptr(),
            n as u64,
            layout,
            HALO2_GPU_CTX.stream.as_raw(),
        )
    };
    if status.code != 0 {
        return Err(status.into());
    }
    Ok((d_nums, d_denoms))
}

/// Variant of [`decode_assigned_to_num_denom_device`] that writes the
/// per-element denominators into a caller-owned device buffer at a given
/// `denom_dst_offset` (in elements). Used by
/// `batch_invert_assigned_device` to concatenate every column's
/// denominators into one large device buffer suitable for the single
/// `batch_invert_device_in_place` call, without per-column D2D copies.
///
/// Returns the freshly-allocated `DeviceBuffer<F>` of decoded numerators
/// for `column`.
pub fn decode_assigned_into_denom_slice_device<F: Field>(
    column: &[GpuAssigned<F>],
    d_denoms_concat: &DeviceBuffer<F>,
    denom_dst_offset: usize,
) -> Result<DeviceBuffer<F>, HaloGpuError> {
    crate::perf_section!("decode_assigned_into_denom_slice_device");
    let n = column.len();
    let dst_end = denom_dst_offset
        .checked_add(n)
        .expect("decode_assigned_into_denom_slice_device: dst_offset + n overflows usize");
    assert!(
        dst_end <= d_denoms_concat.len(),
        "decode_assigned_into_denom_slice_device: dst range [{}, {}) out of bounds for d_denoms_concat.len()={}",
        denom_dst_offset,
        dst_end,
        d_denoms_concat.len(),
    );
    if n == 0 {
        return Ok(DeviceBuffer::<F>::new());
    }
    assert_assigned_kernel_field_is_bn256_fr::<F>();
    ensure_current_device_matches_ctx()?;
    verify_assigned_layout::<F>();

    let (stride_bytes, num_offset, denom_offset) = assigned_layout_offsets::<F>();
    let layout = AssignedLayout {
        stride_bytes,
        num_offset,
        denom_offset,
    };
    let d_raw = column.to_device_on(&HALO2_GPU_CTX)?;
    let d_nums = DeviceBuffer::<F>::with_capacity_on(n, &HALO2_GPU_CTX);
    let d_denoms_at = unsafe {
        (d_denoms_concat.as_mut_raw_ptr() as *mut u8).add(denom_dst_offset * mem::size_of::<F>())
            as *mut libc::c_void
    };

    let status = unsafe {
        _halo2_decode_assigned(
            d_nums.as_mut_raw_ptr(),
            d_denoms_at,
            d_raw.as_raw_ptr(),
            n as u64,
            layout,
            HALO2_GPU_CTX.stream.as_raw(),
        )
    };
    if status.code != 0 {
        return Err(status.into());
    }
    Ok(d_nums)
}
