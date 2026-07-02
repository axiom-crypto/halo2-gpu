use crate::cuda::culib::{
    _halo2_grand_product, _halo2_grand_product_device_inputs, _halo2_grand_product_max_len,
    _halo2_grand_product_workspace_size,
};
use crate::cuda::utils::{
    ensure_current_device_matches_ctx, query_device_free_bytes_for_chunking, FFITraitObject,
    HALO2_GPU_CTX,
};
use crate::cuda::HaloGpuError;
use ff::Field;
use openvm_cuda_common::copy::{cuda_memcpy_on, MemCopyH2D};
use openvm_cuda_common::d_buffer::DeviceBuffer;

pub fn grand_product_gpu<F: Field>(
    output: &mut [F],
    input: &[F],
    prefix: F,
) -> Result<(), HaloGpuError> {
    crate::perf_section!("grand_product");
    ensure_current_device_matches_ctx()?;
    assert_eq!(output.len(), input.len());
    let poly_len = input.len();
    let max_len = unsafe {
        _halo2_grand_product_max_len(poly_len, query_device_free_bytes_for_chunking() as u64)
    };
    let mut chunk_size = poly_len;
    if poly_len > max_len {
        chunk_size = max_len;
    }
    if chunk_size == 0 {
        return Err(HaloGpuError::InvalidParameter {
            context: "grand_product_gpu",
            magnitude: poly_len as u64,
        });
    }

    let mut prefix = prefix;
    let output_obj = FFITraitObject::from_ref(&output[0]);
    let input_obj = FFITraitObject::from_ref(&input[0]);
    let prefix_obj = FFITraitObject::from_ref(&prefix);

    let gp_scratch_bytes =
        unsafe { _halo2_grand_product_workspace_size(chunk_size as u64) } as usize;
    let scratch = DeviceBuffer::<u8>::with_capacity_on(gp_scratch_bytes, &HALO2_GPU_CTX);
    for offset in (0..poly_len).step_by(chunk_size) {
        let this_len = chunk_size.min(poly_len - offset);
        let status = unsafe {
            _halo2_grand_product(
                &output_obj,
                &input_obj,
                &prefix_obj,
                this_len,
                offset,
                scratch.as_mut_raw_ptr(),
                gp_scratch_bytes as u64,
                HALO2_GPU_CTX.stream.as_raw(),
            )
        };
        if status.code != 0 {
            return Err(status.into());
        }
        if offset + this_len < poly_len {
            prefix = output[offset + this_len - 1];
            // print out this to surpress the compiler warning
            log::debug!("prefix at offset {}: {:?}", offset, prefix);
        }
    }
    Ok(())
}

/// Per-chunk scan length for an `output_len`-element device-input scan,
/// derived from current free device memory. Shared by the host-prefix and
/// device-prefix entry points. The common case (input fits in one chunk)
/// returns `output_len`.
fn grand_product_device_chunk_size(output_len: usize) -> usize {
    let max_len = unsafe {
        _halo2_grand_product_max_len(output_len, query_device_free_bytes_for_chunking() as u64)
    };
    output_len.min(max_len)
}

/// Device-input variant of `grand_product_gpu`. Scans the first
/// `output_len` elements of the device-resident `input_device` buffer in
/// place — `input_device[i]` becomes `prefix · ∏_{j=0..=i} input_device[j]`
/// for `i ∈ [0, output_len)` — and returns the consumed buffer to the
/// caller. The scan output stays device-resident; the wrapper performs no
/// device→host copy of the scanned region.
///
/// `input_device` may be larger than `output_len` (the tail beyond
/// `output_len` is untouched). The host-origin `prefix` is staged into a
/// 1-element device buffer once and then carried across chunk boundaries
/// entirely on device (see `grand_product_device_chunked`) — no per-chunk
/// device↔host round-trip. The common single-chunk path (e.g. `log_n=22` at
/// the typical fibonacci circuit size) never crosses a boundary.
pub fn grand_product_device<F: Field>(
    input_device: DeviceBuffer<F>,
    output_len: usize,
    prefix: F,
) -> Result<DeviceBuffer<F>, HaloGpuError> {
    crate::perf_section!("grand_product_device_inputs");
    ensure_current_device_matches_ctx()?;
    assert!(
        output_len <= input_device.len(),
        "grand_product_device: output_len ({}) must not exceed input_device ({})",
        output_len,
        input_device.len(),
    );
    // Host-origin prefix → one 1-element H2D, then a fully device-resident scan.
    let prefix_device = std::slice::from_ref(&prefix).to_device_on(&HALO2_GPU_CTX)?;
    let chunk_size = grand_product_device_chunk_size(output_len);
    grand_product_device_chunked(input_device, output_len, &prefix_device, chunk_size)
}

/// Device-prefix variant of `grand_product_device`. The running prefix is
/// supplied as a device-resident 1-element buffer (`prefix_device`) rather
/// than a host scalar, so a caller that chains scans — e.g. the permutation
/// prover's per-set grand product, whose set `k` starts from set `k-1`'s tail
/// value — can carry the running product from one scan's tail into the next
/// scan's prefix entirely on device, with no device→host→device round-trip
/// and no stream sync between sets.
///
/// `prefix_device` must hold at least one element; only element `0` is read.
/// The buffer is left unmodified (the cross-chunk carry rolls through an
/// internal buffer), so the caller may reuse it freely.
pub fn grand_product_device_with_prefix_device<F: Field>(
    input_device: DeviceBuffer<F>,
    output_len: usize,
    prefix_device: &DeviceBuffer<F>,
) -> Result<DeviceBuffer<F>, HaloGpuError> {
    crate::perf_section!("grand_product_device_inputs");
    ensure_current_device_matches_ctx()?;
    assert!(
        output_len <= input_device.len(),
        "grand_product_device_with_prefix_device: output_len ({}) must not exceed input_device ({})",
        output_len,
        input_device.len(),
    );
    assert!(
        !prefix_device.is_empty(),
        "grand_product_device_with_prefix_device: prefix_device must hold at least one element",
    );
    let chunk_size = grand_product_device_chunk_size(output_len);
    grand_product_device_chunked(input_device, output_len, prefix_device, chunk_size)
}

/// Core chunked device-input scan shared by `grand_product_device` and
/// `grand_product_device_with_prefix_device`. `chunk_size` is the per-FFI
/// scan length (normally `grand_product_device_chunk_size`; taken as a
/// parameter so the multi-chunk boundary carry can be exercised by tests
/// without a memory-sized input).
///
/// Chunk 0 reads the caller's `prefix_device`; the running prefix is then
/// carried across each chunk boundary with a single device→device copy of the
/// last scanned scalar into an internal rolling buffer — no device→host copy
/// and no stream sync (stream ordering guarantees the copy lands before the
/// next chunk's FFI reads it). The single-chunk path allocates nothing and
/// never rolls, and the caller's `prefix_device` is never mutated.
pub(crate) fn grand_product_device_chunked<F: Field>(
    input_device: DeviceBuffer<F>,
    output_len: usize,
    prefix_device: &DeviceBuffer<F>,
    chunk_size: usize,
) -> Result<DeviceBuffer<F>, HaloGpuError> {
    if chunk_size == 0 {
        return Err(HaloGpuError::InvalidParameter {
            context: "grand_product_device_chunked",
            magnitude: output_len as u64,
        });
    }

    let bytes = std::mem::size_of::<F>();
    let input_obj = FFITraitObject::new(input_device.as_raw_ptr() as usize);

    // Rolling running-prefix buffer, allocated lazily on the first boundary.
    let mut rolling: Option<DeviceBuffer<F>> = None;

    for offset in (0..output_len).step_by(chunk_size) {
        let this_len = chunk_size.min(output_len - offset);
        let prefix_ptr = match rolling {
            Some(ref b) => b.as_raw_ptr(),
            None => prefix_device.as_raw_ptr(),
        };
        let prefix_obj = FFITraitObject::new(prefix_ptr as usize);
        let status = unsafe {
            _halo2_grand_product_device_inputs(
                &input_obj,
                &prefix_obj,
                this_len,
                offset,
                HALO2_GPU_CTX.stream.as_raw(),
            )
        };
        if status.code != 0 {
            return Err(status.into());
        }
        if offset + this_len < output_len {
            // Cross-chunk carry. The last scanned scalar of THIS chunk is the
            // next chunk's running prefix; copy it device→device into the
            // rolling buffer. Stream-ordered before the next FFI, so no host
            // round-trip and no stream sync are needed.
            let last_idx = offset + this_len - 1;
            let roll = rolling
                .get_or_insert_with(|| DeviceBuffer::<F>::with_capacity_on(1, &HALO2_GPU_CTX));
            unsafe {
                cuda_memcpy_on::<true, true>(
                    roll.as_mut_raw_ptr(),
                    (input_device.as_raw_ptr() as *const u8).add(last_idx * bytes)
                        as *const libc::c_void,
                    bytes,
                    &HALO2_GPU_CTX,
                )?;
            }
        }
    }

    Ok(input_device)
}
