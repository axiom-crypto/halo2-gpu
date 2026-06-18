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

    let mut prefix = prefix;
    let output_obj = FFITraitObject::from_ref(&output[0]);
    let input_obj = FFITraitObject::from_ref(&input[0]);
    let prefix_obj = FFITraitObject::from_ref(&prefix);

    let gp_scratch_bytes =
        unsafe { _halo2_grand_product_workspace_size(chunk_size as u64) } as usize;
    let scratch = DeviceBuffer::<u8>::with_capacity_on(gp_scratch_bytes, &HALO2_GPU_CTX);
    for offset in (0..poly_len).step_by(chunk_size) {
        let status = unsafe {
            _halo2_grand_product(
                &output_obj,
                &input_obj,
                &prefix_obj,
                chunk_size,
                offset,
                scratch.as_mut_raw_ptr(),
                gp_scratch_bytes as u64,
                HALO2_GPU_CTX.stream.as_raw(),
            )
        };
        if status.code != 0 {
            return Err(status.into());
        }
        if offset + chunk_size < poly_len {
            prefix = output[offset + chunk_size - 1];
            // print out this to surpress the compiler warning
            log::debug!("prefix at offset {}: {:?}", offset, prefix);
        }
    }
    Ok(())
}

/// Device-input variant of `grand_product_gpu`. Scans the first
/// `output_len` elements of the device-resident `input_device` buffer in
/// place — `input_device[i]` becomes `prefix · ∏_{j=0..=i} input_device[j]`
/// for `i ∈ [0, output_len)` — and returns the consumed buffer to the
/// caller. The scan output stays device-resident; the wrapper performs no
/// device→host copy of the scanned region.
///
/// `input_device` may be larger than `output_len` (the tail beyond
/// `output_len` is untouched). Single 32-byte D→H reads happen only at
/// chunk boundaries to roll the running prefix across the chunked FFI loop;
/// the common single-chunk path (e.g. `log_n=22` at the typical fibonacci
/// circuit size) skips that entirely.
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
    let poly_len = output_len;
    let max_len = unsafe {
        _halo2_grand_product_max_len(poly_len, query_device_free_bytes_for_chunking() as u64)
    };
    let mut chunk_size = poly_len;
    if poly_len > max_len {
        chunk_size = max_len;
    }

    let mut prefix_host = prefix;
    let mut prefix_device = std::slice::from_ref(&prefix_host).to_device_on(&HALO2_GPU_CTX)?;
    let input_obj = FFITraitObject::new(input_device.as_raw_ptr() as usize);
    let prefix_obj = FFITraitObject::new(prefix_device.as_raw_ptr() as usize);

    for offset in (0..poly_len).step_by(chunk_size) {
        let status = unsafe {
            _halo2_grand_product_device_inputs(
                &input_obj,
                &prefix_obj,
                chunk_size,
                offset,
                HALO2_GPU_CTX.stream.as_raw(),
            )
        };
        if status.code != 0 {
            return Err(status.into());
        }
        if offset + chunk_size < poly_len {
            // Multi-chunk: pull the last scanned scalar of THIS chunk from
            // device (it becomes the next chunk's running prefix) and
            // re-stage it into the device-side prefix slot. Single-chunk
            // runs (the log_n=22 common case) skip this entirely.
            let last_idx = offset + chunk_size - 1;
            let bytes = std::mem::size_of::<F>();
            unsafe {
                cuda_memcpy_on::<true, false>(
                    &mut prefix_host as *mut F as *mut libc::c_void,
                    (input_device.as_raw_ptr() as *const u8).add(last_idx * bytes)
                        as *const libc::c_void,
                    bytes,
                    &HALO2_GPU_CTX,
                )?;
            }
            HALO2_GPU_CTX.stream.to_host_sync()?;
            std::slice::from_ref(&prefix_host).copy_to_on(&mut prefix_device, &HALO2_GPU_CTX)?;
            log::debug!("prefix at offset {}: {:?}", offset, prefix_host);
        }
    }

    Ok(input_device)
}
