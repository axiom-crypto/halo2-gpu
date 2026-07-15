use crate::cuda::culib::_halo2_batch_invert;
use crate::cuda::utils::{ensure_current_device_matches_ctx, to_device_on_pinned, HALO2_GPU_CTX};
use crate::cuda::HaloGpuError;
use ff::Field;
use openvm_cuda_common::copy::cuda_memcpy_on;
use openvm_cuda_common::d_buffer::DeviceBuffer;
use std::ffi::c_void;
use std::mem;

/// In-place Montgomery batch inversion over `data` via the GPU
/// `_halo2_batch_invert` kernel (single chunk, no length splitting).
///
/// Stages `data` into a `DeviceBuffer<F>`, runs the FFI, and copies the
/// inverted result back into `data`.
pub fn batch_invert_single_gpu<F: Field>(data: &mut [F]) -> Result<(), HaloGpuError> {
    crate::perf_section!("batch_invert");
    ensure_current_device_matches_ctx()?;
    let is_type_fr =
        std::any::TypeId::of::<F>() == std::any::TypeId::of::<halo2curves::bn256::Fr>();
    let is_type_fq =
        std::any::TypeId::of::<F>() == std::any::TypeId::of::<halo2curves::bn256::Fq>();
    let field_type: u32 = if is_type_fr {
        0
    } else if is_type_fq {
        1
    } else {
        panic!("field_type must be Fr or Fq")
    };

    let d_data = to_device_on_pinned(data as &[F])?;
    let status = unsafe {
        _halo2_batch_invert(
            d_data.as_mut_raw_ptr(),
            field_type,
            data.len(),
            HALO2_GPU_CTX.stream.as_raw(),
        )
    };

    if status.code != 0 {
        return Err(status.into());
    }

    let bytes = mem::size_of_val(data);
    unsafe {
        cuda_memcpy_on::<true, false>(
            data.as_mut_ptr() as *mut c_void,
            d_data.as_raw_ptr(),
            bytes,
            &HALO2_GPU_CTX,
        )?;
    }
    HALO2_GPU_CTX.stream.to_host_sync()?;
    Ok(())
}

pub fn batch_invert_gpu<F: Field>(data: &mut [F]) -> Result<(), HaloGpuError> {
    // Single-stream GPU prover: run the whole batch on gpu 0.
    batch_invert_single_gpu(data)
}

/// In-place Montgomery batch inversion over the caller-owned device
/// buffer `d_data`. Pure-device variant of [`batch_invert_single_gpu`]:
/// no H2D, no D2H, no stream sync. Same-stream subsequent kernels see
/// the inverted result.
pub fn batch_invert_device_in_place<F: Field>(
    d_data: &DeviceBuffer<F>,
) -> Result<(), HaloGpuError> {
    crate::perf_section!("batch_invert_device");
    ensure_current_device_matches_ctx()?;
    let is_type_fr =
        std::any::TypeId::of::<F>() == std::any::TypeId::of::<halo2curves::bn256::Fr>();
    let is_type_fq =
        std::any::TypeId::of::<F>() == std::any::TypeId::of::<halo2curves::bn256::Fq>();
    let field_type: u32 = if is_type_fr {
        0
    } else if is_type_fq {
        1
    } else {
        panic!("field_type must be Fr or Fq")
    };

    let status = unsafe {
        _halo2_batch_invert(
            d_data.as_mut_raw_ptr(),
            field_type,
            d_data.len(),
            HALO2_GPU_CTX.stream.as_raw(),
        )
    };
    if status.code != 0 {
        return Err(status.into());
    }
    Ok(())
}
