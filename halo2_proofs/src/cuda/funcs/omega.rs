use crate::arithmetic::DENSE_POWER_DEGREE;
use crate::cuda::culib::{
    _halo2_generate_omega_lut, _halo2_generate_omega_lut_workspace_size,
    _halo2_generate_omega_powers, _halo2_generate_omega_powers_workspace_size,
    _halo2_power_of_omega, _halo2_power_of_omega_workspace_size,
};
use crate::cuda::utils::{ensure_current_device_matches_ctx, FFITraitObject, HALO2_GPU_CTX};
use crate::cuda::HaloGpuError;
use ff::Field;
use openvm_cuda_common::copy::{cuda_memcpy_on, MemCopyH2D};
use openvm_cuda_common::d_buffer::DeviceBuffer;

/// Computes `omega^pow` on GPU using the per-call omega LUT and returns the
/// scalar result. `Err(HaloGpuError::Cuda)` on FFI failure.
pub fn power_of_omega_gpu<F: Field>(
    omega: F,
    omega_lut: &mut [F],
    log_n: u32,
    pow: u32,
) -> Result<F, HaloGpuError> {
    crate::perf_section!("power_of_omega");
    ensure_current_device_matches_ctx()?;
    let res = F::ZERO;
    let res_obj = FFITraitObject::from_ref(&res);
    let lut_obj = FFITraitObject::from_ref(&omega_lut[0]);
    let omega_obj = FFITraitObject::from_ref(&omega);
    let bytes = unsafe { _halo2_power_of_omega_workspace_size(log_n) } as usize;
    let scratch = DeviceBuffer::<u8>::with_capacity_on(bytes, &HALO2_GPU_CTX);
    let status = unsafe {
        _halo2_power_of_omega(
            &res_obj,
            &lut_obj,
            &omega_obj,
            log_n,
            pow,
            scratch.as_mut_raw_ptr(),
            bytes as u64,
            HALO2_GPU_CTX.stream.as_raw(),
        )
    };
    if status.code != 0 {
        return Err(status.into());
    }
    Ok(res)
}

/// Fills `omega_powers[0..output_num]` with `omega^i`; `output_num` must be
/// ≤ `1 << log_n`. The omega scalar is staged into a device buffer, the FFI
/// fills a device-resident result buffer, and the first `output_num`
/// elements are copied back into the caller's host slice.
pub fn generate_omega_powers_gpu<F: Field>(
    omega_powers: &mut [F],
    omega: F,
    log_n: u32,
    output_num: u64, // cut off to [0, output_num)
) -> Result<(), HaloGpuError> {
    crate::perf_section!("generate_omega_powers");
    ensure_current_device_matches_ctx()?;
    assert!(output_num <= (1 << log_n) as u64);
    assert!(omega_powers.len() as u64 >= output_num);

    // Stage omega host→device (32 bytes).
    let omega_device = std::slice::from_ref(&omega).to_device_on(&HALO2_GPU_CTX)?;
    // Allocate the device-resident result buffer at the full LUT length
    // the kernel writes; the Rust-side D2H truncates to `output_num`.
    let full_len = 1usize << log_n;
    let omega_powers_device = DeviceBuffer::<F>::with_capacity_on(full_len, &HALO2_GPU_CTX);
    let bytes = unsafe { _halo2_generate_omega_powers_workspace_size(log_n) } as usize;
    let scratch = DeviceBuffer::<u8>::with_capacity_on(bytes, &HALO2_GPU_CTX);
    let status = unsafe {
        _halo2_generate_omega_powers(
            omega_powers_device.as_mut_raw_ptr(),
            omega_device.as_raw_ptr(),
            log_n,
            scratch.as_mut_raw_ptr(),
            bytes as u64,
            HALO2_GPU_CTX.stream.as_raw(),
        )
    };

    if status.code != 0 {
        return Err(status.into());
    }

    // Copy the first `output_num` elements back into the caller's host slice.
    // The result buffer is staged through a manual copy + sync because the
    // FFI surface does not return a Rust-owned `DeviceBuffer`.
    let copy_bytes = std::mem::size_of::<F>() * (output_num as usize);
    unsafe {
        cuda_memcpy_on::<true, false>(
            omega_powers.as_mut_ptr() as *mut libc::c_void,
            omega_powers_device.as_raw_ptr(),
            copy_bytes,
            &HALO2_GPU_CTX,
        )?;
    }
    HALO2_GPU_CTX.stream.to_host_sync()?;
    Ok(())
}

/// Builds the sparse-twiddle omega LUT (`DENSE_POWER_DEGREE`-based
/// dual-level layout), returned host-side. Used by the FFT and
/// permutation-product kernels for per-thread `ω^i` reconstruction
/// without per-thread `pow()`.
pub fn generate_omega_lut_gpu<F: Field>(omega: F, log_n: u32) -> Result<Vec<F>, HaloGpuError> {
    crate::perf_section!("generate_omega_lut");
    ensure_current_device_matches_ctx()?;
    let dense_degree = DENSE_POWER_DEGREE;
    let low_degree_lut_len = 1 << dense_degree;
    let high_degree_lut_len = 1 << (log_n - dense_degree);
    let omega_lut = vec![F::ZERO; (low_degree_lut_len + high_degree_lut_len) as usize];

    let omega_obj = FFITraitObject::from_ref(&omega);
    let sparse_twiddle_obj = FFITraitObject::from_ref(&omega_lut[0]);
    let bytes = unsafe { _halo2_generate_omega_lut_workspace_size(log_n) } as usize;
    let scratch = DeviceBuffer::<u8>::with_capacity_on(bytes, &HALO2_GPU_CTX);
    let status = unsafe {
        _halo2_generate_omega_lut(
            &sparse_twiddle_obj,
            &omega_obj,
            log_n,
            scratch.as_mut_raw_ptr(),
            bytes as u64,
            HALO2_GPU_CTX.stream.as_raw(),
        )
    };

    if status.code != 0 {
        return Err(status.into());
    }

    Ok(omega_lut)
}
