use super::omega::generate_omega_lut_gpu;
use crate::arithmetic::DENSE_POWER_DEGREE;
use crate::cpu::arithmetic::parallelize;
use crate::cuda::culib::{
    _halo2_cosetfft, _halo2_distribute_powers_zeta, _halo2_divide_by_vanishing_poly,
    _halo2_extended_from_lagrange_vec_device, _halo2_fft_many, _halo2_fft_many_to_device,
    _halo2_fft_many_to_device_workspace_size, _halo2_fft_many_workspace_size, _halo2_fft_normal,
    _halo2_fft_normal_check_memory, _halo2_fft_normal_to_device,
    _halo2_fft_normal_to_device_workspace_size, _halo2_fft_normal_workspace_size,
    _halo2_get_fft_split_radix,
};
use crate::cuda::utils::{
    ensure_current_device_matches_ctx, query_device_free_bytes_for_chunking, FFITraitObject,
    HALO2_GPU_CTX,
};
use crate::cuda::HaloGpuError;
use crate::poly::{Coeff, DevicePolyExt, NttType, Polynomial};
use ff::Field;
use openvm_cuda_common::copy::MemCopyH2D;
use openvm_cuda_common::d_buffer::DeviceBuffer;
use std::ffi::c_void;

pub(crate) fn module_fft_normal_with_scratch(
    data_out: *mut c_void,
    data_in: *const c_void,
    omega_in: *const c_void,
    divisor_in: *const c_void,
    ntt_type: u32,
    log_n: u32,
    scratch: &DeviceBuffer<u8>,
    scratch_bytes: u64,
) -> Result<(), HaloGpuError> {
    crate::perf_section!("fft_normal");
    ensure_current_device_matches_ctx()?;
    let status = unsafe {
        _halo2_fft_normal(
            ntt_type,
            log_n,
            data_in,
            data_out,
            omega_in,
            divisor_in,
            scratch.as_mut_raw_ptr(),
            scratch_bytes,
            HALO2_GPU_CTX.stream.as_raw(),
        )
    };
    if status.code != 0 {
        return Err(status.into());
    }
    Ok(())
}

pub(crate) fn module_fft_normal(
    data_out: *mut c_void,
    data_in: *const c_void,
    omega_in: *const c_void,
    divisor_in: *const c_void,
    ntt_type: u32,
    log_n: u32,
) -> Result<(), HaloGpuError> {
    let bytes = unsafe { _halo2_fft_normal_workspace_size(ntt_type, log_n, log_n) } as usize;
    let scratch = DeviceBuffer::<u8>::with_capacity_on(bytes, &HALO2_GPU_CTX);
    module_fft_normal_with_scratch(
        data_out,
        data_in,
        omega_in,
        divisor_in,
        ntt_type,
        log_n,
        &scratch,
        bytes as u64,
    )
}

/// Strided-gather of per-part device buffers into a contiguous extended
/// buffer: `d_out[i * num_parts + p] = d_parts[p][i]`. Mirrors the host
/// transpose-flatten at `poly/domain.rs::extended_from_lagrange_vec`.
///
/// `d_parts` must contain `num_parts` device pointers (one per part); each
/// part is `n` field elements; `d_out` has capacity `n * num_parts`. The
/// per-call H2D of the pointer table is `num_parts * sizeof(*const c_void)`
/// (≤ 32 bytes at fibonacci) and lives in a throwaway DeviceBuffer.
pub(crate) fn extended_from_lagrange_vec_device<F>(
    d_out: &DeviceBuffer<F>,
    d_parts: &[&DeviceBuffer<F>],
    n: u64,
) -> Result<(), HaloGpuError> {
    crate::perf_section!("extended_from_lagrange_vec_device");
    ensure_current_device_matches_ctx()?;
    let num_parts = d_parts.len() as u32;
    // Build the table of device pointers (the Vec itself is
    // host-resident; each element points at a device buffer) and H2D
    // it into a throwaway device buffer.
    let ptrs_device: Vec<*const c_void> = d_parts.iter().map(|b| b.as_raw_ptr()).collect();
    let d_ptr_table =
        ptrs_device.as_slice().to_device_on(&HALO2_GPU_CTX).map_err(HaloGpuError::from)?;
    let status = unsafe {
        _halo2_extended_from_lagrange_vec_device(
            d_out.as_mut_raw_ptr(),
            d_ptr_table.as_raw_ptr(),
            num_parts,
            n,
            HALO2_GPU_CTX.stream.as_raw(),
        )
    };
    if status.code != 0 {
        return Err(status.into());
    }
    Ok(())
}

/// In-place element-wise multiply by a cycling coset-powers table:
/// `d_a[i] *= d_coset_powers[(i % (coset_powers.len() + 1)) - 1]` when
/// `i_mod != 0` (identity at `i_mod == 0`). `coset_powers` is small
/// (length 2 at fibonacci shape); the H2D into a throwaway DeviceBuffer
/// is negligible (≤ 64 B per call). Mirrors the host arm at
/// `poly/domain.rs::distribute_powers_zeta`.
pub(crate) fn distribute_powers_zeta_device<F>(
    d_a: &DeviceBuffer<F>,
    coset_powers: &[F],
) -> Result<(), HaloGpuError> {
    crate::perf_section!("distribute_powers_zeta_device");
    ensure_current_device_matches_ctx()?;
    let d_coset_powers = coset_powers.to_device_on(&HALO2_GPU_CTX).map_err(HaloGpuError::from)?;
    let n = d_a.len() as u64;
    let status = unsafe {
        _halo2_distribute_powers_zeta(
            d_a.as_mut_raw_ptr(),
            d_coset_powers.as_raw_ptr(),
            coset_powers.len() as u32,
            n,
            HALO2_GPU_CTX.stream.as_raw(),
        )
    };
    if status.code != 0 {
        return Err(status.into());
    }
    Ok(())
}

/// In-place element-wise multiply: `d_poly[i] *= d_t_evals[i % t_len]`.
/// Both buffers are device-resident. Mirrors the host arm at
/// `poly/domain.rs::divide_by_vanishing_poly`.
pub(crate) fn divide_by_vanishing_poly_device<F>(
    d_poly: &DeviceBuffer<F>,
    d_t_evals: &DeviceBuffer<F>,
) -> Result<(), HaloGpuError> {
    crate::perf_section!("divide_by_vanishing_poly_device");
    ensure_current_device_matches_ctx()?;
    let t_len = d_t_evals.len() as u32;
    let n = d_poly.len() as u64;
    let status = unsafe {
        _halo2_divide_by_vanishing_poly(
            d_poly.as_mut_raw_ptr(),
            d_t_evals.as_raw_ptr(),
            t_len,
            n,
            HALO2_GPU_CTX.stream.as_raw(),
        )
    };
    if status.code != 0 {
        return Err(status.into());
    }
    Ok(())
}

#[allow(dead_code)]
/// Device-input / device-output variant of [`module_fft_normal`].
///
/// Supported `ntt_type` values: `FFT`, `iFFT`, and `CosetFFT_Part`.
pub(crate) fn fft_normal_device<F>(
    ntt_type: u32,
    log_n: u32,
    d_input: &DeviceBuffer<F>,
    d_output: &DeviceBuffer<F>,
    omega: F,
    divisor: F,
) -> Result<(), HaloGpuError> {
    crate::perf_section!("fft_normal_device");
    ensure_current_device_matches_ctx()?;
    let bytes =
        unsafe { _halo2_fft_normal_to_device_workspace_size(ntt_type, log_n, log_n) } as usize;
    let scratch = DeviceBuffer::<u8>::with_capacity_on(bytes, &HALO2_GPU_CTX);
    let omega_device = std::slice::from_ref(&omega).to_device_on(&HALO2_GPU_CTX)?;
    let status = unsafe {
        _halo2_fft_normal_to_device(
            ntt_type,
            log_n,
            d_input.as_raw_ptr(),
            d_output.as_mut_raw_ptr(),
            omega_device.as_raw_ptr(),
            &divisor as *const F as *const c_void,
            scratch.as_mut_raw_ptr(),
            bytes as u64,
            HALO2_GPU_CTX.stream.as_raw(),
        )
    };
    if status.code != 0 {
        return Err(status.into());
    }
    Ok(())
}

pub(crate) fn fft_gpu<F: Field>(
    ntt_type: u32,
    a: &mut [F],
    log_n: u32,
    omega: F,
    divisor: F,
) -> Result<(), HaloGpuError> {
    let data_in = a.as_ptr() as *const c_void;
    let data_out = a.as_mut_ptr() as *mut c_void;
    let omega_in = &omega as *const F as *const c_void;
    let divisor_in = &divisor as *const F as *const c_void;
    let is_memory_enough =
        unsafe { _halo2_fft_normal_check_memory(ntt_type, data_in, log_n, log_n) };
    if is_memory_enough {
        module_fft_normal(data_out, data_in, omega_in, divisor_in, ntt_type, log_n)
    } else {
        split_radix_fft_gpu(ntt_type, a, log_n, log_n, omega, divisor)
    }
}

pub(crate) fn cosetfft_gpu<F>(
    ntt_type: u32,
    a: &[F],
    b: &mut [F],
    log_n: u32,
    extend_log_n: u32,
    omega: F,
    divisor: F,
) -> Result<(), HaloGpuError> {
    crate::perf_section!("cosetfft");
    ensure_current_device_matches_ctx()?;
    let bytes = unsafe { _halo2_fft_normal_workspace_size(ntt_type, log_n, extend_log_n) } as usize;
    let scratch = DeviceBuffer::<u8>::with_capacity_on(bytes, &HALO2_GPU_CTX);
    let status = unsafe {
        _halo2_cosetfft(
            ntt_type,
            log_n,
            extend_log_n,
            a.as_ptr() as *const libc::c_void,
            b.as_mut_ptr() as *mut libc::c_void,
            &omega as *const F as *const libc::c_void,
            &divisor as *const F as *const libc::c_void,
            scratch.as_mut_raw_ptr(),
            bytes as u64,
            HALO2_GPU_CTX.stream.as_raw(),
        )
    };

    if status.code != 0 {
        return Err(status.into());
    }
    Ok(())
}

pub(crate) fn cosetfft_gpu_many<F>(
    ntt_type: u32,
    in_many: Vec<FFITraitObject>,
    out_many: Vec<FFITraitObject>,
    log_n: u32,
    extend_log_n: u32,
    omega: F,
    divisor: F,
) -> Result<(), HaloGpuError> {
    crate::perf_section!("cosetfft_many");
    ensure_current_device_matches_ctx()?;
    assert_eq!(in_many.len(), out_many.len());
    let omega_device = std::slice::from_ref(&omega).to_device_on(&HALO2_GPU_CTX)?;
    let divisor_obj = FFITraitObject::from_ref(&divisor);

    let bytes = unsafe { _halo2_fft_many_workspace_size(ntt_type, log_n, extend_log_n) } as usize;
    let scratch = DeviceBuffer::<u8>::with_capacity_on(bytes, &HALO2_GPU_CTX);
    let status = unsafe {
        _halo2_fft_many(
            ntt_type,
            out_many.len() as u32,
            log_n,
            extend_log_n,
            in_many.as_ptr(),
            out_many.as_ptr(),
            omega_device.as_raw_ptr(),
            &divisor_obj,
            scratch.as_mut_raw_ptr(),
            bytes as u64,
            HALO2_GPU_CTX.stream.as_raw(),
        )
    };

    if status.code != 0 {
        return Err(status.into());
    }
    Ok(())
}

/// Device-output variant of `cosetfft_gpu_many`.
///
/// Allocates one `DeviceBuffer<F>` per input polynomial, routes the GPU FFT result directly into
/// those device buffers, and returns them. The caller can then pass the
/// device pointers to a downstream GPU kernel without paying a
/// `cudaMemcpyDeviceToHost` here followed by a `cudaMemcpyHostToDevice`
/// in the next FFI's preamble.
///
/// Callers that ALSO need a host copy (e.g. the lookup CPU evaluator)
/// should `cuda_memcpy_on::<true, false>(...)` from the returned device
/// buffer to a host `Vec<F>` — single FFT pass, single D→H, no waste.
pub(crate) fn cosetfft_many_h2d<F>(
    ntt_type: u32,
    in_many: Vec<FFITraitObject>,
    log_n: u32,
    extend_log_n: u32,
    omega: F,
    divisor: F,
) -> Result<Vec<DeviceBuffer<F>>, HaloGpuError> {
    crate::perf_section!("cosetfft_many_to_device");
    ensure_current_device_matches_ctx()?;
    let num_many = in_many.len();

    // Allocate device-resident output buffers, one per polynomial. Size
    // matches the kernel's actual valid output:
    //   - `cosetFFT` / `icosetFFT` upscale internally, output is
    //     `1 << extend_log_n`.
    //   - `FFT` / `iFFT` / `CosetFFT_Part` keep size `1 << log_n`.
    // (See `CudaFFTManyInfo::data_memory_size` in `cuda/src/ntt.cu`.)
    // The device-output path writes the FFT result directly into
    // these final buffers, so the element count must match the kernel's
    // true output shape exactly.
    let coset_fft_id = NttType::CosetFFT as u32;
    let icoset_fft_id = NttType::iCosetFFT as u32;
    let output_log_n =
        if ntt_type == coset_fft_id || ntt_type == icoset_fft_id { extend_log_n } else { log_n };
    let output_len = 1usize << output_log_n;
    let out_bufs: Vec<DeviceBuffer<F>> = (0..num_many)
        .map(|_| DeviceBuffer::<F>::with_capacity_on(output_len, &HALO2_GPU_CTX))
        .collect();
    let out_objs: Vec<FFITraitObject> =
        out_bufs.iter().map(|b| FFITraitObject::new(b.as_raw_ptr() as usize)).collect();

    let omega_device = std::slice::from_ref(&omega).to_device_on(&HALO2_GPU_CTX)?;
    let divisor_obj = FFITraitObject::from_ref(&divisor);

    let bytes =
        unsafe { _halo2_fft_many_to_device_workspace_size(ntt_type, log_n, extend_log_n) } as usize;
    let scratch = DeviceBuffer::<u8>::with_capacity_on(bytes, &HALO2_GPU_CTX);
    let status = unsafe {
        _halo2_fft_many_to_device(
            ntt_type,
            num_many as u32,
            log_n,
            extend_log_n,
            in_many.as_ptr(),
            out_objs.as_ptr(),
            omega_device.as_raw_ptr(),
            &divisor_obj,
            /* input_on_device */ false,
            scratch.as_mut_raw_ptr(),
            bytes as u64,
            HALO2_GPU_CTX.stream.as_raw(),
        )
    };

    if status.code != 0 {
        return Err(status.into());
    }
    Ok(out_bufs)
}

/// Device-input + device-output variant of `cosetfft_many_h2d`.
///
/// Inputs are device-resident `DeviceBuffer<F>` slices wrapped as
/// `FFITraitObject`. Allocates one fresh `DeviceBuffer<F>` per polynomial
/// for the FFT output and routes the kernel result directly into them on
/// the canonical stream. No host staging on input or output.
pub(crate) fn cosetfft_many_device<F>(
    ntt_type: u32,
    in_many_device: Vec<FFITraitObject>,
    log_n: u32,
    extend_log_n: u32,
    omega: F,
    divisor: F,
) -> Result<Vec<DeviceBuffer<F>>, HaloGpuError> {
    crate::perf_section!("cosetfft_many_device_to_device");
    ensure_current_device_matches_ctx()?;
    let num_many = in_many_device.len();

    let coset_fft_id = NttType::CosetFFT as u32;
    let icoset_fft_id = NttType::iCosetFFT as u32;
    let output_log_n =
        if ntt_type == coset_fft_id || ntt_type == icoset_fft_id { extend_log_n } else { log_n };
    let output_len = 1usize << output_log_n;
    let out_bufs: Vec<DeviceBuffer<F>> = (0..num_many)
        .map(|_| DeviceBuffer::<F>::with_capacity_on(output_len, &HALO2_GPU_CTX))
        .collect();
    let out_objs: Vec<FFITraitObject> =
        out_bufs.iter().map(|b| FFITraitObject::new(b.as_raw_ptr() as usize)).collect();

    let omega_device = std::slice::from_ref(&omega).to_device_on(&HALO2_GPU_CTX)?;
    let divisor_obj = FFITraitObject::from_ref(&divisor);

    let bytes =
        unsafe { _halo2_fft_many_to_device_workspace_size(ntt_type, log_n, extend_log_n) } as usize;
    let scratch = DeviceBuffer::<u8>::with_capacity_on(bytes, &HALO2_GPU_CTX);
    let status = unsafe {
        _halo2_fft_many_to_device(
            ntt_type,
            num_many as u32,
            log_n,
            extend_log_n,
            in_many_device.as_ptr(),
            out_objs.as_ptr(),
            omega_device.as_raw_ptr(),
            &divisor_obj,
            /* input_on_device */ true,
            scratch.as_mut_raw_ptr(),
            bytes as u64,
            HALO2_GPU_CTX.stream.as_raw(),
        )
    };

    if status.code != 0 {
        return Err(status.into());
    }
    Ok(out_bufs)
}

// in-place fft only
pub(crate) fn fft_gpu_many<F>(
    ntt_type: u32,
    inout_many: Vec<FFITraitObject>,
    log_n: u32,
    omega: F,
    divisor: F,
) -> Result<(), HaloGpuError> {
    crate::perf_section!("fft_many");
    ensure_current_device_matches_ctx()?;
    let omega_device = std::slice::from_ref(&omega).to_device_on(&HALO2_GPU_CTX)?;
    let divisor_obj = FFITraitObject::from_ref(&divisor);

    let bytes = unsafe { _halo2_fft_many_workspace_size(ntt_type, log_n, log_n) } as usize;
    let scratch = DeviceBuffer::<u8>::with_capacity_on(bytes, &HALO2_GPU_CTX);
    let status = unsafe {
        _halo2_fft_many(
            ntt_type,
            inout_many.len() as u32,
            log_n,
            log_n,
            inout_many.as_ptr(),
            inout_many.as_ptr(),
            omega_device.as_raw_ptr(),
            &divisor_obj,
            scratch.as_mut_raw_ptr(),
            bytes as u64,
            HALO2_GPU_CTX.stream.as_raw(),
        )
    };

    if status.code != 0 {
        return Err(status.into());
    }
    Ok(())
}

pub(crate) fn ifft_gpu_many<F: Field>(
    ntt_type: u32,
    in_many: Vec<FFITraitObject>,
    out_many: Vec<FFITraitObject>,
    log_n: u32,
    omega: F,
    divisor: F,
) -> Result<(), HaloGpuError> {
    crate::perf_section!("ifft_many");
    ensure_current_device_matches_ctx()?;
    assert_eq!(in_many.len(), out_many.len());
    let omega_device = std::slice::from_ref(&omega).to_device_on(&HALO2_GPU_CTX)?;
    let divisor_obj = FFITraitObject::from_ref(&divisor);

    let bytes = unsafe { _halo2_fft_many_workspace_size(ntt_type, log_n, log_n) } as usize;
    let scratch = DeviceBuffer::<u8>::with_capacity_on(bytes, &HALO2_GPU_CTX);
    let status = unsafe {
        _halo2_fft_many(
            ntt_type,
            in_many.len() as u32,
            log_n,
            log_n,
            in_many.as_ptr(),
            out_many.as_ptr(),
            omega_device.as_raw_ptr(),
            &divisor_obj,
            scratch.as_mut_raw_ptr(),
            bytes as u64,
            HALO2_GPU_CTX.stream.as_raw(),
        )
    };

    if status.code != 0 {
        return Err(status.into());
    }
    Ok(())
}

/// Host-input, Device-output batch iFFT.
///
/// Allocates one `DeviceBuffer<F>` per input polynomial (each of length
/// `1 << log_n`), routes the FFT result directly into those device
/// buffers, and returns the resulting Device-resident
/// `Polynomial<F, Coeff>` items. Reuses the existing
/// `_halo2_fft_many_to_device` FFI (same one used for cosetFFT
/// device-output) with `ntt_type = iFFT`.
///
/// Mirrors `cosetfft_many_h2d`'s device-output pattern; the
/// only differences are the NTT mode (iFFT) and the output basis tag
/// (`Coeff`). VRAM headroom is checked at the caller (`lagrange_to_coeff(_many)_device`
/// in `domain.rs`), so this wrapper does not pre-check.
pub(crate) fn ifft_many_h2d<F>(
    in_many: &[Polynomial<F, crate::poly::LagrangeCoeff>],
    log_n: u32,
    omega: F,
    divisor: F,
) -> Result<Vec<Polynomial<F, Coeff, crate::poly::Device>>, HaloGpuError>
where
    F: Field,
{
    crate::perf_section!("ifft_many_to_device");
    ensure_current_device_matches_ctx()?;
    let num_many = in_many.len();
    if num_many == 0 {
        return Ok(vec![]);
    }
    let n = 1usize << log_n;
    let in_objs: Vec<FFITraitObject> =
        in_many.iter().map(|p| FFITraitObject::from_slice(p.values())).collect();
    let out_bufs: Vec<DeviceBuffer<F>> =
        (0..num_many).map(|_| DeviceBuffer::<F>::with_capacity_on(n, &HALO2_GPU_CTX)).collect();
    let out_objs: Vec<FFITraitObject> =
        out_bufs.iter().map(|b| FFITraitObject::new(b.as_raw_ptr() as usize)).collect();
    let omega_device = std::slice::from_ref(&omega).to_device_on(&HALO2_GPU_CTX)?;
    let divisor_obj = FFITraitObject::from_ref(&divisor);
    let ntt_type = crate::poly::NttType::iFFT as u32;
    let bytes = unsafe {
        crate::cuda::culib::_halo2_fft_many_to_device_workspace_size(ntt_type, log_n, log_n)
    } as usize;
    let scratch = DeviceBuffer::<u8>::with_capacity_on(bytes, &HALO2_GPU_CTX);
    let status = unsafe {
        crate::cuda::culib::_halo2_fft_many_to_device(
            ntt_type,
            num_many as u32,
            log_n,
            log_n,
            in_objs.as_ptr(),
            out_objs.as_ptr(),
            omega_device.as_raw_ptr(),
            &divisor_obj,
            /* input_on_device */ false,
            scratch.as_mut_raw_ptr(),
            bytes as u64,
            HALO2_GPU_CTX.stream.as_raw(),
        )
    };
    if status.code != 0 {
        return Err(status.into());
    }
    // Wrap each Device buffer as a Polynomial<F, Coeff>::Device. The
    // out_objs (FFITraitObjects) drop here; out_bufs ownership passes
    // into the returned Polynomials.
    Ok(out_bufs.into_iter().map(Polynomial::from_device).collect())
}

/// Device-input + device-output batch iFFT.
///
/// Inputs are device-resident `DeviceBuffer<F>` slices wrapped as
/// `FFITraitObject`. Allocates one fresh `DeviceBuffer<F>` per polynomial
/// (each of length `1 << log_n`) for the iFFT output and routes the
/// kernel result directly into them on the canonical stream. The FFI
/// `cudaMemcpyAsync` on the data plane becomes a device-to-device copy
/// (`run_fft_many` keys the copy kind off `input_on_device`).
///
/// Mirrors `cosetfft_many_device`'s device-input structure
/// with the iFFT `ntt_type` and the same omega/divisor semantics as
/// `ifft_many_h2d`. VRAM headroom is gated at each caller
/// (`lagrange_to_coeff_many_device` for the batch device-input arm
/// and `lagrange_to_coeff_device_input` for the single-poly device-input
/// arm in `domain.rs`).
pub(crate) fn ifft_many_device<F>(
    in_many_device: Vec<FFITraitObject>,
    log_n: u32,
    omega: F,
    divisor: F,
) -> Result<Vec<DeviceBuffer<F>>, HaloGpuError>
where
    F: Field,
{
    crate::perf_section!("ifft_many_device_to_device");
    ensure_current_device_matches_ctx()?;
    let num_many = in_many_device.len();
    if num_many == 0 {
        return Ok(vec![]);
    }
    let n = 1usize << log_n;
    let out_bufs: Vec<DeviceBuffer<F>> =
        (0..num_many).map(|_| DeviceBuffer::<F>::with_capacity_on(n, &HALO2_GPU_CTX)).collect();
    let out_objs: Vec<FFITraitObject> =
        out_bufs.iter().map(|b| FFITraitObject::new(b.as_raw_ptr() as usize)).collect();
    let omega_device = std::slice::from_ref(&omega).to_device_on(&HALO2_GPU_CTX)?;
    let divisor_obj = FFITraitObject::from_ref(&divisor);
    let ntt_type = NttType::iFFT as u32;
    let bytes = unsafe {
        crate::cuda::culib::_halo2_fft_many_to_device_workspace_size(ntt_type, log_n, log_n)
    } as usize;
    let scratch = DeviceBuffer::<u8>::with_capacity_on(bytes, &HALO2_GPU_CTX);
    let status = unsafe {
        crate::cuda::culib::_halo2_fft_many_to_device(
            ntt_type,
            num_many as u32,
            log_n,
            log_n,
            in_many_device.as_ptr(),
            out_objs.as_ptr(),
            omega_device.as_raw_ptr(),
            &divisor_obj,
            /* input_on_device */ true,
            scratch.as_mut_raw_ptr(),
            bytes as u64,
            HALO2_GPU_CTX.stream.as_raw(),
        )
    };
    if status.code != 0 {
        return Err(status.into());
    }
    Ok(out_bufs)
}

fn get_fft_split_radix_gpu(
    ntt_type: u32,
    log_n: u32,
    extend_log_n: u32,
) -> Result<u32, HaloGpuError> {
    // `_halo2_get_fft_split_radix` is a host-side decision query: it
    // returns an `i32` (not a `CudaStatus`) where -1 means "insufficient
    // device memory for any split" and 0 means "undefined behaviour"
    // (caller passed a bogus log_n / extend_log_n). Surface both as
    // typed `HaloGpuError` variants.
    let free_bytes = query_device_free_bytes_for_chunking() as u64;
    let log_split_radix: i32 =
        unsafe { _halo2_get_fft_split_radix(ntt_type, log_n, extend_log_n, free_bytes) };
    if log_split_radix == -1 {
        return Err(HaloGpuError::InsufficientGpuMemory {
            context: "get_fft_split_radix_gpu",
            magnitude: log_n as u64,
            free_bytes,
        });
    }
    if log_split_radix == 0 {
        return Err(HaloGpuError::InvalidParameter {
            context: "get_fft_split_radix_gpu",
            magnitude: log_n as u64,
        });
    }
    Ok(log_split_radix as u32)
}

fn split_radix_fft_prologue<F: Field>(
    a: &[F], // fft data
    omega: F,
    tmp: &mut [F],      // tmp buffer
    split_radix: usize, // 1<<log_split
    j: usize,           // iter idx of split_radix
    log_n: u32,         // fft log_n
    log_split_n: u32,   // log_n - log_split
) -> Result<(), HaloGpuError> {
    let dense_degree = DENSE_POWER_DEGREE;
    let omega_lut = generate_omega_lut_gpu(omega, log_n)?;
    let _step: usize = j << (log_split_n as usize);
    parallelize(tmp, |tmp, start| {
        for (i, tmp) in tmp.iter_mut().enumerate() {
            let i = start + i;
            for s in 0..split_radix {
                let tiwddle_idx = (i * j + s * _step) % (1 << log_n);
                let low_idx = tiwddle_idx % (1 << dense_degree);
                let high_idx = (1 << dense_degree) + (tiwddle_idx >> dense_degree);
                let twiddle = omega_lut[low_idx] * omega_lut[high_idx];
                let mut t = a[(i + (s << log_split_n)) % (1 << log_n)];
                t.mul_assign(&twiddle);
                if s == 0 {
                    *tmp = t;
                } else {
                    tmp.add_assign(&t);
                }
            }
        }
    });
    Ok(())
}

fn split_radix_fft_epilogue<F: Field>(
    ntt_type: u32,
    a: &mut [F],
    tmp: &[F],
    divisor: F,
    log_split: u32,
) {
    let ifft_type: u32 = NttType::iFFT.into();
    let icosetfft_type: u32 = NttType::iCosetFFT.into();
    let mask = (1 << log_split) - 1;
    let chunk_size = tmp.len() >> log_split;
    parallelize(a, |a, start| {
        for (idx, a) in a.iter_mut().enumerate() {
            let idx = start + idx;
            *a = tmp[chunk_size * (idx & mask) + (idx >> log_split)];
            if ntt_type == ifft_type || ntt_type == icosetfft_type {
                a.mul_assign(&divisor);
            }
        }
    });
}

#[allow(clippy::uninit_vec)]
/// In-place split-radix FFT over a host slice `a`. Decomposes `1 << log_n`
/// into log_split + log_split_n radix passes. The host-input variant
/// internally H2D's per chunk via `tmp` (vs the device-input
/// `split_radix_fft_inout_gpu` sibling). Follows the module-wide
/// wrapper contract; debug_assert!'s `a.len() == 1 << iternal_log_n`.
pub fn split_radix_fft_gpu<F: Field>(
    ntt_type: u32,
    a: &mut [F],
    log_n: u32,
    extend_log_n: u32,
    omega: F,
    divisor: F,
) -> Result<(), HaloGpuError> {
    crate::perf_section!("split_radix_fft");
    let split_ntt_type = NttType::FFT.into();
    let split_divisor = F::ONE;
    let coset_ntt_type: u32 = NttType::CosetFFT.into();
    let part_coset_ntt_type: u32 = NttType::CosetFFT_Part.into();
    let mut iternal_log_n = log_n;
    if ntt_type == coset_ntt_type {
        iternal_log_n = extend_log_n;
    }
    debug_assert_eq!(
        a.len(),
        1usize << iternal_log_n,
        "split_radix_fft_gpu: a.len() ({}) must equal 1 << iternal_log_n (1 << {})",
        a.len(),
        iternal_log_n
    );
    // mul power before split
    if ntt_type == part_coset_ntt_type {
        let c = divisor; // borrow divisor param slot
        parallelize(a, |a, index| {
            let mut c_power = c.pow_vartime([index as u64, 0, 0, 0]);
            for a in a {
                a.mul_assign(&c_power);
                c_power = c_power * c;
            }
        });
    }

    let log_split = get_fft_split_radix_gpu(ntt_type, log_n, extend_log_n)?;
    let log_split_n = iternal_log_n - log_split;
    let split_radix = 1 << log_split; // radix-x
    let new_omega = omega.pow_vartime([split_radix as u64]);

    let mut tmp: Vec<F> = Vec::with_capacity(1 << iternal_log_n);
    unsafe {
        tmp.set_len(1 << iternal_log_n);
    }

    let split_scratch_bytes =
        unsafe { _halo2_fft_normal_workspace_size(split_ntt_type, log_split_n, log_split_n) }
            as usize;
    let split_scratch = DeviceBuffer::<u8>::with_capacity_on(split_scratch_bytes, &HALO2_GPU_CTX);

    // split-radix prologue and sub-fft
    for (j, tmp) in tmp.chunks_mut(1 << log_split_n).enumerate() {
        split_radix_fft_prologue(a, omega, tmp, split_radix, j, iternal_log_n, log_split_n)?;

        let data_in = tmp.as_ptr() as *const c_void;
        let data_out = tmp.as_mut_ptr() as *mut c_void;
        let omega_in = &new_omega as *const F as *const c_void;
        let divisor_in = &split_divisor as *const F as *const c_void;

        module_fft_normal_with_scratch(
            data_out,
            data_in,
            omega_in,
            divisor_in,
            split_ntt_type,
            log_split_n,
            &split_scratch,
            split_scratch_bytes as u64,
        )?;
    }

    split_radix_fft_epilogue(ntt_type, a, &tmp, divisor, log_split);
    Ok(())
}

#[allow(clippy::uninit_vec)]
pub fn split_radix_fft_inout_gpu<F: Field>(
    ntt_type: u32,
    a: &[F],
    b: &mut [F],
    log_n: u32,
    extend_log_n: u32,
    omega: F,
    divisor: F,
) -> Result<(), HaloGpuError> {
    crate::perf_section!("split_radix_fft_inout");
    let split_ntt_type = NttType::FFT.into();
    let split_divisor = F::ONE;
    let coset_ntt_type: u32 = NttType::CosetFFT.into();
    let part_coset_ntt_type: u32 = NttType::CosetFFT_Part.into();
    let mut iternal_log_n = log_n;
    if ntt_type == coset_ntt_type {
        iternal_log_n = extend_log_n;
    }
    debug_assert_eq!(
        a.len(),
        b.len(),
        "split_radix_fft_inout_gpu: a.len() ({}) must equal b.len() ({})",
        a.len(),
        b.len()
    );
    debug_assert_eq!(
        a.len(),
        1usize << iternal_log_n,
        "split_radix_fft_inout_gpu: a.len() ({}) must equal 1 << iternal_log_n (1 << {})",
        a.len(),
        iternal_log_n
    );
    // mul power before split
    if ntt_type == part_coset_ntt_type {
        panic!("unsupported type NttType::CosetFFT_Part");
    }

    let log_split = get_fft_split_radix_gpu(ntt_type, log_n, extend_log_n)?;
    let log_split_n = iternal_log_n - log_split;
    let split_radix = 1 << log_split; // radix-x
    let new_omega = omega.pow_vartime([split_radix as u64]);

    let mut tmp: Vec<F> = Vec::with_capacity(1 << iternal_log_n);
    unsafe {
        tmp.set_len(1 << iternal_log_n);
    }

    let split_scratch_bytes =
        unsafe { _halo2_fft_normal_workspace_size(split_ntt_type, log_split_n, log_split_n) }
            as usize;
    let split_scratch = DeviceBuffer::<u8>::with_capacity_on(split_scratch_bytes, &HALO2_GPU_CTX);

    // split-radix prologue and sub-fft
    for (j, tmp) in tmp.chunks_mut(1 << log_split_n).enumerate() {
        split_radix_fft_prologue(a, omega, tmp, split_radix, j, iternal_log_n, log_split_n)?;

        let data_in = tmp.as_ptr() as *const c_void;
        let data_out = tmp.as_mut_ptr() as *mut c_void;
        let omega_in = &new_omega as *const F as *const c_void;
        let divisor_in = &split_divisor as *const F as *const c_void;

        module_fft_normal_with_scratch(
            data_out,
            data_in,
            omega_in,
            divisor_in,
            split_ntt_type,
            log_split_n,
            &split_scratch,
            split_scratch_bytes as u64,
        )?;
    }

    split_radix_fft_epilogue(ntt_type, b, &tmp, divisor, log_split);
    Ok(())
}
