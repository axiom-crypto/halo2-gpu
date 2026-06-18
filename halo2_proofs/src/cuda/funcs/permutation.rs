use crate::cuda::culib::{
    _halo2_permutation_product, _halo2_permutation_product_device_inputs,
    _halo2_permutation_product_device_inputs_workspace_size, _halo2_permutation_product_max_len,
    _halo2_permutation_product_workspace_size, _halo2_quotient_permutation,
};
use crate::cuda::utils::{
    ensure_current_device_matches_ctx, query_device_free_bytes_for_chunking, FFITraitObject,
    HALO2_GPU_CTX,
};
use crate::cuda::HaloGpuError;
use ff::{Field, WithSmallOrderMulGroup};
use openvm_cuda_common::copy::{cuda_memcpy_on, MemCopyH2D};
use openvm_cuda_common::d_buffer::DeviceBuffer;
use std::ffi::c_void;
use std::mem;

pub fn permutation_product_gpu<F: Field>(
    modified_values: &mut [F],
    permutations_ffi: &[FFITraitObject],
    values_ffi: &[FFITraitObject],
    beta: F,
    gamma: F,
    delta: F,
    omega: F,
    deltaomega: F,
) -> Result<(), HaloGpuError> {
    crate::perf_section!("permutation_product");
    ensure_current_device_matches_ctx()?;
    let batch_size = permutations_ffi.len();
    let poly_len = modified_values.len();
    let max_len = unsafe {
        _halo2_permutation_product_max_len(poly_len, query_device_free_bytes_for_chunking() as u64)
    };
    let mut chunk_size = poly_len;
    if poly_len > max_len {
        chunk_size = max_len;
    }

    // beta/gamma/delta/omega are constant across every call in a proof; their
    // device staging is re-uploaded per call because there is no shared cache
    // on this path.
    let beta_device = std::slice::from_ref(&beta).to_device_on(&HALO2_GPU_CTX)?;
    let gamma_device = std::slice::from_ref(&gamma).to_device_on(&HALO2_GPU_CTX)?;
    let delta_device = std::slice::from_ref(&delta).to_device_on(&HALO2_GPU_CTX)?;
    let omega_device = std::slice::from_ref(&omega).to_device_on(&HALO2_GPU_CTX)?;
    let mut deltaomega_device = DeviceBuffer::<F>::with_capacity_on(1, &HALO2_GPU_CTX);

    let pp_scratch_bytes =
        unsafe { _halo2_permutation_product_workspace_size(chunk_size as u64) } as usize;
    let scratch = DeviceBuffer::<u8>::with_capacity_on(pp_scratch_bytes, &HALO2_GPU_CTX);
    for offset in (0..poly_len).step_by(chunk_size) {
        let _deltaomega = deltaomega * omega.pow_vartime([offset as u64, 0, 0, 0]);
        std::slice::from_ref(&_deltaomega).copy_to_on(&mut deltaomega_device, &HALO2_GPU_CTX)?;
        let modified_values_obj = FFITraitObject::from_ref(&modified_values[offset]);
        let status = unsafe {
            _halo2_permutation_product(
                &modified_values_obj,
                permutations_ffi.as_ptr(),
                values_ffi.as_ptr(),
                beta_device.as_raw_ptr(),
                gamma_device.as_raw_ptr(),
                delta_device.as_raw_ptr(),
                omega_device.as_raw_ptr(),
                deltaomega_device.as_mut_raw_ptr(),
                chunk_size,
                offset,
                batch_size,
                scratch.as_mut_raw_ptr(),
                pp_scratch_bytes as u64,
                HALO2_GPU_CTX.stream.as_raw(),
            )
        };
        if status.code != 0 {
            return Err(status.into());
        }
    }
    Ok(())
}

/// Device-input + device-output variant of `permutation_product_gpu`.
///
/// `modified_values_device` is a full-length `DeviceBuffer<F>` that the
/// caller initialises to `F::ONE`; the chunk loop writes per-chunk
/// accumulated products in-place. Each `permutations_ffi[i].ptr` and
/// `values_ffi[i].ptr` is a device pointer to a full-length column
/// (caller supplies σ + advice/instance/fixed device pointers).
pub fn permutation_product_device<F: Field>(
    modified_values_device: &mut DeviceBuffer<F>,
    permutations_ffi: &[FFITraitObject],
    values_ffi: &[FFITraitObject],
    beta: F,
    gamma: F,
    delta: F,
    omega: F,
    deltaomega: F,
) -> Result<(), HaloGpuError> {
    crate::perf_section!("permutation_product_device_inputs");
    ensure_current_device_matches_ctx()?;
    let batch_size = permutations_ffi.len();
    let poly_len = modified_values_device.len();
    let max_len = unsafe {
        _halo2_permutation_product_max_len(poly_len, query_device_free_bytes_for_chunking() as u64)
    };
    let mut chunk_size = poly_len;
    if poly_len > max_len {
        chunk_size = max_len;
    }

    // beta/gamma/delta/omega are constant across every call in a proof; their
    // device staging is re-uploaded per call because there is no shared cache
    // on this path.
    let beta_device = std::slice::from_ref(&beta).to_device_on(&HALO2_GPU_CTX)?;
    let gamma_device = std::slice::from_ref(&gamma).to_device_on(&HALO2_GPU_CTX)?;
    let delta_device = std::slice::from_ref(&delta).to_device_on(&HALO2_GPU_CTX)?;
    let omega_device = std::slice::from_ref(&omega).to_device_on(&HALO2_GPU_CTX)?;
    let mut deltaomega_device = DeviceBuffer::<F>::with_capacity_on(1, &HALO2_GPU_CTX);

    let pp_scratch_bytes =
        unsafe { _halo2_permutation_product_device_inputs_workspace_size(chunk_size as u64) }
            as usize;
    let scratch = DeviceBuffer::<u8>::with_capacity_on(pp_scratch_bytes, &HALO2_GPU_CTX);
    let elem_bytes = mem::size_of::<F>();
    let base_ptr = modified_values_device.as_raw_ptr() as *const u8;
    for offset in (0..poly_len).step_by(chunk_size) {
        let _deltaomega = deltaomega * omega.pow_vartime([offset as u64, 0, 0, 0]);
        std::slice::from_ref(&_deltaomega).copy_to_on(&mut deltaomega_device, &HALO2_GPU_CTX)?;
        let chunk_ptr = unsafe { base_ptr.add(offset * elem_bytes) };
        let modified_values_obj = FFITraitObject::new(chunk_ptr as usize);
        let status = unsafe {
            _halo2_permutation_product_device_inputs(
                &modified_values_obj,
                permutations_ffi.as_ptr(),
                values_ffi.as_ptr(),
                beta_device.as_raw_ptr(),
                gamma_device.as_raw_ptr(),
                delta_device.as_raw_ptr(),
                omega_device.as_raw_ptr(),
                deltaomega_device.as_mut_raw_ptr(),
                chunk_size,
                offset,
                batch_size,
                scratch.as_mut_raw_ptr(),
                pp_scratch_bytes as u64,
                HALO2_GPU_CTX.stream.as_raw(),
            )
        };
        if status.code != 0 {
            return Err(status.into());
        }
    }
    Ok(())
}

/// Permutation-quotient kernel dispatch on the crate's explicit CUDA stream.
///
/// All polynomial inputs arrive as device-resident pointer arrays:
///   - `d_values` / `d_l0` / `d_l_last` / `d_l_active_row` are
///     caller-owned device buffers;
///   - the three pointer tables are uploaded once via `cuda-common` and
///     passed to CUDA as device-resident arrays of device pointers.
///
/// The CUDA entrypoint is intentionally thin: it launches the kernel on
/// the same explicit stream and does not hide any extra H2D work or
/// cross-stream synchronization.
#[allow(clippy::too_many_arguments)]
pub(crate) fn module_quotient_permutation(
    // device-borrowed
    d_values: *mut c_void,
    d_l0: *const c_void,
    d_l_last: *const c_void,
    d_l_active_row: *const c_void,
    // device pointer arrays (host slice of device pointers)
    d_perm_prod_cosets_ptrs: &[*const c_void],
    d_perm_cosets_ptrs: &[*const c_void],
    d_column_values_ptrs: &[*const c_void],
    // host-borrowed: single scalars
    h_beta: *const c_void,
    h_gamma: *const c_void,
    h_y: *const c_void,
    h_delta: *const c_void,
    h_delta_start: *const c_void,
    h_current_extended_omega: *const c_void,
    h_omega: *const c_void,
    // metadata
    n_sets: usize,
    chunk_len: usize,
    last_rotation: i32,
    rot_scale: i32,
    isize_: i32,
    poly_length: usize,
) -> Result<(), HaloGpuError> {
    crate::perf_section!("quotient_permutation");
    debug_assert_eq!(d_perm_prod_cosets_ptrs.len(), n_sets);
    let n_perm_cols = d_column_values_ptrs.len();
    debug_assert_eq!(d_perm_cosets_ptrs.len(), n_perm_cols);
    // The last set may be partial when n_perm_cols % chunk_len != 0; the
    // sets always cover all columns and never extend past them.
    debug_assert!(n_perm_cols <= n_sets * chunk_len);
    debug_assert!(n_perm_cols + chunk_len > n_sets * chunk_len);

    debug_assert!(n_perm_cols > 0);

    crate::perf_h2d!(
        "cuda.quotient_permutation.perm_prod_ptrs",
        mem::size_of_val(d_perm_prod_cosets_ptrs)
    );
    let d_perm_prod_ptrs_table = d_perm_prod_cosets_ptrs.to_device_on(&HALO2_GPU_CTX)?;
    crate::perf_h2d!(
        "cuda.quotient_permutation.perm_coset_ptrs",
        mem::size_of_val(d_perm_cosets_ptrs)
    );
    let d_perm_coset_ptrs_table = d_perm_cosets_ptrs.to_device_on(&HALO2_GPU_CTX)?;
    crate::perf_h2d!(
        "cuda.quotient_permutation.column_values_ptrs",
        mem::size_of_val(d_column_values_ptrs)
    );
    let d_column_values_ptrs_table = d_column_values_ptrs.to_device_on(&HALO2_GPU_CTX)?;

    let status = unsafe {
        _halo2_quotient_permutation(
            d_values,
            d_l0,
            d_l_last,
            d_l_active_row,
            d_perm_prod_ptrs_table.as_raw_ptr(),
            d_perm_coset_ptrs_table.as_raw_ptr(),
            d_column_values_ptrs_table.as_raw_ptr(),
            h_beta,
            h_gamma,
            h_y,
            h_delta,
            h_delta_start,
            h_current_extended_omega,
            h_omega,
            n_sets as u64,
            chunk_len as u64,
            n_perm_cols as u64,
            last_rotation,
            rot_scale,
            isize_,
            poly_length as u64,
            HALO2_GPU_CTX.stream.as_raw(),
        )
    };

    if status.code != 0 {
        return Err(status.into());
    }
    Ok(())
}

/// Self-contained host-side dispatch of the permutation-quotient kernel.
/// Allocates device buffers for `values` / `l0` / `l_last` / `l_active_row`,
/// uploads, calls `module_quotient_permutation`, downloads `values` back.
///
/// This is the equivalence-test entry point used in `cuda::tests`. The
/// production prover instead reuses already-resident device buffers via
/// `module_quotient_permutation` directly.
#[allow(clippy::too_many_arguments)]
pub fn permutation_quotient_gpu<F: WithSmallOrderMulGroup<3>>(
    values: &mut [F],
    permutation_product_cosets: &[&[F]],
    permutation_cosets: &[&[F]],
    column_values: &[&[F]],
    l0: &[F],
    l_last: &[F],
    l_active_row: &[F],
    beta: F,
    gamma: F,
    y: F,
    delta_start: F,
    current_extended_omega: F,
    omega: F,
    chunk_len: usize,
    last_rotation: i32,
    rot_scale: i32,
    isize_: i32,
) -> Result<(), HaloGpuError> {
    crate::perf_section!("permutation_quotient_gpu");
    let n_sets = permutation_product_cosets.len();
    assert!(n_sets > 0, "permutation_product_cosets must be non-empty");
    let n_perm_cols = column_values.len();
    assert_eq!(permutation_cosets.len(), n_perm_cols);
    // The last set may be partial when n_perm_cols % chunk_len != 0.
    assert!(n_perm_cols <= n_sets * chunk_len);
    assert!(n_perm_cols + chunk_len > n_sets * chunk_len);
    let length = values.len();
    assert_eq!(l0.len(), length);
    assert_eq!(l_last.len(), length);
    assert_eq!(l_active_row.len(), length);
    for s in permutation_product_cosets {
        assert_eq!(s.len(), length);
    }
    for s in permutation_cosets {
        assert_eq!(s.len(), length);
    }
    for s in column_values {
        assert_eq!(s.len(), length);
    }

    crate::perf_h2d!(
        "cuda.permutation_quotient_gpu.values()",
        mem::size_of_val(values)
    );
    let values_device = values.to_device_on(&HALO2_GPU_CTX)?;
    crate::perf_h2d!("cuda.permutation_quotient_gpu.l0", mem::size_of_val(l0));
    let l0_device = l0.to_device_on(&HALO2_GPU_CTX)?;
    crate::perf_h2d!(
        "cuda.permutation_quotient_gpu.l_last",
        mem::size_of_val(l_last)
    );
    let l_last_device = l_last.to_device_on(&HALO2_GPU_CTX)?;
    crate::perf_h2d!(
        "cuda.permutation_quotient_gpu.l_active_row",
        mem::size_of_val(l_active_row)
    );
    let l_active_row_device = l_active_row.to_device_on(&HALO2_GPU_CTX)?;

    // The permutation kernel reads `perm_prod_cosets` and `perm_cosets`
    // via device pointers. The test path's inputs are host slices, so
    // upload them here. (The production path in `evaluate_h` skips this
    // upload: it gets device pointers directly from
    // `_halo2_fft_many_to_device`.)
    crate::perf_h2d!(
        "cuda.permutation_quotient_gpu.perm_prod_cosets",
        permutation_product_cosets.len() * length * mem::size_of::<F>()
    );
    let perm_prod_devs: Vec<DeviceBuffer<F>> = permutation_product_cosets
        .iter()
        .map(|s| s.to_device_on(&HALO2_GPU_CTX))
        .collect::<Result<_, _>>()?;
    crate::perf_h2d!(
        "cuda.permutation_quotient_gpu.perm_cosets",
        permutation_cosets.len() * length * mem::size_of::<F>()
    );
    let perm_coset_devs: Vec<DeviceBuffer<F>> = permutation_cosets
        .iter()
        .map(|s| s.to_device_on(&HALO2_GPU_CTX))
        .collect::<Result<_, _>>()?;

    let perm_prod_ptrs_device: Vec<*const c_void> =
        perm_prod_devs.iter().map(|b| b.as_raw_ptr()).collect();
    let perm_coset_ptrs_device: Vec<*const c_void> =
        perm_coset_devs.iter().map(|b| b.as_raw_ptr()).collect();

    // Upload column_values host slices to per-column device buffers so
    // the kernel can address them via a device pointer array (matching
    // the production path where the columns are already device-resident).
    crate::perf_h2d!(
        "cuda.permutation_quotient_gpu.column_values",
        column_values.len() * length * mem::size_of::<F>()
    );
    let column_value_devs: Vec<DeviceBuffer<F>> = column_values
        .iter()
        .map(|s| s.to_device_on(&HALO2_GPU_CTX))
        .collect::<Result<_, _>>()?;
    let column_value_ptrs_device: Vec<*const c_void> =
        column_value_devs.iter().map(|b| b.as_raw_ptr()).collect();

    let delta = F::DELTA;

    module_quotient_permutation(
        values_device.as_mut_raw_ptr(),
        l0_device.as_raw_ptr(),
        l_last_device.as_raw_ptr(),
        l_active_row_device.as_raw_ptr(),
        &perm_prod_ptrs_device,
        &perm_coset_ptrs_device,
        &column_value_ptrs_device,
        &beta as *const F as *const c_void,
        &gamma as *const F as *const c_void,
        &y as *const F as *const c_void,
        &delta as *const F as *const c_void,
        &delta_start as *const F as *const c_void,
        &current_extended_omega as *const F as *const c_void,
        &omega as *const F as *const c_void,
        n_sets,
        chunk_len,
        last_rotation,
        rot_scale,
        isize_,
        length,
    )?;

    // D2H values back.
    let size_in_bytes = std::mem::size_of_val::<[F]>(values);
    unsafe {
        cuda_memcpy_on::<true, false>(
            values.as_mut_ptr() as *mut libc::c_void,
            values_device.as_raw_ptr(),
            size_in_bytes,
            &HALO2_GPU_CTX,
        )?;
    }
    HALO2_GPU_CTX.stream.to_host_sync().unwrap();

    Ok(())
}
