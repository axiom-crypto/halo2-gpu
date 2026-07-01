use crate::cuda::culib::{
    _halo2_commit_product, _halo2_commit_product_device_inputs, _halo2_commit_product_max_len,
    _halo2_commit_product_workspace_size, _halo2_permute_expression_pair,
    _halo2_permute_expression_pair_workspace_size, _halo2_quotient_lookups,
};
use crate::cuda::utils::{
    ensure_current_device_matches_ctx, query_device_free_bytes_for_chunking, FFITraitObject,
    HALO2_GPU_CTX,
};
use crate::cuda::HaloGpuError;
use ff::Field;
use openvm_cuda_common::copy::MemCopyH2D;
use openvm_cuda_common::d_buffer::DeviceBuffer;
use std::ffi::c_void;

/// This function is the first step to calculate the lagrange coeff form of Lookup grand product polynomial
/// lookup_product(X) = ((compressed_input(X) + beta)*(compressed_table(X) + gamma))
///                       / ((permuted_input(X) + beta) * (permuted_table(X) + gamma))
/// The ultimate grand product polynomial Z(X) can be calculated as
///    Z(X*g) = lookup_product(X) * Z(X)
///    Z(g) = 1
pub fn lookup_product_gpu<F: Field>(
    lookup_product: &mut [F],
    permuted_input: &[F],
    permuted_table: &[F],
    compressed_input: &[F],
    compressed_table: &[F],
    beta: F,
    gamma: F,
) -> Result<(), HaloGpuError> {
    crate::perf_section!("lookup_product");
    ensure_current_device_matches_ctx()?;
    let poly_len = lookup_product.len();
    assert_eq!(permuted_input.len(), poly_len);
    assert_eq!(permuted_table.len(), poly_len);
    assert_eq!(compressed_input.len(), poly_len);
    assert_eq!(compressed_table.len(), poly_len);

    let max_len = unsafe {
        _halo2_commit_product_max_len(poly_len, query_device_free_bytes_for_chunking() as u64)
    };
    let mut chunk_size = poly_len;
    if poly_len > max_len {
        chunk_size = max_len;
    }

    let lookup_product_obj = FFITraitObject::from_ref(&lookup_product[0]);
    let permuted_input_obj = FFITraitObject::from_ref(&permuted_input[0]);
    let permuted_table_obj = FFITraitObject::from_ref(&permuted_table[0]);
    let compressed_input_obj = FFITraitObject::from_ref(&compressed_input[0]);
    let compressed_table_obj = FFITraitObject::from_ref(&compressed_table[0]);
    let beta_device = std::slice::from_ref(&beta).to_device_on(&HALO2_GPU_CTX)?;
    let gamma_device = std::slice::from_ref(&gamma).to_device_on(&HALO2_GPU_CTX)?;

    let cp_scratch_bytes =
        unsafe { _halo2_commit_product_workspace_size(chunk_size as u64) } as usize;
    let scratch = DeviceBuffer::<u8>::with_capacity_on(cp_scratch_bytes, &HALO2_GPU_CTX);
    for offset in (0..poly_len).step_by(chunk_size) {
        let status = unsafe {
            _halo2_commit_product(
                &lookup_product_obj,
                &permuted_input_obj,
                &permuted_table_obj,
                &compressed_input_obj,
                &compressed_table_obj,
                beta_device.as_raw_ptr(),
                gamma_device.as_raw_ptr(),
                chunk_size,
                offset,
                scratch.as_mut_raw_ptr(),
                cp_scratch_bytes as u64,
                HALO2_GPU_CTX.stream.as_raw(),
            )
        };
        if status.code != 0 {
            return Err(status.into());
        }
    }
    Ok(())
}

/// Device-input + device-output variant of [`lookup_product_gpu`].
///
/// Consumes four caller-owned device buffers (`d_permuted_input`,
/// `d_permuted_table`, `d_compressed_input`, `d_compressed_table`, all of
/// length `n`) and returns a freshly allocated `DeviceBuffer<F>` holding
/// `lookup_product[i] = (compressed_input[i] + β) · (compressed_table[i] + γ)
///                     / ((permuted_input[i] + β) · (permuted_table[i] + γ))`
/// for `i ∈ [0, n)`. No PCIe traffic on the four inputs or the output —
/// the launcher reads/writes only via the caller's device pointers.
///
/// Mirrors `permutation_product_device` in shape: the
/// device-pointer-addressing launcher runs in a single FFI call (no
/// workspace, no per-chunk H2D), so the wrapper bypasses the host
/// variant's `_max_len`-driven chunk loop entirely.
pub fn lookup_product_device<F: Field>(
    d_permuted_input: &DeviceBuffer<F>,
    d_permuted_table: &DeviceBuffer<F>,
    d_compressed_input: &DeviceBuffer<F>,
    d_compressed_table: &DeviceBuffer<F>,
    beta: F,
    gamma: F,
) -> Result<DeviceBuffer<F>, HaloGpuError> {
    crate::perf_section!("lookup_product_device");
    ensure_current_device_matches_ctx()?;
    let poly_len = d_permuted_input.len();
    assert_eq!(d_permuted_table.len(), poly_len);
    assert_eq!(d_compressed_input.len(), poly_len);
    assert_eq!(d_compressed_table.len(), poly_len);

    let d_lookup_product = DeviceBuffer::<F>::with_capacity_on(poly_len, &HALO2_GPU_CTX);
    let beta_device = std::slice::from_ref(&beta).to_device_on(&HALO2_GPU_CTX)?;
    let gamma_device = std::slice::from_ref(&gamma).to_device_on(&HALO2_GPU_CTX)?;

    let lookup_product_obj = FFITraitObject::new(d_lookup_product.as_raw_ptr() as usize);
    let permuted_input_obj = FFITraitObject::new(d_permuted_input.as_raw_ptr() as usize);
    let permuted_table_obj = FFITraitObject::new(d_permuted_table.as_raw_ptr() as usize);
    let compressed_input_obj = FFITraitObject::new(d_compressed_input.as_raw_ptr() as usize);
    let compressed_table_obj = FFITraitObject::new(d_compressed_table.as_raw_ptr() as usize);

    let status = unsafe {
        _halo2_commit_product_device_inputs(
            &lookup_product_obj,
            &permuted_input_obj,
            &permuted_table_obj,
            &compressed_input_obj,
            &compressed_table_obj,
            beta_device.as_raw_ptr(),
            gamma_device.as_raw_ptr(),
            poly_len,
            0,
            HALO2_GPU_CTX.stream.as_raw(),
        )
    };
    if status.code != 0 {
        return Err(status.into());
    }
    Ok(d_lookup_product)
}

/// Computes the numerator part of lookup's quotient polynomial.
///  N(X) = L1(X) + L2(X)*Y + L3(X)*Y^2 + L4(X)*Y^3 + L5(X)*Y^4
/// Lookup identities includes
/// 1. l_active(X)*(A'(X) - S'(X))*(A'(X) - A'(w^(-1)*X)) = 0
/// 2. l0(X)*(A'(X) - S'(X)) = 0
/// 3. l_active(X)*(Z(wX)*(A'(X)+\beta)*(S'(X)+\gamma) - Z(X)*(A(X)+\beta)*(S(X)+\gamma)) = 0
/// 4. l_last(X)*(Z(X)^2 - Z(X)) = 0
/// 5. l0(X)*(1 - Z(X)) = 0
pub(crate) fn module_quotient_lookups(
    values_mem: *mut c_void,                 // N(X)        device
    table_values_mem: *const c_void,         // (A+β)(S+γ)  device
    product_coset_mem: *const c_void,        // Z(X)        device
    permuted_input_coset_mem: *const c_void, // A'(X)       device
    permuted_table_coset_mem: *const c_void, // S'(X)       device
    l0_mem: *const c_void,                   // l0(X)       device
    l_last_mem: *const c_void,               // l_last(X)   device
    l_active_row_mem: *const c_void,         // l_active(X) device
    beta_mem: *const c_void,                 // β           device
    gamma_mem: *const c_void,                // γ           device
    y_mem: *const c_void,                    // y           device
    poly_length: usize,
) -> Result<(), HaloGpuError> {
    crate::perf_section!("quotient_lookups");
    debug_assert!(poly_length > 0, "module_quotient_lookups: poly_length must be > 0");
    debug_assert!(
        !values_mem.is_null()
            && !table_values_mem.is_null()
            && !product_coset_mem.is_null()
            && !permuted_input_coset_mem.is_null()
            && !permuted_table_coset_mem.is_null()
            && !l0_mem.is_null()
            && !l_last_mem.is_null()
            && !l_active_row_mem.is_null()
            && !beta_mem.is_null()
            && !gamma_mem.is_null()
            && !y_mem.is_null(),
        "module_quotient_lookups: all 11 device pointers must be non-null"
    );
    let status = unsafe {
        _halo2_quotient_lookups(
            values_mem,
            table_values_mem,
            product_coset_mem,
            permuted_input_coset_mem,
            permuted_table_coset_mem,
            l0_mem,
            l_last_mem,
            l_active_row_mem,
            beta_mem,
            gamma_mem,
            y_mem,
            poly_length,
            HALO2_GPU_CTX.stream.as_raw(),
        )
    };

    if status.code != 0 {
        return Err(status.into());
    }
    Ok(())
}

/// Device-resident `permute_expression_pair`: produces sorted permutation
/// outputs `(permuted_input, permuted_table)` matching the CPU
/// `permute_expression_pair_seq` byte-for-byte. All inputs and outputs
/// are caller-owned `DeviceBuffer<F>` of length `n`; only `[0,
/// usable_rows)` is read from the inputs and `[usable_rows, n)` is
/// zero-padded in both outputs.
pub fn permute_expression_pair_device<F: Field>(
    d_compressed_input: &DeviceBuffer<F>,
    d_compressed_table: &DeviceBuffer<F>,
    d_permuted_input: &mut DeviceBuffer<F>,
    d_permuted_table: &mut DeviceBuffer<F>,
    n: usize,
    usable_rows: usize,
) -> Result<(), HaloGpuError> {
    crate::perf_section!("permute_expression_pair");
    ensure_current_device_matches_ctx()?;
    assert_eq!(d_compressed_input.len(), n);
    assert_eq!(d_compressed_table.len(), n);
    assert_eq!(d_permuted_input.len(), n);
    assert_eq!(d_permuted_table.len(), n);
    assert!(usable_rows <= n);

    let scratch_bytes =
        unsafe { _halo2_permute_expression_pair_workspace_size(n as u64, usable_rows as u64) }
            as usize;
    let scratch = DeviceBuffer::<u8>::with_capacity_on(scratch_bytes, &HALO2_GPU_CTX);

    let status = unsafe {
        _halo2_permute_expression_pair(
            d_compressed_input.as_raw_ptr(),
            d_compressed_table.as_raw_ptr(),
            d_permuted_input.as_mut_raw_ptr(),
            d_permuted_table.as_mut_raw_ptr(),
            n as u64,
            usable_rows as u64,
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
