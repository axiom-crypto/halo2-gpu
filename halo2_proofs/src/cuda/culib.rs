//! `extern "C"` declarations for the halo2-gpu CUDA kernel launchers.
//!
//! Conventions for every launcher below: raw `*c_void` parameters are
//! caller-owned device pointers; `FFITraitObject*` carries host or
//! device storage via an `is_device` discriminant; `scratch` is a
//! device block sized by the launcher's `_workspace_size` sibling.
//! Launchers enqueue on the passed `stream` and never call
//! `cudaStreamSynchronize` or `cudaSetDevice`. Returned `CudaStatus`
//! maps to `HaloGpuError::Cuda` via the `From` impl; workspace/shape
//! queries are pure host fns and signal failure via sentinel values.

use crate::cuda::error::CudaStatus;
use crate::cuda::utils::FFITraitObject;

/// Byte layout of one `Assigned<F>` element passed to
/// [`_halo2_decode_assigned`]: the per-element stride and the byte
/// offsets of the numerator and denominator field payloads. `#[repr(C)]`
/// and the field order pin the FFI ABI to the C `assigned_layout_t`
/// struct.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct AssignedLayout {
    pub stride_bytes: u32,
    pub num_offset: u32,
    pub denom_offset: u32,
}

#[link(name = "halo2_gpu", kind = "static")]
extern "C" {
    pub fn _halo2_msm_max_length(free_bytes: u64) -> usize;

    pub fn _halo2_multiexp_workspace_size(length: u64) -> u64;

    /// Pippenger MSM with host scalars + host bases.
    pub fn _halo2_multiexp(
        scalar: *const FFITraitObject,
        point: *const FFITraitObject,
        output: *const FFITraitObject,
        length: usize,
        scratch: *mut libc::c_void,
        scratch_bytes: u64,
        stream: *mut libc::c_void,
    ) -> CudaStatus;

    pub fn _halo2_multiexp_device_bases_workspace_size(length: u64) -> u64;

    /// Device-bases MSM. Scalars and output stay on host.
    pub fn _halo2_multiexp_device_bases(
        h_scalar: *const FFITraitObject,
        d_bases: *const libc::c_void,
        h_output: *const FFITraitObject,
        length: usize,
        scratch: *mut libc::c_void,
        scratch_bytes: u64,
        stream: *mut libc::c_void,
    ) -> CudaStatus;

    pub fn _halo2_multiexp_device_scalars_device_bases_workspace_size(length: u64) -> u64;

    /// Fully device-resident MSM; only the 96-byte Jacobian result is copied
    /// back to host (for the host-only transcript / `batch_normalize`).
    pub fn _halo2_multiexp_device_scalars_device_bases(
        d_scalar: *const libc::c_void,
        d_bases: *const libc::c_void,
        h_output: *const FFITraitObject,
        length: usize,
        scratch: *mut libc::c_void,
        scratch_bytes: u64,
        stream: *mut libc::c_void,
    ) -> CudaStatus;

    /// In-place Montgomery-batch inversion over `d_data`. `field_type`
    /// selects the prime field encoding.
    pub fn _halo2_batch_invert(
        d_data: *mut libc::c_void,
        field_type: u32,
        length: usize,
        stream: *mut libc::c_void,
    ) -> CudaStatus;

    pub fn _halo2_commit_product_max_len(length: usize, free_bytes: u64) -> usize;
    pub fn _halo2_commit_product_workspace_size(poly_length: u64) -> u64;

    /// Lookup commit_product kernel:
    /// `(permuted_input + β)·(permuted_table + γ) / ((input + β)·(table + γ))`
    /// over `[offset, offset + length)`.
    pub fn _halo2_commit_product(
        lookup_product: *const FFITraitObject,
        permuted_input: *const FFITraitObject,
        permuted_table: *const FFITraitObject,
        compressed_input: *const FFITraitObject,
        compressed_table: *const FFITraitObject,
        beta_device: *const libc::c_void,
        gamma_device: *const libc::c_void,
        length: usize,
        offset: usize,
        scratch: *mut libc::c_void,
        scratch_bytes: u64,
        stream: *mut libc::c_void,
    ) -> CudaStatus;

    /// Device-input + device-output variant of `_halo2_commit_product`.
    /// Every `.ptr` on the five `FFITraitObject` carriers is a device
    /// pointer to a full-length polynomial; the chunk slice is addressed
    /// via pointer arithmetic on `poly_offset * 32` bytes. The three
    /// compute kernels (`cuda_kernel_lookup_denominator`, `batch_invert`,
    /// `cuda_kernel_lookup_numerator`) run unchanged on the caller's
    /// device buffers — no `cudaMemcpyHostToDevice` for any input, no
    /// `cudaMemcpyDeviceToHost` for the output, and no scratch
    /// allocation (caller owns every buffer; no `_workspace_size` sibling
    /// is emitted). Mirrors the no-scratch shape of
    /// `_halo2_grand_product_device_inputs` and the device-pointer
    /// addressing pattern of `_halo2_permutation_product_device_inputs`.
    pub fn _halo2_commit_product_device_inputs(
        d_lookup_product: *const FFITraitObject,
        d_permuted_input: *const FFITraitObject,
        d_permuted_table: *const FFITraitObject,
        d_compressed_input: *const FFITraitObject,
        d_compressed_table: *const FFITraitObject,
        beta_device: *const libc::c_void,
        gamma_device: *const libc::c_void,
        length: usize,
        offset: usize,
        stream: *mut libc::c_void,
    ) -> CudaStatus;

    pub fn _halo2_permute_expression_pair_workspace_size(n: u64, usable_rows: u64) -> u64;

    /// Sorted permutation pair `(permuted_input, permuted_table)` matching the
    /// host `permute_expression_pair_seq` byte-for-byte under BN254 `Fr::Ord`.
    pub fn _halo2_permute_expression_pair(
        d_compressed_input: *const libc::c_void,
        d_compressed_table: *const libc::c_void,
        d_permuted_input: *mut libc::c_void,
        d_permuted_table: *mut libc::c_void,
        n: u64,
        usable_rows: u64,
        scratch: *mut libc::c_void,
        scratch_bytes: u64,
        stream: *mut libc::c_void,
    ) -> CudaStatus;

    pub fn _halo2_permutation_product_max_len(length: usize, free_bytes: u64) -> usize;
    pub fn _halo2_permutation_product_workspace_size(poly_length: u64) -> u64;

    /// Per-chunk permutation grand-product polynomial.
    pub fn _halo2_permutation_product(
        modified_values: *const FFITraitObject,
        permutations: *const FFITraitObject,
        values: *const FFITraitObject,
        beta_device: *const libc::c_void,
        gamma_device: *const libc::c_void,
        delta_device: *const libc::c_void,
        omega_device: *const libc::c_void,
        deltaomega_device: *mut libc::c_void,
        length: usize,
        offset: usize,
        batch: usize,
        scratch: *mut libc::c_void,
        scratch_bytes: u64,
        stream: *mut libc::c_void,
    ) -> CudaStatus;

    pub fn _halo2_permutation_product_device_inputs_workspace_size(poly_length: u64) -> u64;

    /// Device-input + device-output variant of `_halo2_permutation_product`.
    /// Each `permutations_device[i].ptr` and `values_device[i].ptr` is a
    /// device pointer to a full-length column polynomial; the launcher
    /// addresses the chunk slice via pointer arithmetic on
    /// `poly_offset * 32` bytes, eliminating the per-chunk-per-column
    /// `cudaMemcpyHostToDevice` traffic. `modified_values_device.ptr` is a
    /// device pointer that is read in-place as the running accumulator and
    /// written in-place with the chunk's accumulated product — no initial
    /// H2D and no terminal D2H. Workspace shrinks by `3 × poly_size`
    /// relative to `_halo2_permutation_product` (drops `d_denominators`,
    /// `d_permutations`, `d_values` scratch slots).
    ///
    /// Inputs are device-resident: advice + instance + fixed + σ all flow
    /// in as device pointers; the four scalars stay device-pointer per the
    /// existing precedent. Output `modified_values` is device-resident;
    /// no D2H from this FFI.
    pub fn _halo2_permutation_product_device_inputs(
        modified_values_device: *const FFITraitObject,
        permutations_device: *const FFITraitObject,
        values_device: *const FFITraitObject,
        beta_device: *const libc::c_void,
        gamma_device: *const libc::c_void,
        delta_device: *const libc::c_void,
        omega_device: *const libc::c_void,
        deltaomega_device: *mut libc::c_void,
        length: usize,
        offset: usize,
        batch: usize,
        scratch: *mut libc::c_void,
        scratch_bytes: u64,
        stream: *mut libc::c_void,
    ) -> CudaStatus;

    pub fn _halo2_grand_product_max_len(length: usize, free_bytes: u64) -> usize;
    pub fn _halo2_grand_product_workspace_size(poly_length: u64) -> u64;

    /// Prefix-product scan: `output[i] = prefix · ∏_{j=0..=i} input[j]`
    /// over `[offset, offset + length)`.
    pub fn _halo2_grand_product(
        output: *const FFITraitObject,
        input: *const FFITraitObject,
        prefix: *const FFITraitObject,
        length: usize,
        offset: usize,
        scratch: *mut libc::c_void,
        scratch_bytes: u64,
        stream: *mut libc::c_void,
    ) -> CudaStatus;

    /// Device-input variant of `_halo2_grand_product`. `d_inout.ptr` is a
    /// device pointer to the full-length running-product polynomial; the
    /// in-place scan mutates the chunk slice addressed by `offset` (the
    /// caller's buffer is single-use post-scan, so destructive mutation is
    /// safe). `d_prefix.ptr` is a device pointer to the 32-byte running
    /// prefix. This launcher performs no `cudaMemcpy`; the Rust wrapper
    /// copies the scan result back to the caller's host slice after the
    /// chunked loop.
    pub fn _halo2_grand_product_device_inputs(
        d_inout: *const FFITraitObject,
        d_prefix: *const FFITraitObject,
        length: usize,
        offset: usize,
        stream: *mut libc::c_void,
    ) -> CudaStatus;

    pub fn _halo2_eval_polynomial_workspace_size(length: u64) -> u64;

    /// Device-input eval_polynomial. The 32-byte result stays on
    /// device; the kernel returns its device pointer via `d_result_out`.
    pub fn _halo2_eval_polynomial(
        d_poly: *const libc::c_void,
        point: *const FFITraitObject,
        d_result_out: *mut *mut libc::c_void,
        length: usize,
        scratch: *mut libc::c_void,
        scratch_bytes: u64,
        stream: *mut libc::c_void,
    ) -> CudaStatus;

    pub fn _halo2_eval_poly_batch_max_len(
        poly_length: usize,
        batch_size: usize,
        free_bytes: u64,
    ) -> usize;

    pub fn _halo2_eval_polynomial_batch_workspace_size(poly_length: u64, batch_size: u64) -> u64;

    /// Batched Horner eval over `(poly_in_many[i], eval_points[i])` pairs.
    pub fn _halo2_eval_polynomial_batch(
        poly_in_many: *const FFITraitObject,
        eval_points: *const FFITraitObject,
        eval_result: *const FFITraitObject,
        poly_offset: usize,
        poly_length: usize,
        batch_size: usize,
        scratch: *mut libc::c_void,
        scratch_bytes: u64,
        stream: *mut libc::c_void,
    ) -> CudaStatus;

    /// In-place `d_poly_acc[i] += d_scalar * d_poly_in[i]`; all pointers
    /// device-resident.
    pub fn _halo2_poly_multiply_add(
        d_poly_acc: *mut libc::c_void,
        d_poly_in: *const libc::c_void,
        d_scalar: *const libc::c_void,
        poly_length: usize,
        stream: *mut libc::c_void,
    ) -> CudaStatus;

    /// `d_out[i] = d_a[i] * d_b[i]` elementwise; aliasing output with either input is safe.
    pub fn _halo2_poly_elementwise_multiply(
        d_out: *mut libc::c_void,
        d_a: *const libc::c_void,
        d_b: *const libc::c_void,
        length: u64,
        stream: *mut libc::c_void,
    ) -> CudaStatus;

    /// Decode a device-resident `[Assigned<F>]` raw-bytes array into
    /// separate per-element numerator and denominator device buffers.
    /// `d_raw` is the bytes of the host `&[Assigned<F>]` (uploaded via
    /// `to_device_on`); `stride_bytes` / `num_offset` / `denom_offset`
    /// come from `size_of` / `align_of` on the Rust side (the layout is
    /// pinned by `#[repr(C, u8)]` on `Assigned<F>`). For each element
    /// `i`, writes `d_nums[i]` = numerator (Zero→0, Trivial(x)→x,
    /// Rational(n,_)→n) and `d_denoms[i]` = denominator (Zero/Trivial→1,
    /// Rational(_,d)→d). `n == 0` is a no-op.
    pub fn _halo2_decode_assigned(
        d_nums: *mut libc::c_void,
        d_denoms: *mut libc::c_void,
        d_raw: *const libc::c_void,
        n: u64,
        layout: AssignedLayout,
        stream: *mut libc::c_void,
    ) -> CudaStatus;

    /// In-place `d_poly[i] *= d_t_evals[i % t_len]`.
    pub fn _halo2_divide_by_vanishing_poly(
        d_poly: *mut libc::c_void,
        d_t_evals: *const libc::c_void,
        t_len: u32,
        n: u64,
        stream: *mut libc::c_void,
    ) -> CudaStatus;

    pub fn _halo2_kate_division_workspace_size(n: u64) -> u64;

    /// Polynomial division by a linear factor `(X - u)` on the device:
    /// `d_q(X) = (d_a(X) - d_a(u)) / (X - u)`. All pointers device-resident;
    /// `d_a` is length-n input, `d_q` is length-(n-1) output, `d_u` is a
    /// single 32-byte scalar.
    pub fn _halo2_kate_division_device(
        d_a: *const libc::c_void,
        d_q: *mut libc::c_void,
        d_u: *const libc::c_void,
        n: u64,
        scratch: *mut libc::c_void,
        scratch_bytes: u64,
        stream: *mut libc::c_void,
    ) -> CudaStatus;

    /// Padded variant: writes the length-(n-1) quotient + trailing zeros
    /// into a length-`out_len` output buffer (`out_len >= n-1`) in a
    /// single kernel launch (positions [0, n-1) hold the quotient,
    /// [n-1, out_len) hold zeros).
    pub fn _halo2_kate_division_device_padded(
        d_a: *const libc::c_void,
        d_q: *mut libc::c_void,
        d_u: *const libc::c_void,
        n: u64,
        out_len: u64,
        scratch: *mut libc::c_void,
        scratch_bytes: u64,
        stream: *mut libc::c_void,
    ) -> CudaStatus;

    /// `d_acc[i] -= d_short[i]` for i in [0, short_len). All pointers
    /// device-resident. `d_acc` updated in place; only the first
    /// `short_len` elements are touched.
    pub fn _halo2_poly_sub_short_inplace(
        d_acc: *mut libc::c_void,
        d_short: *const libc::c_void,
        short_len: u64,
        stream: *mut libc::c_void,
    ) -> CudaStatus;

    /// `d_out[i] = d_long[i] - d_short[i]` for i in [0, short_len);
    /// `d_out[i] = d_long[i]` for i in [short_len, long_len). Out-of-place
    /// sibling of `_halo2_poly_sub_short_inplace`: produces a fresh
    /// `d_out` without a D2D clone of `d_long`.
    pub fn _halo2_poly_sub_short_out_of_place(
        d_out: *mut libc::c_void,
        d_long: *const libc::c_void,
        d_short: *const libc::c_void,
        short_len: u64,
        long_len: u64,
        stream: *mut libc::c_void,
    ) -> CudaStatus;

    /// `d_buf[0] -= *d_scalar`. Single index-0 subtract on device. Both
    /// pointers device-resident.
    pub fn _halo2_poly_sub_scalar_at_zero(
        d_buf: *mut libc::c_void,
        d_scalar: *const libc::c_void,
        stream: *mut libc::c_void,
    ) -> CudaStatus;

    /// 1D strided gather `d_out[i*num_parts + p] = d_parts[p][i]`;
    /// `d_parts` is a device array of device pointers.
    pub fn _halo2_extended_from_lagrange_vec_device(
        d_out: *mut libc::c_void,
        d_parts: *const libc::c_void,
        num_parts: u32,
        n: u64,
        stream: *mut libc::c_void,
    ) -> CudaStatus;

    /// In-place `d_a[i] *= d_coset_powers[i_mod - 1]` when `i_mod != 0`,
    /// identity at `i_mod == 0`; `i_mod = i % (coset_powers_len + 1)`.
    pub fn _halo2_distribute_powers_zeta(
        d_a: *mut libc::c_void,
        d_coset_powers: *const libc::c_void,
        coset_powers_len: u32,
        n: u64,
        stream: *mut libc::c_void,
    ) -> CudaStatus;

    pub fn _halo2_multiopen_poly_max_len(
        poly_length: usize,
        batch_size: usize,
        free_bytes: u64,
    ) -> usize;

    pub fn _halo2_multiopen_poly_calculation_workspace_size(
        poly_length: u64,
        batch_size: u64,
    ) -> u64;

    /// Per pair `(poly_in_many[i], evaluate_point[i])` compute Horner
    /// eval and fold into the `challenge_point`-indexed RLC `poly_acc`.
    pub fn _halo2_multiopen_poly_calculation(
        poly_in_many: *const FFITraitObject,
        poly_acc: *const FFITraitObject,
        poly_offset: usize,
        poly_length: usize,
        batch_size: usize,
        challenge_point: *const FFITraitObject,
        evaluate_point: *const FFITraitObject,
        evalaute_result: *const FFITraitObject,
        scratch: *mut libc::c_void,
        scratch_bytes: u64,
        stream: *mut libc::c_void,
    ) -> CudaStatus;

    pub fn cuda_omega_lut(res: *const FFITraitObject, log_n: u32) -> CudaStatus;

    pub fn _halo2_power_of_omega_workspace_size(log_n: u32) -> u64;

    pub fn _halo2_power_of_omega(
        res: *const FFITraitObject,
        omega_lut: *const FFITraitObject,
        omega: *const FFITraitObject,
        log_n: u32,
        pow: u32,
        scratch: *mut libc::c_void,
        scratch_bytes: u64,
        stream: *mut libc::c_void,
    ) -> CudaStatus;

    pub fn _halo2_generate_omega_powers_workspace_size(log_n: u32) -> u64;

    /// Device-only omega-powers generator. `d_omega_powers` and `d_omega`
    /// are caller-allocated **device** pointers: the LUT is written to
    /// `d_omega_powers` and `d_omega` holds the seed root of unity. All
    /// H↔D traffic lives Rust-side in `generate_omega_powers_gpu`.
    pub fn _halo2_generate_omega_powers(
        d_omega_powers: *mut libc::c_void,
        d_omega: *const libc::c_void,
        log_n: u32,
        scratch: *mut libc::c_void,
        scratch_bytes: u64,
        stream: *mut libc::c_void,
    ) -> CudaStatus;

    pub fn _halo2_generate_omega_lut_workspace_size(log_n: u32) -> u64;

    pub fn _halo2_generate_omega_lut(
        sparse_twiddle: *const FFITraitObject,
        omega: *const FFITraitObject,
        log_n: u32,
        scratch: *mut libc::c_void,
        scratch_bytes: u64,
        stream: *mut libc::c_void,
    ) -> CudaStatus;

    pub fn _halo2_generate_omegadelta_workspace_size(
        log_n: u32,
        omega_start: u32,
        omega_end: u32,
        colunm_num: u32,
        colunm_offset: u32,
    ) -> u64;

    pub fn _halo2_generate_omegadelta(
        res: *mut libc::c_void,
        mapping: *const libc::c_void,
        omega: *const libc::c_void,
        delta: *const libc::c_void,
        log_n: u32,
        omega_start: u32,
        omega_end: u32,
        colunm_num: u32,
        colunm_offset: u32,
        scratch: *mut libc::c_void,
        scratch_bytes: u64,
        stream: *mut libc::c_void,
    ) -> CudaStatus;

    pub fn _halo2_fft_normal_workspace_size(ntt_type: u32, log_n: u32, extend_log_n: u32) -> u64;

    pub fn _halo2_fft_normal(
        ntt_type: u32,
        log_n: u32,
        input: *const libc::c_void,
        output: *mut libc::c_void,
        omega: *const libc::c_void,
        divisor: *const libc::c_void,
        scratch: *mut libc::c_void,
        scratch_bytes: u64,
        stream: *mut libc::c_void,
    ) -> CudaStatus;

    pub fn _halo2_fft_normal_check_memory(
        ntt_type: u32,
        input: *const libc::c_void,
        log_n: u32,
        extend_log_n: u32,
    ) -> bool; // always true; ABI shim retained for callers

    pub fn _halo2_fft_normal_to_device_workspace_size(
        ntt_type: u32,
        log_n: u32,
        extend_log_n: u32,
    ) -> u64;

    /// Device-input/device-output variant of `_halo2_fft_normal`.
    /// `d_input == d_output` is permitted; the FFT runs in place on
    /// `d_output` after a D2D pre-copy from `d_input`.
    pub fn _halo2_fft_normal_to_device(
        ntt_type: u32,
        log_n: u32,
        d_input: *const libc::c_void,
        d_output: *mut libc::c_void,
        omega_device: *const libc::c_void,
        h_divisor: *const libc::c_void,
        scratch: *mut libc::c_void,
        scratch_bytes: u64,
        stream: *mut libc::c_void,
    ) -> CudaStatus;

    pub fn _halo2_cosetfft(
        ntt_type: u32,
        log_n: u32,
        extend_log_n: u32,
        input: *const libc::c_void,
        output: *mut libc::c_void,
        omega: *const libc::c_void,
        divisor: *const libc::c_void,
        scratch: *mut libc::c_void,
        scratch_bytes: u64,
        stream: *mut libc::c_void,
    ) -> CudaStatus;

    pub fn _halo2_fft_many_workspace_size(ntt_type: u32, log_n: u32, extend_log_n: u32) -> u64;

    pub fn _halo2_fft_many_to_device_workspace_size(
        ntt_type: u32,
        log_n: u32,
        extend_log_n: u32,
    ) -> u64;

    /// Batch FFT entry point. Host-borrowed input/output slices via
    /// FFITraitObject; `omega_device` is a device-resident 32-byte
    /// scalar (caller uploads via `to_device_on(&HALO2_GPU_CTX)`);
    /// `divisor` is a host-borrowed FFITraitObject (the kernel does its own
    /// 32-byte H2D of the divisor scalar).
    pub fn _halo2_fft_many(
        ntt_type: u32,
        num_many: u32,
        log_n: u32,
        extend_log_n: u32,
        input: *const FFITraitObject,
        output: *const FFITraitObject,
        omega_device: *const libc::c_void,
        divisor: *const FFITraitObject,
        scratch: *mut libc::c_void,
        scratch_bytes: u64,
        stream: *mut libc::c_void,
    ) -> CudaStatus;

    /// Device-output variant of `_halo2_fft_many`. Each `output[i].ptr` is a
    /// device pointer; per-poly copy is D2D rather than D2H. `input_on_device`
    /// selects the data-plane copy kind: `false` reads each `input[i].ptr` as
    /// a host pointer, `true` reads it as a device pointer.
    pub fn _halo2_fft_many_to_device(
        ntt_type: u32,
        num_many: u32,
        log_n: u32,
        extend_log_n: u32,
        input: *const FFITraitObject,
        output: *const FFITraitObject,
        omega_device: *const libc::c_void,
        divisor: *const FFITraitObject,
        input_on_device: bool,
        scratch: *mut libc::c_void,
        scratch_bytes: u64,
        stream: *mut libc::c_void,
    ) -> CudaStatus;

    pub fn _halo2_fft_many_check_memory(ntt_type: u32, log_n: u32, extend_log_n: u32) -> bool; // always true

    pub fn _halo2_get_fft_split_radix(
        ntt_type: u32,
        log_n: u32,
        extend_log_n: u32,
        free_bytes: u64,
    ) -> i32;

    pub fn _halo2_quotient_lookups(
        values: *mut libc::c_void,
        table_values: *const libc::c_void,
        product_coset: *const libc::c_void,
        permuted_input_coset: *const libc::c_void,
        permuted_table_coset: *const libc::c_void,
        l0: *const libc::c_void,
        l_last: *const libc::c_void,
        l_active_row: *const libc::c_void,
        beta: *const libc::c_void,
        gamma: *const libc::c_void,
        y: *const libc::c_void,
        poly_length: usize,
        stream: *mut libc::c_void,
    ) -> CudaStatus;

    /// PLONK permutation argument per-row identities, accumulated into
    /// `values`. Buffer pointers are device-resident; scalar pointers
    /// (`beta`/`gamma`/`y`/`delta`/`omega`) are host (kernel async-memcpys).
    #[allow(clippy::too_many_arguments)]
    pub fn _halo2_quotient_permutation(
        // device-borrowed
        values: *mut libc::c_void,
        l0: *const libc::c_void,
        l_last: *const libc::c_void,
        l_active_row: *const libc::c_void,
        // device-borrowed: pointer tables (arrays of device pointers,
        // one entry per polynomial)
        perm_prod_cosets: *const libc::c_void,
        perm_cosets: *const libc::c_void,
        column_values: *const libc::c_void,
        // host-borrowed: single scalars
        beta: *const libc::c_void,
        gamma: *const libc::c_void,
        y: *const libc::c_void,
        delta: *const libc::c_void,
        delta_start: *const libc::c_void,
        current_extended_omega: *const libc::c_void,
        omega: *const libc::c_void,
        // metadata
        n_sets: u64,
        chunk_len: u64,
        n_perm_cols: u64,
        last_rotation: i32,
        rot_scale: i32,
        isize_: i32,
        poly_length: u64,
        stream: *mut libc::c_void,
    ) -> CudaStatus;

}
