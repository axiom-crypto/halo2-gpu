use std::ffi::c_void;
use std::mem;

use crate::cuda::culib::{
    _halo2_fft_many_to_device_workspace_size, _halo2_fft_normal_check_memory,
    _halo2_fft_normal_workspace_size,
};
use crate::cuda::funcs::{
    module_fft_normal, module_quotient_lookups, split_radix_fft_gpu, split_radix_fft_inout_gpu,
};
use crate::cuda::utils::{as_bytes, HALO2_GPU_CTX};
use crate::poly::NttType;
use ff::Field;
pub use halo2curves::{CurveAffine, CurveExt};
use openvm_cuda_common::copy::{cuda_memcpy_on, MemCopyH2D};
use openvm_cuda_common::d_buffer::DeviceBuffer;

use crate::cuda::HaloGpuError;

use super::funcs::module_quotient_permutation;
use super::utils::query_device_free_bytes_for_chunking;

fn cosetfftpart_module<Scalar>(
    data_out: *mut c_void,
    data_in: *const c_void,
    log_n: u32,
    omega: Scalar,
    omega_part: Scalar,
) -> Result<(), HaloGpuError> {
    module_fft_normal(
        data_out,
        data_in,
        &omega as *const Scalar as *const c_void,
        &omega_part as *const Scalar as *const c_void,
        NttType::CosetFFT_Part.into(),
        log_n,
    )
}

fn ifft_cosetfftpart_module<Scalar>(
    data_out: *mut c_void,
    data_in: *const c_void,
    log_n: u32,
    omega_inv: Scalar,
    divisor: Scalar,
    omega: Scalar,
    omega_part: Scalar,
) -> Result<(), HaloGpuError> {
    // iFFT
    module_fft_normal(
        data_out,
        data_in,
        &omega_inv as *const Scalar as *const c_void,
        &divisor as *const Scalar as *const c_void,
        NttType::iFFT.into(),
        log_n,
    )?;
    // CosetFFT_Part
    module_fft_normal(
        data_out,
        data_out as *const c_void,
        &omega as *const Scalar as *const c_void,
        &omega_part as *const Scalar as *const c_void,
        NttType::CosetFFT_Part.into(),
        log_n,
    )
}

pub(crate) fn ifft_cosetfftpart_gpu<F: Field>(
    a: &[F],
    b: &mut [F],
    log_n: u32,
    extend_log_n: u32,
    omega_inv: F,
    divisor: F,
    omega: F,
    omega_part: F,
) -> Result<(), HaloGpuError> {
    crate::perf_section!("ifft_cosetfftpart");
    assert_eq!(a.len(), b.len());
    assert_eq!(log_n, extend_log_n);
    let out_bytes = mem::size_of_val(b);

    let is_memory_enough = unsafe {
        // if CosetFFT_Part memroy is enough, then iFFT memory is enough
        _halo2_fft_normal_check_memory(
            NttType::CosetFFT_Part.into(),
            std::ptr::null(),
            log_n,
            log_n,
        )
    };

    if is_memory_enough {
        // [host]input → [device]input/output (alloc + H2D fused via MemCopyH2D)
        // SAFETY: `F` here is a halo2 field scalar (POD repr).
        crate::perf_h2d!("ifft_cosetfftpart.in", out_bytes);
        let data_io_device = unsafe { as_bytes(a) }.to_device_on(&HALO2_GPU_CTX).unwrap();
        HALO2_GPU_CTX.stream.synchronize().unwrap();

        ifft_cosetfftpart_module(
            data_io_device.as_mut_raw_ptr(),
            data_io_device.as_raw_ptr(),
            log_n,
            omega_inv,
            divisor,
            omega,
            omega_part,
        )?;

        // then copy back to host
        unsafe {
            crate::perf_d2h!("ifft_cosetfftpart.out", out_bytes);
            cuda_memcpy_on::<true, false>(
                b.as_mut_ptr() as *mut libc::c_void,
                data_io_device.as_raw_ptr(),
                out_bytes,
                &HALO2_GPU_CTX,
            )
            .unwrap();
        }
        HALO2_GPU_CTX.stream.synchronize().unwrap();

        // `data_io_device` will automatically dropped
        Ok(())
    } else {
        log::warn!(" ifft_cosetfftpart_gpu: insufficient gpu memory, using split radix fft");
        split_radix_fft_inout_gpu(NttType::iFFT.into(), a, b, log_n, log_n, omega_inv, divisor)?;
        split_radix_fft_gpu(
            NttType::CosetFFT_Part.into(),
            b,
            log_n,
            log_n,
            omega,
            omega_part,
        )
    }
}

// Device-input/device-output sibling of `ifft_cosetfftpart_gpu`. Skips both the
// H2D prologue and the D2H epilogue: the caller already owns
// `d_a: &DeviceBuffer<F>` and `d_b: &mut DeviceBuffer<F>`. If VRAM is too tight
// for the in-device fast path, returns `InsufficientGpuMemory` rather than
// falling back through `split_radix_fft_*` (which take host slices and would
// defeat the device-only contract).
//
// Retained under `#[cfg(test)]` only: the sole remaining caller is the
// equivalence test in `cuda::tests` that exercises the live host-input sibling
// `ifft_cosetfftpart_gpu` against this device-in/device-out twin.
#[cfg(test)]
pub(crate) fn ifft_cosetfftpart_device<F: Field>(
    d_a: &DeviceBuffer<F>,
    d_b: &mut DeviceBuffer<F>,
    log_n: u32,
    extend_log_n: u32,
    omega_inv: F,
    divisor: F,
    omega: F,
    omega_part: F,
) -> Result<(), HaloGpuError> {
    crate::perf_section!("ifft_cosetfftpart_device_input");
    assert_eq!(d_a.len(), d_b.len());
    assert_eq!(log_n, extend_log_n);

    let is_memory_enough = unsafe {
        _halo2_fft_normal_check_memory(
            NttType::CosetFFT_Part.into(),
            std::ptr::null(),
            log_n,
            log_n,
        )
    };
    if !is_memory_enough {
        let n_bytes = (1usize << log_n) * mem::size_of::<F>();
        return Err(HaloGpuError::InsufficientGpuMemory {
            context: "ifft_cosetfftpart_device.vram_tight",
            magnitude: n_bytes as u64,
            free_bytes: 0,
        });
    }

    ifft_cosetfftpart_module(
        d_b.as_mut_raw_ptr(),
        d_a.as_raw_ptr(),
        log_n,
        omega_inv,
        divisor,
        omega,
        omega_part,
    )
}

pub(crate) fn module_poly_to_coset_device<F: Field>(
    values_mem: *const c_void,
    omega: F,
    omega_part: F,
    log_n: u32,
    length: usize,
) -> Result<DeviceBuffer<u8>, HaloGpuError> {
    let out_bytes = length * mem::size_of::<F>();
    let values_device = DeviceBuffer::<u8>::with_capacity_on(out_bytes, &HALO2_GPU_CTX);
    cosetfftpart_module(
        values_device.as_mut_raw_ptr(),
        values_mem,
        log_n,
        omega,
        omega_part,
    )?;
    Ok(values_device)
}

// Device-input sibling of `module_poly_to_coset_device`. Consumes a
// device-resident `DeviceBuffer<F>` input and returns a fresh
// `DeviceBuffer<F>` output (typed, so it can wrap into
// `Polynomial::<F, _, Device>::from_device`).
pub(crate) fn module_poly_to_coset_device_with_device_input<F: Field>(
    d_in: &DeviceBuffer<F>,
    omega: F,
    omega_part: F,
    log_n: u32,
    length: usize,
) -> Result<DeviceBuffer<F>, HaloGpuError> {
    assert_eq!(d_in.len(), length);
    let values_device = DeviceBuffer::<F>::with_capacity_on(length, &HALO2_GPU_CTX);
    cosetfftpart_module(
        values_device.as_mut_raw_ptr(),
        d_in.as_raw_ptr(),
        log_n,
        omega,
        omega_part,
    )?;
    Ok(values_device)
}

// H2D a dense Lagrange slice into a fresh device buffer + iFFT + CosetFFT_Part.
// Dense inputs are already in Montgomery form, so this path runs no
// packed-polynomial unpack step.
pub(crate) fn dense_lagrange_to_coset_device<F: Field>(
    values: &[F],
    omega_inv: F,
    divisor: F,
    omega: F,
    omega_part: F,
    log_n: u32,
) -> Result<DeviceBuffer<u8>, HaloGpuError> {
    assert_eq!(values.len(), 1 << log_n);
    let out_bytes = mem::size_of_val(values);
    let data_io_device = DeviceBuffer::<u8>::with_capacity_on(out_bytes, &HALO2_GPU_CTX);

    // SAFETY: `F` is POD (halo2 field scalar) — byte cast is sound.
    crate::perf_h2d!("dense_lagrange_to_coset.in", out_bytes);
    unsafe {
        cuda_memcpy_on::<false, true>(
            data_io_device.as_mut_raw_ptr(),
            values.as_ptr() as *const libc::c_void,
            out_bytes,
            &HALO2_GPU_CTX,
        )?;
    }

    ifft_cosetfftpart_module(
        data_io_device.as_mut_raw_ptr(),
        data_io_device.as_raw_ptr(),
        log_n,
        omega_inv,
        divisor,
        omega,
        omega_part,
    )?;
    Ok(data_io_device)
}

// Device-input sibling of `dense_lagrange_to_coset_device`. Skips the H2D
// prologue: caller already owns `d_in: &DeviceBuffer<F>` and the kernel
// consumes it directly. Returns a fresh `DeviceBuffer<F>` so the result can
// be wrapped in `Polynomial::<F, _, Device>::from_device` upstream.
pub(crate) fn dense_lagrange_to_coset_device_with_device_input<F: Field>(
    d_in: &DeviceBuffer<F>,
    omega_inv: F,
    divisor: F,
    omega: F,
    omega_part: F,
    log_n: u32,
) -> Result<DeviceBuffer<F>, HaloGpuError> {
    let n = 1usize << log_n;
    assert_eq!(d_in.len(), n);
    let data_out = DeviceBuffer::<F>::with_capacity_on(n, &HALO2_GPU_CTX);

    ifft_cosetfftpart_module(
        data_out.as_mut_raw_ptr(),
        d_in.as_raw_ptr(),
        log_n,
        omega_inv,
        divisor,
        omega,
        omega_part,
    )?;
    Ok(data_out)
}

/// Lagrange selector polynomial storage for [`QuotientLookupsGpu`].
///
/// The struct's three Lagrange selectors (`l0`, `l_last`, `l_active_row`)
/// may be either struct-owned (host-input [`QuotientLookupsGpu::new`] path —
/// the bytes are copied to device during constructor) or borrowed
/// (`new_with_device_selectors` path — the selectors live in caller-owned
/// device buffers that outlive `self`).
///
/// `Borrowed` carries a raw `*const c_void` whose lifetime must be enforced
/// by the caller; see [`QuotientLookupsGpu::new_with_device_selectors`] for
/// the lifetime contract.
#[derive(Debug)]
enum SelectorRef<F> {
    Owned(DeviceBuffer<F>),
    Borrowed(*const c_void),
}

impl<F> SelectorRef<F> {
    fn as_raw_ptr(&self) -> *const c_void {
        match self {
            Self::Owned(b) => b.as_raw_ptr(),
            Self::Borrowed(p) => *p,
        }
    }
}

/// Lookup-quotient GPU dispatch state.
///
/// Load-once values/selectors/scalars stay device-resident across lookup
/// calls. Per-call dense host inputs are uploaded by `calculate_constraints`;
/// the C++ lookup FFI expects device pointers only.
#[derive(Debug)]
pub struct QuotientLookupsGpu<F> {
    pub values_device: DeviceBuffer<F>,
    l0_selector: SelectorRef<F>,
    l_last_selector: SelectorRef<F>,
    l_active_row_selector: SelectorRef<F>,
    pub beta_device: DeviceBuffer<F>,
    pub gamma_device: DeviceBuffer<F>,
    pub y_device: DeviceBuffer<F>,
    // Reusable per-call scratch for `table_values` H2D.
    pub table_values_device: DeviceBuffer<F>,
    // Host-side scalars retained for `add_permutation_constraints`, whose
    // FFI takes host pointers for these.
    beta: F,
    gamma: F,
    y: F,
    log_n: u32,
    omega_inv: F,
    divisor: F,
    omega: F,
    length: usize,
}

impl<F: Field> QuotientLookupsGpu<F> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        // load once
        values: &[F],
        l0: &[F],
        l_last: &[F],
        l_active_row: &[F],
        // params
        beta: F,
        gamma: F,
        y: F,
        log_n: u32,
        omega_inv: F,
        divisor: F,
        omega: F,
        length: usize,
    ) -> Self {
        crate::perf_section!("quotient_lookups_gpu.new");
        crate::perf_h2d!(
            "quotient_lookups_gpu.new.values",
            std::mem::size_of_val(values)
        );
        let values_device = values.to_device_on(&HALO2_GPU_CTX).unwrap();
        crate::perf_h2d!("quotient_lookups_gpu.new.l0", std::mem::size_of_val(l0));
        let l0_device = l0.to_device_on(&HALO2_GPU_CTX).unwrap();
        crate::perf_h2d!(
            "quotient_lookups_gpu.new.l_last",
            std::mem::size_of_val(l_last)
        );
        let l_last_device = l_last.to_device_on(&HALO2_GPU_CTX).unwrap();
        crate::perf_h2d!(
            "quotient_lookups_gpu.new.l_active_row",
            std::mem::size_of_val(l_active_row)
        );
        let l_active_row_device = l_active_row.to_device_on(&HALO2_GPU_CTX).unwrap();

        // beta/gamma/y don't change across calls — upload once.
        crate::perf_h2d!("quotient_lookups_gpu.new.beta", std::mem::size_of::<F>());
        let beta_device = std::slice::from_ref(&beta)
            .to_device_on(&HALO2_GPU_CTX)
            .unwrap();
        crate::perf_h2d!("quotient_lookups_gpu.new.gamma", std::mem::size_of::<F>());
        let gamma_device = std::slice::from_ref(&gamma)
            .to_device_on(&HALO2_GPU_CTX)
            .unwrap();
        crate::perf_h2d!("quotient_lookups_gpu.new.y", std::mem::size_of::<F>());
        let y_device = std::slice::from_ref(&y)
            .to_device_on(&HALO2_GPU_CTX)
            .unwrap();

        let table_values_device = DeviceBuffer::<F>::with_capacity_on(length, &HALO2_GPU_CTX);

        Self {
            values_device,
            l0_selector: SelectorRef::Owned(l0_device),
            l_last_selector: SelectorRef::Owned(l_last_device),
            l_active_row_selector: SelectorRef::Owned(l_active_row_device),
            beta_device,
            gamma_device,
            y_device,
            table_values_device,
            beta,
            gamma,
            y,
            log_n,
            omega_inv,
            divisor,
            omega,
            length,
        }
    }

    /// Device-input sibling of [`Self::new`]: the partial-quotient
    /// `values_device` and the three Lagrange selector polys (`l0`,
    /// `l_last`, `l_active_row`) arrive as device-resident
    /// `DeviceBuffer<F>`. `values_device` is taken by value (the struct
    /// owns the accumulator); the three selectors are borrowed by raw
    /// pointer — no per-selector D2D copy occurs at construction.
    ///
    /// # Lifetime contract for borrowed selectors
    ///
    /// `l0_device_in`, `l_last_device_in`, and `l_active_row_device_in`
    /// MUST outlive the returned `QuotientLookupsGpu`. The struct holds
    /// raw `*const c_void` pointers into the caller-owned device
    /// buffers; if the caller drops or reallocates those buffers before
    /// the struct, lookup/permutation kernel dispatches will read
    /// freed memory.
    ///
    /// The intended caller is `evaluate_h` (`plonk/evaluation.rs`): the
    /// three selector polys come from `parts[l0_part_idx]` /
    /// `[l_last_part_idx]` / `[l_active_part_idx]` of the per-part
    /// `coeff_to_extended_part_many_device` output, which is allocated
    /// once per part and lives for the entire per-part `for phase_idx`
    /// loop body — i.e. strictly outlives every `QuotientLookupsGpu`
    /// constructed inside that body.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_device_selectors(
        values_device: DeviceBuffer<F>,
        l0_device_in: &DeviceBuffer<F>,
        l_last_device_in: &DeviceBuffer<F>,
        l_active_row_device_in: &DeviceBuffer<F>,
        beta: F,
        gamma: F,
        y: F,
        log_n: u32,
        omega_inv: F,
        divisor: F,
        omega: F,
        length: usize,
    ) -> Result<Self, HaloGpuError> {
        crate::perf_section!("quotient_lookups_gpu.new_with_device_selectors");
        debug_assert_eq!(values_device.len(), length);
        debug_assert_eq!(l0_device_in.len(), length);
        debug_assert_eq!(l_last_device_in.len(), length);
        debug_assert_eq!(l_active_row_device_in.len(), length);

        let l0_selector = SelectorRef::Borrowed(l0_device_in.as_raw_ptr());
        let l_last_selector = SelectorRef::Borrowed(l_last_device_in.as_raw_ptr());
        let l_active_row_selector = SelectorRef::Borrowed(l_active_row_device_in.as_raw_ptr());

        // beta/gamma/y don't change across calls — upload once.
        crate::perf_h2d!(
            "cuda.quotient_lookups_gpu.new_with_device_selectors.beta",
            std::mem::size_of::<F>()
        );
        let beta_device = std::slice::from_ref(&beta).to_device_on(&HALO2_GPU_CTX)?;
        crate::perf_h2d!(
            "cuda.quotient_lookups_gpu.new_with_device_selectors.gamma",
            std::mem::size_of::<F>()
        );
        let gamma_device = std::slice::from_ref(&gamma).to_device_on(&HALO2_GPU_CTX)?;
        crate::perf_h2d!(
            "cuda.quotient_lookups_gpu.new_with_device_selectors.y",
            std::mem::size_of::<F>()
        );
        let y_device = std::slice::from_ref(&y).to_device_on(&HALO2_GPU_CTX)?;

        let table_values_device = DeviceBuffer::<F>::with_capacity_on(length, &HALO2_GPU_CTX);

        Ok(Self {
            values_device,
            l0_selector,
            l_last_selector,
            l_active_row_selector,
            beta_device,
            gamma_device,
            y_device,
            table_values_device,
            beta,
            gamma,
            y,
            log_n,
            omega_inv,
            divisor,
            omega,
            length,
        })
    }

    pub fn is_gpu_memory_enough(log_n: u32, n_perm_sets: usize, n_perm_cols: usize) -> bool {
        // The `ColumnPool` (~2–15 GiB transient depending on column counts)
        // is scoped to the lookups-iter closure in
        // `plonk/prover.rs::create_proof`'s witness phase; it is dropped
        // before `QuotientLookupsGpu::new` runs in the evaluation phase.
        // So `free_gpu_mem` here reflects the post-pool-drop state — no
        // co-resident accounting is needed at this call site.
        //
        // If a future change moves the pool to `OnceCell` or extends its
        // lifetime across phases, this check MUST add
        // `ColumnPool::estimate_resident_bytes(...)` to `required_mem`
        // below.
        let free_gpu_mem = query_device_free_bytes_for_chunking();
        let poly_mem_size = (1usize << log_n) * mem::size_of::<F>();

        // Peak lookup alloc:
        //   resident: values / l0 / l_last / l_active_row /
        //             table_values_device                      (5 polys)
        //   transient: permuted_input / permuted_table /
        //              product_coset device buffers            (3 polys)
        //   FFT scratch (`_halo2_fft_normal_workspace_size`)
        //   beta/gamma/y device scalars                        (3 F)
        // The FFT scratch is transient — `dense_lagrange_to_coset_device`
        // calls run sequentially on the single stream, so only one
        // workspace is alive at any moment (it doesn't stack with the
        // next coset allocation).
        // Queried precisely via `_halo2_fft_normal_workspace_size` rather
        // than a fixed conservative slot — a tight `free_gpu_mem` would
        // otherwise turn into a silent OOM panic in `DeviceBuffer::with_capacity_on`
        // (`expect("GPU allocation failed")`).
        let lookup_resident_bytes = 8 * poly_mem_size;
        let fft_scratch_bytes = unsafe {
            _halo2_fft_normal_workspace_size(
                crate::poly::NttType::CosetFFT_Part.into(),
                log_n,
                log_n,
            )
        } as usize;
        let scalar_residents = 3 * mem::size_of::<F>();

        let permutation_extra = if n_perm_sets == 0 {
            0
        } else {
            let fft_output_bytes = (n_perm_sets + n_perm_cols) * poly_mem_size;
            let fft_workspace_bytes = unsafe {
                _halo2_fft_many_to_device_workspace_size(
                    crate::poly::NttType::CosetFFT_Part.into(),
                    log_n,
                    log_n,
                )
            } as usize;
            let ptr_table_bytes =
                (n_perm_sets + n_perm_cols) * mem::size_of::<*const libc::c_void>();
            let column_values_bytes = n_perm_cols * poly_mem_size;

            fft_output_bytes + fft_workspace_bytes + ptr_table_bytes + column_values_bytes
        };
        let required_mem =
            lookup_resident_bytes + fft_scratch_bytes + scalar_residents + permutation_extra;
        if free_gpu_mem < required_mem {
            log::info!(
                "QuotientLookupsGpu: insufficient gpu memory, free: {}, required: {}, will fall back to normal calculation",
                free_gpu_mem, required_mem
            );
            return false;
        }
        true
    }

    /// Dense-input lookup quotient: uploads the per-call inputs, runs
    /// iFFT + CosetFFT_Part on the permuted polys and CosetFFT_Part on
    /// `product_coset`, then dispatches `_halo2_quotient_lookups`.
    pub fn calculate_constraints(
        &mut self,
        table_values: &[F],
        product_coset: &[F],
        permuted_input_lagrange: &[F],
        permuted_table_lagrange: &[F],
        omega_part: F,
    ) -> Result<(), HaloGpuError> {
        crate::perf_section!("quotient_lookups_gpu.calculate_constraints");
        assert_eq!(self.length, table_values.len());
        assert_eq!(self.length, product_coset.len());
        assert_eq!(self.length, permuted_input_lagrange.len());
        assert_eq!(self.length, permuted_table_lagrange.len());

        let permuted_input_coset = dense_lagrange_to_coset_device::<F>(
            permuted_input_lagrange,
            self.omega_inv,
            self.divisor,
            self.omega,
            omega_part,
            self.log_n,
        )?;
        let permuted_table_coset = dense_lagrange_to_coset_device::<F>(
            permuted_table_lagrange,
            self.omega_inv,
            self.divisor,
            self.omega,
            omega_part,
            self.log_n,
        )?;

        // `product_coset` arrives in LagrangeCoeff form per the prover's
        // contract — runs CosetFFT_Part only (no iFFT step).
        let product_coset_mem = module_poly_to_coset_device::<F>(
            product_coset.as_ptr() as *const c_void,
            self.omega,
            omega_part,
            self.log_n,
            self.length,
        )?;

        // Upload `table_values` into the reusable struct-owned scratch so
        // the FFI sees a device pointer.
        let table_values_bytes = std::mem::size_of_val(table_values);
        crate::perf_h2d!(
            "quotient_lookups_gpu.calculate_constraints.table_values",
            table_values_bytes
        );
        unsafe {
            cuda_memcpy_on::<false, true>(
                self.table_values_device.as_mut_raw_ptr(),
                table_values.as_ptr() as *const libc::c_void,
                table_values_bytes,
                &HALO2_GPU_CTX,
            )?;
        }

        module_quotient_lookups(
            self.values_device.as_mut_raw_ptr(),
            self.table_values_device.as_raw_ptr(),
            product_coset_mem.as_raw_ptr(),
            permuted_input_coset.as_raw_ptr(),
            permuted_table_coset.as_raw_ptr(),
            self.l0_selector.as_raw_ptr(),
            self.l_last_selector.as_raw_ptr(),
            self.l_active_row_selector.as_raw_ptr(),
            self.beta_device.as_raw_ptr(),
            self.gamma_device.as_raw_ptr(),
            self.y_device.as_raw_ptr(),
            self.length,
        )
    }

    /// Sibling of `calculate_constraints` that consumes a
    /// caller-provided device-resident `table_values` buffer instead of
    /// a host slice. The kernel reads `d_table_values` directly, so no
    /// per-call H2D of the table values happens here.
    ///
    /// `d_table_values` must have length `self.length` and be
    /// device-resident on `HALO2_GPU_CTX`. The struct's
    /// `table_values_device` scratch is left untouched and may be
    /// repurposed by future callers without contention.
    ///
    /// Retained as the partial-device-input anchor for the
    /// equivalence test
    /// `test_calculate_constraints_full_device_vs_calculate_constraints_device`
    /// in `cuda::tests`, which pairs it against the fully-device-input
    /// sibling [`Self::calculate_constraints_full_device`] used on
    /// production prover paths.
    pub fn calculate_constraints_device(
        &mut self,
        d_table_values: &DeviceBuffer<F>,
        product_coset: &[F],
        permuted_input_lagrange: &[F],
        permuted_table_lagrange: &[F],
        omega_part: F,
    ) -> Result<(), HaloGpuError> {
        crate::perf_section!("quotient_lookups_gpu.calculate_constraints_device");
        assert_eq!(self.length, d_table_values.len());
        assert_eq!(self.length, product_coset.len());
        assert_eq!(self.length, permuted_input_lagrange.len());
        assert_eq!(self.length, permuted_table_lagrange.len());

        let permuted_input_coset = dense_lagrange_to_coset_device::<F>(
            permuted_input_lagrange,
            self.omega_inv,
            self.divisor,
            self.omega,
            omega_part,
            self.log_n,
        )?;
        let permuted_table_coset = dense_lagrange_to_coset_device::<F>(
            permuted_table_lagrange,
            self.omega_inv,
            self.divisor,
            self.omega,
            omega_part,
            self.log_n,
        )?;

        let product_coset_mem = module_poly_to_coset_device::<F>(
            product_coset.as_ptr() as *const c_void,
            self.omega,
            omega_part,
            self.log_n,
            self.length,
        )?;

        // Skip the H2D of `table_values`: caller owns a Device buffer.
        module_quotient_lookups(
            self.values_device.as_mut_raw_ptr(),
            d_table_values.as_raw_ptr(),
            product_coset_mem.as_raw_ptr(),
            permuted_input_coset.as_raw_ptr(),
            permuted_table_coset.as_raw_ptr(),
            self.l0_selector.as_raw_ptr(),
            self.l_last_selector.as_raw_ptr(),
            self.l_active_row_selector.as_raw_ptr(),
            self.beta_device.as_raw_ptr(),
            self.gamma_device.as_raw_ptr(),
            self.y_device.as_raw_ptr(),
            self.length,
        )
    }

    /// Fully device-input sibling of `calculate_constraints_device`.
    /// Consumes the lookup `product_poly`, `permuted_input`, and
    /// `permuted_table` as device-resident `DeviceBuffer<F>` instead of
    /// host slices, so no per-call H2D of those three buffers fires here.
    ///
    /// The internal CosetFFT_Part helpers are swapped for their
    /// `_device_input` siblings
    /// ([`dense_lagrange_to_coset_device_with_device_input`] and
    /// [`module_poly_to_coset_device_with_device_input`]), so neither
    /// `cuda.dense_lagrange_to_coset.in` nor the `module_poly_to_coset`
    /// host-pointer H2D fires from this path.
    ///
    /// All four `d_*` buffers must have length `self.length` and live on
    /// `HALO2_GPU_CTX`. `g_coset_part` is the coset coordinate for this
    /// extended-domain part (i.e. `domain.g_coset *
    /// current_extended_omega`), matching the `omega_part` argument of
    /// `calculate_constraints_device`.
    pub fn calculate_constraints_full_device(
        &mut self,
        d_table_values: &DeviceBuffer<F>,
        d_product_poly: &DeviceBuffer<F>,
        d_permuted_input: &DeviceBuffer<F>,
        d_permuted_table: &DeviceBuffer<F>,
        g_coset_part: F,
    ) -> Result<(), HaloGpuError> {
        crate::perf_section!("quotient_lookups_gpu.calculate_constraints_full_device");
        assert_eq!(self.length, d_table_values.len());
        assert_eq!(self.length, d_product_poly.len());
        assert_eq!(self.length, d_permuted_input.len());
        assert_eq!(self.length, d_permuted_table.len());

        let permuted_input_coset = dense_lagrange_to_coset_device_with_device_input::<F>(
            d_permuted_input,
            self.omega_inv,
            self.divisor,
            self.omega,
            g_coset_part,
            self.log_n,
        )?;
        let permuted_table_coset = dense_lagrange_to_coset_device_with_device_input::<F>(
            d_permuted_table,
            self.omega_inv,
            self.divisor,
            self.omega,
            g_coset_part,
            self.log_n,
        )?;

        let product_coset_mem = module_poly_to_coset_device_with_device_input::<F>(
            d_product_poly,
            self.omega,
            g_coset_part,
            self.log_n,
            self.length,
        )?;

        module_quotient_lookups(
            self.values_device.as_mut_raw_ptr(),
            d_table_values.as_raw_ptr(),
            product_coset_mem.as_raw_ptr(),
            permuted_input_coset.as_raw_ptr(),
            permuted_table_coset.as_raw_ptr(),
            self.l0_selector.as_raw_ptr(),
            self.l_last_selector.as_raw_ptr(),
            self.l_active_row_selector.as_raw_ptr(),
            self.beta_device.as_raw_ptr(),
            self.gamma_device.as_raw_ptr(),
            self.y_device.as_raw_ptr(),
            self.length,
        )
    }

    /// Apply the permutation-argument quotient contribution to the
    /// already-resident `values_device`. Reuses the in-struct
    /// `beta`/`gamma`/`y`/`omega` plus the on-device `values`/`l0`/
    /// `l_last`/`l_active_row` buffers, so the only per-call H2D
    /// traffic is the three pointer-table uploads
    /// (`permutation_product_cosets_device`, `permutation_cosets_device`,
    /// `column_values_device`) — each `n * sizeof(ptr)` bytes.
    ///
    /// Inputs:
    ///   - `permutation_product_cosets_device`,
    ///     `permutation_cosets_device`, and `column_values_device` are
    ///     slices of *device pointers* (one entry per polynomial).
    ///     Pointers reference per-part device buffers produced upstream
    ///     by `coeff_to_extended_part_many_device` and consumed without
    ///     a per-poly H→D copy of the underlying data.
    ///     Lengths: `n_sets`, `n_perm_cols`, `n_perm_cols`.
    ///
    /// Sets always cover all columns
    /// (`n_perm_cols ≤ n_sets * chunk_len`), but the last set may be
    /// partial when `n_perm_cols % chunk_len != 0`.
    #[allow(clippy::too_many_arguments)]
    pub fn add_permutation_constraints(
        &mut self,
        permutation_product_cosets_device: &[*const c_void],
        permutation_cosets_device: &[*const c_void],
        column_values_device: &[*const c_void],
        last_rotation: i32,
        rot_scale: i32,
        isize_: i32,
        chunk_len: usize,
        delta: F,
        delta_start: F,
        current_extended_omega: F,
    ) -> Result<(), HaloGpuError> {
        crate::perf_section!("quotient_lookups_gpu.add_permutation_constraints");
        let n_sets = permutation_product_cosets_device.len();
        assert!(n_sets > 0);
        let n_perm_cols = column_values_device.len();
        assert_eq!(permutation_cosets_device.len(), n_perm_cols);
        // Last set may be partial; total columns must cover all sets without
        // exceeding `n_sets * chunk_len`.
        assert!(n_perm_cols <= n_sets * chunk_len);
        assert!(n_perm_cols + chunk_len > n_sets * chunk_len);

        module_quotient_permutation(
            self.values_device.as_mut_raw_ptr(),
            self.l0_selector.as_raw_ptr(),
            self.l_last_selector.as_raw_ptr(),
            self.l_active_row_selector.as_raw_ptr(),
            permutation_product_cosets_device,
            permutation_cosets_device,
            column_values_device,
            &self.beta as *const F as *const c_void,
            &self.gamma as *const F as *const c_void,
            &self.y as *const F as *const c_void,
            &delta as *const F as *const c_void,
            &delta_start as *const F as *const c_void,
            &current_extended_omega as *const F as *const c_void,
            &self.omega as *const F as *const c_void,
            n_sets,
            chunk_len,
            last_rotation,
            rot_scale,
            isize_,
            self.length,
        )
    }

    pub fn copy_values_back_to_host(&self, values: &mut [F]) {
        crate::perf_section!("quotient_lookups_gpu.copy_values_back_to_host");
        assert_eq!(self.length, values.len());
        let size_in_bytes = self.length * mem::size_of::<F>();
        unsafe {
            crate::perf_d2h!(
                "quotient_lookups_gpu.copy_values_back_to_host.values_back",
                size_in_bytes
            );
            cuda_memcpy_on::<true, false>(
                values.as_mut_ptr() as *mut libc::c_void,
                self.values_device.as_raw_ptr(),
                size_in_bytes,
                &HALO2_GPU_CTX,
            )
            .unwrap();
        }
        HALO2_GPU_CTX.stream.to_host_sync().unwrap();
    }

    /// Takes ownership of the accumulated values buffer.
    ///
    /// Other resident buffers are dropped with `self`. The returned buffer has
    /// length `self.length`.
    pub fn take_values_device(self) -> DeviceBuffer<F> {
        crate::perf_section!("quotient_lookups_gpu.take_values_device");
        self.values_device
    }
}
