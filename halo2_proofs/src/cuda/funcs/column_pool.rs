use crate::cuda::utils::{query_device_free_bytes_for_chunking, to_device_on_pinned};
use crate::cuda::HaloGpuError;
use crate::poly::DevicePolyExt;
use ff::Field;
use openvm_cuda_common::d_buffer::DeviceBuffer;
use std::ffi::c_void;
use std::mem;

// Per-prove device pool for the `fixed_values` / `advice_values` /
// `instance_values` host-resident column inputs that the device-side
// `compress_expressions_device` and the `_halo2_quotient_device_columns`
// FFI consume.
//
// Lifecycle: lazy-uploaded on first `try_init`; freed on `Drop`. Witness
// columns (advice/instance) vary per prove, so this is not a `OnceCell`
// cache. VRAM is gated all-at-once via `is_gpu_memory_enough`; if the
// gate fails the caller falls back to the CPU `compress_expressions`
// closure.
#[derive(Debug)]
pub struct ColumnPool<F: Field> {
    n: usize,
    fixed_d: Vec<DeviceBuffer<F>>,
    advice_d: Vec<DeviceBuffer<F>>,
    instance_d: Vec<DeviceBuffer<F>>,
    fixed_ptrs_device: Vec<*const c_void>,
    advice_ptrs_device: Vec<*const c_void>,
    instance_ptrs_device: Vec<*const c_void>,
    initialized: bool,
}

impl<F: Field> ColumnPool<F> {
    pub fn new(n: usize) -> Self {
        Self {
            n,
            fixed_d: Vec::new(),
            advice_d: Vec::new(),
            instance_d: Vec::new(),
            fixed_ptrs_device: Vec::new(),
            advice_ptrs_device: Vec::new(),
            instance_ptrs_device: Vec::new(),
            initialized: false,
        }
    }

    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// All-at-once VRAM check. Returns true iff the total device bytes for
    /// fixed/advice/instance columns fits within the chunking-headroom
    /// budget reported by `query_device_free_bytes_for_chunking`. Mirrors
    /// the gating shape used by `QuotientLookupsGpu::is_gpu_memory_enough`.
    pub fn is_gpu_memory_enough(&self, n_fixed: usize, n_advice: usize, n_instance: usize) -> bool {
        let total = Self::estimate_resident_bytes(self.n, n_fixed, n_advice, n_instance);
        let free = query_device_free_bytes_for_chunking() as u64;
        total < free
    }

    /// Static estimator. Used by `QuotientLookupsGpu::is_gpu_memory_enough`
    /// when the lookup engine's check needs to consider a co-resident
    /// pool (defensive accounting; in practice the pool is dropped
    /// before `QuotientLookupsGpu::new`, so the co-residency budget is
    /// zero at the call sites in this crate).
    pub fn estimate_resident_bytes(
        n: usize,
        n_fixed: usize,
        n_advice: usize,
        n_instance: usize,
    ) -> u64 {
        let per_column_bytes = n.saturating_mul(mem::size_of::<F>()) as u64;
        per_column_bytes.saturating_mul((n_fixed + n_advice + n_instance) as u64)
    }

    /// Upload all columns and build the device-pointer arrays. Idempotent:
    /// returns Ok(()) immediately if already initialized. Returns Err on
    /// insufficient VRAM or H2D failure — caller must fall back to host
    /// arm.
    pub fn try_init(
        &mut self,
        fixed_values: &[&[F]],
        advice_values: &[&[F]],
        instance_values: &[&[F]],
    ) -> Result<(), HaloGpuError> {
        self.try_init_inner::<crate::poly::LagrangeCoeff>(
            None,
            fixed_values,
            advice_values,
            instance_values,
        )
    }

    /// Pointer-mirror init that additionally borrows device pointers for
    /// the instance columns. Combined with the borrowed PK fixed Lagrange
    /// mirror and borrowed advice device polynomials, the pool performs
    /// no H2D for any of advice / instance / fixed (when the PK mirror
    /// is populated) — only the PK-mirror-miss path falls through to
    /// uploading fixed columns.
    ///
    /// Safety contract: the caller MUST keep `advice_values_device`,
    /// `instance_values_device`, and any borrowed `pk_fixed_lagrange_mirror`
    /// alive for at least as long as this pool's `Drop`. In the
    /// `create_proof` flow this holds because the per-prove advice and
    /// instance device-poly vectors outlive the per-prove `ColumnPool`.
    pub fn try_init_device<B1, B2, B3>(
        &mut self,
        pk_fixed_lagrange_mirror: Option<&[crate::poly::Polynomial<F, B1, crate::poly::Device>]>,
        fixed_values: &[&[F]],
        advice_values_device: &[crate::poly::Polynomial<F, B2, crate::poly::Device>],
        instance_values_device: &[crate::poly::Polynomial<F, B3, crate::poly::Device>],
    ) -> Result<(), HaloGpuError> {
        if self.initialized {
            return Ok(());
        }
        // Borrowed iff a non-empty device mirror is present and either no
        // host slice is supplied (device-only path) or the host slice
        // matches the mirror's length (cross-check for callers that still
        // pass both). This lets evaluate_h_device pass `&[]` for the host
        // slice when the static fixed columns are already device-resident.
        let fixed_borrowed = match pk_fixed_lagrange_mirror {
            Some(m) => !m.is_empty() && (fixed_values.is_empty() || m.len() == fixed_values.len()),
            None => false,
        };
        let n_fixed_for_gate = if fixed_borrowed {
            0
        } else {
            fixed_values.len()
        };
        if !self.is_gpu_memory_enough(n_fixed_for_gate, 0, 0) {
            return Err(HaloGpuError::InsufficientGpuMemory {
                context: "ColumnPool::try_init_device",
                magnitude: n_fixed_for_gate as u64,
                free_bytes: query_device_free_bytes_for_chunking() as u64,
            });
        }
        crate::perf_section!("column_pool.upload");
        let per_column_bytes = self.n.saturating_mul(mem::size_of::<F>()) as u64;
        let total_bytes: u64 = per_column_bytes.saturating_mul(n_fixed_for_gate as u64);
        crate::perf_h2d!("cuda.column_pool.upload", total_bytes);

        if fixed_borrowed {
            let mirror = pk_fixed_lagrange_mirror.unwrap();
            for (idx, mirror_poly) in mirror.iter().enumerate() {
                debug_assert_eq!(mirror_poly.len(), self.n);
                if let Some(host_col) = fixed_values.get(idx) {
                    debug_assert_eq!(host_col.len(), self.n);
                }
                let d = mirror_poly.device_buf();
                self.fixed_ptrs_device.push(d.as_raw_ptr());
            }
        } else {
            for col in fixed_values {
                debug_assert_eq!(col.len(), self.n);
                let d = to_device_on_pinned(col).map_err(HaloGpuError::from)?;
                self.fixed_ptrs_device.push(d.as_raw_ptr());
                self.fixed_d.push(d);
            }
        }
        for poly in advice_values_device {
            debug_assert_eq!(poly.len(), self.n);
            self.advice_ptrs_device.push(poly.device_buf().as_raw_ptr());
        }
        for poly in instance_values_device {
            debug_assert_eq!(poly.len(), self.n);
            self.instance_ptrs_device
                .push(poly.device_buf().as_raw_ptr());
        }
        self.initialized = true;
        Ok(())
    }

    fn try_init_inner<B>(
        &mut self,
        pk_fixed_lagrange_mirror: Option<&[crate::poly::Polynomial<F, B, crate::poly::Device>]>,
        fixed_values: &[&[F]],
        advice_values: &[&[F]],
        instance_values: &[&[F]],
    ) -> Result<(), HaloGpuError> {
        if self.initialized {
            return Ok(());
        }
        // VRAM accounting: when the PK Lagrange mirror is present and
        // matches `fixed_values` shape (same column count + same per-column
        // length), the fixed-col block is borrowed (zero pool VRAM cost
        // for those columns). Drop fixed from the gate's column count.
        let fixed_borrowed = pk_fixed_lagrange_mirror
            .map(|m| m.len() == fixed_values.len())
            .unwrap_or(false);
        let n_fixed_for_gate = if fixed_borrowed {
            0
        } else {
            fixed_values.len()
        };
        if !self.is_gpu_memory_enough(n_fixed_for_gate, advice_values.len(), instance_values.len())
        {
            return Err(HaloGpuError::InsufficientGpuMemory {
                context: "ColumnPool::try_init",
                magnitude: (n_fixed_for_gate + advice_values.len() + instance_values.len()) as u64,
                free_bytes: query_device_free_bytes_for_chunking() as u64,
            });
        }
        crate::perf_section!("column_pool.upload");
        let per_column_bytes = self.n.saturating_mul(mem::size_of::<F>()) as u64;
        let total_bytes: u64 = per_column_bytes.saturating_mul(
            (n_fixed_for_gate + advice_values.len() + instance_values.len()) as u64,
        );
        crate::perf_h2d!("cuda.column_pool.upload", total_bytes);

        if fixed_borrowed {
            // Borrowed path: read raw Device pointers from the PK mirror.
            // The mirror's lifetime is bound to `'pk`; the pool's
            // `fixed_ptrs_device` retains these raw pointers but never owns
            // the underlying buffers (the corresponding `self.fixed_d`
            // slot stays empty — `num_fixed` reads from
            // `fixed_ptrs_device.len()` instead via the new accessor below).
            let mirror = pk_fixed_lagrange_mirror.unwrap();
            for (mirror_poly, host_col) in mirror.iter().zip(fixed_values.iter()) {
                debug_assert_eq!(host_col.len(), self.n);
                debug_assert_eq!(mirror_poly.len(), self.n);
                let d = mirror_poly.device_buf();
                self.fixed_ptrs_device.push(d.as_raw_ptr());
            }
        } else {
            // H2D path: upload each fixed column into a fresh
            // device buffer owned by this pool.
            for col in fixed_values {
                debug_assert_eq!(col.len(), self.n);
                let d = to_device_on_pinned(col).map_err(HaloGpuError::from)?;
                self.fixed_ptrs_device.push(d.as_raw_ptr());
                self.fixed_d.push(d);
            }
        }
        for col in advice_values {
            debug_assert_eq!(col.len(), self.n);
            let d = to_device_on_pinned(col).map_err(HaloGpuError::from)?;
            self.advice_ptrs_device.push(d.as_raw_ptr());
            self.advice_d.push(d);
        }
        for col in instance_values {
            debug_assert_eq!(col.len(), self.n);
            let d = to_device_on_pinned(col).map_err(HaloGpuError::from)?;
            self.instance_ptrs_device.push(d.as_raw_ptr());
            self.instance_d.push(d);
        }
        self.initialized = true;
        Ok(())
    }

    pub fn fixed_ptrs(&self) -> *const *const c_void {
        self.fixed_ptrs_device.as_ptr()
    }
    pub fn advice_ptrs(&self) -> *const *const c_void {
        self.advice_ptrs_device.as_ptr()
    }
    pub fn instance_ptrs(&self) -> *const *const c_void {
        self.instance_ptrs_device.as_ptr()
    }
    pub fn num_fixed(&self) -> usize {
        // Read from `fixed_ptrs_device` rather than `fixed_d`: the
        // borrowed-mirror path leaves `fixed_d` empty (the PK owns
        // those buffers) while still pushing pointers into
        // `fixed_ptrs_device`.
        self.fixed_ptrs_device.len()
    }
    pub fn num_advice(&self) -> usize {
        // Read from the pointer array, not `advice_d`: the
        // pointer-mirror init path borrows device pointers from the
        // caller's per-prove advice polynomials and leaves `advice_d`
        // empty.
        self.advice_ptrs_device.len()
    }
    pub fn num_instance(&self) -> usize {
        self.instance_ptrs_device.len()
    }
}
