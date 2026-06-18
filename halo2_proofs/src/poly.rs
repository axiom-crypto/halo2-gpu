//! Contains utilities for performing arithmetic over univariate polynomials in
//! various forms, including computing commitments to them and provably opening
//! the committed polynomials at arbitrary points.

use crate::helpers::SerdePrimeField;
use crate::plonk::Assigned;
use crate::SerdeFormat;

use std::fmt::Debug;
use std::io;
use std::marker::{PhantomData, Sync};
use std::mem;
use std::ops::{Add, Deref, DerefMut, Index, IndexMut, Mul, Range, RangeFrom, RangeFull, Sub};

use group::ff::Field;

#[cfg(test)]
use group::ff::BatchInvert;

use openvm_cuda_common::copy::{cuda_memcpy_on, MemCopyH2D};
use openvm_cuda_common::d_buffer::DeviceBuffer;
use openvm_cuda_common::error::MemCopyError;
use openvm_cuda_common::stream::GpuDeviceCtx;

use crate::cpu::arithmetic::parallelize;
use crate::cuda::culib::_halo2_poly_elementwise_multiply;
#[cfg(test)]
use crate::cuda::funcs::batch_invert_gpu;
use crate::cuda::funcs::{batch_invert_device_in_place, decode_assigned_into_denom_slice_device};
use crate::cuda::utils::HALO2_GPU_CTX;
use crate::cuda::HaloGpuError;

/// Generic commitment scheme structures
pub mod commitment;
mod domain;
mod query;
mod strategy;

/// Inner product argument commitment scheme. Kept for compile-surface drop-in
/// compatibility with snark-verifier; openvm itself does not exercise this path.
pub mod ipa;

/// KZG commitment scheme
pub mod kzg;

#[cfg(test)]
mod multiopen_test;

pub use domain::*;
pub use query::{PolyRef, ProverQuery, VerifierQuery};
pub use strategy::{Guard, VerificationStrategy};

/// This is an error that could occur during proving or circuit synthesis.
#[derive(Debug)]
pub enum Error {
    /// OpeningProof is not well-formed
    OpeningError,
    /// Caller needs to re-sample a point
    SamplingError,
}

/// The basis over which a polynomial is described.
pub trait Basis: Copy + Debug + Send + Sync {}

/// The polynomial is defined as coefficients
#[derive(Clone, Copy, Debug)]
pub struct Coeff;
impl Basis for Coeff {}

/// The polynomial is defined as coefficients of Lagrange basis polynomials
#[derive(Clone, Copy, Debug)]
pub struct LagrangeCoeff;
impl Basis for LagrangeCoeff {}

/// The polynomial is defined as coefficients of Lagrange basis polynomials in
/// an extended size domain which supports multiplication
#[derive(Clone, Copy, Debug)]
pub struct ExtendedLagrangeCoeff;
impl Basis for ExtendedLagrangeCoeff {}

/// Residency marker for a [`Polynomial`]. Implementors associate the backing
/// container type via the [`Storage::Backing`] GAT: `Vec<F>` for [`Host`] and
/// `DeviceBuffer<F>` for [`Device`]. Generic over `F` so a single marker can
/// type every scalar choice the prover instantiates.
pub trait Storage: 'static {
    type Backing<F>;

    /// Length of the backing container in elements.
    fn backing_len<F>(b: &Self::Backing<F>) -> usize;

    /// Compile-time tag distinguishing the two storage flavours. Used by
    /// the few code paths that are generic over `S` and want to take a
    /// runtime-fast branch without virtual dispatch (the optimiser folds
    /// the branch since `IS_DEVICE` is `const`).
    const IS_DEVICE: bool;
}

/// Marker indicating a host-resident polynomial whose coefficients live in a
/// `Vec<F>`.
#[derive(Clone, Copy, Debug)]
pub struct Host;

/// Marker indicating a device-resident polynomial whose coefficients live in a
/// `DeviceBuffer<F>` on the shared CUDA stream.
#[derive(Clone, Copy, Debug)]
pub struct Device;

impl Storage for Host {
    type Backing<F> = Vec<F>;
    fn backing_len<F>(b: &Vec<F>) -> usize {
        b.len()
    }
    const IS_DEVICE: bool = false;
}

impl Storage for Device {
    type Backing<F> = DeviceBuffer<F>;
    fn backing_len<F>(b: &DeviceBuffer<F>) -> usize {
        b.len()
    }
    const IS_DEVICE: bool = true;
}

/// Represents a univariate polynomial defined over a field and a particular
/// basis, parameterised by its storage residency.
pub struct Polynomial<F, B, S: Storage = Host> {
    storage: S::Backing<F>,
    _marker: PhantomData<B>,
}

/// Additive type aliases that make residency visible at the function-signature
/// level. The underlying generic [`Polynomial`] stays the canonical type;
/// these aliases give readers a one-token signal of the storage residency
/// without forcing them to inspect the generic parameter list.
pub type HostPoly<F, B> = Polynomial<F, B, Host>;
/// Device-resident counterpart of [`HostPoly`]; see its docs for rationale.
pub type DevicePoly<F, B> = Polynomial<F, B, Device>;

/// Owned residency-tagged polynomial used at boundary sites where the
/// producer's residency choice is decided at runtime (typically by GPU
/// availability or VRAM-fallback feature gates) but the consumer must dispatch
/// per residency at compile time.
pub enum MaybeDevice<F, B> {
    Host(Polynomial<F, B, Host>),
    Device(Polynomial<F, B, Device>),
}

impl<F, B> std::fmt::Debug for MaybeDevice<F, B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MaybeDevice::Host(p) => f.debug_tuple("MaybeDevice::Host").field(p).finish(),
            MaybeDevice::Device(p) => f.debug_tuple("MaybeDevice::Device").field(p).finish(),
        }
    }
}

impl<F, B> MaybeDevice<F, B> {
    pub fn len(&self) -> usize {
        match self {
            MaybeDevice::Host(p) => p.len(),
            MaybeDevice::Device(p) => p.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn is_device(&self) -> bool {
        matches!(self, MaybeDevice::Device(_))
    }

    /// Consume and return a host-resident polynomial; emits a D2H if the
    /// inner carrier is device-resident.
    pub fn into_host_polynomial(self) -> Polynomial<F, B, Host> {
        match self {
            MaybeDevice::Host(p) => p,
            MaybeDevice::Device(p) => p.materialize_host(),
        }
    }
}

impl<F: Clone, B> MaybeDevice<F, B> {
    /// Borrow the values as a host slice; for the device arm this performs a
    /// D2H per call into an owned `Vec<F>`, returned as `Cow::Owned`.
    pub fn values_host(&self) -> std::borrow::Cow<'_, [F]> {
        match self {
            MaybeDevice::Host(p) => std::borrow::Cow::Borrowed(p.values()),
            MaybeDevice::Device(p) => std::borrow::Cow::Owned(p.to_host().into_values()),
        }
    }

    /// Return a host-resident polynomial without consuming the source; the
    /// device arm performs a D2H copy.
    pub fn to_host_polynomial(&self) -> Polynomial<F, B, Host> {
        match self {
            MaybeDevice::Host(p) => Polynomial::new(p.values().to_vec()),
            MaybeDevice::Device(p) => p.to_host(),
        }
    }
}

impl<F, B> From<Polynomial<F, B, Host>> for MaybeDevice<F, B> {
    fn from(p: Polynomial<F, B, Host>) -> Self {
        MaybeDevice::Host(p)
    }
}

impl<F, B> From<Polynomial<F, B, Device>> for MaybeDevice<F, B> {
    fn from(p: Polynomial<F, B, Device>) -> Self {
        MaybeDevice::Device(p)
    }
}

impl<F, B, S: Storage> std::fmt::Debug for Polynomial<F, B, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Polynomial")
            .field("len", &S::backing_len::<F>(&self.storage))
            .field("residency", &if S::IS_DEVICE { "Device" } else { "Host" })
            .finish()
    }
}

// Safety contract:
//
// `DeviceBuffer<F>` (the inner storage of a `Polynomial<_, _, Device>`) lives
// behind `HALO2_GPU_CTX.stream` — a single explicit `cudaStream_t` shared
// across the entire prover (see `cuda/utils.rs`). All device work serializes
// through that stream; cross-thread races on the underlying device memory are
// impossible by construction. Host-resident polynomials carry a `Vec<F>`
// whose `Send`/`Sync` behaviour is the standard one.
unsafe impl<F, B, S: Storage> Sync for Polynomial<F, B, S> {}
unsafe impl<F: Send, B, S: Storage> Send for Polynomial<F, B, S> {}

impl<F: Clone, B> Clone for Polynomial<F, B, Host> {
    fn clone(&self) -> Self {
        Self {
            storage: self.storage.clone(),
            _marker: PhantomData,
        }
    }
}

impl<F, B> Polynomial<F, B, Host> {
    /// Construct a host-resident polynomial directly from `Vec<F>`.
    pub fn new(val: Vec<F>) -> Self {
        Self {
            storage: val,
            _marker: PhantomData,
        }
    }

    /// Direct host slice accessor.
    pub fn values(&self) -> &[F] {
        self.storage.as_slice()
    }

    /// Direct mutable host slice accessor.
    pub fn values_mut(&mut self) -> &mut [F] {
        self.storage.as_mut_slice()
    }

    /// Consume the polynomial and return the owned `Vec<F>` of host
    /// coefficients.
    pub fn into_values(self) -> Vec<F> {
        self.storage
    }

    /// Iterate over the values.
    pub fn iter(&self) -> impl Iterator<Item = &F> {
        self.storage.iter()
    }

    /// Iterate over the values mutably.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut F> {
        self.storage.iter_mut()
    }

    pub fn to_device_on(
        &self,
        device_ctx: &GpuDeviceCtx,
    ) -> Result<Polynomial<F, B, Device>, MemCopyError> {
        Ok(Polynomial {
            storage: self.values().to_device_on(device_ctx)?,
            _marker: PhantomData,
        })
    }
}

impl<F, B> Polynomial<F, B, Device> {
    /// Construct a device-resident polynomial wrapping `buf`.
    ///
    /// Callers MUST verify VRAM headroom before allocating the underlying
    /// `DeviceBuffer<F>` (typically via
    /// `crate::cuda::modules::QuotientLookupsGpu::is_gpu_memory_enough` or
    /// equivalent free-bytes check); by the time this constructor is reached
    /// the buffer is already allocated, so we do not re-check here.
    pub fn from_device(buf: DeviceBuffer<F>) -> Self {
        Self {
            storage: buf,
            _marker: PhantomData,
        }
    }

    /// Returns the underlying device buffer.
    pub fn device_buf(&self) -> &DeviceBuffer<F> {
        &self.storage
    }

    /// Consume the polynomial and return the owned device buffer.
    pub fn into_device_buf(self) -> DeviceBuffer<F> {
        self.storage
    }

    /// D→H copy producing a host-resident polynomial without consuming the
    /// device-resident original. Emits a `tracing::warn!` on the
    /// `halo2_proofs::poly` target so unexpected materializations show up in
    /// trace output.
    pub fn to_host(&self) -> Polynomial<F, B, Host> {
        let buf = &self.storage;
        let n = buf.len();
        let bytes = n * mem::size_of::<F>();
        tracing::warn!(
            target: "halo2_proofs::poly",
            "device->host copy (to_host): {} elements ({} MiB)",
            n,
            bytes >> 20,
        );
        crate::perf_d2h!("poly.to_host", bytes);
        let mut host: Vec<F> = Vec::with_capacity(n);
        // SAFETY: set_len precedes the cuda_memcpy_on that fully initialises
        // `n` elements; `F` is a halo2 field scalar (POD repr).
        unsafe {
            host.set_len(n);
            cuda_memcpy_on::<true, false>(
                host.as_mut_ptr() as *mut libc::c_void,
                buf.as_raw_ptr(),
                bytes,
                &HALO2_GPU_CTX,
            )
            .expect("D2H to_host copy failed");
        }
        HALO2_GPU_CTX
            .stream
            .to_host_sync()
            .expect("stream sync after D2H to_host failed");
        Polynomial {
            storage: host,
            _marker: PhantomData,
        }
    }

    /// D→H copy that consumes the device polynomial and returns the
    /// host-resident equivalent. Emits a `tracing::warn!` on the
    /// `halo2_proofs::poly` target so unexpected materializations show up
    /// in trace output.
    pub fn materialize_host(self) -> Polynomial<F, B, Host> {
        let buf = self.storage;
        let n = buf.len();
        let bytes = n * mem::size_of::<F>();
        tracing::warn!(
            target: "halo2_proofs::poly",
            "device->host materialization: {} elements ({} MiB)",
            n,
            bytes >> 20,
        );
        crate::perf_d2h!("poly.materialize_host", bytes);
        let mut host: Vec<F> = Vec::with_capacity(n);
        // SAFETY: `set_len` precedes the cuda_memcpy_on that fully
        // initialises `n` elements; `F` is a halo2 field scalar (POD repr).
        unsafe {
            host.set_len(n);
            cuda_memcpy_on::<true, false>(
                host.as_mut_ptr() as *mut libc::c_void,
                buf.as_raw_ptr(),
                bytes,
                &HALO2_GPU_CTX,
            )
            .expect("D2H materialization copy failed");
        }
        HALO2_GPU_CTX
            .stream
            .to_host_sync()
            .expect("stream sync after D2H materialization failed");
        Polynomial {
            storage: host,
            _marker: PhantomData,
        }
    }
}

impl<F, B, S: Storage> Polynomial<F, B, S> {
    /// Number of coefficients.
    pub fn len(&self) -> usize {
        S::backing_len::<F>(&self.storage)
    }

    /// `true` if there are no coefficients.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Synonym for [`Polynomial::len`].
    pub fn num_coeffs(&self) -> usize {
        self.len()
    }
}

impl<F, B> AsRef<[F]> for Polynomial<F, B, Host> {
    fn as_ref(&self) -> &[F] {
        self.values()
    }
}

impl<F, B> AsMut<[F]> for Polynomial<F, B, Host> {
    fn as_mut(&mut self) -> &mut [F] {
        self.values_mut()
    }
}

impl<F, B> Deref for Polynomial<F, B, Host> {
    type Target = [F];

    fn deref(&self) -> &[F] {
        self.values()
    }
}

impl<F, B> DerefMut for Polynomial<F, B, Host> {
    fn deref_mut(&mut self) -> &mut [F] {
        self.values_mut()
    }
}

impl<F, B> Index<usize> for Polynomial<F, B, Host> {
    type Output = F;

    fn index(&self, index: usize) -> &F {
        &self.values()[index]
    }
}

impl<F, B> IndexMut<usize> for Polynomial<F, B, Host> {
    fn index_mut(&mut self, index: usize) -> &mut F {
        &mut self.values_mut()[index]
    }
}

impl<F, B> Index<Range<usize>> for Polynomial<F, B, Host> {
    type Output = [F];

    fn index(&self, index: Range<usize>) -> &[F] {
        &self.values()[index]
    }
}

impl<F, B> Index<RangeFrom<usize>> for Polynomial<F, B, Host> {
    type Output = [F];

    fn index(&self, index: RangeFrom<usize>) -> &[F] {
        &self.values()[index]
    }
}

impl<F, B> IndexMut<Range<usize>> for Polynomial<F, B, Host> {
    fn index_mut(&mut self, index: Range<usize>) -> &mut [F] {
        &mut self.values_mut()[index]
    }
}

impl<F, B> IndexMut<RangeFrom<usize>> for Polynomial<F, B, Host> {
    fn index_mut(&mut self, index: RangeFrom<usize>) -> &mut [F] {
        &mut self.values_mut()[index]
    }
}

impl<F, B> Index<RangeFull> for Polynomial<F, B, Host> {
    type Output = [F];

    fn index(&self, _index: RangeFull) -> &[F] {
        self.values()
    }
}

impl<F, B> IndexMut<RangeFull> for Polynomial<F, B, Host> {
    fn index_mut(&mut self, index: RangeFull) -> &mut [F] {
        &mut self.values_mut()[index]
    }
}

impl<F: Clone, B> Polynomial<F, B, Host> {
    /// Fallible clone of a host-resident polynomial. Wraps `Vec::clone` in
    /// the same `Result` shape as [`Polynomial<F, B, Device>::try_clone`] so
    /// callers can write residency-generic code.
    pub fn try_clone(&self) -> Result<Self, HaloGpuError> {
        Ok(Self {
            storage: self.storage.clone(),
            _marker: PhantomData,
        })
    }
}

impl<F, B> Polynomial<F, B, Device> {
    /// Fallible clone of a device-resident polynomial. Allocates a new
    /// `DeviceBuffer<F>` and submits a D→D copy
    /// (`cudaMemcpyAsync(DeviceToDevice)`) on the explicit stream.
    pub fn try_clone(&self) -> Result<Self, HaloGpuError> {
        use openvm_cuda_common::copy::MemCopyD2D;
        let dst = self.storage.device_copy_on(&HALO2_GPU_CTX)?;
        Ok(Self {
            storage: dst,
            _marker: PhantomData,
        })
    }
}

impl<F: Field> Polynomial<F, Coeff, Host> {
    /// Evaluate the polynomial at `point` using the rayon-parallel CPU Horner
    /// implementation.
    pub fn eval_at(&self, point: F) -> F {
        crate::arithmetic::eval_polynomial(&self.storage, point)
    }
}

impl<F: Field> Polynomial<F, Coeff, Device> {
    /// Evaluate the polynomial at `point` on-GPU via `eval_polynomial_device`.
    /// The kernel tags its own 32-byte result D2H under
    /// `cuda.eval_polynomial_device.result`.
    pub fn eval_at(&self, point: F) -> F {
        crate::cuda::funcs::eval_polynomial_device(&self.storage, point)
            .expect("eval_polynomial_device failed in Polynomial::eval_at")
    }
}

impl<F> Polynomial<F, Coeff, Device> {
    /// Splits a Device-resident `Polynomial<F, Coeff, Device>` into pieces of
    /// length `chunk_len` each. Self is consumed; the parent allocation is
    /// freed after the loop completes. Peak transient device memory is
    /// roughly `2 * parent.len() * sizeof::<F>()` during the loop (parent +
    /// all chunks alive simultaneously); this matches the prior host path's
    /// `to_vec()` per chunk transient.
    ///
    /// At fibonacci shape `parent.len() == n * quotient_poly_degree`,
    /// `chunk_len == n`, `num_chunks == quotient_poly_degree` (3 for outer,
    /// 4 for wrapper); both tile cleanly.
    pub fn chunks_device(self, chunk_len: usize) -> Vec<Polynomial<F, Coeff, Device>> {
        use openvm_cuda_common::copy::cuda_memcpy_on;
        let parent = self.storage;
        let total_len = parent.len();
        assert!(
            chunk_len > 0 && total_len.is_multiple_of(chunk_len),
            "chunks_device: total_len {} not divisible by chunk_len {}",
            total_len,
            chunk_len
        );
        let num_chunks = total_len / chunk_len;
        let parent_base = parent.as_raw_ptr() as *const u8;
        let elem_bytes = mem::size_of::<F>();
        let chunk_bytes = chunk_len * elem_bytes;

        let mut chunks: Vec<Polynomial<F, Coeff, Device>> = Vec::with_capacity(num_chunks);
        for i in 0..num_chunks {
            let dst: DeviceBuffer<F> =
                DeviceBuffer::<F>::with_capacity_on(chunk_len, &HALO2_GPU_CTX);
            // SAFETY: chunk_bytes <= parent allocation remaining; offset is in-bounds
            // because i * chunk_len < total_len for all i < num_chunks.
            unsafe {
                let src = parent_base.add(i * chunk_bytes) as *const libc::c_void;
                cuda_memcpy_on::<true, true>(
                    dst.as_mut_raw_ptr(),
                    src,
                    chunk_bytes,
                    &HALO2_GPU_CTX,
                )
                .expect("D2D copy in chunks_device failed");
            }
            chunks.push(Polynomial::from_device(dst));
        }
        drop(parent);
        chunks
    }
}

impl<F: SerdePrimeField, B> Polynomial<F, B, Host> {
    /// Reads polynomial from buffer using `SerdePrimeField::read`.
    pub(crate) fn read<R: io::Read>(reader: &mut R, format: SerdeFormat) -> Self {
        let mut poly_len = [0u8; 4];
        reader.read_exact(&mut poly_len).unwrap();
        let poly_len = u32::from_be_bytes(poly_len);
        let values: Vec<F> = (0..poly_len)
            .map(|_| F::read(reader, format).unwrap())
            .collect();
        Self::new(values)
    }

    /// Writes polynomial to buffer using `SerdePrimeField::write`.
    pub(crate) fn write<W: io::Write>(&self, writer: &mut W, format: SerdeFormat) {
        let values = self.values();
        writer
            .write_all(&(values.len() as u32).to_be_bytes())
            .unwrap();
        for value in values.iter() {
            value.write(writer, format).unwrap();
        }
    }
}

/// Inverts the per-cell denominators on device, then reconstructs each
/// column's `numerator * inv_denom` product into a per-column
/// `DeviceBuffer<F>` via the `_halo2_poly_elementwise_multiply` kernel.
/// Returns one `Polynomial<F, LagrangeCoeff, Device>` per input column.
///
/// Each `&[Assigned<F>]` column is uploaded as raw `#[repr(C, u8)]`
/// bytes (single H2D, no host enum-decode pass) and decoded on device
/// by `_halo2_decode_assigned` into a per-column numerator buffer plus
/// a slice into the concatenated denominator buffer. The denominators
/// are then batch-inverted in place and the numerator * inv-denom
/// product is reduced per column with the existing elementwise
/// multiply kernel. Cells with `None` denominator (`Assigned::Zero` /
/// `Assigned::Trivial`) get a `F::ONE` denominator emitted directly by
/// the decode kernel so the multiply preserves their numerator
/// unchanged.
///
/// All work is enqueued on `HALO2_GPU_CTX.stream`; the function returns
/// before the kernels complete. Same-stream subsequent device ops see
/// the result. Host reads require an explicit `to_host_sync()`.
pub(crate) fn batch_invert_assigned_device<F: Field, PR>(
    assigned: impl AsRef<[PR]>,
) -> Result<Vec<Polynomial<F, LagrangeCoeff, Device>>, HaloGpuError>
where
    PR: AsRef<[Assigned<F>]> + Send + Sync,
{
    let assigned = assigned.as_ref();
    if assigned.is_empty() {
        return Ok(vec![]);
    }
    let n = assigned[0].as_ref().len();
    let n_cols = assigned.len();

    let d_inv_denoms: DeviceBuffer<F> =
        DeviceBuffer::<F>::with_capacity_on(n * n_cols, &HALO2_GPU_CTX);

    let mut nums_per_col: Vec<DeviceBuffer<F>> = Vec::with_capacity(n_cols);
    for (col_idx, poly_assigned) in assigned.iter().enumerate() {
        let poly_slice = poly_assigned.as_ref();
        assert_eq!(
            poly_slice.len(),
            n,
            "batch_invert_assigned_device: column {} has length {} but column 0 has length {}",
            col_idx,
            poly_slice.len(),
            n,
        );
        let d_nums =
            decode_assigned_into_denom_slice_device(poly_slice, &d_inv_denoms, col_idx * n)?;
        nums_per_col.push(d_nums);
    }

    batch_invert_device_in_place(&d_inv_denoms)?;

    let mut results: Vec<Polynomial<F, LagrangeCoeff, Device>> = Vec::with_capacity(n_cols);
    for (col_idx, d_col) in nums_per_col.into_iter().enumerate() {
        let d_inv_denoms_col_ptr: *const std::ffi::c_void =
            unsafe { d_inv_denoms.as_ptr().add(col_idx * n) as *const std::ffi::c_void };
        let status = unsafe {
            _halo2_poly_elementwise_multiply(
                d_col.as_mut_raw_ptr(),
                d_col.as_raw_ptr(),
                d_inv_denoms_col_ptr,
                n as u64,
                HALO2_GPU_CTX.stream.as_raw(),
            )
        };
        if status.code != 0 {
            return Err(status.into());
        }
        results.push(Polynomial::from_device(d_col));
    }
    Ok(results)
}

impl<F: Field> Polynomial<Assigned<F>, LagrangeCoeff> {
    pub fn invert(
        &self,
        inv_denoms: impl ExactSizeIterator<Item = F>,
    ) -> Polynomial<F, LagrangeCoeff> {
        let src = self.values();
        assert_eq!(inv_denoms.len(), src.len());
        let values: Vec<F> = src
            .iter()
            .zip(inv_denoms)
            .map(|(a, inv_den)| a.numerator() * inv_den)
            .collect();
        Polynomial::new(values)
    }
}

impl<'a, F: Field, B: Basis> Add<&'a Polynomial<F, B, Host>> for Polynomial<F, B, Host> {
    type Output = Polynomial<F, B, Host>;

    fn add(mut self, rhs: &'a Polynomial<F, B, Host>) -> Polynomial<F, B, Host> {
        let rhs_slice = rhs.values();
        parallelize(self.values_mut(), |lhs, start| {
            for (lhs, rhs) in lhs.iter_mut().zip(rhs_slice[start..].iter()) {
                *lhs += *rhs;
            }
        });
        self
    }
}

impl<'a, F: Field, B: Basis> Sub<&'a Polynomial<F, B, Host>> for Polynomial<F, B, Host> {
    type Output = Polynomial<F, B, Host>;

    fn sub(mut self, rhs: &'a Polynomial<F, B, Host>) -> Polynomial<F, B, Host> {
        let rhs_slice = rhs.values();
        parallelize(self.values_mut(), |lhs, start| {
            for (lhs, rhs) in lhs.iter_mut().zip(rhs_slice[start..].iter()) {
                *lhs -= *rhs;
            }
        });
        self
    }
}

impl<F: Field> Polynomial<F, LagrangeCoeff, Host> {
    /// Rotates the values in a Lagrange basis polynomial by `Rotation`
    pub fn rotate(&self, rotation: Rotation) -> Polynomial<F, LagrangeCoeff, Host> {
        let mut values = self.values().to_vec();
        if rotation.0 < 0 {
            values.rotate_right((-rotation.0) as usize);
        } else {
            values.rotate_left(rotation.0 as usize);
        }
        Polynomial::new(values)
    }
}

impl<F: Field, B: Basis> Mul<F> for Polynomial<F, B, Host> {
    type Output = Polynomial<F, B, Host>;

    fn mul(mut self, rhs: F) -> Polynomial<F, B, Host> {
        if rhs == F::ZERO {
            return Polynomial::new(vec![F::ZERO; self.len()]);
        }
        if rhs == F::ONE {
            return self;
        }

        parallelize(self.values_mut(), |lhs, _| {
            for lhs in lhs.iter_mut() {
                *lhs *= rhs;
            }
        });

        self
    }
}

impl<F: Field, B: Basis> Sub<F> for &Polynomial<F, B, Host> {
    type Output = Polynomial<F, B, Host>;

    fn sub(self, rhs: F) -> Polynomial<F, B, Host> {
        let mut res = self.clone();
        res.values_mut()[0] -= rhs;
        res
    }
}

/// Describes the relative rotation of a vector. Negative numbers represent
/// reverse (leftmost) rotations and positive numbers represent forward (rightmost)
/// rotations. Zero represents no rotation.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Rotation(pub i32);

impl Rotation {
    /// The current location in the evaluation domain
    pub fn cur() -> Rotation {
        Rotation(0)
    }

    /// The previous location in the evaluation domain
    pub fn prev() -> Rotation {
        Rotation(-1)
    }

    /// The next location in the evaluation domain
    pub fn next() -> Rotation {
        Rotation(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::poly::{
        batch_invert_assigned, batch_invert_assigned_gpu, batch_invert_assigned_par,
    };
    use std::time::Instant;

    #[test]
    fn test_batch_invert_par() {
        use halo2curves::bn256::Fr;

        let min_k = 19;
        let max_k = 26;

        for k in min_k..=max_k {
            let n = 1 << k;
            let assigned = (0..n)
                .map(|j| {
                    let num = Fr::from(1_u64);
                    let denom = Fr::from(j as u64);
                    Assigned::from((num, denom))
                })
                .collect::<Vec<_>>();
            let poly = Polynomial::<_, LagrangeCoeff>::new(assigned);
            let polys = vec![poly];

            let seq_time = Instant::now();
            let seq = batch_invert_assigned(polys.clone());
            let seq_time = seq_time.elapsed();

            let par_time = Instant::now();
            let par = batch_invert_assigned_par(polys);
            let par_time = par_time.elapsed();
            assert_eq!(seq[0].values(), par[0].values());
            println!(
                "batch invert of 1 poly of size {}: seq = {:?}, par = {:?}, speedup = {}",
                n,
                seq_time,
                par_time,
                seq_time.as_micros() as f64 / par_time.as_micros() as f64
            );
        }
    }

    #[test]
    #[allow(unused_variables)]
    fn test_batch_invert_gpu() {
        use std::time::Instant;

        use rand::thread_rng;

        fn test_field<F: Field>() {
            let is_type_fr =
                std::any::TypeId::of::<F>() == std::any::TypeId::of::<halo2curves::bn256::Fr>();
            let is_type_fq =
                std::any::TypeId::of::<F>() == std::any::TypeId::of::<halo2curves::bn256::Fq>();
            if is_type_fr {
                println!("test_batch_invert_gpu, field type: Fr");
            }
            if is_type_fq {
                println!("test_batch_invert_gpu, field type: Fq");
            }

            let n = 1 << 20;
            let mut rng = thread_rng();
            let scalars = (0..n).map(|_| F::random(&mut rng)).collect::<Vec<_>>();
            let mut result_cpu = scalars.clone();
            let mut result_gpu = scalars.clone();

            // correctness and warmup
            result_cpu.batch_invert();
            batch_invert_gpu(&mut result_gpu).unwrap();
            assert_eq!(result_cpu, result_gpu);

            // benchmark
            let mut result_cpu = scalars.clone();
            let mut result_gpu = scalars.clone();

            let cpu_time = Instant::now();
            result_cpu.batch_invert();
            let cpu_time = cpu_time.elapsed();
            let cpu_micros = f64::from(cpu_time.as_micros() as u32);

            let gpu_time = Instant::now();
            batch_invert_gpu(&mut result_gpu).unwrap();
            let gpu_time = gpu_time.elapsed();
            let gpu_micros = f64::from(gpu_time.as_micros() as u32);

            println!(
                "[num = {}] cpu_time: {:?}, gpu_time: {:?}, speedup: {}",
                n,
                cpu_time,
                gpu_time,
                cpu_micros / gpu_micros
            );
        }

        test_field::<halo2curves::bn256::Fr>();
        test_field::<halo2curves::bn256::Fq>();
    }

    #[test]
    fn test_batch_invert_assigned_gpu() {
        use std::time::Instant;

        use halo2curves::bn256::Fr;

        let min_k = 19;
        let max_k = 22;

        for k in min_k..=max_k {
            let n = 1 << k;
            let assigned = (0..n)
                .map(|j| {
                    let num = Fr::from(1_u64);
                    let denom = Fr::from(j as u64 + 1);
                    Assigned::from((num, denom))
                })
                .collect::<Vec<_>>();
            let poly = Polynomial::<_, LagrangeCoeff>::new(assigned);
            let polys = vec![poly];

            // warmup
            let cpu_poly = polys.clone();
            let gpu_poly = polys.clone();
            let res_par = batch_invert_assigned_par(cpu_poly);
            let res_gpu = batch_invert_assigned_gpu(gpu_poly).unwrap();
            assert_eq!(res_par[0].values()[0], res_gpu[0].values()[0]);

            // benchmark
            let cpu_poly = polys.clone();
            let gpu_poly = polys.clone();
            let par_time = Instant::now();
            let _res_par = batch_invert_assigned_par(cpu_poly);
            let par_time = par_time.elapsed();
            let gpu_time = Instant::now();
            let _res_gpu = batch_invert_assigned_gpu(gpu_poly);
            let gpu_time = gpu_time.elapsed();

            println!(
                "batch invert of 1 poly of size {}: par = {:?}, gpu = {:?}, speedup = {}",
                n,
                par_time,
                gpu_time,
                par_time.as_micros() as f64 / gpu_time.as_micros() as f64
            );
        }
    }

    #[test]
    fn test_decode_assigned_to_num_denom_device_vs_host() {
        use crate::cuda::funcs::decode_assigned_to_num_denom_device;
        use halo2curves::bn256::Fr;
        use openvm_cuda_common::copy::cuda_memcpy_on;
        use rand::rngs::OsRng;
        use std::ffi::c_void;

        fn run_one(log_n: u32) {
            let n: usize = 1usize << log_n;
            // Exercise all 3 variants in varied positions: every 3rd
            // element is Zero / Trivial / Rational, plus a salt that
            // shifts the pattern so consecutive columns sample
            // different variant orderings.
            let column: Vec<Assigned<Fr>> = (0..n)
                .map(|j| match (j + 7) % 3 {
                    0 => Assigned::Zero,
                    1 => Assigned::Trivial(Fr::random(OsRng)),
                    _ => Assigned::Rational(Fr::random(OsRng), Fr::random(OsRng)),
                })
                .collect();

            // Host oracle — the same numerator/denominator decode the GPU
            // kernel performs, done as two `par_iter` passes; the
            // byte-exact reference for the GPU decode kernel.
            let host_nums: Vec<Fr> = column.iter().map(|a| a.numerator()).collect();
            let host_denoms: Vec<Fr> = column
                .iter()
                .map(|a| a.denominator().unwrap_or(Fr::ONE))
                .collect();

            let (d_nums, d_denoms) = decode_assigned_to_num_denom_device(&column)
                .expect("decode_assigned_to_num_denom_device failed");

            let mut gpu_nums = vec![Fr::ZERO; n];
            let mut gpu_denoms = vec![Fr::ZERO; n];
            unsafe {
                cuda_memcpy_on::<true, false>(
                    gpu_nums.as_mut_ptr() as *mut c_void,
                    d_nums.as_raw_ptr(),
                    n * std::mem::size_of::<Fr>(),
                    &HALO2_GPU_CTX,
                )
                .expect("D2H of decoded nums failed");
                cuda_memcpy_on::<true, false>(
                    gpu_denoms.as_mut_ptr() as *mut c_void,
                    d_denoms.as_raw_ptr(),
                    n * std::mem::size_of::<Fr>(),
                    &HALO2_GPU_CTX,
                )
                .expect("D2H of decoded denoms failed");
            }
            HALO2_GPU_CTX.stream.to_host_sync().unwrap();

            for (i, ((h_n, g_n), (h_d, g_d))) in host_nums
                .iter()
                .zip(gpu_nums.iter())
                .zip(host_denoms.iter().zip(gpu_denoms.iter()))
                .enumerate()
            {
                assert_eq!(h_n, g_n, "numerator mismatch at log_n={log_n}, idx={i}");
                assert_eq!(h_d, g_d, "denominator mismatch at log_n={log_n}, idx={i}");
            }
        }

        for &log_n in &[12u32, 18, 20] {
            run_one(log_n);
        }
    }

    #[test]
    #[ignore = "heavy"]
    fn test_batch_invert_assigned_device_vs_host() {
        use halo2curves::bn256::Fr;
        use openvm_cuda_common::copy::cuda_memcpy_on;
        use rand::rngs::OsRng;
        use std::ffi::c_void;

        fn run_one(log_n: u32) {
            let n: usize = 1usize << log_n;
            let n_cols: usize = 3;

            let columns: Vec<Vec<Assigned<Fr>>> = (0..n_cols)
                .map(|col_idx| {
                    (0..n)
                        .map(|j| match (col_idx + j) % 3 {
                            0 => Assigned::Zero,
                            1 => Assigned::Trivial(Fr::random(OsRng)),
                            _ => {
                                let num = Fr::random(OsRng);
                                let denom = Fr::random(OsRng);
                                Assigned::from((num, denom))
                            }
                        })
                        .collect()
                })
                .collect();

            let host_input: Vec<&[Assigned<Fr>]> = columns.iter().map(|c| c.as_slice()).collect();
            let device_input: Vec<&[Assigned<Fr>]> = host_input.clone();

            let host_polys: Vec<Polynomial<Fr, LagrangeCoeff>> =
                batch_invert_assigned_gpu(host_input).expect("host-output batch_invert failed");

            let device_polys: Vec<Polynomial<Fr, LagrangeCoeff, Device>> =
                batch_invert_assigned_device(device_input)
                    .expect("device-output batch_invert failed");

            assert_eq!(host_polys.len(), n_cols);
            assert_eq!(device_polys.len(), n_cols);

            for (col_idx, (host_poly, device_poly)) in
                host_polys.iter().zip(device_polys.iter()).enumerate()
            {
                let host_vals = host_poly.values();
                let mut device_vals = vec![Fr::ZERO; n];
                unsafe {
                    cuda_memcpy_on::<true, false>(
                        device_vals.as_mut_ptr() as *mut c_void,
                        device_poly.device_buf().as_raw_ptr(),
                        n * std::mem::size_of::<Fr>(),
                        &HALO2_GPU_CTX,
                    )
                    .expect("D2H of device-output column failed");
                }
                HALO2_GPU_CTX.stream.to_host_sync().unwrap();
                for (i, (h, d)) in host_vals.iter().zip(device_vals.iter()).enumerate() {
                    assert_eq!(
                        h, d,
                        "device vs host disagree at log_n={log_n}, col={col_idx}, idx={i}"
                    );
                }
            }
        }

        for &log_n in &[20u32, 22, 23] {
            run_one(log_n);
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_eval_polynomial() {
        use std::time::Instant;

        use halo2curves::bn256::Fr as Scalar;
        use rand::thread_rng;

        use crate::{arithmetic::eval_polynomial, cuda::funcs::eval_polynomial_gpu};
        let max_base_size = 25;
        let min_base_size = 6;

        let N: usize = 1 << max_base_size;
        let mut rng = thread_rng();
        let poly = (0..N).map(|_| Scalar::random(&mut rng)).collect::<Vec<_>>();
        let point = Scalar::random(&mut rng);

        println!("test_eval_polynomial");

        for k in min_base_size..=max_base_size {
            let n = 1 << k;
            // correctness and warmup
            let cpu_result = eval_polynomial(&poly[..n], point);
            let gpu_result = eval_polynomial_gpu(&poly[..n], point).unwrap();
            assert_eq!(cpu_result, gpu_result);

            // benchmark
            let cpu_time = Instant::now();
            let _cpu_result = eval_polynomial(&poly[..n], point);
            let cpu_time = cpu_time.elapsed();
            let _cpu_micros = f64::from(cpu_time.as_micros() as u32);

            let gpu_time = Instant::now();
            let _gpu_result = eval_polynomial_gpu(&poly[..n], point);
            let gpu_time = gpu_time.elapsed();
            let _gpu_micros = f64::from(gpu_time.as_micros() as u32);
            let data_transfer_size = n * 32;
            let data_transfer_bandwidth =
                (data_transfer_size as f64 / 1024.0 / 1024.0 / 1024.0) / gpu_time.as_secs_f64();

            println!(
                "[k = {}] cpu_time: {:?}, gpu_time: {:?}, PCI-E bandwidth: {} GB/s",
                k, cpu_time, gpu_time, data_transfer_bandwidth
            );
        }
    }

    #[test]
    fn test_batch_eval_polynomial() {
        use std::time::Instant;

        use halo2curves::bn256::Fr as Scalar;
        use rand::thread_rng;

        use crate::{
            arithmetic::eval_polynomial,
            cuda::{funcs::batch_eval_polynomial_gpu, utils::FFITraitObject},
        };
        let mut rng = thread_rng();

        let batch_size = 31;
        let max_base_size = 21;
        let min_base_size = 6;
        println!("test_batch_eval_polynomial");
        println!("batch_size: {}", batch_size);

        for log_n in min_base_size..=max_base_size {
            let poly_len: usize = 1 << log_n;
            let ploy_in_many = (0..batch_size)
                .map(|_| {
                    (0..poly_len)
                        .map(|_| Scalar::random(&mut rng))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let eval_point = (0..batch_size)
                .map(|_| Scalar::random(&mut rng))
                .collect::<Vec<_>>();
            let eval_point_gpu = eval_point.clone();

            let get_polys_ffi_in = |polys: Vec<&Vec<Scalar>>| {
                polys
                    .iter()
                    .map(|poly| FFITraitObject::from_slice(poly.as_slice()))
                    .collect::<Vec<FFITraitObject>>()
            };
            let poly_in_many_ori = get_polys_ffi_in(ploy_in_many.iter().collect());

            // correctness and warmup
            let cpu_result = (0..batch_size)
                .map(|idx| eval_polynomial(&ploy_in_many[idx], eval_point[idx]))
                .collect::<Vec<_>>();
            let mut gpu_result = vec![Scalar::zero(); batch_size];
            batch_eval_polynomial_gpu(
                &poly_in_many_ori,
                &eval_point_gpu,
                &mut gpu_result,
                poly_len,
            )
            .unwrap();
            assert_eq!(cpu_result, gpu_result);

            // benchmark
            let cpu_time = Instant::now();
            let _cpu_result = (0..batch_size)
                .map(|idx| eval_polynomial(&ploy_in_many[idx], eval_point[idx]))
                .collect::<Vec<_>>();
            let cpu_time = cpu_time.elapsed();
            let _cpu_micros = f64::from(cpu_time.as_micros() as u32);

            let gpu_time = Instant::now();
            let mut gpu_result = vec![Scalar::zero(); batch_size];
            batch_eval_polynomial_gpu(
                &poly_in_many_ori,
                &eval_point_gpu,
                &mut gpu_result,
                poly_len,
            )
            .unwrap();
            let gpu_time = gpu_time.elapsed();
            let _gpu_micros = f64::from(gpu_time.as_micros() as u32);
            let data_transfer_size = batch_size * poly_len * 32;
            let data_transfer_bandwidth =
                (data_transfer_size as f64 / 1024.0 / 1024.0 / 1024.0) / gpu_time.as_secs_f64();

            println!(
                "[k = {}] cpu_time: {:?}, gpu_time: {:?}, PCI-E bandwidth: {} GB/s",
                log_n, cpu_time, gpu_time, data_transfer_bandwidth
            );
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_poly_multiply_add() {
        use halo2curves::bn256::Fr as Scalar;
        use rand::thread_rng;
        use std::time::Instant;

        use crate::cuda::funcs::poly_multiply_add_single_gpu;
        let mut rng = thread_rng();
        let max_base_size = 25;
        let min_base_size = 6;

        let N: usize = 1 << max_base_size;
        let point = Scalar::random(&mut rng);
        let poly_with_max_degree = (0..N).map(|_| Scalar::random(&mut rng)).collect::<Vec<_>>();

        println!("test_poly_multiply_add");

        for k in min_base_size..=max_base_size {
            let n = 1 << k;
            let poly_in: Polynomial<Scalar, Coeff> =
                Polynomial::new(poly_with_max_degree[..n].to_vec());
            let mut acc_poly: Polynomial<Scalar, Coeff> =
                Polynomial::new((0..n).map(|_| Scalar::zero()).collect::<Vec<_>>());

            // correctness and warmup
            let cpu_result = poly_in.clone() * point + &acc_poly;
            poly_multiply_add_single_gpu(acc_poly.values_mut(), poly_in.values(), point).unwrap();
            assert_eq!(cpu_result.values(), acc_poly.values());

            // benchmark
            let cpu_time = Instant::now();
            let _cpu_result = poly_in.clone() * point + &acc_poly;
            let cpu_time = cpu_time.elapsed();
            let cpu_micros = f64::from(cpu_time.as_micros() as u32);

            let gpu_time = Instant::now();
            poly_multiply_add_single_gpu(acc_poly.values_mut(), poly_in.values(), point).unwrap();
            let gpu_time = gpu_time.elapsed();
            let gpu_micros = f64::from(gpu_time.as_micros() as u32);

            let data_transfer_size = n * 32 * 2;
            let data_transfer_size = data_transfer_size as f64 / 1024.0 / 1024.0;
            println!(
                "[k = {}] cpu_time: {:?}, gpu_time: {:?}, speedup: {}, data_transfer_size: {} MB",
                k,
                cpu_time,
                gpu_time,
                cpu_micros / gpu_micros,
                data_transfer_size
            );
        }
    }

    #[test]
    fn test_multiopen_poly_calculation() {
        use halo2curves::bn256::Fr as Scalar;
        use rand::thread_rng;
        use std::time::Instant;

        use crate::{
            arithmetic::{eval_polynomial, powers},
            cuda::{funcs::multiopen_poly_calculation_gpu, utils::FFITraitObject},
        };
        let mut rng = thread_rng();

        let batch_size = 31;
        let max_base_size = 21;
        let min_base_size = 6;
        println!("test_multiopen_poly_calculation");
        println!("single gpu, batch_size: {}", batch_size);

        let v = Scalar::random(&mut rng);
        let challenge_point = (0..batch_size)
            .zip(powers(v))
            .map(|(_, power_of_v)| power_of_v)
            .collect::<Vec<_>>();

        for log_n in min_base_size..=max_base_size {
            let poly_length = 1 << log_n;

            let evaluate_point = (0..batch_size)
                .map(|_| Scalar::random(&mut rng))
                .collect::<Vec<_>>();
            let evaluate_point_gpu = evaluate_point.clone();
            let poly_acc: Polynomial<Scalar, Coeff> =
                Polynomial::new((0..poly_length).map(|_| Scalar::zero()).collect::<Vec<_>>());
            let mut poly_vec: Vec<Polynomial<Scalar, Coeff>> = Vec::with_capacity(batch_size);
            for _ in 0..batch_size {
                poly_vec.push(Polynomial::new(
                    (0..poly_length)
                        .map(|_| Scalar::random(&mut rng))
                        .collect::<Vec<_>>(),
                ));
            }
            let get_slice_polys_ffi_in = |polys: Vec<&Polynomial<Scalar, Coeff>>| {
                polys
                    .iter()
                    .map(|poly| FFITraitObject::from_slice(poly.values()))
                    .collect::<Vec<FFITraitObject>>()
            };

            // correctness and warmup
            let mut poly_acc_cpu = poly_acc.clone();
            for (i, power_of_v) in (0..batch_size).zip(challenge_point.clone()) {
                poly_acc_cpu = poly_vec[i].clone() * power_of_v + &poly_acc_cpu;
            }
            let eval_result_cpu = evaluate_point
                .iter()
                .enumerate()
                .map(|(i, point)| eval_polynomial(poly_vec[i].values(), *point))
                .collect::<Vec<_>>();

            let mut eval_result_gpu = vec![Scalar::zero(); batch_size];
            let mut poly_acc_gpu = poly_acc.clone();
            multiopen_poly_calculation_gpu(
                get_slice_polys_ffi_in(poly_vec.iter().collect()),
                challenge_point.clone(),
                poly_acc_gpu.values_mut(), // multiply_add
                evaluate_point_gpu.clone(),
                &mut eval_result_gpu, // evaluation
                poly_length,
            )
            .unwrap();
            assert_eq!(eval_result_cpu, eval_result_gpu);
            assert_eq!(poly_acc_cpu.values(), poly_acc_gpu.values());

            // benchmark
            let mut poly_acc_cpu = poly_acc.clone();
            let cpu_time = Instant::now();
            for (i, power_of_v) in (0..batch_size).zip(challenge_point.clone()) {
                poly_acc_cpu = poly_vec[i].clone() * power_of_v + &poly_acc_cpu;
            }
            let _eval_result_cpu = evaluate_point
                .iter()
                .enumerate()
                .map(|(i, point)| eval_polynomial(poly_vec[i].values(), *point))
                .collect::<Vec<_>>();
            let cpu_time = cpu_time.elapsed();
            let cpu_micros = f64::from(cpu_time.as_micros() as u32);

            let mut eval_result_gpu = vec![Scalar::zero(); batch_size];
            let mut poly_acc_gpu = poly_acc.clone();
            let gpu_time = Instant::now();
            multiopen_poly_calculation_gpu(
                get_slice_polys_ffi_in(poly_vec.iter().collect()),
                challenge_point.clone(),
                poly_acc_gpu.values_mut(), // multiply_add
                evaluate_point_gpu.clone(),
                &mut eval_result_gpu, // evaluation
                poly_length,
            )
            .unwrap();
            let gpu_time = gpu_time.elapsed();
            let gpu_micros = f64::from(gpu_time.as_micros() as u32);

            let data_transfer_size = (batch_size + 1) * poly_length * 32;
            let data_transfer_size = data_transfer_size as f64 / 1024.0 / 1024.0;
            println!(
                "[k = {}] cpu_time: {:?}, gpu_time: {:?}, speedup: {}, data_transfer_size: {} MB",
                log_n,
                cpu_time,
                gpu_time,
                cpu_micros / gpu_micros,
                data_transfer_size
            );
        }
    }
}
