//! This module provides an implementation of a variant of (Turbo)[PLONK][plonk]
//! that is designed specifically for the polynomial commitment scheme described
//! in the [Halo][halo] paper.
//!
//! [halo]: https://eprint.iacr.org/2019/1021
//! [plonk]: https://eprint.iacr.org/2019/953
use blake2b_simd::Params as Blake2bParams;
use group::ff::{Field, FromUniformBytes, PrimeField};

use crate::arithmetic::CurveAffine;
use crate::cuda::utils::{query_device_free_bytes_for_chunking, HALO2_GPU_CTX};
use crate::helpers::{
    polynomial_slice_byte_length, read_polynomial_vec, write_polynomial_slice, SerdeCurveAffine,
    SerdePrimeField,
};
use crate::poly::{
    Coeff, DevicePolyExt, EvaluationDomain, HostPolyExt, LagrangeCoeff, PinnedEvaluationDomain,
    PolyIo, Polynomial,
};
use crate::transcript::{ChallengeScalar, EncodedChallenge, Transcript};
use crate::SerdeFormat;
use once_cell::sync::OnceCell;
use openvm_cuda_common::copy::MemCopyH2D;

mod assigned;
mod circuit;
mod error;
pub(crate) mod evaluation;
mod keygen;
mod logging;
pub(crate) mod lookup;
pub mod permutation;
mod vanishing;

mod prover;
mod verifier;

pub use assigned::*;
pub use circuit::*;
pub use error::*;
pub use keygen::*;
pub use prover::*;
pub use verifier::*;

use evaluation::Evaluator;

use std::io;

/// This is a verifying key which allows for the verification of proofs for a
/// particular circuit.
#[derive(Clone, Debug)]
pub struct VerifyingKey<C: CurveAffine> {
    domain: EvaluationDomain<C::Scalar>,
    fixed_commitments: Vec<C>,
    permutation: permutation::VerifyingKey<C>,
    cs: ConstraintSystem<C::Scalar>,
    /// Cached maximum degree of `cs` (which doesn't change after construction).
    cs_degree: usize,
    /// The representative of this `VerifyingKey` in transcripts.
    transcript_repr: C::Scalar,
    selectors: Vec<Vec<bool>>,
    /// Whether selector compression is turned on or not.
    compress_selectors: bool,
}

impl<C: SerdeCurveAffine> VerifyingKey<C>
where
    C::Scalar: SerdePrimeField + FromUniformBytes<64>, // the FromUniformBytes<64> should not be necessary: currently serialization always stores a Blake2b hash of verifying key; this should be removed
{
    /// Writes a verifying key to a buffer.
    ///
    /// Writes a curve element according to `format`:
    /// - `Processed`: Writes a compressed curve element with coordinates in standard form.
    ///   Writes a field element in standard form, with endianness specified by the
    ///   `PrimeField` implementation.
    /// - Otherwise: Writes an uncompressed curve element with coordinates in Montgomery form
    ///   Writes a field element into raw bytes in its internal Montgomery representation,
    ///   WITHOUT performing the expensive Montgomery reduction.
    pub fn write<W: io::Write>(&self, writer: &mut W, format: SerdeFormat) -> io::Result<()> {
        // Version byte that will be checked on read.
        writer.write_all(&[0x02])?;
        writer.write_all(&self.domain.k().to_le_bytes())?;
        writer.write_all(&[self.compress_selectors as u8])?;
        writer.write_all(&(self.fixed_commitments.len() as u32).to_le_bytes())?;
        for commitment in &self.fixed_commitments {
            commitment.write(writer, format)?;
        }
        self.permutation.write(writer, format)?;

        if !self.compress_selectors {
            assert!(self.selectors.is_empty());
        }
        // write self.selectors
        for selector in &self.selectors {
            // since `selector` is filled with `bool`, we pack them 8 at a time into bytes and then write
            for bits in selector.chunks(8) {
                writer.write_all(&[crate::helpers::pack(bits)])?;
            }
        }
        Ok(())
    }

    /// Reads a verification key from a buffer.
    ///
    /// Reads a curve element from the buffer and parses it according to the `format`:
    /// - `Processed`: Reads a compressed curve element and decompresses it.
    ///   Reads a field element in standard form, with endianness specified by the
    ///   `PrimeField` implementation, and checks that the element is less than the modulus.
    /// - `RawBytes`: Reads an uncompressed curve element with coordinates in Montgomery form.
    ///   Checks that field elements are less than modulus, and then checks that the point is on the curve.
    /// - `RawBytesUnchecked`: Reads an uncompressed curve element with coordinates in Montgomery form;
    ///   does not perform any checks
    pub fn read<R: io::Read, ConcreteCircuit: Circuit<C::Scalar>>(
        reader: &mut R,
        format: SerdeFormat,
        #[cfg(feature = "circuit-params")] params: ConcreteCircuit::Params,
    ) -> io::Result<Self> {
        let mut version_byte = [0u8; 1];
        reader.read_exact(&mut version_byte)?;
        if 0x02 != version_byte[0] {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unexpected version byte",
            ));
        }
        let mut k = [0u8; 4];
        reader.read_exact(&mut k)?;
        let k = u32::from_le_bytes(k);
        let mut compress_selectors = [0u8; 1];
        reader.read_exact(&mut compress_selectors)?;
        if compress_selectors[0] != 0 && compress_selectors[0] != 1 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unexpected compress_selectors not boolean",
            ));
        }
        let compress_selectors = compress_selectors[0] == 1;
        let (domain, cs, _) = keygen::create_domain::<C, ConcreteCircuit>(
            k,
            #[cfg(feature = "circuit-params")]
            params,
        );
        let mut num_fixed_columns = [0u8; 4];
        reader.read_exact(&mut num_fixed_columns)?;
        let num_fixed_columns = u32::from_le_bytes(num_fixed_columns);

        let fixed_commitments: Vec<_> = (0..num_fixed_columns)
            .map(|_| C::read(reader, format))
            .collect::<io::Result<_>>()?;

        let permutation = permutation::VerifyingKey::read(reader, &cs.permutation, format)?;

        let (cs, selectors) = if compress_selectors {
            // read selectors
            let selectors: Vec<Vec<bool>> = vec![vec![false; 1 << k]; cs.num_selectors]
                .into_iter()
                .map(|mut selector| {
                    let mut selector_bytes = vec![0u8; selector.len().div_ceil(8)];
                    reader.read_exact(&mut selector_bytes)?;
                    for (bits, byte) in selector.chunks_mut(8).zip(selector_bytes) {
                        crate::helpers::unpack(byte, bits);
                    }
                    Ok(selector)
                })
                .collect::<io::Result<_>>()?;
            let (cs, _) = cs.compress_selectors(selectors.clone());
            (cs, selectors)
        } else {
            // we still need to replace selectors with fixed Expressions in `cs`
            let fake_selectors = vec![vec![false]; cs.num_selectors];
            let (cs, _) = cs.directly_convert_selectors_to_fixed(fake_selectors);
            (cs, vec![])
        };

        Ok(Self::from_parts(
            domain,
            fixed_commitments,
            permutation,
            cs,
            selectors,
            compress_selectors,
        ))
    }

    /// Writes a verifying key to a vector of bytes using [`Self::write`].
    pub fn to_bytes(&self, format: SerdeFormat) -> Vec<u8> {
        let mut bytes = Vec::<u8>::with_capacity(self.bytes_length());
        Self::write(self, &mut bytes, format).expect("Writing to vector should not fail");
        bytes
    }

    /// Reads a verification key from a slice of bytes using [`Self::read`].
    pub fn from_bytes<ConcreteCircuit: Circuit<C::Scalar>>(
        mut bytes: &[u8],
        format: SerdeFormat,
        #[cfg(feature = "circuit-params")] params: ConcreteCircuit::Params,
    ) -> io::Result<Self> {
        Self::read::<_, ConcreteCircuit>(
            &mut bytes,
            format,
            #[cfg(feature = "circuit-params")]
            params,
        )
    }
}

impl<C: CurveAffine> VerifyingKey<C> {
    fn bytes_length(&self) -> usize {
        8 + (self.fixed_commitments.len() * C::default().to_bytes().as_ref().len())
            + self.permutation.bytes_length()
            + self.selectors.len()
                * (self
                    .selectors
                    .first()
                    .map(|selector| selector.len().div_ceil(8))
                    .unwrap_or(0))
    }

    fn from_parts(
        domain: EvaluationDomain<C::Scalar>,
        fixed_commitments: Vec<C>,
        permutation: permutation::VerifyingKey<C>,
        cs: ConstraintSystem<C::Scalar>,
        selectors: Vec<Vec<bool>>,
        compress_selectors: bool,
    ) -> Self
    where
        C::Scalar: FromUniformBytes<64>,
    {
        // Compute cached values.
        let cs_degree = cs.degree();

        let mut vk = Self {
            domain,
            fixed_commitments,
            permutation,
            cs,
            cs_degree,
            // Temporary, this is not pinned.
            transcript_repr: C::Scalar::ZERO,
            selectors,
            compress_selectors,
        };

        let mut hasher = Blake2bParams::new()
            .hash_length(64)
            .personal(b"Halo2-Verify-Key")
            .to_state();

        let s = format!("{:?}", vk.pinned());

        hasher.update(&(s.len() as u64).to_le_bytes());
        hasher.update(s.as_bytes());

        // Hash in final Blake2bState
        vk.transcript_repr = C::Scalar::from_uniform_bytes(hasher.finalize().as_array());

        vk
    }

    /// Hashes a verification key into a transcript.
    pub fn hash_into<E: EncodedChallenge<C>, T: Transcript<C, E>>(
        &self,
        transcript: &mut T,
    ) -> io::Result<()> {
        transcript.common_scalar(self.transcript_repr)?;

        Ok(())
    }

    /// Obtains a pinned representation of this verification key that contains
    /// the minimal information necessary to reconstruct the verification key.
    pub fn pinned(&self) -> PinnedVerificationKey<'_, C> {
        PinnedVerificationKey {
            base_modulus: C::Base::MODULUS,
            scalar_modulus: C::Scalar::MODULUS,
            domain: self.domain.pinned(),
            fixed_commitments: &self.fixed_commitments,
            permutation: &self.permutation,
            cs: self.cs.pinned(),
        }
    }

    /// Returns commitments of fixed polynomials
    pub fn fixed_commitments(&self) -> &Vec<C> {
        &self.fixed_commitments
    }

    /// Returns `VerifyingKey` of permutation
    pub fn permutation(&self) -> &permutation::VerifyingKey<C> {
        &self.permutation
    }

    /// Returns `ConstraintSystem`
    pub fn cs(&self) -> &ConstraintSystem<C::Scalar> {
        &self.cs
    }

    /// Returns representative of this `VerifyingKey` in transcripts
    pub fn transcript_repr(&self) -> C::Scalar {
        self.transcript_repr
    }
}

/// Minimal representation of a verification key that can be used to identify
/// its active contents.
// Load-bearing: fields are read via the auto-derived `Debug` whose output is
// hashed into `vk.transcript_repr` via Blake2b at the `vk.pinned()` call site
// in `keygen_vk_custom`. Removing fields would change the transcript
// representation, breaking proof/verifier compatibility.
#[allow(dead_code)]
#[derive(Debug)]
pub struct PinnedVerificationKey<'a, C: CurveAffine> {
    base_modulus: &'static str,
    scalar_modulus: &'static str,
    domain: PinnedEvaluationDomain<'a, C::Scalar>,
    cs: PinnedConstraintSystem<'a, C::Scalar>,
    fixed_commitments: &'a Vec<C>,
    permutation: &'a permutation::VerifyingKey<C>,
}
/// This is a proving key which allows for the creation of proofs for a
/// particular circuit.
#[derive(Debug)]
pub struct ProvingKey<C: CurveAffine> {
    vk: VerifyingKey<C>,
    l0: Polynomial<C::Scalar, Coeff>,
    l_last: Polynomial<C::Scalar, Coeff>,
    l_active_row: Polynomial<C::Scalar, Coeff>,
    fixed_values: Vec<Polynomial<C::Scalar, LagrangeCoeff>>,
    fixed_polys: Vec<Polynomial<C::Scalar, Coeff>>,
    permutation: permutation::ProvingKey<C>,
    ev: Evaluator<C>,
    /// Device-resident Coeff mirror of `fixed_polys`. Lazy-populated on
    /// first `fixed_polys_device()` access. Empty after `Clone`, by
    /// design — matches `ParamsKZG`'s `OnceCell` invalidation contract.
    /// `ProverQuery::poly` reads through this mirror to keep PK
    /// polynomials device-resident through the multiopen pipeline.
    fixed_polys_device: OnceCell<Vec<Polynomial<C::Scalar, Coeff, crate::poly::Device>>>,
    /// Device-resident Lagrange mirror of `fixed_values`. Consumed by
    /// `ColumnPool::try_init_device`
    /// so the per-prove pool can skip re-uploading the fixed columns.
    /// Empty after `Clone`.
    fixed_values_device: OnceCell<Vec<Polynomial<C::Scalar, LagrangeCoeff, crate::poly::Device>>>,
    /// Device-resident Coeff mirror of `permutation.polys`.
    /// Lazy-populated; empty after `Clone`. Mirrors the lifecycle of
    /// `fixed_polys_device`.
    permutation_polys_device: OnceCell<Vec<Polynomial<C::Scalar, Coeff, crate::poly::Device>>>,
    /// Device-resident Coeff mirror of `l0`. Lazy-populated on first
    /// `l0_device()` access; empty after `Clone`. Borrowed by
    /// `evaluate_h_device` so the per-prove cosetFFT consumes the L-poly
    /// device pointer directly instead of paying an H2D each call.
    l0_device: OnceCell<Polynomial<C::Scalar, Coeff, crate::poly::Device>>,
    /// Device-resident Coeff mirror of `l_last`. Same lifecycle as
    /// `l0_device`.
    l_last_device: OnceCell<Polynomial<C::Scalar, Coeff, crate::poly::Device>>,
    /// Device-resident Coeff mirror of `l_active_row`. Same lifecycle as
    /// `l0_device`.
    l_active_row_device: OnceCell<Polynomial<C::Scalar, Coeff, crate::poly::Device>>,
}

impl<C: CurveAffine> Clone for ProvingKey<C> {
    /// Cloning a ProvingKey clones the host polys but EMPTIES the Device
    /// `OnceCell` mirrors. A clone regenerates its device mirrors lazily
    /// on first use — matches `ParamsKZG`'s OnceCell pattern at
    /// `poly/kzg/commitment.rs:39-42, 461-476, 491-507`. This avoids
    /// action-at-a-distance Device-pointer aliasing under partial-PK
    /// serialization (e.g. `pk.to_bytes()` round-trips that re-construct
    /// host polys).
    fn clone(&self) -> Self {
        Self {
            vk: self.vk.clone(),
            l0: self.l0.clone(),
            l_last: self.l_last.clone(),
            l_active_row: self.l_active_row.clone(),
            fixed_values: self.fixed_values.clone(),
            fixed_polys: self.fixed_polys.clone(),
            permutation: self.permutation.clone(),
            ev: self.ev.clone(),
            fixed_polys_device: OnceCell::new(),
            fixed_values_device: OnceCell::new(),
            permutation_polys_device: OnceCell::new(),
            l0_device: OnceCell::new(),
            l_last_device: OnceCell::new(),
            l_active_row_device: OnceCell::new(),
        }
    }
}

impl<C: CurveAffine> ProvingKey<C>
where
    C::Scalar: FromUniformBytes<64>,
{
    /// Get the underlying [`VerifyingKey`].
    pub fn get_vk(&self) -> &VerifyingKey<C> {
        &self.vk
    }

    /// Gets the total number of bytes in the serialization of `self`
    fn bytes_length(&self) -> usize {
        let scalar_len = C::Scalar::default().to_repr().as_ref().len();
        self.vk.bytes_length()
            + 12
            + scalar_len * (self.l0.len() + self.l_last.len() + self.l_active_row.len())
            + polynomial_slice_byte_length(&self.fixed_values)
            + polynomial_slice_byte_length(&self.fixed_polys)
            + self.permutation.bytes_length()
    }
}

impl<C: CurveAffine> ProvingKey<C> {
    /// Lazy device mirror of `fixed_polys` (Coeff form). Returns `None`
    /// if VRAM-gated out or H2D fails — the host-arm fallback in
    /// `shplonk::prover` then handles the upload per-rotation-set.
    ///
    /// Gates eagerly on `query_device_free_bytes_for_chunking()` before
    /// any H2D. On error the cell is left empty so the caller can fall
    /// back transparently.
    pub(crate) fn fixed_polys_device(
        &self,
    ) -> Option<&[Polynomial<C::Scalar, Coeff, crate::poly::Device>]> {
        if let Some(v) = self.fixed_polys_device.get() {
            return Some(v.as_slice());
        }
        try_init_pk_device_mirror::<C, Coeff>(
            &self.fixed_polys,
            "pk.fixed_polys_device.init",
            &self.fixed_polys_device,
        )
        .map(|v| v.as_slice())
    }

    /// Lazy device mirror of `fixed_values` (Lagrange form). Returns `None`
    /// if VRAM-gated out. Borrowed by
    /// `ColumnPool::try_init_device`
    /// to drop the per-prove fixed-col H2D.
    pub(crate) fn fixed_values_device(
        &self,
    ) -> Option<&[Polynomial<C::Scalar, LagrangeCoeff, crate::poly::Device>]> {
        if let Some(v) = self.fixed_values_device.get() {
            return Some(v.as_slice());
        }
        try_init_pk_device_mirror::<C, LagrangeCoeff>(
            &self.fixed_values,
            "pk.fixed_values_device.init",
            &self.fixed_values_device,
        )
        .map(|v| v.as_slice())
    }

    /// Lazy device mirror of `permutation.polys` (Coeff form). Returns
    /// `None` if VRAM-gated out. Same F-1 path as `fixed_polys_device`.
    pub(crate) fn permutation_polys_device(
        &self,
    ) -> Option<&[Polynomial<C::Scalar, Coeff, crate::poly::Device>]> {
        if let Some(v) = self.permutation_polys_device.get() {
            return Some(v.as_slice());
        }
        try_init_pk_device_mirror::<C, Coeff>(
            self.permutation.polys(),
            "pk.permutation_polys_device.init",
            &self.permutation_polys_device,
        )
        .map(|v| v.as_slice())
    }

    /// Lazy device mirror of `l0`. Returns `None` if VRAM-gated out.
    /// Borrowed by `evaluate_h_device` so the per-prove cosetFFT consumes
    /// the L-poly device pointer instead of paying an H2D per call.
    pub(crate) fn l0_device(&self) -> Option<&Polynomial<C::Scalar, Coeff, crate::poly::Device>> {
        if let Some(v) = self.l0_device.get() {
            return Some(v);
        }
        try_init_pk_device_mirror_one::<C, Coeff>(&self.l0, "pk.l0_device.init", &self.l0_device)
    }

    /// Lazy device mirror of `l_last`. Same contract as `l0_device`.
    pub(crate) fn l_last_device(
        &self,
    ) -> Option<&Polynomial<C::Scalar, Coeff, crate::poly::Device>> {
        if let Some(v) = self.l_last_device.get() {
            return Some(v);
        }
        try_init_pk_device_mirror_one::<C, Coeff>(
            &self.l_last,
            "pk.l_last_device.init",
            &self.l_last_device,
        )
    }

    /// Lazy device mirror of `l_active_row`. Same contract as `l0_device`.
    pub(crate) fn l_active_row_device(
        &self,
    ) -> Option<&Polynomial<C::Scalar, Coeff, crate::poly::Device>> {
        if let Some(v) = self.l_active_row_device.get() {
            return Some(v);
        }
        try_init_pk_device_mirror_one::<C, Coeff>(
            &self.l_active_row,
            "pk.l_active_row_device.init",
            &self.l_active_row_device,
        )
    }
}

/// Helper for the three PK device-mirror lazy initializers. Centralizes
/// the VRAM gate + per-poly H2D loop + `perf_h2d!` byte-trace so the three
/// accessor methods on `ProvingKey` stay symmetric and the audit trail
/// surfaces consistently across mirrors.
fn try_init_pk_device_mirror<'pk, C: CurveAffine, B>(
    host: &[Polynomial<C::Scalar, B, crate::poly::Host>],
    perf_tag: &'static str,
    cell: &'pk OnceCell<Vec<Polynomial<C::Scalar, B, crate::poly::Device>>>,
) -> Option<&'pk Vec<Polynomial<C::Scalar, B, crate::poly::Device>>>
where
    B: 'static,
{
    let elem_bytes = std::mem::size_of::<C::Scalar>();
    let total_bytes: usize = host.iter().map(|p| p.len() * elem_bytes).sum();
    let free_bytes = query_device_free_bytes_for_chunking();
    // Conservative gate: require ≥ 2× the mirror size free (leaves room for
    // a transient ColumnPool / Sibling pool peak co-resident). Matches
    // ParamsKZG's pattern of "fail open" on `to_device_on` — we return
    // None rather than panic, letting the host-arm fallback engage.
    if free_bytes < total_bytes.saturating_mul(2) {
        tracing::warn!(
            target: "halo2_vram_fallback",
            site = "try_init_pk_device_mirror.headroom_gate",
            perf_tag,
            free_bytes = free_bytes as u64,
            needed_bytes = total_bytes as u64,
            "VRAM fallback fired: PK device mirror skipped (2× headroom gate failed); caller falls back to host arm"
        );
        return None;
    }
    let mut mirror: Vec<Polynomial<C::Scalar, B, crate::poly::Device>> =
        Vec::with_capacity(host.len());
    for poly in host {
        match poly.values().to_device_on(&HALO2_GPU_CTX) {
            Ok(d) => mirror.push(Polynomial::from_device(d)),
            Err(e) => {
                log::warn!("PK device mirror {}: H2D failed ({:?}); skip", perf_tag, e);
                return None;
            }
        }
    }
    // `perf_h2d!` requires a literal label, so emit the tracing event
    // directly with the runtime label string (kind="h2d", label=perf_tag,
    // bytes=total_bytes). Same target ("halo2_perf") as the macro.
    crate::cuda::utils::__perf_reexports::info!(
        target: "halo2_perf",
        kind = "h2d",
        label = perf_tag,
        bytes = total_bytes as u64,
    );
    // `set` returns Err if another thread populated the cell first; in
    // that case the other thread's mirror is the one we keep (drop ours).
    let _ = cell.set(mirror);
    cell.get()
}

/// Single-polynomial variant of [`try_init_pk_device_mirror`] for the
/// three L-polys (`l0`, `l_last`, `l_active_row`) that are stored as
/// individual `Polynomial` values rather than as a slice.
fn try_init_pk_device_mirror_one<'pk, C: CurveAffine, B>(
    host: &Polynomial<C::Scalar, B, crate::poly::Host>,
    perf_tag: &'static str,
    cell: &'pk OnceCell<Polynomial<C::Scalar, B, crate::poly::Device>>,
) -> Option<&'pk Polynomial<C::Scalar, B, crate::poly::Device>>
where
    B: 'static,
{
    let total_bytes: usize = host.len() * std::mem::size_of::<C::Scalar>();
    let free_bytes = query_device_free_bytes_for_chunking();
    if free_bytes < total_bytes {
        tracing::error!(
            target: "halo2_vram_fallback",
            site = "try_init_pk_device_mirror_one.headroom_gate",
            perf_tag,
            free_bytes = free_bytes as u64,
            needed_bytes = total_bytes as u64,
            "not enough vram"
        );
        return None;
    }
    let mirror = match host.to_device_on(&HALO2_GPU_CTX) {
        Ok(d) => d,
        Err(e) => {
            log::warn!("PK device mirror {}: H2D failed ({:?}); skip", perf_tag, e);
            return None;
        }
    };
    crate::cuda::utils::__perf_reexports::info!(
        target: "halo2_perf",
        kind = "h2d",
        label = perf_tag,
        bytes = total_bytes as u64,
    );
    let _ = cell.set(mirror);
    cell.get()
}

impl<C: SerdeCurveAffine> ProvingKey<C>
where
    C::Scalar: SerdePrimeField + FromUniformBytes<64>,
{
    /// Writes a proving key to a buffer.
    ///
    /// Writes a curve element according to `format`:
    /// - `Processed`: Writes a compressed curve element with coordinates in standard form.
    ///   Writes a field element in standard form, with endianness specified by the
    ///   `PrimeField` implementation.
    /// - Otherwise: Writes an uncompressed curve element with coordinates in Montgomery form
    ///   Writes a field element into raw bytes in its internal Montgomery representation,
    ///   WITHOUT performing the expensive Montgomery reduction.
    ///
    /// Does so by first writing the verifying key and then serializing the rest of the data (in the form of field polynomials)
    pub fn write<W: io::Write>(&self, writer: &mut W, format: SerdeFormat) -> io::Result<()> {
        self.vk.write(writer, format)?;
        self.l0.write_poly(writer, format);
        self.l_last.write_poly(writer, format);
        self.l_active_row.write_poly(writer, format);
        write_polynomial_slice(&self.fixed_values, writer, format);
        write_polynomial_slice(&self.fixed_polys, writer, format);
        self.permutation.write(writer, format);
        Ok(())
    }

    /// Reads a proving key from a buffer.
    /// Does so by reading verification key first, and then deserializing the rest of the file into the remaining proving key data.
    ///
    /// Reads a curve element from the buffer and parses it according to the `format`:
    /// - `Processed`: Reads a compressed curve element and decompresses it.
    ///   Reads a field element in standard form, with endianness specified by the
    ///   `PrimeField` implementation, and checks that the element is less than the modulus.
    /// - `RawBytes`: Reads an uncompressed curve element with coordinates in Montgomery form.
    ///   Checks that field elements are less than modulus, and then checks that the point is on the curve.
    /// - `RawBytesUnchecked`: Reads an uncompressed curve element with coordinates in Montgomery form;
    ///   does not perform any checks
    pub fn read<R: io::Read, ConcreteCircuit: Circuit<C::Scalar>>(
        reader: &mut R,
        format: SerdeFormat,
        #[cfg(feature = "circuit-params")] params: ConcreteCircuit::Params,
    ) -> io::Result<Self> {
        let vk = VerifyingKey::<C>::read::<R, ConcreteCircuit>(
            reader,
            format,
            #[cfg(feature = "circuit-params")]
            params,
        )?;
        let l0 = Polynomial::read_poly(reader, format);
        let l_last = Polynomial::read_poly(reader, format);
        let l_active_row = Polynomial::read_poly(reader, format);
        let fixed_values = read_polynomial_vec(reader, format);
        let fixed_polys = read_polynomial_vec(reader, format);
        let permutation = permutation::ProvingKey::read(reader, format);
        let ev = Evaluator::new(vk.cs());
        Ok(Self {
            vk,
            l0,
            l_last,
            l_active_row,
            fixed_values,
            fixed_polys,
            permutation,
            ev,
            fixed_polys_device: OnceCell::new(),
            fixed_values_device: OnceCell::new(),
            permutation_polys_device: OnceCell::new(),
            l0_device: OnceCell::new(),
            l_last_device: OnceCell::new(),
            l_active_row_device: OnceCell::new(),
        })
    }

    /// Writes a proving key to a vector of bytes using [`Self::write`].
    pub fn to_bytes(&self, format: SerdeFormat) -> Vec<u8> {
        let mut bytes = Vec::<u8>::with_capacity(self.bytes_length());
        Self::write(self, &mut bytes, format).expect("Writing to vector should not fail");
        bytes
    }

    /// Reads a proving key from a slice of bytes using [`Self::read`].
    pub fn from_bytes<ConcreteCircuit: Circuit<C::Scalar>>(
        mut bytes: &[u8],
        format: SerdeFormat,
        #[cfg(feature = "circuit-params")] params: ConcreteCircuit::Params,
    ) -> io::Result<Self> {
        Self::read::<_, ConcreteCircuit>(
            &mut bytes,
            format,
            #[cfg(feature = "circuit-params")]
            params,
        )
    }
}

impl<C: CurveAffine> VerifyingKey<C> {
    /// Get the underlying [`EvaluationDomain`].
    pub fn get_domain(&self) -> &EvaluationDomain<C::Scalar> {
        &self.domain
    }
}

#[derive(Clone, Copy, Debug)]
struct Theta;
type ChallengeTheta<F> = ChallengeScalar<F, Theta>;

#[derive(Clone, Copy, Debug)]
struct Beta;
type ChallengeBeta<F> = ChallengeScalar<F, Beta>;

#[derive(Clone, Copy, Debug)]
struct Gamma;
type ChallengeGamma<F> = ChallengeScalar<F, Gamma>;

#[derive(Clone, Copy, Debug)]
struct Y;
type ChallengeY<F> = ChallengeScalar<F, Y>;

#[derive(Clone, Copy, Debug)]
struct X;
type ChallengeX<F> = ChallengeScalar<F, X>;
