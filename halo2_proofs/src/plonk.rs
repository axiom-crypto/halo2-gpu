//! This module provides an implementation of a variant of (Turbo)[PLONK][plonk]
//! that is designed specifically for the polynomial commitment scheme described
//! in the [Halo][halo] paper.
//!
//! [halo]: https://eprint.iacr.org/2019/1021
//! [plonk]: https://eprint.iacr.org/2019/953
use group::ff::FromUniformBytes;

use crate::arithmetic::CurveAffine;
use crate::cuda::utils::{query_device_free_bytes_for_chunking, HALO2_GPU_CTX};
use crate::helpers::{SerdeCurveAffine, SerdePrimeField};
use crate::poly::{Coeff, DevicePolyExt, EvaluationDomain, HostPolyExt, LagrangeCoeff, Polynomial};
use crate::transcript::{ChallengeScalar, EncodedChallenge, Transcript, TranscriptWrite};
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

use std::borrow::Cow;
use std::io;

// Canonical proving/verifying keys; the GPU prover/verifier rebuild their
// `Gpu*` forks from these. See `ARCHITECTURE.md`.
pub use halo2_axiom::plonk::{ProvingKey, VerifyingKey};

// Canonical synthesis frontend. See `ARCHITECTURE.md`.
pub use halo2_axiom::plonk::{
    Advice, AdviceQuery, Any, Assigned, Assignment, Challenge, Circuit, Column, ColumnType,
    Constraint, ConstraintSystem, Constraints, Error, Expression, FirstPhase, Fixed, FixedQuery,
    FloorPlanner, Gate, Instance, InstanceQuery, Phase, SecondPhase, Selector, TableColumn,
    ThirdPhase, VirtualCell, VirtualCells,
};

/// GPU-side verifying key (not serialized; the canonical one is
/// [`VerifyingKey`]). Holds the GPU-crate forks the verifier attaches to,
/// rebuilt via [`GpuVerifyingKey::from_host`].
#[derive(Clone, Debug)]
pub struct GpuVerifyingKey<C: CurveAffine> {
    pub(crate) domain: EvaluationDomain<C::Scalar>,
    pub(crate) fixed_commitments: Vec<C>,
    pub(crate) permutation: permutation::VerifyingKey<C>,
    pub(crate) cs: GpuConstraintSystem<C::Scalar>,
    /// Cached maximum degree of `cs` (which doesn't change after construction).
    pub(crate) cs_degree: usize,
    /// The representative of this `VerifyingKey` in transcripts.
    pub(crate) transcript_repr: C::Scalar,
}

impl<C: CurveAffine> GpuVerifyingKey<C>
where
    C::Scalar: FromUniformBytes<64>,
{
    /// Rebuilds the GPU verifying key from a canonical [`VerifyingKey`].
    /// Pure host: no device traffic, no kernel launch.
    pub fn from_host(vk: &VerifyingKey<C>) -> Self {
        let cs = GpuConstraintSystem::from(vk.cs());
        let cs_degree = cs.degree();
        let hdomain = vk.get_domain();
        let domain =
            EvaluationDomain::new(hdomain.get_quotient_poly_degree() as u32 + 1, hdomain.k());
        let permutation = permutation::VerifyingKey {
            commitments: vk.permutation().commitments().clone(),
        };
        GpuVerifyingKey {
            domain,
            fixed_commitments: vk.fixed_commitments().clone(),
            permutation,
            cs,
            cs_degree,
            transcript_repr: vk.transcript_repr(),
        }
    }

    /// Hashes the (canonical) verifying-key representative into a transcript.
    pub fn hash_into<E: EncodedChallenge<C>, T: Transcript<C, E>>(
        &self,
        transcript: &mut T,
    ) -> io::Result<()> {
        transcript.common_scalar(self.transcript_repr)?;
        Ok(())
    }
}

/// GPU-side proving key. Wraps the canonical [`ProvingKey`] (`inner`) plus the
/// GPU-crate composing forks (`cs`/`domain`/`ev`) rebuilt from it and the lazy
/// device mirrors of the pk polynomials.
///
/// `inner` is a [`Cow`] so the wrapper can take ownership
/// ([`from_host`](Self::from_host)) or *borrow*
/// ([`from_host_ref`](Self::from_host_ref)) a canonical key. The borrowed path
/// clones no host polys — only the cheap `cs`/`domain`/`ev` rebuild — so the
/// per-proof [`create_proof`] hot path pays no deep copy.
#[derive(Debug)]
pub struct GpuProvingKey<'a, C: CurveAffine> {
    /// Canonical proving key: serialization, host-fallback polys, and vk
    /// metadata. Owned or borrowed; derefs to `&ProvingKey<C>`.
    inner: Cow<'a, ProvingKey<C>>,
    /// GPU `ConstraintSystem`, holding the lookup/permutation Arguments the
    /// prover's `commit`/`commit_permuted` attach to.
    cs: GpuConstraintSystem<C::Scalar>,
    domain: EvaluationDomain<C::Scalar>,
    /// GPU quotient evaluator.
    ev: Evaluator<C>,
    /// Cached `cs.degree()`, constant after construction; avoids rescanning
    /// gates/lookups on the hot proof path.
    cs_degree: usize,
    /// Cached vk `transcript_repr`. Lets `hash_into` run without re-borrowing
    /// the `FromUniformBytes`-bounded vk, keeping `create_proof`'s looser bound.
    transcript_repr: C::Scalar,
    /// Lazy device mirrors of the pk polynomials. Emptied on `Clone` and
    /// regenerated on first use (matches `ParamsKZG`'s `OnceCell` contract).
    fixed_polys_device: OnceCell<Vec<Polynomial<C::Scalar, Coeff, crate::poly::Device>>>,
    fixed_values_device: OnceCell<Vec<Polynomial<C::Scalar, LagrangeCoeff, crate::poly::Device>>>,
    permutation_polys_device: OnceCell<Vec<Polynomial<C::Scalar, Coeff, crate::poly::Device>>>,
    l0_device: OnceCell<Polynomial<C::Scalar, Coeff, crate::poly::Device>>,
    l_last_device: OnceCell<Polynomial<C::Scalar, Coeff, crate::poly::Device>>,
    l_active_row_device: OnceCell<Polynomial<C::Scalar, Coeff, crate::poly::Device>>,
    /// Lagrange σ-columns, distinct from `permutation_polys_device` (Coeff form).
    permutation_lagrange_device:
        OnceCell<Vec<Polynomial<C::Scalar, LagrangeCoeff, crate::poly::Device>>>,
}

impl<'a, C: CurveAffine> Clone for GpuProvingKey<'a, C> {
    /// Clones `inner` and the GPU composing types but empties the device
    /// `OnceCell` mirrors, so the clone regenerates them lazily on first use.
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            cs: self.cs.clone(),
            domain: self.domain.clone(),
            ev: self.ev.clone(),
            cs_degree: self.cs_degree,
            transcript_repr: self.transcript_repr,
            fixed_polys_device: OnceCell::new(),
            fixed_values_device: OnceCell::new(),
            permutation_polys_device: OnceCell::new(),
            l0_device: OnceCell::new(),
            l_last_device: OnceCell::new(),
            l_active_row_device: OnceCell::new(),
            permutation_lagrange_device: OnceCell::new(),
        }
    }
}

impl<'a, C: CurveAffine> GpuProvingKey<'a, C>
where
    C::Scalar: FromUniformBytes<64>,
{
    /// Wraps a canonical [`ProvingKey`] by ownership. Used by the serde path
    /// and tests.
    pub fn from_host(inner: ProvingKey<C>) -> GpuProvingKey<'static, C> {
        GpuProvingKey::from_cow(Cow::Owned(inner))
    }

    /// Wraps a canonical [`ProvingKey`] by *borrowing* it: the per-proof hot
    /// path. Clones no host polys, so [`create_proof`] takes a `&ProvingKey`
    /// without a per-proof deep copy.
    pub fn from_host_ref(inner: &'a ProvingKey<C>) -> Self {
        GpuProvingKey::from_cow(Cow::Borrowed(inner))
    }

    /// Shared constructor for the owned/borrowed paths. Pure host rebuild: no
    /// device traffic, no kernel launch. Device mirrors start empty and
    /// populate lazily at first prove.
    fn from_cow(inner: Cow<'a, ProvingKey<C>>) -> Self {
        let cs = GpuConstraintSystem::from(inner.get_vk().cs());
        let hdomain = inner.get_vk().get_domain();
        let domain =
            EvaluationDomain::new(hdomain.get_quotient_poly_degree() as u32 + 1, hdomain.k());
        let ev = Evaluator::new(&cs);
        let cs_degree = cs.degree();
        let transcript_repr = inner.get_vk().transcript_repr();
        GpuProvingKey {
            inner,
            cs,
            domain,
            ev,
            cs_degree,
            transcript_repr,
            fixed_polys_device: OnceCell::new(),
            fixed_values_device: OnceCell::new(),
            permutation_polys_device: OnceCell::new(),
            l0_device: OnceCell::new(),
            l_last_device: OnceCell::new(),
            l_active_row_device: OnceCell::new(),
            permutation_lagrange_device: OnceCell::new(),
        }
    }

    /// Get the underlying canonical [`VerifyingKey`].
    pub fn get_vk(&self) -> &VerifyingKey<C> {
        self.inner.get_vk()
    }
}

impl<'a, C: CurveAffine> GpuProvingKey<'a, C> {
    /// Hashes the verifying-key representative into a transcript via the GPU
    /// transcript trait. Kept off the `FromUniformBytes`-bounded block so
    /// `create_proof` keeps its looser scalar bound.
    pub fn hash_into<E: EncodedChallenge<C>, T: Transcript<C, E>>(
        &self,
        transcript: &mut T,
    ) -> io::Result<()> {
        transcript.common_scalar(self.transcript_repr)?;
        Ok(())
    }

    /// Writes the permutation σ-poly evaluations at `x` into the transcript.
    pub(in crate::plonk) fn evaluate_permutation<
        E: EncodedChallenge<C>,
        T: TranscriptWrite<C, E>,
    >(
        &self,
        x: ChallengeX<C>,
        transcript: &mut T,
    ) -> Result<(), GpuError> {
        crate::perf_section!("permutation_pk.evaluate");
        for eval in self
            .inner
            .permutation()
            .polys()
            .iter()
            .map(|poly| crate::arithmetic::eval_polynomial(poly, *x))
        {
            transcript.write_scalar(eval)?;
        }
        Ok(())
    }

    /// Lazy device mirror of `inner.fixed_polys()` (Coeff form). `None` if
    /// VRAM-gated out or H2D fails, leaving the host-arm fallback to upload.
    pub(crate) fn fixed_polys_device(
        &self,
    ) -> Option<&[Polynomial<C::Scalar, Coeff, crate::poly::Device>]> {
        if let Some(v) = self.fixed_polys_device.get() {
            return Some(v.as_slice());
        }
        try_init_pk_device_mirror::<C, Coeff>(
            self.inner.fixed_polys(),
            "pk.fixed_polys_device.init",
            &self.fixed_polys_device,
        )
        .map(|v| v.as_slice())
    }

    /// Lazy device mirror of `inner.fixed_values()` (Lagrange form). Returns
    /// `None` if VRAM-gated out.
    pub(crate) fn fixed_values_device(
        &self,
    ) -> Option<&[Polynomial<C::Scalar, LagrangeCoeff, crate::poly::Device>]> {
        if let Some(v) = self.fixed_values_device.get() {
            return Some(v.as_slice());
        }
        try_init_pk_device_mirror::<C, LagrangeCoeff>(
            self.inner.fixed_values(),
            "pk.fixed_values_device.init",
            &self.fixed_values_device,
        )
        .map(|v| v.as_slice())
    }

    /// Lazy device mirror of `inner.permutation().polys()` (Coeff form).
    pub(crate) fn permutation_polys_device(
        &self,
    ) -> Option<&[Polynomial<C::Scalar, Coeff, crate::poly::Device>]> {
        if let Some(v) = self.permutation_polys_device.get() {
            return Some(v.as_slice());
        }
        try_init_pk_device_mirror::<C, Coeff>(
            self.inner.permutation().polys(),
            "pk.permutation_polys_device.init",
            &self.permutation_polys_device,
        )
        .map(|v| v.as_slice())
    }

    /// Lazy device mirror of `inner.permutation().permutations()` (Lagrange
    /// σ-columns); distinct from `permutation_polys_device` (Coeff form).
    pub(crate) fn permutation_lagrange_device(
        &self,
    ) -> Option<&[Polynomial<C::Scalar, LagrangeCoeff, crate::poly::Device>]> {
        if let Some(v) = self.permutation_lagrange_device.get() {
            return Some(v.as_slice());
        }
        try_init_pk_device_mirror::<C, LagrangeCoeff>(
            self.inner.permutation().permutations(),
            "pk.permutation.permutations_device.init",
            &self.permutation_lagrange_device,
        )
        .map(|v| v.as_slice())
    }

    /// Lazy device mirror of `inner.l0()`. Returns `None` if VRAM-gated out.
    pub(crate) fn l0_device(&self) -> Option<&Polynomial<C::Scalar, Coeff, crate::poly::Device>> {
        if let Some(v) = self.l0_device.get() {
            return Some(v);
        }
        try_init_pk_device_mirror_one::<C, Coeff>(
            self.inner.l0(),
            "pk.l0_device.init",
            &self.l0_device,
        )
    }

    /// Lazy device mirror of `inner.l_last()`. Same contract as `l0_device`.
    pub(crate) fn l_last_device(
        &self,
    ) -> Option<&Polynomial<C::Scalar, Coeff, crate::poly::Device>> {
        if let Some(v) = self.l_last_device.get() {
            return Some(v);
        }
        try_init_pk_device_mirror_one::<C, Coeff>(
            self.inner.l_last(),
            "pk.l_last_device.init",
            &self.l_last_device,
        )
    }

    /// Lazy device mirror of `inner.l_active_row()`. Same contract as `l0_device`.
    pub(crate) fn l_active_row_device(
        &self,
    ) -> Option<&Polynomial<C::Scalar, Coeff, crate::poly::Device>> {
        if let Some(v) = self.l_active_row_device.get() {
            return Some(v);
        }
        try_init_pk_device_mirror_one::<C, Coeff>(
            self.inner.l_active_row(),
            "pk.l_active_row_device.init",
            &self.l_active_row_device,
        )
    }
}

// Serialization delegates entirely to the canonical `ProvingKey`, so the bytes
// are CPU/GPU-identical.
impl<'a, C> GpuProvingKey<'a, C>
where
    // `SerdeCurveAffine`/`SerdePrimeField` are the canonical halo2-axiom serde
    // traits (re-exported by `crate::helpers`); this bound is what the delegated
    // `inner.write`/`inner.read` require.
    C: SerdeCurveAffine,
    C::Scalar: SerdePrimeField + FromUniformBytes<64>,
{
    /// Writes the proving key to a buffer (canonical halo2-axiom serialization).
    pub fn write<W: io::Write>(
        &self,
        writer: &mut W,
        format: halo2_axiom::SerdeFormat,
    ) -> io::Result<()> {
        self.inner.write(writer, format)
    }

    /// Reads a canonical [`ProvingKey`] and wraps it via
    /// [`GpuProvingKey::from_host`]; device mirrors start empty (lazy).
    pub fn read<R: io::Read, ConcreteCircuit: halo2_axiom::plonk::Circuit<C::Scalar>>(
        reader: &mut R,
        format: halo2_axiom::SerdeFormat,
        #[cfg(feature = "circuit-params")] params: ConcreteCircuit::Params,
    ) -> io::Result<Self> {
        let inner = ProvingKey::<C>::read::<R, ConcreteCircuit>(
            reader,
            format,
            #[cfg(feature = "circuit-params")]
            params,
        )?;
        Ok(Self::from_host(inner))
    }

    /// Writes the proving key to a vector of bytes using [`Self::write`].
    pub fn to_bytes(&self, format: halo2_axiom::SerdeFormat) -> Vec<u8> {
        self.inner.to_bytes(format)
    }

    /// Reads a proving key from a slice of bytes using [`Self::read`].
    pub fn from_bytes<ConcreteCircuit: halo2_axiom::plonk::Circuit<C::Scalar>>(
        bytes: &[u8],
        format: halo2_axiom::SerdeFormat,
        #[cfg(feature = "circuit-params")] params: ConcreteCircuit::Params,
    ) -> io::Result<Self> {
        let inner = ProvingKey::<C>::from_bytes::<ConcreteCircuit>(
            bytes,
            format,
            #[cfg(feature = "circuit-params")]
            params,
        )?;
        Ok(Self::from_host(inner))
    }
}

/// Shared lazy initializer for the slice-valued PK device mirrors: VRAM gate,
/// per-poly H2D loop, and `perf_h2d!` byte-trace.
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
    // Require >= 2x the mirror size free, leaving room for a transient pool
    // peak co-resident; fail open (return None for the host fallback) rather
    // than panic.
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
    // `perf_h2d!` requires a literal label; emit directly to use the runtime
    // `perf_tag` on the same "halo2_perf" target.
    crate::cuda::utils::__perf_reexports::info!(
        target: "halo2_perf",
        kind = "h2d",
        label = perf_tag,
        bytes = total_bytes as u64,
    );
    // If another thread populated the cell first, keep theirs and drop ours.
    let _ = cell.set(mirror);
    cell.get()
}

/// Single-polynomial variant of [`try_init_pk_device_mirror`] for the L-polys
/// (`l0`, `l_last`, `l_active_row`).
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
