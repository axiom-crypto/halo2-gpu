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

// The canonical proving/verifying key structs are the halo2-axiom CPU types:
// one source-of-truth struct family, byte-identical serialization for CPU and
// GPU by construction. The GPU crate keeps its own forked composing types,
// renamed `Gpu*` (`GpuConstraintSystem`/`GpuExpression`/`GpuGate`/`GpuColumn`/…,
// exported by `pub use circuit::*` above), which the GPU prover/verifier operate
// on after a cheap host-only rebuild (`GpuProvingKey`/`GpuVerifyingKey::from_host`)
// via the `From<&canonical ConstraintSystem> for GpuConstraintSystem` bridge.
pub use halo2_axiom::plonk::{ProvingKey, VerifyingKey};

// Canonical frontend (the ESCALATE re-export). Consumers (`halo2-base`,
// `snark-verifier`, `openvm`) and the GPU crate's own keygen/prover synthesis
// resolve these names to the canonical halo2-axiom types, so `impl Circuit` and
// `configure(&mut ConstraintSystem)` are canonical end-to-end. The `Gpu*` forks
// (re-exported above) remain the backend's working types, rebuilt from canonical
// via the `From` bridge.
pub use halo2_axiom::plonk::{
    Advice, AdviceQuery, Any, Assigned, Assignment, Challenge, Circuit, Column, ColumnType,
    Constraint, ConstraintSystem, Constraints, Error, Expression, Fixed, FirstPhase, FixedQuery,
    FloorPlanner, Gate, Instance, InstanceQuery, Phase, SecondPhase, Selector, TableColumn,
    ThirdPhase, VirtualCell, VirtualCells,
};

/// GPU-side verifying key. NOT serialized — the canonical, serialized
/// verifying key is [`VerifyingKey`] (halo2-axiom). `GpuVerifyingKey` holds the
/// GPU-crate forks (`cs`/`domain`/`permutation`) that the GPU verifier's
/// inherent methods (`commit`/`evaluate`/`queries`) attach to, rebuilt from a
/// canonical [`VerifyingKey`] via [`GpuVerifyingKey::from_host`].
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
    /// Rebuilds the GPU verifying key from a canonical halo2-axiom
    /// [`VerifyingKey`]. PURE HOST: a `ConstraintSystem` field-copy, a
    /// reconstructed `EvaluationDomain::new(j, k)`, and a `Vec<C>` clone of the
    /// permutation/fixed commitments — no device traffic, no kernel launch.
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

/// GPU-side proving key. Wraps the canonical halo2-axiom [`ProvingKey`]
/// (`inner`: serde source-of-truth + host-fallback polys + vk metadata) and
/// holds the GPU-crate composing forks (`cs`/`domain`/`ev`) rebuilt from it,
/// plus the lazy device mirrors of the pk polynomials.
///
/// `inner` is a [`Cow`] so the wrapper can be built either by taking ownership
/// of a canonical key ([`from_host`](Self::from_host), used by serde/tests) or
/// by *borrowing* one ([`from_host_ref`](Self::from_host_ref), used by
/// [`create_proof`] on the per-proof hot path). The borrowed path performs ZERO
/// host-poly clones — only the cheap `cs`/`domain`/`ev` rebuild — so consumers
/// holding a canonical `&ProvingKey` pay no per-proof deep copy.
#[derive(Debug)]
pub struct GpuProvingKey<'a, C: CurveAffine> {
    /// Canonical halo2-axiom proving key: serialization, host-fallback polys
    /// (`l0`/`fixed_values`/`fixed_polys`/`permutation`), and vk metadata
    /// (`fixed_commitments`/`transcript_repr`/selectors). Owned or borrowed via
    /// [`Cow`]; auto-derefs to `&ProvingKey<C>` at every use site.
    inner: Cow<'a, ProvingKey<C>>,
    /// GPU `ConstraintSystem`, rebuilt from `inner.get_vk().cs()`. Holds the
    /// GPU `lookup`/`permutation` Arguments whose inherent `commit_permuted`/
    /// `commit` methods the prover calls.
    cs: GpuConstraintSystem<C::Scalar>,
    /// GPU evaluation domain, reconstructed via `EvaluationDomain::new(j, k)`.
    domain: EvaluationDomain<C::Scalar>,
    /// GPU quotient evaluator, `Evaluator::new(&self.cs)`.
    ev: Evaluator<C>,
    // (the `cs` field below is the GPU fork `GpuConstraintSystem`, rebuilt from
    // the canonical cs via the `From` bridge — see `from_cow`)
    /// Cached maximum degree of `cs` (constant after construction). Avoids
    /// rescanning all gates/lookups via `cs.degree()` on the hot proof path
    /// (the permutation commit and the quotient `EvaluatorVkView`).
    cs_degree: usize,
    /// Cached `transcript_repr` of the canonical vk, copied at `from_host`.
    /// Lets `hash_into` write the vk representative without re-borrowing the
    /// canonical vk (whose `get_vk()` accessor is `FromUniformBytes`-bounded),
    /// so the prover's `create_proof` keeps its original (looser) scalar bound.
    transcript_repr: C::Scalar,
    /// Device-resident Coeff mirror of `inner.fixed_polys()`. Lazy; empty after
    /// `Clone`, matching `ParamsKZG`'s `OnceCell` invalidation contract.
    fixed_polys_device: OnceCell<Vec<Polynomial<C::Scalar, Coeff, crate::poly::Device>>>,
    /// Device-resident Lagrange mirror of `inner.fixed_values()`. Lazy.
    fixed_values_device: OnceCell<Vec<Polynomial<C::Scalar, LagrangeCoeff, crate::poly::Device>>>,
    /// Device-resident Coeff mirror of `inner.permutation().polys()`. Lazy.
    permutation_polys_device: OnceCell<Vec<Polynomial<C::Scalar, Coeff, crate::poly::Device>>>,
    /// Device-resident Coeff mirror of `inner.l0()`. Lazy.
    l0_device: OnceCell<Polynomial<C::Scalar, Coeff, crate::poly::Device>>,
    /// Device-resident Coeff mirror of `inner.l_last()`. Lazy.
    l_last_device: OnceCell<Polynomial<C::Scalar, Coeff, crate::poly::Device>>,
    /// Device-resident Coeff mirror of `inner.l_active_row()`. Lazy.
    l_active_row_device: OnceCell<Polynomial<C::Scalar, Coeff, crate::poly::Device>>,
    /// Device-resident Lagrange mirror of `inner.permutation().permutations()`
    /// (the σ-columns). Distinct from `permutation_polys_device` (Coeff form);
    /// consumed by `permutation::Argument::commit`'s `permutation_product_device`.
    /// Lazy; empty after `Clone`.
    permutation_lagrange_device:
        OnceCell<Vec<Polynomial<C::Scalar, LagrangeCoeff, crate::poly::Device>>>,
}

impl<'a, C: CurveAffine> Clone for GpuProvingKey<'a, C> {
    /// Cloning clones the canonical `inner` and the rebuilt GPU composing types
    /// but EMPTIES the device `OnceCell` mirrors — a clone regenerates them
    /// lazily on first use (matches `ParamsKZG`'s OnceCell pattern). `inner`'s
    /// own device mirrors are likewise empty after its `Clone`.
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
    /// Wraps a canonical halo2-axiom [`ProvingKey`] (taking ownership) for GPU
    /// proving. Used by the serde/deserialize path and tests. See
    /// [`from_cow`](Self::from_cow) for the (host-only) rebuild details.
    pub fn from_host(inner: ProvingKey<C>) -> GpuProvingKey<'static, C> {
        GpuProvingKey::from_cow(Cow::Owned(inner))
    }

    /// Wraps a canonical halo2-axiom [`ProvingKey`] by *borrowing* it for GPU
    /// proving. This is the per-proof hot path: it performs NO clone of the host
    /// proving-key polynomials — only the cheap `cs`/`domain`/`ev` rebuild —
    /// letting [`create_proof`] accept a borrowed canonical `&ProvingKey` from a
    /// consumer and avoid a per-proof deep copy. Device mirrors still populate
    /// lazily from the borrowed host polys, exactly as in the owned path.
    pub fn from_host_ref(inner: &'a ProvingKey<C>) -> Self {
        GpuProvingKey::from_cow(Cow::Borrowed(inner))
    }

    /// Shared constructor for the owned/borrowed paths.
    ///
    /// PURE HOST rebuild — performs ZERO device traffic and ZERO kernel
    /// launches: a `ConstraintSystem` field-copy from `inner.get_vk().cs()`
    /// (the equivalence-critical `Expression`/`Argument` map), a reconstructed
    /// `EvaluationDomain::new(j, k)`, and `Evaluator::new(&cs)`. Device mirrors
    /// start empty and populate lazily at first prove with the existing VRAM
    /// gate + host fallback — same H2D count and trigger points as before.
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
    /// Hashes the canonical verifying-key representative into a transcript.
    /// Reads the cached `transcript_repr` scalar and writes it via the GPU
    /// transcript trait (the transcript trait is a GPU fork, so we cannot
    /// delegate to the canonical vk's own `hash_into`). Kept off the
    /// `FromUniformBytes`-bounded block so `create_proof` keeps its scalar bound.
    pub fn hash_into<E: EncodedChallenge<C>, T: Transcript<C, E>>(
        &self,
        transcript: &mut T,
    ) -> io::Result<()> {
        transcript.common_scalar(self.transcript_repr)?;
        Ok(())
    }

    /// Hashes the permutation σ-poly evaluations at `x` into the transcript.
    /// (Was `permutation::ProvingKey::evaluate`; reads `inner`'s host polys.)
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

    /// Lazy device mirror of `inner.fixed_polys()` (Coeff form). Returns `None`
    /// if VRAM-gated out or H2D fails — the host-arm fallback then handles the
    /// upload per-rotation-set.
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
    /// σ-columns). Distinct from `permutation_polys_device` (the Coeff mirror);
    /// consumed by `permutation::Argument::commit`.
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

// Serialization delegates entirely to the canonical halo2-axiom `ProvingKey`,
// so the bytes are CPU/GPU-identical by construction. The serde domain is
// therefore halo2-axiom's (`halo2_axiom::SerdeFormat` + `halo2_axiom::helpers`
// serde traits + a `halo2_axiom::plonk::Circuit` for cs reconstruction on read);
// reading reproduces today's lazy behavior — device mirrors start empty.
impl<'a, C> GpuProvingKey<'a, C>
where
    // The gpu serde traits are blanket-impl'd over `CurveAffine + SerdeObject`,
    // which also satisfies halo2-axiom's identical blanket impls — so the
    // delegated `inner.write`/`inner.read` calls (bounded on halo2-axiom's
    // serde traits) resolve. halo2-axiom's `helpers` module is private and not
    // nameable here, hence bounding on the gpu-side traits.
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

    /// Reads a canonical halo2-axiom [`ProvingKey`] and wraps it via
    /// [`GpuProvingKey::from_host`]; the device mirrors start empty (lazy),
    /// reproducing today's behavior.
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

/// Helper for the PK device-mirror lazy initializers. Centralizes the VRAM
/// gate, the per-poly H2D loop, and the `perf_h2d!` byte-trace so the accessor
/// methods stay symmetric and the audit trail surfaces consistently across mirrors.
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
