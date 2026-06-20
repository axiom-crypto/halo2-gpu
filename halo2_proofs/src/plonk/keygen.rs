//! GPU-local key generation.
//!
//! These functions take a **GPU** [`Params`](crate::poly::commitment::Params)
//! and a **GPU** [`Circuit`](crate::plonk::Circuit) and return the **canonical**
//! halo2-axiom [`ProvingKey`]/[`VerifyingKey`] (the serde source-of-truth). This
//! matches the signatures the patched `halo2-base`/`openvm` stack calls
//! (`keygen_pk2(params, circuit, _)`, `keygen_vk_custom(params, circuit, _)`,
//! `keygen_pk(params, vk, circuit)`), while keeping the GPU `Params`/`Circuit`
//! traits on the input side.
//!
//! # Phase 1 status (STUB)
//!
//! The bodies are intentionally `unimplemented!()` for the unified-pk Phase-1
//! spike: the goal of Phase 1 is to make the workspace gate type-check past
//! keygen *without* the not-yet-existing halo2-axiom `from_parts` constructors.
//! A diverging body type-checks as any return type, so the signatures alone
//! re-establish the GPU-typed keygen surface the consumer stack resolves
//! against.
//!
//! ## Phase 2 (real body — designed, not yet wired)
//!
//! The real bodies port the GPU-local keygen recovered at commit `b222e4d`
//! (`git show b222e4d:halo2_proofs/src/plonk/keygen.rs`), which is already
//! GPU-accelerated and must stay so — **no CPU-MSM regression**:
//!
//! * fixed-column commitments via **GPU MSM** `params.commit_lagrange(poly,
//!   Blind::default())` (one per fixed/selector poly);
//! * fixed/basis polynomials via **GPU FFT** `domain.lagrange_to_coeff[_many]`;
//! * permutation vk/pk via the existing GPU `permutation::keygen::Assembly`.
//!
//! The only new work versus `b222e4d` is *assembling the canonical key types*
//! from those GPU-computed pieces. Under the (compiler-confirmed) frontend
//! re-export, `ConcreteCircuit::configure(&mut canonical_cs)` builds the
//! canonical `ConstraintSystem` natively, so the only halo2-axiom additions
//! needed are the 4 `from_parts` constructors (`VerifyingKey::from_parts`
//! un-private, `ProvingKey::from_parts`, `permutation::VerifyingKey::from_commitments`,
//! `permutation::ProvingKey::from_parts`). Those land in Phase 2.

use group::ff::FromUniformBytes;

use super::{Circuit, Error, ProvingKey, VerifyingKey};
use crate::arithmetic::CurveAffine;
use crate::poly::commitment::Params;

/// Generate a [`VerifyingKey`] from an instance of [`Circuit`].
///
/// By default, selector compression is turned **off**.
pub fn keygen_vk<'params, C, P, ConcreteCircuit>(
    _params: &P,
    _circuit: &ConcreteCircuit,
) -> Result<VerifyingKey<C>, Error>
where
    C: CurveAffine,
    P: Params<'params, C> + Sync,
    ConcreteCircuit: Circuit<C::Scalar>,
    C::Scalar: FromUniformBytes<64>,
{
    unimplemented!("Phase-1 stub: GPU-local keygen_vk body lands in Phase 2 (b222e4d port assembling canonical keys via halo2-axiom from_parts)")
}

/// Generate a [`VerifyingKey`] from an instance of [`Circuit`].
///
/// The selector compression optimization is turned on only if
/// `compress_selectors` is `true`.
pub fn keygen_vk_custom<'params, C, P, ConcreteCircuit>(
    _params: &P,
    _circuit: &ConcreteCircuit,
    _compress_selectors: bool,
) -> Result<VerifyingKey<C>, Error>
where
    C: CurveAffine,
    P: Params<'params, C> + Sync,
    ConcreteCircuit: Circuit<C::Scalar>,
    C::Scalar: FromUniformBytes<64>,
{
    unimplemented!("Phase-1 stub: GPU-local keygen_vk_custom body lands in Phase 2 (b222e4d port assembling canonical keys via halo2-axiom from_parts)")
}

/// Generate a [`ProvingKey`] from a [`VerifyingKey`] and an instance of
/// [`Circuit`].
pub fn keygen_pk<'params, C, P, ConcreteCircuit>(
    _params: &P,
    _vk: VerifyingKey<C>,
    _circuit: &ConcreteCircuit,
) -> Result<ProvingKey<C>, Error>
where
    C: CurveAffine,
    C::Scalar: FromUniformBytes<64>,
    P: Params<'params, C> + Sync,
    ConcreteCircuit: Circuit<C::Scalar>,
{
    unimplemented!("Phase-1 stub: GPU-local keygen_pk body lands in Phase 2 (b222e4d port assembling canonical keys via halo2-axiom from_parts)")
}

/// Generate a [`ProvingKey`] from an instance of [`Circuit`]. A
/// [`VerifyingKey`] is generated in the process.
pub fn keygen_pk2<'params, C, P, ConcreteCircuit>(
    _params: &P,
    _circuit: &ConcreteCircuit,
    _compress_selectors: bool,
) -> Result<ProvingKey<C>, Error>
where
    C: CurveAffine,
    C::Scalar: FromUniformBytes<64>,
    P: Params<'params, C> + Sync,
    ConcreteCircuit: Circuit<C::Scalar>,
{
    unimplemented!("Phase-1 stub: GPU-local keygen_pk2 body lands in Phase 2 (b222e4d port assembling canonical keys via halo2-axiom from_parts)")
}
