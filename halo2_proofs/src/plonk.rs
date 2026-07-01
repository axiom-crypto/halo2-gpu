//! This module provides an implementation of a variant of (Turbo)[PLONK][plonk]
//! that is designed specifically for the polynomial commitment scheme described
//! in the [Halo][halo] paper.
//!
//! [halo]: https://eprint.iacr.org/2019/1021
//! [plonk]: https://eprint.iacr.org/2019/953
use group::ff::FromUniformBytes;

use crate::arithmetic::CurveAffine;
use crate::cuda::funcs::batch_eval_polynomial_device_out;
use crate::cuda::utils::{query_device_free_bytes_for_chunking, HALO2_GPU_CTX};
use crate::cuda::HaloGpuError;
use crate::poly::{Coeff, DevicePolyExt, EvaluationDomain, HostPolyExt, LagrangeCoeff, Polynomial};
use crate::transcript::{ChallengeScalar, EncodedChallenge, Transcript, TranscriptWrite};
use crate::{SerdeCurveAffine, SerdePrimeField};
use once_cell::sync::OnceCell;
use openvm_cuda_common::copy::{cuda_memcpy_on, MemCopyH2D};
use openvm_cuda_common::d_buffer::DeviceBuffer;

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
// `Gpu*` forks from these.
pub use halo2_axiom::plonk::{ProvingKey, VerifyingKey};

// Canonical synthesis frontend.
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
pub struct GpuVerifyingKey<'vk, C: CurveAffine> {
    pub(crate) domain: EvaluationDomain<'vk, C::Scalar>,
    pub(crate) fixed_commitments: Vec<C>,
    pub(crate) permutation: permutation::VerifyingKey<C>,
    pub(crate) cs: GpuConstraintSystem<C::Scalar>,
    /// Cached maximum degree of `cs` (which doesn't change after construction).
    pub(crate) cs_degree: usize,
    /// The representative of this `VerifyingKey` in transcripts.
    pub(crate) transcript_repr: C::Scalar,
}

impl<'vk, C: CurveAffine> GpuVerifyingKey<'vk, C>
where
    C::Scalar: FromUniformBytes<64>,
{
    /// Rebuilds the GPU verifying key from a canonical [`VerifyingKey`].
    /// Pure host: no device traffic, no kernel launch.
    pub fn from_host(vk: &'vk VerifyingKey<C>) -> Self {
        let cs = GpuConstraintSystem::from(vk.cs());
        let cs_degree = cs.degree();
        let domain = EvaluationDomain::from_host_domain(vk.get_domain());
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
    domain: EvaluationDomain<'a, C::Scalar>,
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
    /// Empties the device `OnceCell` mirrors so the clone regenerates them lazily.
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
    /// Wraps a canonical [`ProvingKey`] by *borrowing* it: the per-proof hot
    /// path. Clones no host polys, so [`create_proof`] takes a `&ProvingKey`
    /// without a per-proof deep copy.
    ///
    /// Pure host rebuild: no device traffic, no kernel launch. Device mirrors
    /// start empty and populate lazily at first prove.
    pub fn from_host(inner: &'a ProvingKey<C>) -> Self {
        let cs = GpuConstraintSystem::from(inner.get_vk().cs());
        let hdomain = inner.get_vk().get_domain();
        let domain = EvaluationDomain::from_host_domain(hdomain);
        let ev = Evaluator::new(&cs);
        let cs_degree = cs.degree();
        let transcript_repr = inner.get_vk().transcript_repr();
        GpuProvingKey {
            inner: Cow::Borrowed(inner),
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
        for eval in self.permutation_sigma_evals(*x)? {
            transcript.write_scalar(eval)?;
        }
        Ok(())
    }

    /// Permutation σ-poly evaluations at `x`, in slot order (slot `i` =
    /// `inner.permutation().polys()[i]`) — the exact order
    /// [`evaluate_permutation`] absorbs them via `write_scalar`.
    ///
    /// Device-out fast path (σ mirror resident): every σ Coeff poly is
    /// evaluated at `x` into one `DeviceBuffer<F>` via
    /// [`batch_eval_polynomial_device_out`] — no per-eval `to_host_sync`, and no
    /// H2D since the σ polys are already device-resident — then the whole result
    /// buffer is D2H'd once. Host-Horner fallback (mirror `None`, VRAM-gated):
    /// the pre-existing `eval_polynomial` loop, which also serves as the
    /// byte-equivalence oracle for the device path.
    fn permutation_sigma_evals(&self, x: C::Scalar) -> Result<Vec<C::Scalar>, GpuError> {
        use ff::Field;
        if let Some(sigma) = self.permutation_polys_device() {
            let n = sigma.len();
            if n == 0 {
                return Ok(Vec::new());
            }
            // σ polys are already device-resident — collect their buffers (no H2D)
            // and evaluate all at `x` in one device-out batch (no per-eval sync).
            let d_sigma: Vec<&DeviceBuffer<C::Scalar>> =
                sigma.iter().map(|p| p.device_buf()).collect();
            let eval_points = vec![x; n];
            let d_out = batch_eval_polynomial_device_out(&d_sigma, &eval_points)?;

            // ONE batched D2H of the whole σ-eval buffer (was one synced 32-byte
            // D2H per eval). Counted so the perf/nsys workflow attributes it.
            let mut evals = vec![C::Scalar::ZERO; n];
            let bytes = n * std::mem::size_of::<C::Scalar>();
            crate::perf_d2h!("cuda.permutation_sigma_eval.result", bytes as u64);
            unsafe {
                cuda_memcpy_on::<true, false>(
                    evals.as_mut_ptr() as *mut libc::c_void,
                    d_out.as_raw_ptr(),
                    bytes,
                    &HALO2_GPU_CTX,
                )
            }
            .map_err(HaloGpuError::from)?;
            HALO2_GPU_CTX.stream.to_host_sync()?;
            Ok(evals)
        } else {
            // VRAM-gated fallback / equivalence oracle: host-Horner eval, slot
            // order preserved.
            Ok(self
                .inner
                .permutation()
                .polys()
                .iter()
                .map(|poly| crate::arithmetic::eval_polynomial(poly, x))
                .collect())
        }
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
    /// `None` only if the H2D upload fails.
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

    /// Lazy device mirror of `inner.l0()`. Returns `None` only if the H2D upload fails.
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
    // Canonical halo2-axiom serde bounds required by the delegated `inner.write`/`inner.read`.
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

    /// Writes the proving key to a vector of bytes using [`Self::write`].
    pub fn to_bytes(&self, format: halo2_axiom::SerdeFormat) -> Vec<u8> {
        self.inner.to_bytes(format)
    }
}

/// Shared lazy initializer for the slice-valued PK device mirrors: per-poly
/// H2D loop and `perf_h2d!` byte-trace.
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

#[cfg(test)]
mod permutation_sigma_eval_tests {
    //! Byte-equivalence oracle for the device-out permutation-PK σ eval
    //! ([`GpuProvingKey::permutation_sigma_evals`]).
    //!
    //! The device-out path evaluates every σ Coeff poly of the (device-resident)
    //! PK permutation mirror at the challenge `x` in one batched
    //! `batch_eval_polynomial_device_out` + one D2H. This test asserts that
    //! result equals the CPU `eval_polynomial` reference over
    //! `permutation().polys()` element-for-element **in slot order** (slot `i` =
    //! poly `i` = the exact order `evaluate_permutation` absorbs via
    //! `write_scalar`). The circuit enables equality on four columns, so a
    //! slot-order bug (swap/reverse) or a wrong-point bug is observable.

    use super::*;
    use crate::arithmetic::eval_polynomial;
    use crate::circuit::{Layouter, SimpleFloorPlanner, Value};
    use crate::poly::kzg::commitment::ParamsKZG;
    use crate::poly::Rotation;
    use halo2curves::bn256::{Bn256, Fr, G1Affine};
    use rand_core::OsRng;

    /// `1 << K` rows. K >= 14 clears `GPU_MSM_THRESHOLD` so the real GPU device
    /// paths run (matching the other end-to-end GPU tests).
    const K: u32 = 14;

    /// A circuit with four equality-enabled columns (a, b, c, instance) => four
    /// σ permutation polys. Deliberately more than a swap needs, so the
    /// order-sensitive assertion has teeth.
    #[derive(Clone)]
    struct SigmaCircuit;

    #[derive(Clone)]
    struct SigmaConfig {
        a: Column<Advice>,
        b: Column<Advice>,
        c: Column<Advice>,
        q: Column<Fixed>,
    }

    impl Circuit<Fr> for SigmaCircuit {
        type Config = SigmaConfig;
        type FloorPlanner = SimpleFloorPlanner;
        #[cfg(feature = "circuit-params")]
        type Params = ();

        fn without_witnesses(&self) -> Self {
            Self
        }

        fn configure(meta: &mut ConstraintSystem<Fr>) -> SigmaConfig {
            let a = meta.advice_column();
            let b = meta.advice_column();
            let c = meta.advice_column();
            let q = meta.fixed_column();
            let instance = meta.instance_column();

            meta.enable_equality(a);
            meta.enable_equality(b);
            meta.enable_equality(c);
            meta.enable_equality(instance);

            meta.create_gate("mul", |meta| {
                let a = meta.query_advice(a, Rotation::cur());
                let b = meta.query_advice(b, Rotation::cur());
                let c = meta.query_advice(c, Rotation::cur());
                let q = meta.query_fixed(q, Rotation::cur());
                vec![q * (a * b - c)]
            });

            SigmaConfig { a, b, c, q }
        }

        fn synthesize(
            &self,
            config: SigmaConfig,
            mut layouter: impl Layouter<Fr>,
        ) -> Result<(), Error> {
            layouter.assign_region(
                || "r",
                |mut region| {
                    region.assign_advice(config.a, 0, Value::known(Fr::from(3)));
                    region.assign_advice(config.b, 0, Value::known(Fr::from(4)));
                    region.assign_advice(config.c, 0, Value::known(Fr::from(12)));
                    region.assign_fixed(config.q, 0, Fr::from(1));
                    Ok(())
                },
            )
        }
    }

    #[test]
    fn permutation_sigma_evals_match_host_eval_in_slot_order() {
        let params = ParamsKZG::<Bn256>::setup(K, OsRng);
        let circuit = SigmaCircuit;
        let vk = keygen_vk(&params, &circuit).expect("gpu keygen_vk");
        let pk = keygen_pk(&params, vk, &circuit).expect("gpu keygen_pk");
        let gpk = GpuProvingKey::<G1Affine>::from_host(&pk);

        // The device σ mirror MUST be resident, else the test would silently
        // compare host-vs-host and never exercise the device-out path.
        assert!(
            gpk.permutation_polys_device().is_some(),
            "permutation σ device mirror not resident — test would not exercise the device-out path"
        );

        let host_sigma = gpk.inner.permutation().polys();
        let num_sigma = host_sigma.len();
        assert!(
            num_sigma >= 3,
            "need >= 3 σ polys for an order-sensitive check, got {num_sigma}"
        );

        // A non-trivial fixed challenge (deterministic for reproducibility).
        let x = Fr::from(0x9e37_79b9_7f4a_7c15u64);

        // Host oracle: CPU eval_polynomial over the σ Coeff polys, slot order.
        let host_evals: Vec<Fr> = host_sigma.iter().map(|p| eval_polynomial(p, x)).collect();

        // Device-out path under test.
        let device_evals = gpk.permutation_sigma_evals(x).expect("device-out σ evals");

        assert_eq!(
            device_evals.len(),
            num_sigma,
            "device-out σ eval count must match the number of σ polys"
        );
        for i in 0..num_sigma {
            assert_eq!(
                device_evals[i], host_evals[i],
                "σ eval mismatch at slot {i}/{num_sigma} \
                 (device-out must equal host eval_polynomial in write_scalar order)"
            );
        }
    }
}
