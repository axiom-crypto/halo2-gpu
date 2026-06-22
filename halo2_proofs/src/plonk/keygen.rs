//! GPU-local key generation.
//!
//! These functions take a **GPU** [`Params`](crate::poly::commitment::Params)
//! and a **canonical** [`Circuit`](crate::plonk::Circuit) and return the
//! **canonical** halo2-axiom [`ProvingKey`]/[`VerifyingKey`] (the serde
//! source-of-truth). This matches the signatures the patched `halo2-base`/
//! `openvm` stack calls (`keygen_pk2`, `keygen_vk_custom`, `keygen_pk`), while
//! keeping the GPU `Params` on the input side.
//!
//! Assembles the canonical key types via halo2-axiom's `from_parts`
//! constructors. GPU acceleration is preserved by construction:
//!
//! * fixed-column + selector commitments via **GPU MSM** `params.commit_lagrange`;
//! * fixed/basis/σ polynomials via **GPU iFFT** `domain.lagrange_to_coeff[_many]`;
//! * permutation vk/pk via the GPU [`permutation::keygen::Assembly`].
//!
//! The circuit is synthesized through the **canonical** `Circuit`/`Assignment`
//! frontend (`Assembly` implements `halo2_axiom::plonk::Assignment`), so the
//! constraint system is built natively as the canonical `ConstraintSystem` —
//! no reverse cs translation. The only halo2-axiom additions needed are the
//! `from_parts`/`from_commitments` constructors.

use std::marker::PhantomData;
use std::ops::Range;

use group::ff::{Field, FromUniformBytes};
use group::Curve;

#[cfg(feature = "profile")]
use crate::{end_timer, start_timer};

use super::{permutation, Circuit, GpuError, ProvingKey, VerifyingKey};
use crate::arithmetic::CurveAffine;
use crate::circuit::Value;
use crate::cpu::arithmetic::parallelize;
use crate::cpu::poly::batch_invert_assigned;
use crate::plonk::{
    Advice, Any, Assigned, Assignment, Challenge, Column, ConstraintSystem, Fixed, FloorPlanner,
    GpuAssigned, Instance, Selector,
};
use crate::poly::{
    commitment::{Blind, Params},
    EvaluationDomain, LagrangeCoeff, Polynomial,
};

/// Builds the GPU evaluation domain + canonical constraint system + circuit
/// config by running the **canonical** `Circuit::configure` against a fresh
/// canonical `ConstraintSystem`. The domain is the GPU fork (the FFT/MSM
/// engine); the canonical-typed vk domain is reconstructed identically from
/// `degree`+`k` in [`keygen_pk_impl`]/[`keygen_vk_custom`].
fn create_domain<C, ConcreteCircuit>(
    k: u32,
    #[cfg(feature = "circuit-params")] params: ConcreteCircuit::Params,
) -> (
    EvaluationDomain<C::Scalar>,
    ConstraintSystem<C::Scalar>,
    ConcreteCircuit::Config,
)
where
    C: CurveAffine,
    ConcreteCircuit: Circuit<C::Scalar>,
{
    let mut cs = ConstraintSystem::default();
    #[cfg(feature = "circuit-params")]
    let config = ConcreteCircuit::configure_with_params(&mut cs, params);
    #[cfg(not(feature = "circuit-params"))]
    let config = ConcreteCircuit::configure(&mut cs);

    let degree = cs.degree();
    let domain = EvaluationDomain::new(degree as u32, k);

    (domain, cs, config)
}

/// Assembly accumulator for keygen synthesis. Implements the **canonical**
/// `Assignment` trait, so it is driven by the canonical `Circuit::synthesize`.
struct Assembly<F: Field> {
    k: u32,
    /// Canonical-`Assigned` fixed columns; converted to the device-repr
    /// `GpuAssigned` only at the batch-inversion boundary below.
    fixed: Vec<Polynomial<Assigned<F>, LagrangeCoeff>>,
    permutation: permutation::keygen::Assembly,
    selectors: Vec<Vec<bool>>,
    // A range of available rows for assignment and copies.
    usable_rows: Range<usize>,
    _marker: PhantomData<F>,
}

impl<F: Field> Assignment<F> for Assembly<F> {
    fn enter_region<NR, N>(&mut self, _: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
    }

    fn exit_region(&mut self) {}

    fn enable_selector<A, AR>(
        &mut self,
        _: A,
        selector: &Selector,
        row: usize,
    ) -> Result<(), halo2_axiom::plonk::Error>
    where
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        if !self.usable_rows.contains(&row) {
            return Err(halo2_axiom::plonk::Error::NotEnoughRowsAvailable { current_k: self.k });
        }

        self.selectors[selector.index()][row] = true;

        Ok(())
    }

    fn query_instance(
        &self,
        _: Column<Instance>,
        row: usize,
    ) -> Result<Value<F>, halo2_axiom::plonk::Error> {
        if !self.usable_rows.contains(&row) {
            return Err(halo2_axiom::plonk::Error::NotEnoughRowsAvailable { current_k: self.k });
        }

        // There is no instance in this context.
        Ok(Value::unknown())
    }

    fn assign_advice<'v>(
        &mut self,
        _: Column<Advice>,
        _: usize,
        _: Value<Assigned<F>>,
    ) -> Value<&'v Assigned<F>> {
        Value::unknown()
    }

    fn assign_fixed(&mut self, column: Column<Fixed>, row: usize, to: Assigned<F>) {
        if !self.usable_rows.contains(&row) {
            panic!(
                "Assign Fixed {:?}",
                GpuError::not_enough_rows_available(self.k)
            );
        }

        *self
            .fixed
            .get_mut(column.index())
            .and_then(|v| v.get_mut(row))
            .unwrap_or_else(|| {
                panic!(
                    "{:?}",
                    GpuError::Canonical(halo2_axiom::plonk::Error::BoundsFailure)
                )
            }) = to;
    }

    fn copy(
        &mut self,
        left_column: Column<Any>,
        left_row: usize,
        right_column: Column<Any>,
        right_row: usize,
    ) {
        if !self.usable_rows.contains(&left_row) || !self.usable_rows.contains(&right_row) {
            panic!("{:?}", GpuError::not_enough_rows_available(self.k));
        }

        self.permutation
            .copy(left_column, left_row, right_column, right_row)
            .unwrap_or_else(|err| panic!("{err:?}"))
    }

    fn fill_from_row(
        &mut self,
        column: Column<Fixed>,
        from_row: usize,
        to: Value<Assigned<F>>,
    ) -> Result<(), halo2_axiom::plonk::Error> {
        if !self.usable_rows.contains(&from_row) {
            return Err(halo2_axiom::plonk::Error::NotEnoughRowsAvailable { current_k: self.k });
        }

        let col = self
            .fixed
            .get_mut(column.index())
            .ok_or(halo2_axiom::plonk::Error::BoundsFailure)?;

        // Canonical `Value::assign()` is `pub(crate)`; extract the known value
        // via the public `map` closure (erroring on unknown, as `assign` does).
        let mut filler = None;
        to.map(|v| filler = Some(v));
        let filler = filler.ok_or(halo2_axiom::plonk::Error::Synthesis)?;
        for row in self.usable_rows.clone().skip(from_row) {
            col[row] = filler;
        }

        Ok(())
    }

    fn get_challenge(&self, _: Challenge) -> Value<F> {
        Value::unknown()
    }

    fn annotate_column<A, AR>(&mut self, _annotation: A, _column: Column<Any>)
    where
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
    }

    fn push_namespace<NR, N>(&mut self, _: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
    }

    fn pop_namespace(&mut self, _: Option<String>) {}
}

/// Generate a [`VerifyingKey`] from an instance of [`Circuit`].
///
/// By default, selector compression is turned **off**.
pub fn keygen_vk<'params, C, P, ConcreteCircuit>(
    params: &P,
    circuit: &ConcreteCircuit,
) -> Result<VerifyingKey<C>, GpuError>
where
    C: CurveAffine,
    P: Params<'params, C> + Sync,
    ConcreteCircuit: Circuit<C::Scalar>,
    C::Scalar: FromUniformBytes<64>,
{
    keygen_vk_custom(params, circuit, false)
}

/// Generate a [`VerifyingKey`] from an instance of [`Circuit`].
///
/// The selector compression optimization is turned on only if
/// `compress_selectors` is `true`.
pub fn keygen_vk_custom<'params, C, P, ConcreteCircuit>(
    params: &P,
    circuit: &ConcreteCircuit,
    compress_selectors: bool,
) -> Result<VerifyingKey<C>, GpuError>
where
    C: CurveAffine,
    P: Params<'params, C> + Sync,
    ConcreteCircuit: Circuit<C::Scalar>,
    C::Scalar: FromUniformBytes<64>,
{
    let (domain, cs, config) = create_domain::<C, ConcreteCircuit>(
        params.k(),
        #[cfg(feature = "circuit-params")]
        circuit.params(),
    );
    // The vk domain is the canonical-typed twin of the GPU `domain`, built
    // identically from `(degree, k)` so the vk is byte-identical to CPU keygen.
    let degree = cs.degree();

    if (params.n() as usize) < cs.minimum_rows() {
        return Err(GpuError::not_enough_rows_available(params.k()));
    }

    #[cfg(feature = "profile")]
    let assembly_time = start_timer!(|| "create assembly object");
    let mut assembly: Assembly<C::Scalar> = Assembly {
        k: params.k(),
        fixed: vec![domain.empty_lagrange_assigned(); cs.num_fixed_columns()],
        permutation: permutation::keygen::Assembly::new(params.n() as usize, cs.permutation()),
        selectors: vec![vec![false; params.n() as usize]; cs.num_selectors()],
        usable_rows: 0..params.n() as usize - (cs.blinding_factors() + 1),
        _marker: PhantomData,
    };
    #[cfg(feature = "profile")]
    end_timer!(assembly_time);

    // Synthesize the circuit to obtain URS.
    #[cfg(feature = "profile")]
    let synthesize_time = start_timer!(|| "synthesize");
    ConcreteCircuit::FloorPlanner::synthesize(
        &mut assembly,
        circuit,
        config,
        cs.constants().clone(),
    )?;
    #[cfg(feature = "profile")]
    end_timer!(synthesize_time);

    #[cfg(feature = "profile")]
    let batch_invert_time = start_timer!(|| "batch invert fixed columns");
    let mut fixed = batch_invert_fixed::<C::Scalar>(&assembly.fixed);
    #[cfg(feature = "profile")]
    end_timer!(batch_invert_time);
    let (cs, selector_polys) = if compress_selectors {
        cs.compress_selectors(assembly.selectors.clone())
    } else {
        // After this, the ConstraintSystem has no selectors: `verify` doesn't
        // need them and `keygen_pk` regenerates `cs` from scratch anyway.
        let selectors = std::mem::take(&mut assembly.selectors);
        cs.directly_convert_selectors_to_fixed(selectors)
    };
    fixed.extend(
        selector_polys
            .into_iter()
            .map(|poly| domain.lagrange_from_vec(poly)),
    );

    let permutation_vk = assembly
        .permutation
        .build_vk(params, &domain, cs.permutation());

    // GPU MSM commitment per fixed/selector column.
    let fixed_commitments = fixed
        .iter()
        .map(|poly| params.commit_lagrange(poly, Blind::default()).to_affine())
        .collect();

    let vk_domain = halo2_axiom::poly::EvaluationDomain::new(degree as u32, params.k());
    Ok(VerifyingKey::from_parts(
        vk_domain,
        fixed_commitments,
        permutation_vk,
        cs,
        assembly.selectors,
        compress_selectors,
    ))
}

/// Generate a [`ProvingKey`] from a [`VerifyingKey`] and an instance of
/// [`Circuit`].
pub fn keygen_pk<'params, C, P, ConcreteCircuit>(
    params: &P,
    vk: VerifyingKey<C>,
    circuit: &ConcreteCircuit,
) -> Result<ProvingKey<C>, GpuError>
where
    C: CurveAffine,
    C::Scalar: FromUniformBytes<64>,
    P: Params<'params, C> + Sync,
    ConcreteCircuit: Circuit<C::Scalar>,
{
    let compress_selectors = vk.compress_selectors();
    keygen_pk_impl(params, Some(vk), circuit, compress_selectors)
}

/// Generate a [`ProvingKey`] from an instance of [`Circuit`]. A
/// [`VerifyingKey`] is generated in the process.
pub fn keygen_pk2<'params, C, P, ConcreteCircuit>(
    params: &P,
    circuit: &ConcreteCircuit,
    compress_selectors: bool,
) -> Result<ProvingKey<C>, GpuError>
where
    C: CurveAffine,
    C::Scalar: FromUniformBytes<64>,
    P: Params<'params, C> + Sync,
    ConcreteCircuit: Circuit<C::Scalar>,
{
    keygen_pk_impl(params, None, circuit, compress_selectors)
}

/// Shared body: builds the canonical `ProvingKey` from either a precomputed
/// `VerifyingKey` or a freshly-generated one (the latter shares the fixed-column
/// FFTs). GPU-accelerated throughout (MSM commitments, iFFTs); no CPU-MSM.
fn keygen_pk_impl<'params, C, P, ConcreteCircuit>(
    params: &P,
    vk: Option<VerifyingKey<C>>,
    circuit: &ConcreteCircuit,
    compress_selectors: bool,
) -> Result<ProvingKey<C>, GpuError>
where
    C: CurveAffine,
    C::Scalar: FromUniformBytes<64>,
    P: Params<'params, C> + Sync,
    ConcreteCircuit: Circuit<C::Scalar>,
{
    let (domain, cs, config) = create_domain::<C, ConcreteCircuit>(
        params.k(),
        #[cfg(feature = "circuit-params")]
        circuit.params(),
    );
    let degree = cs.degree();

    if (params.n() as usize) < cs.minimum_rows() {
        return Err(GpuError::not_enough_rows_available(params.k()));
    }

    #[cfg(feature = "profile")]
    let assembly_time = start_timer!(|| "create assembly object");
    let mut assembly: Assembly<C::Scalar> = Assembly {
        k: params.k(),
        fixed: vec![domain.empty_lagrange_assigned(); cs.num_fixed_columns()],
        permutation: permutation::keygen::Assembly::new(params.n() as usize, cs.permutation()),
        selectors: vec![vec![false; params.n() as usize]; cs.num_selectors()],
        usable_rows: 0..params.n() as usize - (cs.blinding_factors() + 1),
        _marker: PhantomData,
    };
    #[cfg(feature = "profile")]
    end_timer!(assembly_time);

    // Synthesize the circuit to obtain URS.
    #[cfg(feature = "profile")]
    let synthesize_time = start_timer!(|| "synthesize");
    ConcreteCircuit::FloorPlanner::synthesize(
        &mut assembly,
        circuit,
        config,
        cs.constants().clone(),
    )?;
    #[cfg(feature = "profile")]
    end_timer!(synthesize_time);

    #[cfg(feature = "profile")]
    let batch_invert_time = start_timer!(|| "batch invert fixed columns");
    let mut fixed = batch_invert_fixed::<C::Scalar>(&assembly.fixed);
    #[cfg(feature = "profile")]
    end_timer!(batch_invert_time);
    let (cs, selector_polys) = if compress_selectors {
        cs.compress_selectors(assembly.selectors.clone())
    } else {
        let selectors = std::mem::take(&mut assembly.selectors);
        cs.directly_convert_selectors_to_fixed(selectors)
    };
    fixed.extend(
        selector_polys
            .into_iter()
            .map(|poly| domain.lagrange_from_vec(poly)),
    );

    let permutation_pk =
        assembly
            .permutation
            .clone()
            .build_pk(params, &domain, cs.permutation())?;

    let vk = match vk {
        Some(vk) => vk,
        None => {
            let permutation_vk = assembly
                .permutation
                .build_vk(params, &domain, cs.permutation());

            // GPU MSM commitment per fixed/selector column.
            let fixed_commitments = fixed
                .iter()
                .map(|poly| params.commit_lagrange(poly, Blind::default()).to_affine())
                .collect();

            let vk_domain = halo2_axiom::poly::EvaluationDomain::new(degree as u32, params.k());
            VerifyingKey::from_parts(
                vk_domain,
                fixed_commitments,
                permutation_vk,
                cs,
                assembly.selectors,
                compress_selectors,
            )
        }
    };

    // GPU iFFT: fixed Lagrange -> Coeff.
    let fixed_polys = domain.lagrange_to_coeff_many(&fixed)?;

    let blinding_factors = vk.cs().blinding_factors();

    // Compute l_0(X).
    let mut l0 = domain.empty_lagrange();
    l0[0] = C::Scalar::ONE;
    let l0 = domain.lagrange_to_coeff(l0)?;

    // Compute l_blind(X): 1 on each blinding-factor row, 0 elsewhere.
    let mut l_blind = domain.empty_lagrange();
    for evaluation in l_blind[..].iter_mut().rev().take(blinding_factors) {
        *evaluation = C::Scalar::ONE;
    }

    // Compute l_last(X): 1 on the first inactive row, 0 elsewhere.
    let mut l_last = domain.empty_lagrange();
    l_last[params.n() as usize - blinding_factors - 1] = C::Scalar::ONE;

    // Compute l_active_row(X).
    let one = C::Scalar::ONE;
    let mut l_active_row = domain.empty_lagrange();
    parallelize(&mut l_active_row, |values, start| {
        for (i, value) in values.iter_mut().enumerate() {
            let idx = i + start;
            *value = one - (l_last[idx] + l_blind[idx]);
        }
    });

    let l_last = domain.lagrange_to_coeff(l_last)?;
    let l_active_row = domain.lagrange_to_coeff(l_active_row)?;

    Ok(ProvingKey::from_parts(
        vk,
        l0,
        l_last,
        l_active_row,
        fixed,
        fixed_polys,
        permutation_pk,
    ))
}

/// Synthesis→device boundary for keygen: reinterpret the per-cell canonical
/// `Assigned` fixed columns into the device-repr `GpuAssigned`, then run the
/// (CPU) batch inversion shared with the device-path equivalence oracle.
fn batch_invert_fixed<F: Field>(
    fixed: &[Polynomial<Assigned<F>, LagrangeCoeff>],
) -> Vec<Polynomial<F, LagrangeCoeff>> {
    let columns: Vec<Vec<GpuAssigned<F>>> = fixed
        .iter()
        .map(|poly| poly.iter().map(|a| GpuAssigned::from(*a)).collect())
        .collect();
    batch_invert_assigned::<F, _>(columns)
}
