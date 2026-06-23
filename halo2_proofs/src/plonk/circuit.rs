use super::{lookup, permutation};
use crate::dev::metadata;
use crate::poly::Rotation;
use core::cmp::max;
use core::ops::{Add, Mul};
use ff::Field;
use itertools::Itertools;
use sealed::SealedPhase;
use std::collections::HashMap;
use std::env::var;
use std::fmt::Debug;
use std::{
    convert::TryFrom,
    ops::{Neg, Sub},
};

mod compress_selectors;

/// A column type
pub trait GpuColumnType:
    'static + Sized + Copy + std::fmt::Debug + PartialEq + Eq + Into<GpuAny>
{
    /// Return expression from cell
    fn query_cell<F: Field>(&self, index: usize, at: Rotation) -> GpuExpression<F>;
}

/// A column with an index and type
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct GpuColumn<C: GpuColumnType> {
    pub index: usize,
    column_type: C,
}

impl<C: GpuColumnType> GpuColumn<C> {
    #[cfg(test)]
    pub fn new(index: usize, column_type: C) -> Self {
        GpuColumn { index, column_type }
    }

    /// Index of this column.
    pub fn index(&self) -> usize {
        self.index
    }

    /// Type of this column.
    pub fn column_type(&self) -> &C {
        &self.column_type
    }

    /// Return expression from column at a relative position
    pub fn query_cell<F: Field>(&self, at: Rotation) -> GpuExpression<F> {
        self.column_type.query_cell(self.index, at)
    }

    /// Return expression from column at the current row
    pub fn cur<F: Field>(&self) -> GpuExpression<F> {
        self.query_cell(Rotation::cur())
    }

    /// Return expression from column at the next row
    pub fn next<F: Field>(&self) -> GpuExpression<F> {
        self.query_cell(Rotation::next())
    }

    /// Return expression from column at the previous row
    pub fn prev<F: Field>(&self) -> GpuExpression<F> {
        self.query_cell(Rotation::prev())
    }

    /// Return expression from column at the specified rotation
    pub fn rot<F: Field>(&self, rotation: i32) -> GpuExpression<F> {
        self.query_cell(Rotation(rotation))
    }
}

impl<C: GpuColumnType> Ord for GpuColumn<C> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // This ordering is consensus-critical! The layouters rely on deterministic column
        // orderings.
        match self.column_type.into().cmp(&other.column_type.into()) {
            // Indices are assigned within column types.
            std::cmp::Ordering::Equal => self.index.cmp(&other.index),
            order => order,
        }
    }
}

impl<C: GpuColumnType> PartialOrd for GpuColumn<C> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

pub(crate) mod sealed {

    /// Phase of advice column
    #[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
    pub struct Phase(pub(super) u8);

    impl Phase {
        pub fn prev(&self) -> Option<Phase> {
            self.0.checked_sub(1).map(Phase)
        }
        pub fn next(&self) -> Phase {
            assert!(self.0 < 2, "The API only supports three phases");
            Phase(self.0 + 1)
        }
        #[allow(clippy::wrong_self_convention)]
        pub fn to_u8(&self) -> u8 {
            self.0
        }
    }

    impl SealedPhase for Phase {
        fn to_sealed(self) -> Phase {
            self
        }
    }

    /// Sealed trait to help keep `Phase` private.
    pub trait SealedPhase {
        fn to_sealed(self) -> Phase;
    }
}

/// Phase of advice column
pub trait GpuPhase: SealedPhase {}

impl<P: SealedPhase> GpuPhase for P {}

/// First phase
#[derive(Debug)]
pub struct GpuFirstPhase;

impl SealedPhase for super::GpuFirstPhase {
    fn to_sealed(self) -> sealed::Phase {
        sealed::Phase(0)
    }
}

/// Second phase
#[derive(Debug)]
pub struct GpuSecondPhase;

impl SealedPhase for super::GpuSecondPhase {
    fn to_sealed(self) -> sealed::Phase {
        sealed::Phase(1)
    }
}

/// Third phase
#[derive(Debug)]
pub struct GpuThirdPhase;

impl SealedPhase for super::GpuThirdPhase {
    fn to_sealed(self) -> sealed::Phase {
        sealed::Phase(2)
    }
}

/// An advice column
#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub struct GpuAdvice {
    pub(crate) phase: sealed::Phase,
}

impl Default for GpuAdvice {
    fn default() -> GpuAdvice {
        GpuAdvice {
            phase: GpuFirstPhase.to_sealed(),
        }
    }
}

impl GpuAdvice {
    /// Returns `GpuAdvice` in given `Phase`
    pub fn new<P: GpuPhase>(phase: P) -> GpuAdvice {
        GpuAdvice {
            phase: phase.to_sealed(),
        }
    }

    /// Phase of this column
    pub fn phase(&self) -> u8 {
        self.phase.0
    }
}

impl std::fmt::Debug for GpuAdvice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug_struct = f.debug_struct("Advice");
        // Only show advice's phase if it's not in first phase.
        if self.phase != GpuFirstPhase.to_sealed() {
            debug_struct.field("phase", &self.phase);
        }
        debug_struct.finish()
    }
}

/// A fixed column
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct GpuFixed;

/// An instance column
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct GpuInstance;

/// An enum over the GpuAdvice, GpuFixed, GpuInstance structs
#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub enum GpuAny {
    /// An GpuAdvice variant
    Advice(GpuAdvice),
    /// A GpuFixed variant
    Fixed,
    /// An GpuInstance variant
    Instance,
}

impl GpuAny {
    /// Returns GpuAdvice variant in `GpuFirstPhase`
    pub fn advice() -> GpuAny {
        GpuAny::Advice(GpuAdvice::default())
    }

    /// Returns GpuAdvice variant in given `Phase`
    pub fn advice_in<P: GpuPhase>(phase: P) -> GpuAny {
        GpuAny::Advice(GpuAdvice::new(phase))
    }
}

impl std::fmt::Debug for GpuAny {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuAny::Advice(advice) => {
                let mut debug_struct = f.debug_struct("Advice");
                // Only show advice's phase if it's not in first phase.
                if advice.phase != GpuFirstPhase.to_sealed() {
                    debug_struct.field("phase", &advice.phase);
                }
                debug_struct.finish()
            }
            GpuAny::Fixed => f.debug_struct("Fixed").finish(),
            GpuAny::Instance => f.debug_struct("Instance").finish(),
        }
    }
}

impl Ord for GpuAny {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // This ordering is consensus-critical! The layouters rely on deterministic column
        // orderings.
        match (self, other) {
            (GpuAny::Instance, GpuAny::Instance) | (GpuAny::Fixed, GpuAny::Fixed) => {
                std::cmp::Ordering::Equal
            }
            (GpuAny::Advice(lhs), GpuAny::Advice(rhs)) => lhs.phase.cmp(&rhs.phase),
            // Across column types, sort GpuInstance < GpuAdvice < GpuFixed.
            (GpuAny::Instance, GpuAny::Advice(_))
            | (GpuAny::Advice(_), GpuAny::Fixed)
            | (GpuAny::Instance, GpuAny::Fixed) => std::cmp::Ordering::Less,
            (GpuAny::Fixed, GpuAny::Instance)
            | (GpuAny::Fixed, GpuAny::Advice(_))
            | (GpuAny::Advice(_), GpuAny::Instance) => std::cmp::Ordering::Greater,
        }
    }
}

impl PartialOrd for GpuAny {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl GpuColumnType for GpuAdvice {
    fn query_cell<F: Field>(&self, index: usize, at: Rotation) -> GpuExpression<F> {
        GpuExpression::Advice(GpuAdviceQuery {
            index: None,
            column_index: index,
            rotation: at,
            phase: self.phase,
        })
    }
}
impl GpuColumnType for GpuFixed {
    fn query_cell<F: Field>(&self, index: usize, at: Rotation) -> GpuExpression<F> {
        GpuExpression::Fixed(GpuFixedQuery {
            index: None,
            column_index: index,
            rotation: at,
        })
    }
}
impl GpuColumnType for GpuInstance {
    fn query_cell<F: Field>(&self, index: usize, at: Rotation) -> GpuExpression<F> {
        GpuExpression::Instance(GpuInstanceQuery {
            index: None,
            column_index: index,
            rotation: at,
        })
    }
}
impl GpuColumnType for GpuAny {
    fn query_cell<F: Field>(&self, index: usize, at: Rotation) -> GpuExpression<F> {
        match self {
            GpuAny::Advice(GpuAdvice { phase }) => GpuExpression::Advice(GpuAdviceQuery {
                index: None,
                column_index: index,
                rotation: at,
                phase: *phase,
            }),
            GpuAny::Fixed => GpuExpression::Fixed(GpuFixedQuery {
                index: None,
                column_index: index,
                rotation: at,
            }),
            GpuAny::Instance => GpuExpression::Instance(GpuInstanceQuery {
                index: None,
                column_index: index,
                rotation: at,
            }),
        }
    }
}

impl From<GpuAdvice> for GpuAny {
    fn from(advice: GpuAdvice) -> GpuAny {
        GpuAny::Advice(advice)
    }
}

impl From<GpuFixed> for GpuAny {
    fn from(_: GpuFixed) -> GpuAny {
        GpuAny::Fixed
    }
}

impl From<GpuInstance> for GpuAny {
    fn from(_: GpuInstance) -> GpuAny {
        GpuAny::Instance
    }
}

impl From<GpuColumn<GpuAdvice>> for GpuColumn<GpuAny> {
    fn from(advice: GpuColumn<GpuAdvice>) -> GpuColumn<GpuAny> {
        GpuColumn {
            index: advice.index(),
            column_type: GpuAny::Advice(advice.column_type),
        }
    }
}

impl From<GpuColumn<GpuFixed>> for GpuColumn<GpuAny> {
    fn from(advice: GpuColumn<GpuFixed>) -> GpuColumn<GpuAny> {
        GpuColumn {
            index: advice.index(),
            column_type: GpuAny::Fixed,
        }
    }
}

impl From<GpuColumn<GpuInstance>> for GpuColumn<GpuAny> {
    fn from(advice: GpuColumn<GpuInstance>) -> GpuColumn<GpuAny> {
        GpuColumn {
            index: advice.index(),
            column_type: GpuAny::Instance,
        }
    }
}

impl TryFrom<GpuColumn<GpuAny>> for GpuColumn<GpuAdvice> {
    type Error = &'static str;

    fn try_from(any: GpuColumn<GpuAny>) -> Result<Self, Self::Error> {
        match any.column_type() {
            GpuAny::Advice(advice) => Ok(GpuColumn {
                index: any.index(),
                column_type: *advice,
            }),
            _ => Err("Cannot convert into GpuColumn<GpuAdvice>"),
        }
    }
}

impl TryFrom<GpuColumn<GpuAny>> for GpuColumn<GpuFixed> {
    type Error = &'static str;

    fn try_from(any: GpuColumn<GpuAny>) -> Result<Self, Self::Error> {
        match any.column_type() {
            GpuAny::Fixed => Ok(GpuColumn {
                index: any.index(),
                column_type: GpuFixed,
            }),
            _ => Err("Cannot convert into GpuColumn<GpuFixed>"),
        }
    }
}

impl TryFrom<GpuColumn<GpuAny>> for GpuColumn<GpuInstance> {
    type Error = &'static str;

    fn try_from(any: GpuColumn<GpuAny>) -> Result<Self, Self::Error> {
        match any.column_type() {
            GpuAny::Instance => Ok(GpuColumn {
                index: any.index(),
                column_type: GpuInstance,
            }),
            _ => Err("Cannot convert into GpuColumn<GpuInstance>"),
        }
    }
}

/// A selector, representing a fixed boolean value per row of the circuit.
///
/// Selectors can be used to conditionally enable (portions of) gates:
/// ```
/// use halo2_proofs::poly::Rotation;
/// # use halo2curves::pasta::Fp;
/// # use halo2_proofs::plonk::ConstraintSystem;
///
/// # let mut meta = GpuConstraintSystem::<Fp>::default();
/// let a = meta.advice_column();
/// let b = meta.advice_column();
/// let s = meta.selector();
///
/// meta.create_gate("foo", |meta| {
///     let a = meta.query_advice(a, Rotation::prev());
///     let b = meta.query_advice(b, Rotation::cur());
///     let s = meta.query_selector(s);
///
///     // On rows where the selector is enabled, a is constrained to equal b.
///     // On rows where the selector is disabled, a and b can take any value.
///     vec![s * (a - b)]
/// });
/// ```
///
/// Selectors are disabled on all rows by default, and must be explicitly enabled on each
/// row when required:
/// ```
/// use halo2_proofs::{
///     circuit::{Chip, Layouter, Value},
///     plonk::{GpuAdvice, GpuColumn, Error, GpuSelector},
/// };
/// use ff::Field;
/// # use halo2_proofs::plonk::Fixed;
///
/// struct Config {
///     a: GpuColumn<GpuAdvice>,
///     b: GpuColumn<GpuAdvice>,
///     s: GpuSelector,
/// }
///
/// fn circuit_logic<F: Field, C: Chip<F>>(chip: C, mut layouter: impl Layouter<F>) -> Result<(), Error> {
///     let config = chip.config();
///     # let config: Config = todo!();
///     layouter.assign_region(|| "bar", |mut region| {
///         region.assign_advice(config.a, 0, Value::known(F::ONE));
///         region.assign_advice(config.b, 1, Value::known(F::ONE));
///         config.s.enable(&mut region, 1)
///     })?;
///     Ok(())
/// }
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GpuSelector(pub(crate) usize, bool);

impl GpuSelector {
    /// Is this selector "simple"? Simple selectors can only be multiplied
    /// by expressions that contain no other simple selectors.
    pub fn is_simple(&self) -> bool {
        self.1
    }

    /// Returns index of this selector
    pub fn index(&self) -> usize {
        self.0
    }

    /// Return expression from selector
    pub fn expr<F: Field>(&self) -> GpuExpression<F> {
        GpuExpression::Selector(*self)
    }
}

/// Query of fixed column at a certain relative location
#[derive(Copy, Clone, Debug)]
pub struct GpuFixedQuery {
    /// Query index
    pub(crate) index: Option<usize>,
    /// GpuColumn index
    pub(crate) column_index: usize,
    /// Rotation of this query
    pub(crate) rotation: Rotation,
}

impl GpuFixedQuery {
    /// GpuColumn index
    pub fn column_index(&self) -> usize {
        self.column_index
    }

    /// Rotation of this query
    pub fn rotation(&self) -> Rotation {
        self.rotation
    }
}

/// Query of advice column at a certain relative location
#[derive(Copy, Clone, Debug)]
pub struct GpuAdviceQuery {
    /// Query index
    pub(crate) index: Option<usize>,
    /// GpuColumn index
    pub(crate) column_index: usize,
    /// Rotation of this query
    pub(crate) rotation: Rotation,
    /// Phase of this advice column
    pub(crate) phase: sealed::Phase,
}

impl GpuAdviceQuery {
    /// GpuColumn index
    pub fn column_index(&self) -> usize {
        self.column_index
    }

    /// Rotation of this query
    pub fn rotation(&self) -> Rotation {
        self.rotation
    }

    /// Phase of this advice column
    pub fn phase(&self) -> u8 {
        self.phase.0
    }
}

/// Query of instance column at a certain relative location
#[derive(Copy, Clone, Debug)]
pub struct GpuInstanceQuery {
    /// Query index
    pub(crate) index: Option<usize>,
    /// GpuColumn index
    pub(crate) column_index: usize,
    /// Rotation of this query
    pub(crate) rotation: Rotation,
}

impl GpuInstanceQuery {
    /// GpuColumn index
    pub fn column_index(&self) -> usize {
        self.column_index
    }

    /// Rotation of this query
    pub fn rotation(&self) -> Rotation {
        self.rotation
    }
}

/// A fixed column of a lookup table.
///
/// A lookup table can be loaded into this column via [`Layouter::assign_table`]. Columns
/// can currently only contain a single table, but they may be used in multiple lookup
/// arguments via [`GpuConstraintSystem::lookup`].
///
/// Lookup table columns are always "encumbered" by the lookup arguments they are used in;
/// they cannot simultaneously be used as general fixed columns.
///
/// [`Layouter::assign_table`]: crate::circuit::Layouter::assign_table
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct GpuTableColumn {
    /// The fixed column that this table column is stored in.
    ///
    /// # Security
    ///
    /// This inner column MUST NOT be exposed in the public API, or else chip developers
    /// can load lookup tables into their circuits without default-value-filling the
    /// columns, which can cause soundness bugs.
    inner: GpuColumn<GpuFixed>,
}

impl GpuTableColumn {
    /// Returns inner column
    pub fn inner(&self) -> GpuColumn<GpuFixed> {
        self.inner
    }
}

/// A challenge squeezed from transcript after advice columns at the phase have been committed.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct GpuChallenge {
    index: usize,
    pub(crate) phase: sealed::Phase,
}

impl GpuChallenge {
    /// Index of this challenge.
    pub fn index(&self) -> usize {
        self.index
    }

    /// Phase of this challenge.
    pub fn phase(&self) -> u8 {
        self.phase.0
    }

    /// Return GpuExpression
    pub fn expr<F: Field>(&self) -> GpuExpression<F> {
        GpuExpression::Challenge(*self)
    }
}

/// Low-degree expression representing an identity that must hold over the committed columns.
#[derive(Clone)]
pub enum GpuExpression<F> {
    /// This is a constant polynomial
    Constant(F),
    /// This is a virtual selector
    Selector(GpuSelector),
    /// This is a fixed column queried at a certain relative location
    Fixed(GpuFixedQuery),
    /// This is an advice (witness) column queried at a certain relative location
    Advice(GpuAdviceQuery),
    /// This is an instance (external) column queried at a certain relative location
    Instance(GpuInstanceQuery),
    /// This is a challenge
    Challenge(GpuChallenge),
    /// This is a negated polynomial
    Negated(Box<GpuExpression<F>>),
    /// This is the sum of two polynomials
    Sum(Box<GpuExpression<F>>, Box<GpuExpression<F>>),
    /// This is the product of two polynomials
    Product(Box<GpuExpression<F>>, Box<GpuExpression<F>>),
    /// This is a scaled polynomial
    Scaled(Box<GpuExpression<F>>, F),
}

impl<F: Field> GpuExpression<F> {
    /// Make side effects
    pub fn query_cells(&mut self, cells: &mut GpuVirtualCells<'_, F>) {
        match self {
            GpuExpression::Constant(_) => (),
            GpuExpression::Selector(selector) => {
                if !cells.queried_selectors.contains(selector) {
                    cells.queried_selectors.push(*selector);
                }
            }
            GpuExpression::Fixed(query) => {
                if query.index.is_none() {
                    let col = GpuColumn {
                        index: query.column_index,
                        column_type: GpuFixed,
                    };
                    cells.queried_cells.push((col, query.rotation).into());
                    query.index = Some(cells.meta.query_fixed_index(col, query.rotation));
                }
            }
            GpuExpression::Advice(query) => {
                if query.index.is_none() {
                    let col = GpuColumn {
                        index: query.column_index,
                        column_type: GpuAdvice { phase: query.phase },
                    };
                    cells.queried_cells.push((col, query.rotation).into());
                    query.index = Some(cells.meta.query_advice_index(col, query.rotation));
                }
            }
            GpuExpression::Instance(query) => {
                if query.index.is_none() {
                    let col = GpuColumn {
                        index: query.column_index,
                        column_type: GpuInstance,
                    };
                    cells.queried_cells.push((col, query.rotation).into());
                    query.index = Some(cells.meta.query_instance_index(col, query.rotation));
                }
            }
            GpuExpression::Challenge(_) => (),
            GpuExpression::Negated(a) => a.query_cells(cells),
            GpuExpression::Sum(a, b) => {
                a.query_cells(cells);
                b.query_cells(cells);
            }
            GpuExpression::Product(a, b) => {
                a.query_cells(cells);
                b.query_cells(cells);
            }
            GpuExpression::Scaled(a, _) => a.query_cells(cells),
        };
    }

    /// Evaluate the polynomial using the provided closures to perform the
    /// operations.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate<T>(
        &self,
        constant: &impl Fn(F) -> T,
        selector_column: &impl Fn(GpuSelector) -> T,
        fixed_column: &impl Fn(GpuFixedQuery) -> T,
        advice_column: &impl Fn(GpuAdviceQuery) -> T,
        instance_column: &impl Fn(GpuInstanceQuery) -> T,
        challenge: &impl Fn(GpuChallenge) -> T,
        negated: &impl Fn(T) -> T,
        sum: &impl Fn(T, T) -> T,
        product: &impl Fn(T, T) -> T,
        scaled: &impl Fn(T, F) -> T,
    ) -> T {
        match self {
            GpuExpression::Constant(scalar) => constant(*scalar),
            GpuExpression::Selector(selector) => selector_column(*selector),
            GpuExpression::Fixed(query) => fixed_column(*query),
            GpuExpression::Advice(query) => advice_column(*query),
            GpuExpression::Instance(query) => instance_column(*query),
            GpuExpression::Challenge(value) => challenge(*value),
            GpuExpression::Negated(a) => {
                let a = a.evaluate(
                    constant,
                    selector_column,
                    fixed_column,
                    advice_column,
                    instance_column,
                    challenge,
                    negated,
                    sum,
                    product,
                    scaled,
                );
                negated(a)
            }
            GpuExpression::Sum(a, b) => {
                let a = a.evaluate(
                    constant,
                    selector_column,
                    fixed_column,
                    advice_column,
                    instance_column,
                    challenge,
                    negated,
                    sum,
                    product,
                    scaled,
                );
                let b = b.evaluate(
                    constant,
                    selector_column,
                    fixed_column,
                    advice_column,
                    instance_column,
                    challenge,
                    negated,
                    sum,
                    product,
                    scaled,
                );
                sum(a, b)
            }
            GpuExpression::Product(a, b) => {
                let a = a.evaluate(
                    constant,
                    selector_column,
                    fixed_column,
                    advice_column,
                    instance_column,
                    challenge,
                    negated,
                    sum,
                    product,
                    scaled,
                );
                let b = b.evaluate(
                    constant,
                    selector_column,
                    fixed_column,
                    advice_column,
                    instance_column,
                    challenge,
                    negated,
                    sum,
                    product,
                    scaled,
                );
                product(a, b)
            }
            GpuExpression::Scaled(a, f) => {
                let a = a.evaluate(
                    constant,
                    selector_column,
                    fixed_column,
                    advice_column,
                    instance_column,
                    challenge,
                    negated,
                    sum,
                    product,
                    scaled,
                );
                scaled(a, *f)
            }
        }
    }

    /// Evaluate over selectors only: `selector_fn` on selectors, `combine` on
    /// Sum/Product, `identity` everywhere else.
    pub fn evaluate_selectors<T: Clone>(
        &self,
        identity: T,
        selector_fn: &impl Fn(GpuSelector) -> T,
        combine: &impl Fn(T, T) -> T,
    ) -> T {
        self.evaluate(
            &|_| identity.clone(),
            selector_fn,
            &|_| identity.clone(),
            &|_| identity.clone(),
            &|_| identity.clone(),
            &|_| identity.clone(),
            &|a| a,
            combine,
            combine,
            &|a, _| a,
        )
    }

    /// Evaluate the polynomial lazily using the provided closures to perform the
    /// operations.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_lazy<T: PartialEq>(
        &self,
        constant: &impl Fn(F) -> T,
        selector_column: &impl Fn(GpuSelector) -> T,
        fixed_column: &impl Fn(GpuFixedQuery) -> T,
        advice_column: &impl Fn(GpuAdviceQuery) -> T,
        instance_column: &impl Fn(GpuInstanceQuery) -> T,
        challenge: &impl Fn(GpuChallenge) -> T,
        negated: &impl Fn(T) -> T,
        sum: &impl Fn(T, T) -> T,
        product: &impl Fn(T, T) -> T,
        scaled: &impl Fn(T, F) -> T,
        zero: &T,
    ) -> T {
        match self {
            GpuExpression::Constant(scalar) => constant(*scalar),
            GpuExpression::Selector(selector) => selector_column(*selector),
            GpuExpression::Fixed(query) => fixed_column(*query),
            GpuExpression::Advice(query) => advice_column(*query),
            GpuExpression::Instance(query) => instance_column(*query),
            GpuExpression::Challenge(value) => challenge(*value),
            GpuExpression::Negated(a) => {
                let a = a.evaluate_lazy(
                    constant,
                    selector_column,
                    fixed_column,
                    advice_column,
                    instance_column,
                    challenge,
                    negated,
                    sum,
                    product,
                    scaled,
                    zero,
                );
                negated(a)
            }
            GpuExpression::Sum(a, b) => {
                let a = a.evaluate_lazy(
                    constant,
                    selector_column,
                    fixed_column,
                    advice_column,
                    instance_column,
                    challenge,
                    negated,
                    sum,
                    product,
                    scaled,
                    zero,
                );
                let b = b.evaluate_lazy(
                    constant,
                    selector_column,
                    fixed_column,
                    advice_column,
                    instance_column,
                    challenge,
                    negated,
                    sum,
                    product,
                    scaled,
                    zero,
                );
                sum(a, b)
            }
            GpuExpression::Product(a, b) => {
                let (a, b) = if a.complexity() <= b.complexity() {
                    (a, b)
                } else {
                    (b, a)
                };
                let a = a.evaluate_lazy(
                    constant,
                    selector_column,
                    fixed_column,
                    advice_column,
                    instance_column,
                    challenge,
                    negated,
                    sum,
                    product,
                    scaled,
                    zero,
                );

                if a == *zero {
                    a
                } else {
                    let b = b.evaluate_lazy(
                        constant,
                        selector_column,
                        fixed_column,
                        advice_column,
                        instance_column,
                        challenge,
                        negated,
                        sum,
                        product,
                        scaled,
                        zero,
                    );
                    product(a, b)
                }
            }
            GpuExpression::Scaled(a, f) => {
                let a = a.evaluate_lazy(
                    constant,
                    selector_column,
                    fixed_column,
                    advice_column,
                    instance_column,
                    challenge,
                    negated,
                    sum,
                    product,
                    scaled,
                    zero,
                );
                scaled(a, *f)
            }
        }
    }

    fn write_identifier<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        match self {
            GpuExpression::Constant(scalar) => write!(writer, "{:?}", scalar),
            GpuExpression::Selector(selector) => write!(writer, "selector[{}]", selector.0),
            GpuExpression::Fixed(query) => {
                write!(
                    writer,
                    "fixed[{}][{}]",
                    query.column_index, query.rotation.0
                )
            }
            GpuExpression::Advice(query) => {
                write!(
                    writer,
                    "advice[{}][{}]",
                    query.column_index, query.rotation.0
                )
            }
            GpuExpression::Instance(query) => {
                write!(
                    writer,
                    "instance[{}][{}]",
                    query.column_index, query.rotation.0
                )
            }
            GpuExpression::Challenge(challenge) => {
                write!(writer, "challenge[{}]", challenge.index())
            }
            GpuExpression::Negated(a) => {
                writer.write_all(b"(-")?;
                a.write_identifier(writer)?;
                writer.write_all(b")")
            }
            GpuExpression::Sum(a, b) => {
                writer.write_all(b"(")?;
                a.write_identifier(writer)?;
                writer.write_all(b"+")?;
                b.write_identifier(writer)?;
                writer.write_all(b")")
            }
            GpuExpression::Product(a, b) => {
                writer.write_all(b"(")?;
                a.write_identifier(writer)?;
                writer.write_all(b"*")?;
                b.write_identifier(writer)?;
                writer.write_all(b")")
            }
            GpuExpression::Scaled(a, f) => {
                a.write_identifier(writer)?;
                write!(writer, "*{:?}", f)
            }
        }
    }

    /// Identifier for this expression. Expressions with identical identifiers
    /// do the same calculation (but the expressions don't need to be exactly equal
    /// in how they are composed e.g. `1 + 2` and `2 + 1` can have the same identifier).
    pub fn identifier(&self) -> String {
        let mut cursor = std::io::Cursor::new(Vec::new());
        self.write_identifier(&mut cursor).unwrap();
        String::from_utf8(cursor.into_inner()).unwrap()
    }

    /// Compute the degree of this polynomial
    pub fn degree(&self) -> usize {
        match self {
            GpuExpression::Constant(_) => 0,
            GpuExpression::Selector(_) => 1,
            GpuExpression::Fixed(_) => 1,
            GpuExpression::Advice(_) => 1,
            GpuExpression::Instance(_) => 1,
            GpuExpression::Challenge(_) => 0,
            GpuExpression::Negated(poly) => poly.degree(),
            GpuExpression::Sum(a, b) => max(a.degree(), b.degree()),
            GpuExpression::Product(a, b) => a.degree() + b.degree(),
            GpuExpression::Scaled(poly, _) => poly.degree(),
        }
    }

    /// Approximate the computational complexity of this expression.
    pub fn complexity(&self) -> usize {
        match self {
            GpuExpression::Constant(_) => 0,
            GpuExpression::Selector(_) => 1,
            GpuExpression::Fixed(_) => 1,
            GpuExpression::Advice(_) => 1,
            GpuExpression::Instance(_) => 1,
            GpuExpression::Challenge(_) => 0,
            GpuExpression::Negated(poly) => poly.complexity() + 5,
            GpuExpression::Sum(a, b) => a.complexity() + b.complexity() + 15,
            GpuExpression::Product(a, b) => a.complexity() + b.complexity() + 30,
            GpuExpression::Scaled(poly, _) => poly.complexity() + 30,
        }
    }

    /// Square this expression.
    pub fn square(self) -> Self {
        self.clone() * self
    }

    /// Returns whether or not this expression contains a simple `GpuSelector`.
    fn contains_simple_selector(&self) -> bool {
        self.evaluate_selectors(false, &|selector| selector.is_simple(), &|a, b| a || b)
    }

    /// Extracts a simple selector from this gate, if present
    fn extract_simple_selector(&self) -> Option<GpuSelector> {
        let op = |a, b| match (a, b) {
            (Some(a), None) | (None, Some(a)) => Some(a),
            (Some(_), Some(_)) => panic!("two simple selectors cannot be in the same expression"),
            _ => None,
        };
        self.evaluate_selectors(
            None,
            &|selector| selector.is_simple().then_some(selector),
            &op,
        )
    }

    /// Extracts all used instance columns in this expression
    pub fn extract_instances(&self) -> Vec<usize> {
        self.evaluate(
            &|_| vec![],
            &|_| vec![],
            &|_| vec![],
            &|_| vec![],
            &|query| vec![query.column_index],
            &|_| vec![],
            &|a| a,
            &|mut a, b| {
                a.extend(b);
                a.into_iter().unique().collect()
            },
            &|mut a, b| {
                a.extend(b);
                a.into_iter().unique().collect()
            },
            &|a, _| a,
        )
    }

    /// Extracts all used advice columns in this expression
    pub fn extract_advices(&self) -> Vec<usize> {
        self.evaluate(
            &|_| vec![],
            &|_| vec![],
            &|_| vec![],
            &|query| vec![query.column_index],
            &|_| vec![],
            &|_| vec![],
            &|a| a,
            &|mut a, b| {
                a.extend(b);
                a.into_iter().unique().collect()
            },
            &|mut a, b| {
                a.extend(b);
                a.into_iter().unique().collect()
            },
            &|a, _| a,
        )
    }

    /// Extracts all used fixed columns in this expression
    pub fn extract_fixed(&self) -> Vec<usize> {
        self.evaluate(
            &|_| vec![],
            &|_| vec![],
            &|query| vec![query.column_index],
            &|_| vec![],
            &|_| vec![],
            &|_| vec![],
            &|a| a,
            &|mut a, b| {
                a.extend(b);
                a.into_iter().unique().collect()
            },
            &|mut a, b| {
                a.extend(b);
                a.into_iter().unique().collect()
            },
            &|a, _| a,
        )
    }
}

impl<F: std::fmt::Debug> std::fmt::Debug for GpuExpression<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuExpression::Constant(scalar) => f.debug_tuple("Constant").field(scalar).finish(),
            GpuExpression::Selector(selector) => f.debug_tuple("Selector").field(selector).finish(),
            // Skip enum variant and print query struct directly to maintain backwards compatibility.
            GpuExpression::Fixed(query) => {
                let mut debug_struct = f.debug_struct("Fixed");
                match query.index {
                    None => debug_struct.field("query_index", &query.index),
                    Some(idx) => debug_struct.field("query_index", &idx),
                };
                debug_struct
                    .field("column_index", &query.column_index)
                    .field("rotation", &query.rotation)
                    .finish()
            }
            GpuExpression::Advice(query) => {
                let mut debug_struct = f.debug_struct("Advice");
                match query.index {
                    None => debug_struct.field("query_index", &query.index),
                    Some(idx) => debug_struct.field("query_index", &idx),
                };
                debug_struct
                    .field("column_index", &query.column_index)
                    .field("rotation", &query.rotation);
                // Only show advice's phase if it's not in first phase.
                if query.phase != GpuFirstPhase.to_sealed() {
                    debug_struct.field("phase", &query.phase);
                }
                debug_struct.finish()
            }
            GpuExpression::Instance(query) => {
                let mut debug_struct = f.debug_struct("Instance");
                match query.index {
                    None => debug_struct.field("query_index", &query.index),
                    Some(idx) => debug_struct.field("query_index", &idx),
                };
                debug_struct
                    .field("column_index", &query.column_index)
                    .field("rotation", &query.rotation)
                    .finish()
            }
            GpuExpression::Challenge(challenge) => {
                f.debug_tuple("Challenge").field(challenge).finish()
            }
            GpuExpression::Negated(poly) => f.debug_tuple("Negated").field(poly).finish(),
            GpuExpression::Sum(a, b) => f.debug_tuple("Sum").field(a).field(b).finish(),
            GpuExpression::Product(a, b) => f.debug_tuple("Product").field(a).field(b).finish(),
            GpuExpression::Scaled(poly, scalar) => {
                f.debug_tuple("Scaled").field(poly).field(scalar).finish()
            }
        }
    }
}

impl<F: Field> Neg for GpuExpression<F> {
    type Output = GpuExpression<F>;
    fn neg(self) -> Self::Output {
        GpuExpression::Negated(Box::new(self))
    }
}

impl<F: Field> Add for GpuExpression<F> {
    type Output = GpuExpression<F>;
    fn add(self, rhs: GpuExpression<F>) -> GpuExpression<F> {
        if self.contains_simple_selector() || rhs.contains_simple_selector() {
            panic!("attempted to use a simple selector in an addition");
        }
        GpuExpression::Sum(Box::new(self), Box::new(rhs))
    }
}

impl<F: Field> Sub for GpuExpression<F> {
    type Output = GpuExpression<F>;
    fn sub(self, rhs: GpuExpression<F>) -> GpuExpression<F> {
        if self.contains_simple_selector() || rhs.contains_simple_selector() {
            panic!("attempted to use a simple selector in a subtraction");
        }
        GpuExpression::Sum(Box::new(self), Box::new(-rhs))
    }
}

impl<F: Field> Mul for GpuExpression<F> {
    type Output = GpuExpression<F>;
    fn mul(self, rhs: GpuExpression<F>) -> GpuExpression<F> {
        if self.contains_simple_selector() && rhs.contains_simple_selector() {
            panic!("attempted to multiply two expressions containing simple selectors");
        }
        GpuExpression::Product(Box::new(self), Box::new(rhs))
    }
}

impl<F: Field> Mul<F> for GpuExpression<F> {
    type Output = GpuExpression<F>;
    fn mul(self, rhs: F) -> GpuExpression<F> {
        GpuExpression::Scaled(Box::new(self), rhs)
    }
}

/// A "virtual cell" is a PLONK cell that has been queried at a particular relative offset
/// within a custom gate.
///
/// Populated for structural parity with the canonical `VirtualCell`; the GPU
/// quotient evaluator drives off `GpuGate::polys`, so these fields are unread.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct GpuVirtualCell {
    pub(crate) column: GpuColumn<GpuAny>,
    pub(crate) rotation: Rotation,
}

impl<Col: Into<GpuColumn<GpuAny>>> From<(Col, Rotation)> for GpuVirtualCell {
    fn from((column, rotation): (Col, Rotation)) -> Self {
        GpuVirtualCell {
            column: column.into(),
            rotation,
        }
    }
}

/// An individual polynomial constraint.
///
/// These are returned by the closures passed to `GpuConstraintSystem::create_gate`.
#[derive(Debug)]
pub struct GpuConstraint<F: Field> {
    name: String,
    poly: GpuExpression<F>,
}

impl<F: Field> From<GpuExpression<F>> for GpuConstraint<F> {
    fn from(poly: GpuExpression<F>) -> Self {
        GpuConstraint {
            name: "".to_string(),
            poly,
        }
    }
}

impl<F: Field, S: AsRef<str>> From<(S, GpuExpression<F>)> for GpuConstraint<F> {
    fn from((name, poly): (S, GpuExpression<F>)) -> Self {
        GpuConstraint {
            name: name.as_ref().to_string(),
            poly,
        }
    }
}

impl<F: Field> From<GpuExpression<F>> for Vec<GpuConstraint<F>> {
    fn from(poly: GpuExpression<F>) -> Self {
        vec![GpuConstraint {
            name: "".to_string(),
            poly,
        }]
    }
}

/// A set of polynomial constraints with a common selector.
///
/// ```
/// use halo2_proofs::{plonk::{GpuConstraints, GpuExpression}, poly::Rotation};
/// use halo2curves::pasta::Fp;
/// # use halo2_proofs::plonk::ConstraintSystem;
///
/// # let mut meta = GpuConstraintSystem::<Fp>::default();
/// let a = meta.advice_column();
/// let b = meta.advice_column();
/// let c = meta.advice_column();
/// let s = meta.selector();
///
/// meta.create_gate("foo", |meta| {
///     let next = meta.query_advice(a, Rotation::next());
///     let a = meta.query_advice(a, Rotation::cur());
///     let b = meta.query_advice(b, Rotation::cur());
///     let c = meta.query_advice(c, Rotation::cur());
///     let s_ternary = meta.query_selector(s);
///
///     let one_minus_a = GpuExpression::Constant(Fp::one()) - a.clone();
///
///     GpuConstraints::with_selector(
///         s_ternary,
///         std::array::IntoIter::new([
///             ("a is boolean", a.clone() * one_minus_a.clone()),
///             ("next == a ? b : c", next - (a * b + one_minus_a * c)),
///         ]),
///     )
/// });
/// ```
///
/// Note that the use of `std::array::IntoIter::new` is only necessary if you need to
/// support Rust 1.51 or 1.52. If your minimum supported Rust version is 1.53 or greater,
/// you can pass an array directly.
#[derive(Debug)]
pub struct GpuConstraints<F: Field, C: Into<GpuConstraint<F>>, Iter: IntoIterator<Item = C>> {
    selector: GpuExpression<F>,
    constraints: Iter,
}

impl<F: Field, C: Into<GpuConstraint<F>>, Iter: IntoIterator<Item = C>> GpuConstraints<F, C, Iter> {
    /// Constructs a set of constraints that are controlled by the given selector.
    ///
    /// Each constraint `c` in `iterator` will be converted into the constraint
    /// `selector * c`.
    pub fn with_selector(selector: GpuExpression<F>, constraints: Iter) -> Self {
        GpuConstraints {
            selector,
            constraints,
        }
    }
}

fn apply_selector_to_constraint<F: Field, C: Into<GpuConstraint<F>>>(
    (selector, c): (GpuExpression<F>, C),
) -> GpuConstraint<F> {
    let constraint: GpuConstraint<F> = c.into();
    GpuConstraint {
        name: constraint.name,
        poly: selector * constraint.poly,
    }
}

type ApplySelectorToConstraint<F, C> = fn((GpuExpression<F>, C)) -> GpuConstraint<F>;
type ConstraintsIterator<F, C, I> = std::iter::Map<
    std::iter::Zip<std::iter::Repeat<GpuExpression<F>>, I>,
    ApplySelectorToConstraint<F, C>,
>;

impl<F: Field, C: Into<GpuConstraint<F>>, Iter: IntoIterator<Item = C>> IntoIterator
    for GpuConstraints<F, C, Iter>
{
    type Item = GpuConstraint<F>;
    type IntoIter = ConstraintsIterator<F, C, Iter::IntoIter>;

    fn into_iter(self) -> Self::IntoIter {
        std::iter::repeat(self.selector)
            .zip(self.constraints)
            .map(apply_selector_to_constraint)
    }
}

/// GpuGate
#[derive(Clone, Debug)]
pub struct GpuGate<F: Field> {
    pub(crate) name: String,
    pub(crate) constraint_names: Vec<String>,
    pub(crate) polys: Vec<GpuExpression<F>>,
    /// Queried selectors, tracked separately for gate debug checks. Kept for
    /// parity with the canonical `Gate`; the GPU evaluator does not read them.
    #[allow(dead_code)]
    pub(crate) queried_selectors: Vec<GpuSelector>,
    #[allow(dead_code)]
    pub(crate) queried_cells: Vec<GpuVirtualCell>,
}

impl<F: Field> GpuGate<F> {
    /// Returns the gate name.
    pub fn name(&self) -> &str {
        self.name.as_str()
    }

    /// Returns the name of the constraint at index `constraint_index`.
    pub fn constraint_name(&self, constraint_index: usize) -> &str {
        self.constraint_names[constraint_index].as_str()
    }

    /// Returns constraints of this gate
    pub fn polynomials(&self) -> &[GpuExpression<F>] {
        &self.polys
    }

    #[allow(dead_code)]
    pub(crate) fn queried_selectors(&self) -> &[GpuSelector] {
        &self.queried_selectors
    }

    #[allow(dead_code)]
    pub(crate) fn queried_cells(&self) -> &[GpuVirtualCell] {
        &self.queried_cells
    }
}

/// This is a description of the circuit environment, such as the gate, column and
/// permutation arrangements.
#[derive(Debug, Clone)]
pub struct GpuConstraintSystem<F: Field> {
    pub(crate) num_fixed_columns: usize,
    pub num_advice_columns: usize,
    pub num_instance_columns: usize,
    pub(crate) num_selectors: usize,
    pub(crate) num_challenges: usize,

    /// Contains the phase for each advice column. Should have same length as num_advice_columns.
    pub(crate) advice_column_phase: Vec<sealed::Phase>,
    /// Contains the phase for each challenge. Should have same length as num_challenges.
    pub(crate) challenge_phase: Vec<sealed::Phase>,

    /// This is a cached vector that maps virtual selectors to the concrete
    /// fixed column that they were compressed into. This is just used by dev
    /// tooling right now.
    pub(crate) selector_map: Vec<GpuColumn<GpuFixed>>,
    pub gates: Vec<GpuGate<F>>,
    pub advice_queries: Vec<(GpuColumn<GpuAdvice>, Rotation)>,
    // Contains an integer for each advice column
    // identifying how many distinct queries it has
    // so far; should be same length as num_advice_columns.
    num_advice_queries: Vec<usize>,
    pub instance_queries: Vec<(GpuColumn<GpuInstance>, Rotation)>,
    pub fixed_queries: Vec<(GpuColumn<GpuFixed>, Rotation)>,

    // Permutation argument for performing equality constraints
    pub permutation: permutation::Argument,

    // Vector of lookup arguments, where each corresponds to a sequence of
    // input expressions and a sequence of table expressions involved in the lookup.
    pub lookups: Vec<lookup::Argument<F>>,

    // List of indexes of GpuFixed columns which are associated to a circuit-general GpuColumn tied to their annotation.
    pub(crate) general_column_annotations: HashMap<metadata::Column, String>,

    // Vector of fixed columns, which can be used to store constant values
    // that are copied into advice columns.
    pub(crate) constants: Vec<GpuColumn<GpuFixed>>,

    pub(crate) minimum_degree: Option<usize>,
}

impl<F: Field> Default for GpuConstraintSystem<F> {
    fn default() -> GpuConstraintSystem<F> {
        GpuConstraintSystem {
            num_fixed_columns: 0,
            num_advice_columns: 0,
            num_instance_columns: 0,
            num_selectors: 0,
            num_challenges: 0,
            advice_column_phase: Vec::new(),
            challenge_phase: Vec::new(),
            selector_map: vec![],
            gates: vec![],
            fixed_queries: Vec::new(),
            advice_queries: Vec::new(),
            num_advice_queries: Vec::new(),
            instance_queries: Vec::new(),
            permutation: permutation::Argument::new(),
            lookups: Vec::new(),
            general_column_annotations: HashMap::new(),
            constants: vec![],
            minimum_degree: None,
        }
    }
}

impl<F: Field> GpuConstraintSystem<F> {
    /// Enables this fixed column to be used for global constant assignments.
    ///
    /// # Side-effects
    ///
    /// The column will be equality-enabled.
    pub fn enable_constant(&mut self, column: GpuColumn<GpuFixed>) {
        if !self.constants.contains(&column) {
            self.constants.push(column);
            self.enable_equality(column);
        }
    }

    /// Enable the ability to enforce equality over cells in this column
    pub fn enable_equality<C: Into<GpuColumn<GpuAny>>>(&mut self, column: C) {
        let column = column.into();
        self.query_any_index(column, Rotation::cur());
        self.permutation.add_column(column);
    }

    /// Add a lookup argument for some input expressions and table columns.
    ///
    /// `table_map` returns a map between input expressions and the table columns
    /// they need to match.
    pub fn lookup<S: AsRef<str>>(
        &mut self,
        name: S,
        table_map: impl FnOnce(&mut GpuVirtualCells<'_, F>) -> Vec<(GpuExpression<F>, GpuTableColumn)>,
    ) -> usize {
        let mut cells = GpuVirtualCells::new(self);
        let table_map = table_map(&mut cells)
            .into_iter()
            .map(|(mut input, table)| {
                if input.contains_simple_selector() {
                    panic!("expression containing simple selector supplied to lookup argument");
                }
                let mut table = cells.query_fixed(table.inner(), Rotation::cur());
                input.query_cells(&mut cells);
                table.query_cells(&mut cells);
                (input, table)
            })
            .collect();
        let index = self.lookups.len();

        self.lookups
            .push(lookup::Argument::new(name.as_ref(), table_map));

        index
    }

    /// Add a lookup argument for some input expressions and table expressions.
    ///
    /// `table_map` returns a map between input expressions and the table expressions
    /// they need to match.
    pub fn lookup_any<S: AsRef<str>>(
        &mut self,
        name: S,
        table_map: impl FnOnce(&mut GpuVirtualCells<'_, F>) -> Vec<(GpuExpression<F>, GpuExpression<F>)>,
    ) -> usize {
        let mut cells = GpuVirtualCells::new(self);
        let table_map = table_map(&mut cells)
            .into_iter()
            .map(|(mut input, mut table)| {
                input.query_cells(&mut cells);
                table.query_cells(&mut cells);
                (input, table)
            })
            .collect();
        let index = self.lookups.len();

        self.lookups
            .push(lookup::Argument::new(name.as_ref(), table_map));

        index
    }

    fn query_fixed_index(&mut self, column: GpuColumn<GpuFixed>, at: Rotation) -> usize {
        // Return existing query, if it exists
        for (index, fixed_query) in self.fixed_queries.iter().enumerate() {
            if fixed_query == &(column, at) {
                return index;
            }
        }

        // Make a new query
        let index = self.fixed_queries.len();
        self.fixed_queries.push((column, at));

        index
    }

    pub(crate) fn query_advice_index(
        &mut self,
        column: GpuColumn<GpuAdvice>,
        at: Rotation,
    ) -> usize {
        // Return existing query, if it exists
        for (index, advice_query) in self.advice_queries.iter().enumerate() {
            if advice_query == &(column, at) {
                return index;
            }
        }

        // Make a new query
        let index = self.advice_queries.len();
        self.advice_queries.push((column, at));
        self.num_advice_queries[column.index] += 1;

        index
    }

    fn query_instance_index(&mut self, column: GpuColumn<GpuInstance>, at: Rotation) -> usize {
        // Return existing query, if it exists
        for (index, instance_query) in self.instance_queries.iter().enumerate() {
            if instance_query == &(column, at) {
                return index;
            }
        }

        // Make a new query
        let index = self.instance_queries.len();
        self.instance_queries.push((column, at));

        index
    }

    fn query_any_index(&mut self, column: GpuColumn<GpuAny>, at: Rotation) -> usize {
        match column.column_type() {
            GpuAny::Advice(_) => {
                self.query_advice_index(GpuColumn::<GpuAdvice>::try_from(column).unwrap(), at)
            }
            GpuAny::Fixed => {
                self.query_fixed_index(GpuColumn::<GpuFixed>::try_from(column).unwrap(), at)
            }
            GpuAny::Instance => {
                self.query_instance_index(GpuColumn::<GpuInstance>::try_from(column).unwrap(), at)
            }
        }
    }

    pub(crate) fn get_advice_query_index(
        &self,
        column: GpuColumn<GpuAdvice>,
        at: Rotation,
    ) -> usize {
        for (index, advice_query) in self.advice_queries.iter().enumerate() {
            if advice_query == &(column, at) {
                return index;
            }
        }

        panic!("get_advice_query_index called for non-existent query");
    }

    pub(crate) fn get_fixed_query_index(&self, column: GpuColumn<GpuFixed>, at: Rotation) -> usize {
        for (index, fixed_query) in self.fixed_queries.iter().enumerate() {
            if fixed_query == &(column, at) {
                return index;
            }
        }

        panic!("get_fixed_query_index called for non-existent query");
    }

    pub(crate) fn get_instance_query_index(
        &self,
        column: GpuColumn<GpuInstance>,
        at: Rotation,
    ) -> usize {
        for (index, instance_query) in self.instance_queries.iter().enumerate() {
            if instance_query == &(column, at) {
                return index;
            }
        }

        panic!("get_instance_query_index called for non-existent query");
    }

    pub fn get_any_query_index(&self, column: GpuColumn<GpuAny>, at: Rotation) -> usize {
        match column.column_type() {
            GpuAny::Advice(_) => {
                self.get_advice_query_index(GpuColumn::<GpuAdvice>::try_from(column).unwrap(), at)
            }
            GpuAny::Fixed => {
                self.get_fixed_query_index(GpuColumn::<GpuFixed>::try_from(column).unwrap(), at)
            }
            GpuAny::Instance => self
                .get_instance_query_index(GpuColumn::<GpuInstance>::try_from(column).unwrap(), at),
        }
    }

    /// Sets the minimum degree required by the circuit, which can be set to a
    /// larger amount than actually needed. This can be used, for example, to
    /// force the permutation argument to involve more columns in the same set.
    pub fn set_minimum_degree(&mut self, degree: usize) {
        self.minimum_degree = Some(degree);
    }

    /// Creates a new gate.
    ///
    /// # Panics
    ///
    /// A gate is required to contain polynomial constraints. This method will panic if
    /// `constraints` returns an empty iterator.
    pub fn create_gate<C: Into<GpuConstraint<F>>, Iter: IntoIterator<Item = C>, S: AsRef<str>>(
        &mut self,
        name: S,
        constraints: impl FnOnce(&mut GpuVirtualCells<'_, F>) -> Iter,
    ) {
        let mut cells = GpuVirtualCells::new(self);
        let constraints = constraints(&mut cells);
        let (constraint_names, polys): (_, Vec<_>) = constraints
            .into_iter()
            .map(|c| c.into())
            .map(|mut c: GpuConstraint<F>| {
                c.poly.query_cells(&mut cells);
                (c.name, c.poly)
            })
            .unzip();

        let queried_selectors = cells.queried_selectors;
        let queried_cells = cells.queried_cells;

        assert!(
            !polys.is_empty(),
            "Gates must contain at least one constraint."
        );

        self.gates.push(GpuGate {
            name: name.as_ref().to_string(),
            constraint_names,
            polys,
            queried_selectors,
            queried_cells,
        });
    }

    /// This will compress selectors together depending on their provided
    /// assignments. This `GpuConstraintSystem` will then be modified to add new
    /// fixed columns (representing the actual selectors) and will return the
    /// polynomials for those columns. Finally, an internal map is updated to
    /// find which fixed column corresponds with a given `GpuSelector`.
    ///
    /// Do not call this twice. Yes, this should be a builder pattern instead.
    pub fn compress_selectors(mut self, selectors: Vec<Vec<bool>>) -> (Self, Vec<Vec<F>>) {
        // The number of provided selector assignments must be the number we
        // counted for this constraint system.
        assert_eq!(selectors.len(), self.num_selectors);

        // Compute the maximal degree of every selector. We only consider the
        // expressions in gates, as lookup arguments cannot support simple
        // selectors. Selectors that are complex or do not appear in any gates
        // will have degree zero.
        let mut degrees = vec![0; selectors.len()];
        for expr in self.gates.iter().flat_map(|gate| gate.polys.iter()) {
            if let Some(selector) = expr.extract_simple_selector() {
                degrees[selector.0] = max(degrees[selector.0], expr.degree());
            }
        }

        // We will not increase the degree of the constraint system, so we limit
        // ourselves to the largest existing degree constraint.
        let max_degree = self.degree();

        let mut new_columns = vec![];
        let (polys, selector_assignment) = compress_selectors::process(
            selectors
                .into_iter()
                .zip(degrees)
                .enumerate()
                .map(
                    |(i, (activations, max_degree))| compress_selectors::SelectorDescription {
                        selector: i,
                        activations,
                        max_degree,
                    },
                )
                .collect(),
            max_degree,
            || {
                let column = self.fixed_column();
                new_columns.push(column);
                GpuExpression::Fixed(GpuFixedQuery {
                    index: Some(self.query_fixed_index(column, Rotation::cur())),
                    column_index: column.index,
                    rotation: Rotation::cur(),
                })
            },
        );

        let mut selector_map = vec![None; selector_assignment.len()];
        let mut selector_replacements = vec![None; selector_assignment.len()];
        for assignment in selector_assignment {
            selector_replacements[assignment.selector] = Some(assignment.expression);
            selector_map[assignment.selector] = Some(new_columns[assignment.combination_index]);
        }

        self.selector_map = selector_map
            .into_iter()
            .map(|a| a.unwrap())
            .collect::<Vec<_>>();
        let selector_replacements = selector_replacements
            .into_iter()
            .map(|a| a.unwrap())
            .collect::<Vec<_>>();
        self.replace_selectors_with_fixed(&selector_replacements);

        (self, polys)
    }

    /// Does not combine selectors and directly replaces them everywhere with fixed columns.
    pub fn directly_convert_selectors_to_fixed(
        mut self,
        selectors: Vec<Vec<bool>>,
    ) -> (Self, Vec<Vec<F>>) {
        // The number of provided selector assignments must be the number we
        // counted for this constraint system.
        assert_eq!(selectors.len(), self.num_selectors);

        let (polys, selector_replacements): (Vec<_>, Vec<_>) = selectors
            .into_iter()
            .map(|selector| {
                let poly = selector
                    .iter()
                    .map(|b| if *b { F::ONE } else { F::ZERO })
                    .collect::<Vec<_>>();
                let column = self.fixed_column();
                let rotation = Rotation::cur();
                let expr = GpuExpression::Fixed(GpuFixedQuery {
                    index: Some(self.query_fixed_index(column, rotation)),
                    column_index: column.index,
                    rotation,
                });
                (poly, expr)
            })
            .unzip();

        self.replace_selectors_with_fixed(&selector_replacements);
        self.num_selectors = 0;

        (self, polys)
    }

    fn replace_selectors_with_fixed(&mut self, selector_replacements: &[GpuExpression<F>]) {
        fn replace_selectors<F: Field>(
            expr: &mut GpuExpression<F>,
            selector_replacements: &[GpuExpression<F>],
            must_be_nonsimple: bool,
        ) {
            *expr = expr.evaluate(
                &|constant| GpuExpression::Constant(constant),
                &|selector| {
                    if must_be_nonsimple {
                        // Simple selectors are prohibited from appearing in
                        // expressions in the lookup argument by
                        // `GpuConstraintSystem`.
                        assert!(!selector.is_simple());
                    }

                    selector_replacements[selector.0].clone()
                },
                &|query| GpuExpression::Fixed(query),
                &|query| GpuExpression::Advice(query),
                &|query| GpuExpression::Instance(query),
                &|challenge| GpuExpression::Challenge(challenge),
                &|a| -a,
                &|a, b| a + b,
                &|a, b| a * b,
                &|a, f| a * f,
            );
        }

        // Substitute selectors for the real fixed columns in all gates
        for expr in self.gates.iter_mut().flat_map(|gate| gate.polys.iter_mut()) {
            replace_selectors(expr, selector_replacements, false);
        }

        // Substitute non-simple selectors for the real fixed columns in all
        // lookup expressions
        for expr in self.lookups.iter_mut().flat_map(|lookup| {
            lookup
                .input_expressions
                .iter_mut()
                .chain(lookup.table_expressions.iter_mut())
        }) {
            replace_selectors(expr, selector_replacements, true);
        }
    }

    /// Allocate a new (simple) selector. Simple selectors cannot be added to
    /// expressions nor multiplied by other expressions containing simple
    /// selectors. Also, simple selectors may not appear in lookup argument
    /// inputs.
    pub fn selector(&mut self) -> GpuSelector {
        let index = self.num_selectors;
        self.num_selectors += 1;
        GpuSelector(index, true)
    }

    /// Allocate a new complex selector that can appear anywhere
    /// within expressions.
    pub fn complex_selector(&mut self) -> GpuSelector {
        let index = self.num_selectors;
        self.num_selectors += 1;
        GpuSelector(index, false)
    }

    /// Allocates a new fixed column that can be used in a lookup table.
    pub fn lookup_table_column(&mut self) -> GpuTableColumn {
        GpuTableColumn {
            inner: self.fixed_column(),
        }
    }

    /// Allocate a new fixed column
    pub fn fixed_column(&mut self) -> GpuColumn<GpuFixed> {
        let tmp = GpuColumn {
            index: self.num_fixed_columns,
            column_type: GpuFixed,
        };
        self.num_fixed_columns += 1;
        tmp
    }

    /// Allocate a new advice column at `GpuFirstPhase`
    pub fn advice_column(&mut self) -> GpuColumn<GpuAdvice> {
        self.advice_column_in(GpuFirstPhase)
    }

    /// Allocate a new advice column in given phase
    pub fn advice_column_in<P: GpuPhase>(&mut self, phase: P) -> GpuColumn<GpuAdvice> {
        let phase = phase.to_sealed();
        if let Some(previous_phase) = phase.prev() {
            self.assert_phase_exists(
                previous_phase,
                format!("Column<Advice> in later phase {:?}", phase).as_str(),
            );
        }

        let tmp = GpuColumn {
            index: self.num_advice_columns,
            column_type: GpuAdvice { phase },
        };
        self.num_advice_columns += 1;
        self.num_advice_queries.push(0);
        self.advice_column_phase.push(phase);
        tmp
    }

    /// Allocate a new instance column
    pub fn instance_column(&mut self) -> GpuColumn<GpuInstance> {
        let tmp = GpuColumn {
            index: self.num_instance_columns,
            column_type: GpuInstance,
        };
        self.num_instance_columns += 1;
        tmp
    }

    /// Requests a challenge that is usable after the given phase.
    pub fn challenge_usable_after<P: GpuPhase>(&mut self, phase: P) -> GpuChallenge {
        let phase = phase.to_sealed();
        self.assert_phase_exists(
            phase,
            format!("Challenge usable after phase {:?}", phase).as_str(),
        );

        let tmp = GpuChallenge {
            index: self.num_challenges,
            phase,
        };
        self.num_challenges += 1;
        self.challenge_phase.push(phase);
        tmp
    }

    /// Helper funciotn to assert phase exists, to make sure phase-aware resources
    /// are allocated in order, and to avoid any phase to be skipped accidentally
    /// to cause unexpected issue in the future.
    fn assert_phase_exists(&self, phase: sealed::Phase, resource: &str) {
        self.advice_column_phase
            .iter()
            .find(|advice_column_phase| **advice_column_phase == phase)
            .unwrap_or_else(|| {
                panic!(
                    "No GpuColumn<GpuAdvice> is used in phase {:?} while allocating a new {:?}",
                    phase, resource
                )
            });
    }

    pub(crate) fn phases(&self) -> impl Iterator<Item = sealed::Phase> {
        let max_phase = self
            .advice_column_phase
            .iter()
            .max()
            .map(|phase| phase.0)
            .unwrap_or_default();
        (0..=max_phase).map(sealed::Phase)
    }

    /// Compute the degree of the constraint system (the maximum degree of all
    /// constraints).
    pub fn degree(&self) -> usize {
        // The permutation argument will serve alongside the gates, so must be
        // accounted for.
        let mut degree = self.permutation.required_degree();

        // The lookup argument also serves alongside the gates and must be accounted
        // for.
        degree = std::cmp::max(
            degree,
            self.lookups
                .iter()
                .map(|l| l.required_degree())
                .max()
                .unwrap_or(1),
        );

        // Account for each gate to ensure our quotient polynomial is the
        // correct degree and that our extended domain is the right size.
        degree = std::cmp::max(
            degree,
            self.gates
                .iter()
                .flat_map(|gate| gate.polynomials().iter().map(|poly| poly.degree()))
                .max()
                .unwrap_or(0),
        );

        fn get_max_degree() -> usize {
            var("MAX_DEGREE")
                .unwrap_or_else(|_| "5".to_string())
                .parse()
                .expect("Cannot parse MAX_DEGREE env var as usize")
        }
        degree = std::cmp::min(degree, get_max_degree());

        std::cmp::max(degree, self.minimum_degree.unwrap_or(1))
    }

    /// Compute the number of blinding factors necessary to perfectly blind
    /// each of the prover's witness polynomials.
    pub fn blinding_factors(&self) -> usize {
        // All of the prover's advice columns are evaluated at no more than
        let factors = *self.num_advice_queries.iter().max().unwrap_or(&1);
        // distinct points during gate checks.

        // - The permutation argument witness polynomials are evaluated at most 3 times.
        // - Each lookup argument has independent witness polynomials, and they are
        //   evaluated at most 2 times.
        let factors = std::cmp::max(3, factors);

        // Each polynomial is evaluated at most an additional time during
        // multiopen (at x_3 to produce q_evals):
        let factors = factors + 1;

        // h(x) is derived by the other evaluations so it does not reveal
        // anything; in fact it does not even appear in the proof.

        // h(x_3) is also not revealed; the verifier only learns a single
        // evaluation of a polynomial in x_1 which has h(x_3) and another random
        // polynomial evaluated at x_3 as coefficients -- this random polynomial
        // is "random_poly" in the vanishing argument.

        // Add an additional blinding factor as a slight defense against
        // off-by-one errors.
        factors + 1
    }

    /// Returns the minimum necessary rows that need to exist in order to
    /// account for e.g. blinding factors.
    pub fn minimum_rows(&self) -> usize {
        self.blinding_factors() // m blinding factors
            + 1 // for l_{-(m + 1)} (l_last)
            + 1 // for l_0 (just for extra breathing room for the permutation
                // argument, to essentially force a separation in the
                // permutation polynomial between the roles of l_last, l_0
                // and the interstitial values.)
            + 1 // for at least one row
    }

    /// Returns number of fixed columns
    pub fn num_fixed_columns(&self) -> usize {
        self.num_fixed_columns
    }

    /// Returns number of advice columns
    pub fn num_advice_columns(&self) -> usize {
        self.num_advice_columns
    }

    /// Returns number of instance columns
    pub fn num_instance_columns(&self) -> usize {
        self.num_instance_columns
    }

    /// Returns number of selectors
    pub fn num_selectors(&self) -> usize {
        self.num_selectors
    }

    /// Returns number of challenges
    pub fn num_challenges(&self) -> usize {
        self.num_challenges
    }

    /// Returns phase of advice columns
    pub fn advice_column_phase(&self) -> Vec<u8> {
        self.advice_column_phase
            .iter()
            .map(|phase| phase.0)
            .collect()
    }

    /// Returns phase of challenges
    pub fn challenge_phase(&self) -> Vec<u8> {
        self.challenge_phase.iter().map(|phase| phase.0).collect()
    }

    /// Returns gates
    pub fn gates(&self) -> &Vec<GpuGate<F>> {
        &self.gates
    }

    /// Returns general column annotations
    pub fn general_column_annotations(&self) -> &HashMap<metadata::Column, String> {
        &self.general_column_annotations
    }

    /// Returns advice queries
    pub fn advice_queries(&self) -> &Vec<(GpuColumn<GpuAdvice>, Rotation)> {
        &self.advice_queries
    }

    /// Returns instance queries
    pub fn instance_queries(&self) -> &Vec<(GpuColumn<GpuInstance>, Rotation)> {
        &self.instance_queries
    }

    /// Returns fixed queries
    pub fn fixed_queries(&self) -> &Vec<(GpuColumn<GpuFixed>, Rotation)> {
        &self.fixed_queries
    }

    /// Returns permutation argument
    pub fn permutation(&self) -> &permutation::Argument {
        &self.permutation
    }

    /// Returns lookup arguments
    pub fn lookups(&self) -> &Vec<lookup::Argument<F>> {
        &self.lookups
    }

    /// Returns constants
    pub fn constants(&self) -> &Vec<GpuColumn<GpuFixed>> {
        &self.constants
    }
}

/// Exposes the "virtual cells" that can be queried while creating a custom gate or lookup
/// table.
#[derive(Debug)]
pub struct GpuVirtualCells<'a, F: Field> {
    meta: &'a mut GpuConstraintSystem<F>,
    queried_selectors: Vec<GpuSelector>,
    queried_cells: Vec<GpuVirtualCell>,
}

impl<'a, F: Field> GpuVirtualCells<'a, F> {
    fn new(meta: &'a mut GpuConstraintSystem<F>) -> Self {
        GpuVirtualCells {
            meta,
            queried_selectors: vec![],
            queried_cells: vec![],
        }
    }

    /// Query a selector at the current position.
    pub fn query_selector(&mut self, selector: GpuSelector) -> GpuExpression<F> {
        self.queried_selectors.push(selector);
        GpuExpression::Selector(selector)
    }

    /// Query a fixed column at a relative position
    pub fn query_fixed(&mut self, column: GpuColumn<GpuFixed>, at: Rotation) -> GpuExpression<F> {
        self.queried_cells.push((column, at).into());
        GpuExpression::Fixed(GpuFixedQuery {
            index: Some(self.meta.query_fixed_index(column, at)),
            column_index: column.index,
            rotation: at,
        })
    }

    /// Query an advice column at a relative position
    pub fn query_advice(&mut self, column: GpuColumn<GpuAdvice>, at: Rotation) -> GpuExpression<F> {
        self.queried_cells.push((column, at).into());
        GpuExpression::Advice(GpuAdviceQuery {
            index: Some(self.meta.query_advice_index(column, at)),
            column_index: column.index,
            rotation: at,
            phase: column.column_type().phase,
        })
    }

    /// Query an instance column at a relative position
    pub fn query_instance(
        &mut self,
        column: GpuColumn<GpuInstance>,
        at: Rotation,
    ) -> GpuExpression<F> {
        self.queried_cells.push((column, at).into());
        GpuExpression::Instance(GpuInstanceQuery {
            index: Some(self.meta.query_instance_index(column, at)),
            column_index: column.index,
            rotation: at,
        })
    }

    /// Query an GpuAny column at a relative position
    pub fn query_any<C: Into<GpuColumn<GpuAny>>>(
        &mut self,
        column: C,
        at: Rotation,
    ) -> GpuExpression<F> {
        let column = column.into();
        match column.column_type() {
            GpuAny::Advice(_) => {
                self.query_advice(GpuColumn::<GpuAdvice>::try_from(column).unwrap(), at)
            }
            GpuAny::Fixed => self.query_fixed(GpuColumn::<GpuFixed>::try_from(column).unwrap(), at),
            GpuAny::Instance => {
                self.query_instance(GpuColumn::<GpuInstance>::try_from(column).unwrap(), at)
            }
        }
    }

    /// Query a challenge
    pub fn query_challenge(&mut self, challenge: GpuChallenge) -> GpuExpression<F> {
        GpuExpression::Challenge(challenge)
    }
}

// Conversions from the canonical halo2-axiom constraint-system types into the
// GPU-crate forks. Equivalence-critical: a bug here silently desyncs the
// rebuilt GPU cs/Evaluator/Arguments from what keygen produced. Query `index`
// is set to `None` (not readable from halo2-axiom and unread by the GPU prover)
// and backfilled below for the verifier path.

impl<F: Field> From<&halo2_axiom::plonk::Expression<F>> for GpuExpression<F> {
    fn from(e: &halo2_axiom::plonk::Expression<F>) -> Self {
        use halo2_axiom::plonk::Expression as HExpr;
        match e {
            HExpr::Constant(f) => GpuExpression::Constant(*f),
            HExpr::Selector(s) => GpuExpression::Selector(GpuSelector(s.index(), s.is_simple())),
            HExpr::Fixed(q) => GpuExpression::Fixed(GpuFixedQuery {
                index: None,
                column_index: q.column_index(),
                rotation: q.rotation(),
            }),
            HExpr::Advice(q) => GpuExpression::Advice(GpuAdviceQuery {
                index: None,
                column_index: q.column_index(),
                rotation: q.rotation(),
                phase: sealed::Phase(q.phase()),
            }),
            HExpr::Instance(q) => GpuExpression::Instance(GpuInstanceQuery {
                index: None,
                column_index: q.column_index(),
                rotation: q.rotation(),
            }),
            HExpr::Challenge(c) => GpuExpression::Challenge(GpuChallenge {
                index: c.index(),
                phase: sealed::Phase(c.phase()),
            }),
            HExpr::Negated(a) => GpuExpression::Negated(Box::new(GpuExpression::from(a.as_ref()))),
            HExpr::Sum(a, b) => GpuExpression::Sum(
                Box::new(GpuExpression::from(a.as_ref())),
                Box::new(GpuExpression::from(b.as_ref())),
            ),
            HExpr::Product(a, b) => GpuExpression::Product(
                Box::new(GpuExpression::from(a.as_ref())),
                Box::new(GpuExpression::from(b.as_ref())),
            ),
            HExpr::Scaled(a, f) => {
                GpuExpression::Scaled(Box::new(GpuExpression::from(a.as_ref())), *f)
            }
        }
    }
}

impl From<&halo2_axiom::plonk::Column<halo2_axiom::plonk::Fixed>> for GpuColumn<GpuFixed> {
    fn from(c: &halo2_axiom::plonk::Column<halo2_axiom::plonk::Fixed>) -> Self {
        GpuColumn {
            index: c.index(),
            column_type: GpuFixed,
        }
    }
}

impl From<&halo2_axiom::plonk::Column<halo2_axiom::plonk::Instance>> for GpuColumn<GpuInstance> {
    fn from(c: &halo2_axiom::plonk::Column<halo2_axiom::plonk::Instance>) -> Self {
        GpuColumn {
            index: c.index(),
            column_type: GpuInstance,
        }
    }
}

impl From<&halo2_axiom::plonk::Column<halo2_axiom::plonk::Advice>> for GpuColumn<GpuAdvice> {
    fn from(c: &halo2_axiom::plonk::Column<halo2_axiom::plonk::Advice>) -> Self {
        GpuColumn {
            index: c.index(),
            column_type: GpuAdvice {
                phase: sealed::Phase(c.column_type().phase()),
            },
        }
    }
}

impl From<&halo2_axiom::plonk::Column<halo2_axiom::plonk::Any>> for GpuColumn<GpuAny> {
    fn from(c: &halo2_axiom::plonk::Column<halo2_axiom::plonk::Any>) -> Self {
        let column_type = match c.column_type() {
            halo2_axiom::plonk::Any::Advice(a) => GpuAny::Advice(GpuAdvice {
                phase: sealed::Phase(a.phase()),
            }),
            halo2_axiom::plonk::Any::Fixed => GpuAny::Fixed,
            halo2_axiom::plonk::Any::Instance => GpuAny::Instance,
        };
        GpuColumn {
            index: c.index(),
            column_type,
        }
    }
}

impl<F: Field> From<&halo2_axiom::plonk::Gate<F>> for GpuGate<F> {
    fn from(g: &halo2_axiom::plonk::Gate<F>) -> Self {
        GpuGate {
            name: g.name().to_string(),
            constraint_names: Vec::new(),
            polys: g.polynomials().iter().map(GpuExpression::from).collect(),
            queried_selectors: Vec::new(),
            queried_cells: Vec::new(),
        }
    }
}

impl<F: Field> From<&halo2_axiom::plonk::ConstraintSystem<F>> for GpuConstraintSystem<F> {
    fn from(cs: &halo2_axiom::plonk::ConstraintSystem<F>) -> Self {
        let mut out = GpuConstraintSystem {
            num_fixed_columns: cs.num_fixed_columns(),
            num_advice_columns: cs.num_advice_columns(),
            num_instance_columns: cs.num_instance_columns(),
            num_selectors: cs.num_selectors(),
            num_challenges: cs.num_challenges(),
            advice_column_phase: cs
                .advice_column_phase()
                .into_iter()
                .map(sealed::Phase)
                .collect(),
            challenge_phase: cs
                .challenge_phase()
                .into_iter()
                .map(sealed::Phase)
                .collect(),
            selector_map: cs.selector_map().iter().map(GpuColumn::from).collect(),
            gates: cs.gates().iter().map(GpuGate::from).collect(),
            advice_queries: cs
                .advice_queries()
                .iter()
                .map(|(c, r)| (GpuColumn::from(c), *r))
                .collect(),
            num_advice_queries: cs.num_advice_queries().clone(),
            instance_queries: cs
                .instance_queries()
                .iter()
                .map(|(c, r)| (GpuColumn::from(c), *r))
                .collect(),
            fixed_queries: cs
                .fixed_queries()
                .iter()
                .map(|(c, r)| (GpuColumn::from(c), *r))
                .collect(),
            permutation: permutation::Argument::from(cs.permutation()),
            // `lookup::Argument` is not nameable from this crate (halo2-axiom's
            // `lookup` module is private); rebuild it inline from public accessors.
            lookups: cs
                .lookups()
                .iter()
                .map(|la| lookup::Argument {
                    name: la.name().to_string(),
                    input_expressions: la
                        .input_expressions()
                        .iter()
                        .map(GpuExpression::from)
                        .collect(),
                    table_expressions: la
                        .table_expressions()
                        .iter()
                        .map(GpuExpression::from)
                        .collect(),
                })
                .collect(),
            // Dev-tooling metadata, unread post-keygen; rebuilt empty.
            general_column_annotations: HashMap::new(),
            constants: cs.constants().iter().map(GpuColumn::from).collect(),
            minimum_degree: cs.minimum_degree(),
        };
        // Backfill the per-query `index` the verifier reads, by position in the
        // order-preserving query lists (matches what keygen's cs assigns).
        let fq = out.fixed_queries.clone();
        let aq = out.advice_queries.clone();
        let iq = out.instance_queries.clone();
        for gate in &mut out.gates {
            for poly in &mut gate.polys {
                assign_expr_query_indices(poly, &fq, &aq, &iq);
            }
        }
        for lookup in &mut out.lookups {
            for e in lookup
                .input_expressions
                .iter_mut()
                .chain(lookup.table_expressions.iter_mut())
            {
                assign_expr_query_indices(e, &fq, &aq, &iq);
            }
        }
        out
    }
}

/// Sets each query's `index` to the position of its `(column, rotation)` in the
/// cs's query list, matching what `query_*_index` assigns at keygen time.
fn assign_expr_query_indices<F: Field>(
    expr: &mut GpuExpression<F>,
    fixed_queries: &[(GpuColumn<GpuFixed>, Rotation)],
    advice_queries: &[(GpuColumn<GpuAdvice>, Rotation)],
    instance_queries: &[(GpuColumn<GpuInstance>, Rotation)],
) {
    match expr {
        GpuExpression::Fixed(q) => {
            q.index = Some(
                fixed_queries
                    .iter()
                    .position(|(c, r)| c.index == q.column_index && *r == q.rotation)
                    .expect("fixed query must exist in cs.fixed_queries"),
            );
        }
        GpuExpression::Advice(q) => {
            q.index = Some(
                advice_queries
                    .iter()
                    .position(|(c, r)| {
                        c.index == q.column_index
                            && c.column_type.phase == q.phase
                            && *r == q.rotation
                    })
                    .expect("advice query must exist in cs.advice_queries"),
            );
        }
        GpuExpression::Instance(q) => {
            q.index = Some(
                instance_queries
                    .iter()
                    .position(|(c, r)| c.index == q.column_index && *r == q.rotation)
                    .expect("instance query must exist in cs.instance_queries"),
            );
        }
        GpuExpression::Negated(a) => {
            assign_expr_query_indices(a, fixed_queries, advice_queries, instance_queries)
        }
        GpuExpression::Sum(a, b) | GpuExpression::Product(a, b) => {
            assign_expr_query_indices(a, fixed_queries, advice_queries, instance_queries);
            assign_expr_query_indices(b, fixed_queries, advice_queries, instance_queries);
        }
        GpuExpression::Scaled(a, _) => {
            assign_expr_query_indices(a, fixed_queries, advice_queries, instance_queries)
        }
        GpuExpression::Constant(_) | GpuExpression::Selector(_) | GpuExpression::Challenge(_) => {}
    }
}
