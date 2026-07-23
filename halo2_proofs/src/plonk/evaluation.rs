use super::{GpuConstraintSystem, GpuExpression};

use crate::cpu::arithmetic::parallelize;
use crate::cuda::error::CudaStatus;
use crate::cuda::funcs::ColumnPool;
use crate::cuda::modules::QuotientLookupsGpu;
use crate::cuda::utils::{
    ensure_current_device_matches_ctx, query_device_free_bytes_for_chunking, FFITraitObject,
    HALO2_GPU_CTX,
};
use crate::cuda::HaloGpuError;
use crate::plonk::{lookup, permutation, GpuAny, GpuGate, GpuProvingKey};
use crate::poly::{
    Basis, Coeff, Device, DevicePolyExt, EvaluationDomain, ExtendedLagrangeCoeff, LagrangeCoeff,
    Polynomial, Rotation,
};
use ff::{Field, PrimeField, WithSmallOrderMulGroup};
use halo2curves::CurveAffine;
use openvm_cuda_common::copy::MemCopyH2D;
use openvm_cuda_common::d_buffer::DeviceBuffer;

// a view of the VK fields that is used by the evaluator
// this is useful for tests and separating out what's needed
pub(crate) struct EvaluatorVkView<'a, F: Field> {
    pub(crate) blinding_factors: usize,
    pub(crate) cs_degree: usize,
    pub(crate) permutation_argument: &'a permutation::Argument,
    pub(crate) domain: &'a EvaluationDomain<'a, F>,
}

/// Return the index in the polynomial of size `isize` after rotation `rot`.
pub(crate) fn get_rotation_idx(idx: usize, rot: i32, rot_scale: i32, isize: i32) -> usize {
    (((idx as i32) + (rot * rot_scale)).rem_euclid(isize)) as usize
}

/// Value used in a calculation
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd)]
pub enum ValueSource {
    /// This is a constant value
    Constant(usize),
    /// This is an intermediate value
    Intermediate(usize),
    /// This is a fixed column
    Fixed(usize, usize),
    /// This is an advice (witness) column
    Advice(usize, usize),
    /// This is an instance (external) column
    Instance(usize, usize),
    /// This is a challenge
    Challenge(usize),
    /// beta
    Beta(),
    /// gamma
    Gamma(),
    /// theta
    Theta(),
    /// y
    Y(),
    /// Previous value
    PreviousValue(),
}

impl Default for ValueSource {
    fn default() -> Self {
        ValueSource::Constant(0)
    }
}

impl ValueSource {
    /// Get the value for this source
    #[cfg(test)]
    pub fn get<F: Field, B: Basis>(
        &self,
        rotations: &[usize],
        constants: &[F],
        intermediates: &[F],
        fixed_values: &[Polynomial<F, B>],
        advice_values: &[Polynomial<F, B>],
        instance_values: &[Polynomial<F, B>],
        challenges: &[F],
        beta: &F,
        gamma: &F,
        theta: &F,
        y: &F,
        previous_value: &F,
    ) -> F {
        match self {
            ValueSource::Constant(idx) => constants[*idx],
            ValueSource::Intermediate(idx) => intermediates[*idx],
            ValueSource::Fixed(column_index, rotation) => {
                fixed_values[*column_index][rotations[*rotation]]
            }
            ValueSource::Advice(column_index, rotation) => {
                advice_values[*column_index][rotations[*rotation]]
            }
            ValueSource::Instance(column_index, rotation) => {
                instance_values[*column_index][rotations[*rotation]]
            }
            ValueSource::Challenge(index) => challenges[*index],
            ValueSource::Beta() => *beta,
            ValueSource::Gamma() => *gamma,
            ValueSource::Theta() => *theta,
            ValueSource::Y() => *y,
            ValueSource::PreviousValue() => *previous_value,
        }
    }
    pub fn encode(&self, rotations: &[i32]) -> u64 {
        let (src, idx, rotation) = match self {
            ValueSource::Fixed(column_index, rotation) => (0, *column_index, rotations[*rotation]),
            ValueSource::Instance(column_index, rotation) => {
                (1, *column_index, rotations[*rotation])
            }
            ValueSource::Advice(column_index, rotation) => (2, *column_index, rotations[*rotation]),
            ValueSource::Intermediate(idx) => (3, *idx, 0),
            ValueSource::Constant(idx) => (4, *idx, 0),
            ValueSource::Challenge(idx) => (5, *idx, 0),
            _ => unreachable!(),
        };

        let idx = idx as u64;
        let sign = (rotation < 0) as u64;
        let abs = rotation.unsigned_abs() as u64;
        // big-endian: 4-bit src | 20-bit idx | 15-bit rot abs | 1-bit rot sign, takes 40bit
        src | (idx << 4) | (abs << 24) | (sign << 39)
    }
}

/// Calculation
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Calculation {
    /// This is an addition
    Add(ValueSource, ValueSource),
    /// This is a subtraction
    Sub(ValueSource, ValueSource),
    /// This is a product
    Mul(ValueSource, ValueSource),
    /// This is a square
    Square(ValueSource),
    /// This is a double
    Double(ValueSource),
    /// This is a negation
    Negate(ValueSource),
    /// This is Horner's rule: `val = a; val = val * c + b[]`
    Horner(ValueSource, Vec<ValueSource>, ValueSource),
    /// This is a simple assignment
    Store(ValueSource),
}

#[derive(Clone, Debug, PartialEq)]
enum CombineType {
    Zero,
    One,
    NegOne,
    Two,
}

#[derive(Clone, Debug, PartialEq)]
enum CalcDegree {
    One,
    Two,
}

/// Packed device-consumable rule emitted by [`Calculation::encode`].
///
/// The `(Vec<CalcRule>, Vec<u64>)` pair returned by
/// [`GraphEvaluator::encode_for_device`] is the metadata format consumed
/// verbatim by the `_halo2_quotient*` FFI family: the host pointer is
/// cast `*const u128 as *const u64` and the device decoder
/// (`cuda/include/kernel/quotient.h` `Rule = uint4`) reads the same
/// 128 bits as a `uint4`. No byte-order normalisation is performed —
/// both host (x86_64) and device (NVIDIA SM) are little-endian.
///
/// # `ValueSource` — 40-bit packed `u64` (`ValueSource::encode`)
///
/// ```text
///   bit  0..3   src   : 4-bit tag    (Fixed=0, Instance=1, Advice=2,
///                                     Intermediate=3, Constant=4,
///                                     Challenge=5, Dummy=6)
///   bit  4..23  idx   : 20-bit column / intermediate / constant index
///   bit 24..38  rot   : 15-bit absolute rotation value
///   bit 39      sign  : 1-bit rotation sign (1 = negative)
/// ```
///
/// `Beta()`, `Gamma()`, `Theta()`, `Y()`, and `PreviousValue()` are
/// never encoded as a `ValueSource`; they appear only in the unused
/// start / factor slots of `Calculation::Horner` and are carried into
/// the kernel via `expr_constants` (slot 4 holds `theta`, etc.).
///
/// # `Calculation` — 128-bit packed `CalcRule` (`Calculation::encode`)
///
/// ```text
///   low u64  ( bit  0..63):
///     bit  0..39  a      : encoded ValueSource (operand a)
///     bit 40..43  c1     : 4-bit CombineType (Zero=0, One=1,
///                                             NegOne=2, Two=3)
///     bit 44..47  d      : 4-bit CalcDegree (One=0, Two=1)
///     bit 48..63  unused
///
///   high u64 ( bit 64..127):
///     bit  0..39  b      : encoded ValueSource (operand b)
///     bit 40..43  c2     : 4-bit CombineType
///     bit 44..63  unused
/// ```
///
/// The C++ side decodes via `decode_value` (`cuda/quotient/quotient.cu`)
/// and `evaluate` kernel-side unpack (`cuda/include/kernel/quotient.h`).
///
/// `Horner` is never emitted as a `CalcRule`; it terminates the graph
/// and is serialised separately via `Calculation::encode_vp` into a
/// flat `Vec<u64>` whose entries are individually `ValueSource::encode`
/// results.
#[derive(Clone, Copy, Debug)]
pub struct CalcRule(#[allow(dead_code)] pub u128); // note allow(dead_code) because the field is never read in rust, but read through FFI

impl Calculation {
    /// Get the resulting value of this calculation
    #[cfg(test)]
    pub fn evaluate<F: Field, B: Basis>(
        &self,
        rotations: &[usize],
        constants: &[F],
        intermediates: &[F],
        fixed_values: &[Polynomial<F, B>],
        advice_values: &[Polynomial<F, B>],
        instance_values: &[Polynomial<F, B>],
        challenges: &[F],
        beta: &F,
        gamma: &F,
        theta: &F,
        y: &F,
        previous_value: &F,
    ) -> F {
        let get_value = |value: &ValueSource| {
            value.get(
                rotations,
                constants,
                intermediates,
                fixed_values,
                advice_values,
                instance_values,
                challenges,
                beta,
                gamma,
                theta,
                y,
                previous_value,
            )
        };
        match self {
            Calculation::Add(a, b) => get_value(a) + get_value(b),
            Calculation::Sub(a, b) => get_value(a) - get_value(b),
            Calculation::Mul(a, b) => get_value(a) * get_value(b),
            Calculation::Square(v) => get_value(v).square(),
            Calculation::Double(v) => get_value(v).double(),
            Calculation::Negate(v) => -get_value(v),
            Calculation::Horner(start_value, parts, factor) => {
                let factor = get_value(factor);
                let mut value = get_value(start_value);
                for part in parts.iter() {
                    value = value * factor + get_value(part);
                }
                value
            }
            Calculation::Store(v) => get_value(v),
        }
    }

    fn encode(&self, rotations: &[i32]) -> CalcRule {
        let dummy_var = 6;
        let combines = [
            CombineType::Zero,
            CombineType::One,
            CombineType::NegOne,
            CombineType::Two,
        ];
        let degrees = [CalcDegree::One, CalcDegree::Two];

        let (a, b, c1, c2, d) = match self {
            Calculation::Add(a, b) => (
                a.encode(rotations),
                b.encode(rotations),
                CombineType::One,
                CombineType::One,
                CalcDegree::One,
            ),
            Calculation::Sub(a, b) => (
                a.encode(rotations),
                b.encode(rotations),
                CombineType::One,
                CombineType::NegOne,
                CalcDegree::One,
            ),
            Calculation::Mul(a, b) => (
                a.encode(rotations),
                b.encode(rotations),
                CombineType::One,
                CombineType::Zero,
                CalcDegree::Two,
            ),
            Calculation::Negate(a) => (
                a.encode(rotations),
                dummy_var,
                CombineType::NegOne,
                CombineType::Zero,
                CalcDegree::One,
            ),
            Calculation::Store(a) => (
                a.encode(rotations),
                dummy_var,
                CombineType::One,
                CombineType::Zero,
                CalcDegree::One,
            ),
            Calculation::Double(a) => (
                a.encode(rotations),
                dummy_var,
                CombineType::Two,
                CombineType::Zero,
                CalcDegree::One,
            ),
            Calculation::Square(a) => (
                a.encode(rotations),
                a.encode(rotations),
                CombineType::One,
                CombineType::Zero,
                CalcDegree::Two,
            ),
            _ => unreachable!(),
        };
        let c1 = combines.iter().position(|c| *c == c1).unwrap() as u64;
        let c2 = combines.iter().position(|c| *c == c2).unwrap() as u64;
        let d = degrees.iter().position(|deg| *deg == d).unwrap() as u64;

        // a: 40bit | c1: 4bit | d: 4bit | reseverd: 16bit
        let rule1 = a | (c1 << 40) | (d << 44);
        // b: 40bit | c2: 4bit | reseverd: 20bit
        let rule2 = b | (c2 << 40);
        let rule: u128 = (rule1 as u128) | ((rule2 as u128) << 64);
        CalcRule(rule)
    }

    fn encode_vp(&self, rotations: &[i32]) -> Vec<u64> {
        match self {
            Calculation::Horner(_, parts, _) => parts
                .iter()
                .map(|part| part.encode(rotations))
                .collect::<Vec<u64>>(),
            _ => unreachable!(),
        }
    }
}

/// Evaluator
#[derive(Clone, Default, Debug)]
pub struct Evaluator<C: CurveAffine> {
    ///  Custom gates evalution
    pub custom_gates: GraphEvaluator<C>,
    ///  Lookups evalution
    pub lookups: Vec<GraphEvaluator<C>>,
}

/// GraphEvaluator
#[derive(Clone, Debug)]
pub struct GraphEvaluator<C: CurveAffine> {
    /// Constants
    pub constants: Vec<C::ScalarExt>,
    /// Rotations
    pub rotations: Vec<i32>,
    /// Calculations
    pub calculations: Vec<CalculationInfo>,
    /// Number of intermediates
    pub num_intermediates: usize,
}

/// EvaluationData
#[cfg(test)]
#[derive(Default, Debug)]
pub struct EvaluationData<C: CurveAffine> {
    /// Intermediates
    pub intermediates: Vec<C::ScalarExt>,
    /// Rotations
    pub rotations: Vec<usize>,
}

/// CaluclationInfo
#[derive(Clone, Debug)]
pub struct CalculationInfo {
    /// Calculation
    pub calculation: Calculation,
    /// Target
    pub target: usize,
}

impl<C: CurveAffine> Evaluator<C> {
    pub fn new_inner(
        gates: &[GpuGate<C::ScalarExt>],
        lookups: &[lookup::Argument<C::ScalarExt>],
    ) -> Self {
        let mut ev = Evaluator::default();

        // Custom gates
        let mut parts = Vec::new();
        for gate in gates.iter() {
            parts.extend(
                gate.polynomials()
                    .iter()
                    .map(|poly| ev.custom_gates.add_expression(poly)),
            );
        }
        ev.custom_gates.add_calculation(Calculation::Horner(
            ValueSource::PreviousValue(),
            parts,
            ValueSource::Y(),
        ));

        // Lookups
        for lookup in lookups.iter() {
            let mut graph = GraphEvaluator::default();

            let mut evaluate_lc = |expressions: &Vec<GpuExpression<_>>| {
                let parts = expressions
                    .iter()
                    .map(|expr| graph.add_expression(expr))
                    .collect();
                graph.add_calculation(Calculation::Horner(
                    ValueSource::Constant(0),
                    parts,
                    ValueSource::Theta(),
                ))
            };

            // Input coset
            let compressed_input_coset = evaluate_lc(&lookup.input_expressions);
            // table coset
            let compressed_table_coset = evaluate_lc(&lookup.table_expressions);
            // z(\omega X) (a'(X) + \beta) (s'(X) + \gamma)
            let right_gamma = graph.add_calculation(Calculation::Add(
                compressed_table_coset,
                ValueSource::Gamma(),
            ));
            let lc = graph.add_calculation(Calculation::Add(
                compressed_input_coset,
                ValueSource::Beta(),
            ));
            graph.add_calculation(Calculation::Mul(lc, right_gamma));
            ev.lookups.push(graph);
        }

        ev
    }

    /// Creates a new evaluation structure
    pub fn new(cs: &GpuConstraintSystem<C::ScalarExt>) -> Self {
        Self::new_inner(cs.gates(), cs.lookups())
    }

    fn encode(&self) -> (Vec<CalcRule>, Vec<u64>) {
        self.custom_gates.encode_for_device()
    }
}

impl<C: CurveAffine> Default for GraphEvaluator<C> {
    fn default() -> Self {
        Self {
            // Fixed positions to allow easy access
            constants: vec![
                C::ScalarExt::ZERO,
                C::ScalarExt::ONE,
                C::ScalarExt::from(2u64),
            ],
            rotations: Vec::new(),
            calculations: Vec::new(),
            num_intermediates: 0,
        }
    }
}

impl<C: CurveAffine> GraphEvaluator<C> {
    /// Adds a rotation
    pub(crate) fn add_rotation(&mut self, rotation: &Rotation) -> usize {
        let position = self.rotations.iter().position(|&c| c == rotation.0);
        match position {
            Some(pos) => pos,
            None => {
                self.rotations.push(rotation.0);
                self.rotations.len() - 1
            }
        }
    }

    /// Adds a constant
    pub(crate) fn add_constant(&mut self, constant: &C::ScalarExt) -> ValueSource {
        let position = self.constants.iter().position(|&c| c == *constant);
        ValueSource::Constant(match position {
            Some(pos) => pos,
            None => {
                self.constants.push(*constant);
                self.constants.len() - 1
            }
        })
    }

    /// Adds a calculation.
    /// Currently does the simplest thing possible: just stores the
    /// resulting value so the result can be reused  when that calculation
    /// is done multiple times.
    pub(crate) fn add_calculation(&mut self, calculation: Calculation) -> ValueSource {
        let existing_calculation = self
            .calculations
            .iter()
            .find(|c| c.calculation == calculation);
        match existing_calculation {
            Some(existing_calculation) => ValueSource::Intermediate(existing_calculation.target),
            None => {
                let target = self.num_intermediates;
                self.calculations.push(CalculationInfo {
                    calculation,
                    target,
                });
                self.num_intermediates += 1;
                ValueSource::Intermediate(target)
            }
        }
    }

    /// Generates an optimized evaluation for the expression
    pub(crate) fn add_expression(&mut self, expr: &GpuExpression<C::ScalarExt>) -> ValueSource {
        match expr {
            GpuExpression::Constant(scalar) => self.add_constant(scalar),
            GpuExpression::Selector(_selector) => unreachable!(),
            GpuExpression::Fixed(query) => {
                let rot_idx = self.add_rotation(&query.rotation);
                self.add_calculation(Calculation::Store(ValueSource::Fixed(
                    query.column_index,
                    rot_idx,
                )))
            }
            GpuExpression::Advice(query) => {
                let rot_idx = self.add_rotation(&query.rotation);
                self.add_calculation(Calculation::Store(ValueSource::Advice(
                    query.column_index,
                    rot_idx,
                )))
            }
            GpuExpression::Instance(query) => {
                let rot_idx = self.add_rotation(&query.rotation);
                self.add_calculation(Calculation::Store(ValueSource::Instance(
                    query.column_index,
                    rot_idx,
                )))
            }
            GpuExpression::Challenge(challenge) => self.add_calculation(Calculation::Store(
                ValueSource::Challenge(challenge.index()),
            )),
            GpuExpression::Negated(a) => match **a {
                GpuExpression::Constant(scalar) => self.add_constant(&-scalar),
                _ => {
                    let result_a = self.add_expression(a);
                    match result_a {
                        ValueSource::Constant(0) => result_a,
                        _ => self.add_calculation(Calculation::Negate(result_a)),
                    }
                }
            },
            GpuExpression::Sum(a, b) => {
                // Undo subtraction stored as a + (-b) in expressions
                match &**b {
                    GpuExpression::Negated(b_int) => {
                        let result_a = self.add_expression(a);
                        let result_b = self.add_expression(b_int);
                        if result_a == ValueSource::Constant(0) {
                            self.add_calculation(Calculation::Negate(result_b))
                        } else if result_b == ValueSource::Constant(0) {
                            result_a
                        } else {
                            self.add_calculation(Calculation::Sub(result_a, result_b))
                        }
                    }
                    _ => {
                        let result_a = self.add_expression(a);
                        let result_b = self.add_expression(b);
                        if result_a == ValueSource::Constant(0) {
                            result_b
                        } else if result_b == ValueSource::Constant(0) {
                            result_a
                        } else if result_a <= result_b {
                            self.add_calculation(Calculation::Add(result_a, result_b))
                        } else {
                            self.add_calculation(Calculation::Add(result_b, result_a))
                        }
                    }
                }
            }
            GpuExpression::Product(a, b) => {
                let result_a = self.add_expression(a);
                let result_b = self.add_expression(b);
                if result_a == ValueSource::Constant(0) || result_b == ValueSource::Constant(0) {
                    ValueSource::Constant(0)
                } else if result_a == ValueSource::Constant(1) {
                    result_b
                } else if result_b == ValueSource::Constant(1) {
                    result_a
                } else if result_a == ValueSource::Constant(2) {
                    self.add_calculation(Calculation::Double(result_b))
                } else if result_b == ValueSource::Constant(2) {
                    self.add_calculation(Calculation::Double(result_a))
                } else if result_a == result_b {
                    self.add_calculation(Calculation::Square(result_a))
                } else if result_a <= result_b {
                    self.add_calculation(Calculation::Mul(result_a, result_b))
                } else {
                    self.add_calculation(Calculation::Mul(result_b, result_a))
                }
            }
            GpuExpression::Scaled(a, f) => {
                if *f == C::ScalarExt::ZERO {
                    ValueSource::Constant(0)
                } else if *f == C::ScalarExt::ONE {
                    self.add_expression(a)
                } else {
                    let cst = self.add_constant(f);
                    let result_a = self.add_expression(a);
                    self.add_calculation(Calculation::Mul(result_a, cst))
                }
            }
        }
    }

    /// Build the metadata graph for an expression list that will be
    /// Horner-folded by `theta` on the kernel side, matching the shape
    /// used by `lookup.commit_permuted`'s host closure
    /// (`acc * theta + expression`).
    ///
    /// Slot semantics for the kernel's hard-coded fold:
    /// - `expr_constants[4]` is the Horner factor (caller places
    ///   `theta`);
    /// - `Calculation::Horner(Constant(0), parts, Y())` provides
    ///   shape; `Y()` is unused (`encode_vp` only serialises `parts`);
    ///   `Constant(0)` is unused (the kernel inits `val = 0`).
    pub(crate) fn for_compress(expressions: &[GpuExpression<C::ScalarExt>]) -> Self {
        let mut graph = GraphEvaluator::<C>::default();
        let parts: Vec<ValueSource> = expressions
            .iter()
            .map(|expr| graph.add_expression(expr))
            .collect();
        graph.add_calculation(Calculation::Horner(
            ValueSource::Constant(0),
            parts,
            ValueSource::Y(),
        ));
        graph
    }

    /// Serialise this graph's calculations into the device-consumable
    /// `(intermediate_rules, value_part_rules)` pair. The terminal
    /// `Horner` calculation is split off into `value_part_rules` via
    /// [`Calculation::encode_vp`]; all preceding calculations are
    /// packed into [`CalcRule`] entries via [`Calculation::encode`].
    /// See [`CalcRule`] for the on-wire layout.
    pub(crate) fn encode_for_device(&self) -> (Vec<CalcRule>, Vec<u64>) {
        let n = self.calculations.len() - 1;
        let intermediate_rules = self.calculations[..n]
            .iter()
            .map(|c| c.calculation.encode(&self.rotations))
            .collect();
        let value_part_rules = self.calculations[n].calculation.encode_vp(&self.rotations);
        (intermediate_rules, value_part_rules)
    }

    /// Flatten this graph for the `_halo2_quotient_device_columns*` FFI
    /// when the graph references `Beta`/`Gamma`/`Theta`/`Y` or contains
    /// internal `Horner` calculations.
    ///
    /// Returns `(extended_constants, intermediate_rules, value_part_rules)`:
    /// - `extended_constants` = `self.constants` with `[theta, beta,
    ///   gamma, y]` appended at fixed trailing slots; the kernel reads
    ///   them via the same `d_constants` buffer slot as ordinary
    ///   constants. Any `Beta`/`Gamma`/`Theta`/`Y` `ValueSource` in
    ///   `self.calculations` is rewritten to `Constant(slot)` pointing
    ///   at the appended slot.
    /// - `intermediate_rules` = each internal calc packed as a
    ///   [`CalcRule`]. Internal `Horner(start, parts, factor)` calcs
    ///   are unrolled into a sequence of `Mul`/`Add` intermediates
    ///   `acc_{i+1} = acc_i * factor + parts[i]`; the unrolled
    ///   intermediates are renumbered and downstream references are
    ///   remapped to the new indices.
    /// - `value_part_rules` = the terminal calc's parts. If the
    ///   terminal is a `Horner`, its parts are serialised directly. For
    ///   any other terminal shape (e.g. the lookup-evaluator's
    ///   `Mul(Add(_, Beta), Add(_, Gamma))`), a synthetic single-part
    ///   `Horner(Constant(0), [Intermediate(term)], Constant(0))` is
    ///   emitted; the kernel's value-part loop yields
    ///   `val = 0 * y + term = term`.
    pub(crate) fn encode_for_device_with_runtime_constants(
        &self,
        theta: C::ScalarExt,
        beta: C::ScalarExt,
        gamma: C::ScalarExt,
        y: C::ScalarExt,
    ) -> (Vec<C::ScalarExt>, Vec<CalcRule>, Vec<u64>) {
        let mut constants = self.constants.clone();
        let theta_idx = constants.len();
        constants.push(theta);
        let beta_idx = constants.len();
        constants.push(beta);
        let gamma_idx = constants.len();
        constants.push(gamma);
        let y_idx = constants.len();
        constants.push(y);

        let rewrite = |vs: &ValueSource, remap: &[usize]| -> ValueSource {
            match vs {
                ValueSource::Beta() => ValueSource::Constant(beta_idx),
                ValueSource::Gamma() => ValueSource::Constant(gamma_idx),
                ValueSource::Theta() => ValueSource::Constant(theta_idx),
                ValueSource::Y() => ValueSource::Constant(y_idx),
                ValueSource::Intermediate(j) => ValueSource::Intermediate(remap[*j]),
                other => *other,
            }
        };

        let mut new_calcs: Vec<Calculation> = Vec::with_capacity(self.calculations.len());
        let mut remap: Vec<usize> = Vec::with_capacity(self.calculations.len());

        let (terminal_info, internal_calcs) = self
            .calculations
            .split_last()
            .expect("GraphEvaluator must have at least one calculation");

        for info in internal_calcs {
            match &info.calculation {
                Calculation::Horner(start, parts, factor) => {
                    let start = rewrite(start, &remap);
                    let factor = rewrite(factor, &remap);
                    if parts.is_empty() {
                        new_calcs.push(Calculation::Store(start));
                        remap.push(new_calcs.len() - 1);
                        continue;
                    }
                    let mut acc = start;
                    for part in parts {
                        let part = rewrite(part, &remap);
                        new_calcs.push(Calculation::Mul(acc, factor));
                        let mul_idx = new_calcs.len() - 1;
                        new_calcs.push(Calculation::Add(ValueSource::Intermediate(mul_idx), part));
                        acc = ValueSource::Intermediate(new_calcs.len() - 1);
                    }
                    remap.push(new_calcs.len() - 1);
                }
                other => {
                    let rewritten = match other {
                        Calculation::Add(a, b) => {
                            Calculation::Add(rewrite(a, &remap), rewrite(b, &remap))
                        }
                        Calculation::Sub(a, b) => {
                            Calculation::Sub(rewrite(a, &remap), rewrite(b, &remap))
                        }
                        Calculation::Mul(a, b) => {
                            Calculation::Mul(rewrite(a, &remap), rewrite(b, &remap))
                        }
                        Calculation::Square(v) => Calculation::Square(rewrite(v, &remap)),
                        Calculation::Double(v) => Calculation::Double(rewrite(v, &remap)),
                        Calculation::Negate(v) => Calculation::Negate(rewrite(v, &remap)),
                        Calculation::Store(v) => Calculation::Store(rewrite(v, &remap)),
                        Calculation::Horner(..) => unreachable!(),
                    };
                    new_calcs.push(rewritten);
                    remap.push(new_calcs.len() - 1);
                }
            }
        }

        let terminal_calc = match &terminal_info.calculation {
            Calculation::Horner(start, parts, factor) => Calculation::Horner(
                rewrite(start, &remap),
                parts.iter().map(|p| rewrite(p, &remap)).collect(),
                rewrite(factor, &remap),
            ),
            other => {
                let rewritten = match other {
                    Calculation::Add(a, b) => {
                        Calculation::Add(rewrite(a, &remap), rewrite(b, &remap))
                    }
                    Calculation::Sub(a, b) => {
                        Calculation::Sub(rewrite(a, &remap), rewrite(b, &remap))
                    }
                    Calculation::Mul(a, b) => {
                        Calculation::Mul(rewrite(a, &remap), rewrite(b, &remap))
                    }
                    Calculation::Square(v) => Calculation::Square(rewrite(v, &remap)),
                    Calculation::Double(v) => Calculation::Double(rewrite(v, &remap)),
                    Calculation::Negate(v) => Calculation::Negate(rewrite(v, &remap)),
                    Calculation::Store(v) => Calculation::Store(rewrite(v, &remap)),
                    Calculation::Horner(..) => unreachable!(),
                };
                new_calcs.push(rewritten);
                let term_idx = new_calcs.len() - 1;
                Calculation::Horner(
                    ValueSource::Constant(0),
                    vec![ValueSource::Intermediate(term_idx)],
                    ValueSource::Constant(0),
                )
            }
        };
        new_calcs.push(terminal_calc);

        let n = new_calcs.len() - 1;
        let intermediate_rules = new_calcs[..n]
            .iter()
            .map(|c| c.encode(&self.rotations))
            .collect();
        let value_part_rules = new_calcs[n].encode_vp(&self.rotations);

        (constants, intermediate_rules, value_part_rules)
    }

    /// Creates a new evaluation structure
    #[cfg(test)]
    pub fn instance(&self) -> EvaluationData<C> {
        EvaluationData {
            intermediates: vec![C::ScalarExt::ZERO; self.num_intermediates],
            rotations: vec![0usize; self.rotations.len()],
        }
    }

    #[cfg(test)]
    pub fn evaluate<B: Basis>(
        &self,
        data: &mut EvaluationData<C>,
        fixed: &[Polynomial<C::ScalarExt, B>],
        advice: &[Polynomial<C::ScalarExt, B>],
        instance: &[Polynomial<C::ScalarExt, B>],
        challenges: &[C::ScalarExt],
        beta: &C::ScalarExt,
        gamma: &C::ScalarExt,
        theta: &C::ScalarExt,
        y: &C::ScalarExt,
        previous_value: &C::ScalarExt,
        idx: usize,
        rot_scale: i32,
        isize: i32,
    ) -> C::ScalarExt {
        // All rotation index values
        for (rot_idx, rot) in self.rotations.iter().enumerate() {
            data.rotations[rot_idx] = get_rotation_idx(idx, *rot, rot_scale, isize);
        }

        // All calculations, with cached intermediate results
        for calc in self.calculations.iter() {
            data.intermediates[calc.target] = calc.calculation.evaluate(
                &data.rotations,
                &self.constants,
                &data.intermediates,
                fixed,
                advice,
                instance,
                challenges,
                beta,
                gamma,
                theta,
                y,
                previous_value,
            );
        }

        // Return the result of the last calculation (if any)
        if let Some(calc) = self.calculations.last() {
            data.intermediates[calc.target]
        } else {
            C::ScalarExt::ZERO
        }
    }
}

/// Simple evaluation of an expression
pub fn evaluate<F: Field, B: Basis>(
    expression: &GpuExpression<F>,
    size: usize,
    rot_scale: i32,
    fixed: &[Polynomial<F, B>],
    advice: &[Polynomial<F, B>],
    instance: &[Polynomial<F, B>],
    challenges: &[F],
) -> Vec<F> {
    let mut values = vec![F::ZERO; size];
    let isize = size as i32;
    parallelize(&mut values, |values, start| {
        for (i, value) in values.iter_mut().enumerate() {
            let idx = start + i;
            *value = expression.evaluate(
                &|scalar| scalar,
                &|_| panic!("virtual selectors are removed during optimization"),
                &|query| {
                    fixed[query.column_index]
                        [get_rotation_idx(idx, query.rotation.0, rot_scale, isize)]
                },
                &|query| {
                    advice[query.column_index]
                        [get_rotation_idx(idx, query.rotation.0, rot_scale, isize)]
                },
                &|query| {
                    instance[query.column_index]
                        [get_rotation_idx(idx, query.rotation.0, rot_scale, isize)]
                },
                &|challenge| challenges[challenge.index()],
                &|a| -a,
                &|a, b| a + &b,
                &|a, b| a * b,
                &|a, scalar| a * scalar,
            );
        }
    });
    values
}

pub(in crate::plonk) fn quotient_device<C: CurveAffine>(
    evaluator: &Evaluator<C>,
    num_quotient: usize,
    pool: &crate::cuda::funcs::column_pool::ColumnPool<C::ScalarExt>,
    challenges: &[C::ScalarExt],
    rot_scale: i32,
    y: C::ScalarExt,
    prev_values: Option<&DeviceBuffer<C::ScalarExt>>,
) -> Result<DeviceBuffer<C::ScalarExt>, HaloGpuError> {
    ensure_current_device_matches_ctx()?;
    debug_assert!(pool.is_initialized());

    let custom_gates = &evaluator.custom_gates;
    let (intermediate_rules, value_part_rules) = evaluator.encode();

    let num_fixed = pool.num_fixed() as u64;
    let num_advice = pool.num_advice() as u64;
    let num_instance = pool.num_instance() as u64;

    let batch_size = unsafe {
        _halo2_evaluate_h_max_rows(
            intermediate_rules.as_ptr() as *const u64,
            intermediate_rules.len() as u64,
            value_part_rules.len() as u64,
            num_quotient as u64,
            num_fixed,
            num_instance,
            num_advice,
            challenges.len() as u64,
            custom_gates.constants.len() as u64,
            rot_scale as u64,
            0_u64,
            num_quotient as u64,
            query_device_free_bytes_for_chunking() as u64,
        ) as usize
    };
    log::debug!("gpu_batch_size: {} (device-columns)", batch_size);

    let d_values: DeviceBuffer<C::ScalarExt> =
        DeviceBuffer::<C::ScalarExt>::with_capacity_on(num_quotient, &HALO2_GPU_CTX);

    let mut row_cursor = 0usize;
    while row_cursor < num_quotient {
        let chunk_len = (num_quotient - row_cursor).min(batch_size);
        let batch_row_start = row_cursor as u64;
        let batch_row_end = (row_cursor + chunk_len) as u64;

        let challenges_ffi = FFITraitObject::new(challenges.as_ptr() as usize);
        let constant_ffi = get_poly_ffi(&evaluator.custom_gates.constants, 0);
        let expr_constant = vec![
            C::ScalarExt::ZERO,
            C::ScalarExt::ONE,
            -C::ScalarExt::ONE,
            C::ScalarExt::from(2),
            y,
        ];
        let expr_constant_ffi = get_poly_ffi(&expr_constant, 0);
        let intermediate_rule_ffi = intermediate_rules.as_ptr() as *const u64;
        let value_part_rules_ffi = value_part_rules.as_ptr();

        let scratch_bytes = unsafe {
            _halo2_quotient_workspace_size(
                intermediate_rule_ffi,
                intermediate_rules.len() as u64,
                value_part_rules.len() as u64,
                num_quotient as u64,
                num_fixed,
                num_instance,
                num_advice,
                challenges.len() as u64,
                evaluator.custom_gates.constants.len() as u64,
                rot_scale as u64,
                batch_row_start,
                batch_row_end,
            )
        } as usize;
        let scratch = DeviceBuffer::<u8>::with_capacity_on(scratch_bytes, &HALO2_GPU_CTX);
        let elem_bytes = std::mem::size_of::<C::ScalarExt>();
        let d_out_chunk_ptr =
            unsafe { (d_values.as_mut_raw_ptr() as *mut u8).add(row_cursor * elem_bytes) }
                as *mut libc::c_void;
        // Seed this chunk's value-part Horner from the prior circuit's
        // accumulated values over the same row range; null for the first
        // circuit, which seeds zero.
        let prev_chunk_ptr: *const libc::c_void = match prev_values {
            Some(prev) => {
                (unsafe { (prev.as_raw_ptr() as *const u8).add(row_cursor * elem_bytes) })
                    as *const libc::c_void
            }
            None => std::ptr::null(),
        };

        let err = unsafe {
            let scratch_ptr = scratch.as_mut_raw_ptr();
            _halo2_quotient_device_columns_device_out(
                pool.fixed_ptrs(),
                num_fixed,
                pool.instance_ptrs(),
                num_instance,
                pool.advice_ptrs(),
                num_advice,
                challenges_ffi,
                challenges.len() as u64,
                constant_ffi,
                evaluator.custom_gates.constants.len() as u64,
                expr_constant_ffi,
                intermediate_rule_ffi,
                intermediate_rules.len() as u64,
                value_part_rules_ffi,
                value_part_rules.len() as u64,
                d_out_chunk_ptr,
                num_quotient as u64,
                prev_chunk_ptr,
                rot_scale as u64,
                batch_row_start,
                batch_row_end,
                scratch_ptr,
                scratch_bytes as u64,
                HALO2_GPU_CTX.stream.as_raw(),
            )
        };
        if err.code != 0 {
            return Err(err.into());
        }
        row_cursor += chunk_len;
    }

    Ok(d_values)
}

// computes the h polynomial
// which is the comressed polynomial of
// 1. gate polynomials
// 2. permutation polynomials
// 3. lookup gate polynomials
//
// The gate and lookup gates requires an intepreter for the gate expressions
// otherwise other computations are cosetffts and inverse ffts, as well as elementwise fixed
// expressions over C::ScalarExt
// some limitations of the current impl is that it only supports BN256.
pub(in crate::plonk) fn evaluate_h_device<C: CurveAffine>(
    evaluator: &Evaluator<C>,
    pk: &GpuProvingKey<'_, C>,
    advice_polys: &[&[Polynomial<C::ScalarExt, Coeff, Device>]],
    instance_polys: &[&[Polynomial<C::ScalarExt, Coeff, Device>]],
    challenges: &[C::ScalarExt],
    y: C::ScalarExt,
    beta: C::ScalarExt,
    gamma: C::ScalarExt,
    theta: C::ScalarExt,
    lookups: &[Vec<lookup::prover::Committed<C>>],
    permutations: &[permutation::prover::Committed<C>],
) -> Result<crate::poly::MaybeDevice<C::ScalarExt, ExtendedLagrangeCoeff>, crate::plonk::GpuError>
where
    C::ScalarExt: WithSmallOrderMulGroup<3>,
{
    // Evaluator view over the GPU `cs`/`domain` held on the `GpuProvingKey`.
    let view = EvaluatorVkView {
        blinding_factors: pk.cs.blinding_factors(),
        cs_degree: pk.cs_degree,
        permutation_argument: &pk.cs.permutation,
        domain: &pk.domain,
    };
    evaluate_h_inner(
        evaluator,
        &view,
        pk.l0_device().expect("l0 device failed to transport"),
        pk.l_last_device()
            .expect("l_last device failed to transport"),
        pk.l_active_row_device()
            .expect("l_active_row device failed to transport"),
        pk.fixed_values_device()
            .expect("fixed_values device failed to transport"),
        pk.permutation_polys_device()
            .expect("permutation_polys device failed to transport"),
        pk.fixed_polys_device()
            .expect("fixed_polys device failed to transport"),
        advice_polys,
        instance_polys,
        challenges,
        y,
        beta,
        gamma,
        theta,
        lookups,
        permutations,
    )
}

/// Evaluate h poly
#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_h_inner<C: CurveAffine>(
    evaluator: &Evaluator<C>,
    view: &EvaluatorVkView<'_, C::ScalarExt>,
    pk_l0: &Polynomial<C::ScalarExt, Coeff, Device>,
    pk_l_last: &Polynomial<C::ScalarExt, Coeff, Device>,
    pk_l_active_row: &Polynomial<C::ScalarExt, Coeff, Device>,
    pk_fixed_values: &[Polynomial<C::ScalarExt, LagrangeCoeff, Device>],
    pk_permutation_polys: &[Polynomial<C::ScalarExt, Coeff, Device>],
    pk_fixed_polys: &[Polynomial<C::ScalarExt, Coeff, Device>],
    advice_polys: &[&[Polynomial<C::ScalarExt, Coeff, Device>]],
    instance_polys: &[&[Polynomial<C::ScalarExt, Coeff, Device>]],
    challenges: &[C::ScalarExt],
    y: C::ScalarExt,
    beta: C::ScalarExt,
    gamma: C::ScalarExt,
    theta: C::ScalarExt,
    lookups: &[Vec<lookup::prover::Committed<C>>],
    permutations: &[permutation::prover::Committed<C>],
) -> Result<crate::poly::MaybeDevice<C::ScalarExt, ExtendedLagrangeCoeff>, crate::plonk::GpuError>
where
    C::ScalarExt: WithSmallOrderMulGroup<3>,
{
    crate::perf_section!("evaluate_h");
    let domain = view.domain;
    let size: usize = 1 << domain.k() as usize;
    let mut d_table_values = DeviceBuffer::<C::Scalar>::with_capacity_on(size, &HALO2_GPU_CTX);

    let rot_scale = 1;
    let extended_omega = domain.get_extended_omega();
    let _omega = domain.get_omega();
    let isize = size as i32;
    let one = C::ScalarExt::ONE;
    let p = view.permutation_argument;
    let num_parts = domain.extended_len() >> domain.k();

    // Calculate the quotient polynomial for each part
    let mut current_extended_omega = one;

    let parts_device_refs: Vec<&Polynomial<C::ScalarExt, Coeff, crate::poly::Device>> =
        pk_fixed_polys
            .iter()
            .chain([pk_l0, pk_l_last, pk_l_active_row])
            .chain(advice_polys.iter().flat_map(|p| p.iter()))
            .chain(instance_polys.iter().flat_map(|p| p.iter()))
            .collect();

    let value_parts: Vec<crate::poly::MaybeDevice<C::ScalarExt, LagrangeCoeff>> = (0..num_parts)
        .map(
            |_i| -> Result<
                crate::poly::MaybeDevice<C::ScalarExt, LagrangeCoeff>,
                crate::plonk::GpuError,
            > {
                let parts: Vec<Polynomial<C::ScalarExt, LagrangeCoeff, crate::poly::Device>> = {
                    crate::perf_section!("coeff_to_extended_part");
                    domain
                        .coeff_to_extended_part_many_device_device_inputs(parts_device_refs.clone(), current_extended_omega)?
                        .into_iter()
                        .map(Polynomial::<C::ScalarExt, LagrangeCoeff, crate::poly::Device>::from_device)
                        .collect()
                };

                let mut offset = 0;
                let fixed = &parts[offset..(offset + pk_fixed_polys.len())];
                offset += pk_fixed_polys.len();

                offset += 3;

                // Calculate the advice and instance cosets
                let advice: Vec<&[Polynomial<C::Scalar, LagrangeCoeff, crate::poly::Device>]> =
                    advice_polys
                        .iter()
                        .map(|advice_polys| {
                            let advice = &parts[offset..(offset + advice_polys.len())];
                            offset += advice_polys.len();

                            advice
                        })
                        .collect();
                let instance: Vec<&[Polynomial<C::Scalar, LagrangeCoeff, crate::poly::Device>]> =
                    instance_polys
                        .iter()
                        .map(|instance_polys| {
                            let instance = &parts[offset..(offset + instance_polys.len())];
                            offset += instance_polys.len();

                            instance
                        })
                        .collect();

                let l0_part_idx = pk_fixed_polys.len();
                let l_last_part_idx = pk_fixed_polys.len() + 1;
                let l_active_part_idx = pk_fixed_polys.len() + 2;

                let num_circuits = advice.len();
                assert_eq!(num_circuits, instance.len());
                assert_eq!(num_circuits, lookups.len());
                assert_eq!(num_circuits, permutations.len());

                // Accumulator carried across the circuit batch: each circuit
                // seeds its custom-gates value-part Horner from the previous
                // circuit's accumulated quotient, reproducing the CPU reference
                // fold `values = values * y + contribution` over the batch.
                let mut acc: Option<DeviceBuffer<C::ScalarExt>> = None;

                assert_eq!(pk_fixed_values.len(), pk_fixed_polys.len());
                for (((advice, instance), lookups), permutation) in advice
                    .iter()
                    .zip(instance.iter())
                    .zip(lookups.iter())
                    .zip(permutations.iter())
                {
                    let mut part_pool = ColumnPool::<C::Scalar>::new(size);
                    part_pool
                        .try_init_device::<_, _, _>(Some(fixed), &[], advice, instance)?;

                    // Custom gates part
                    let quotient_values_device = {
                        crate::perf_section!("custom_gates");
                        let calc_count = evaluator.custom_gates.calculations.len();
                        log::debug!(
                            "custom_gates.calculations.len={}, size={}, using quotient_gpu",
                            calc_count,
                            size,
                        );

                        quotient_device(
                            evaluator,
                            size,
                            &part_pool,
                            challenges,
                            rot_scale,
                            y,
                            acc.as_ref(),
                        )?
                    };

                    let mut gpu_compute = QuotientLookupsGpu::new_with_device_selectors(
                        quotient_values_device,
                        parts[l0_part_idx].device_buf(),
                        parts[l_last_part_idx].device_buf(),
                        parts[l_active_part_idx].device_buf(),
                        beta,
                        gamma,
                        y,
                        domain.k(),
                        *domain.omega_inv(),
                        *domain.ifft_divisor(),
                        *domain.omega(),
                       isize as usize,
                    )?;

                    // Permutations
                    {
                        crate::perf_section!("permutation_quotient_poly_part");
                        let sets = &permutation.sets;
                        if !sets.is_empty() {
                            let blinding_factors = view.blinding_factors;
                            let last_rotation = Rotation(-((blinding_factors + 1) as i32));
                            let chunk_len = view.cs_degree - 2;
                            let delta_start = beta * &C::Scalar::ZETA;

                            let column_values_device_ptrs: Vec<*const std::ffi::c_void> = p
                                .columns
                                .iter()
                                .map(|column| match column.column_type() {
                                    GpuAny::Advice(_) => advice[column.index()].device_buf().as_raw_ptr(),
                                    GpuAny::Fixed => fixed[column.index()].device_buf().as_raw_ptr(),
                                    GpuAny::Instance => instance[column.index()].device_buf().as_raw_ptr(),
                                })
                                .collect();

                            crate::perf_section!("permutations");

                            let (permutation_product_cosets_device, permutation_cosets_device) = {
                                crate::perf_section!("permutation_coset_fft");
                                let permutation_product_cosets_device = domain
                                    .coeff_to_extended_part_many_device(
                                        sets.iter()
                                            .map(|set| &set.permutation_product_poly)
                                            .collect::<Vec<_>>(),
                                        current_extended_omega,
                                    )?;
                                // The fixed permutation (sigma) cosets are recomputed at
                                // this circuit's coset shift: `current_extended_omega`
                                // advances across the batch, so each circuit transforms
                                // `pk_permutation_polys` at its own shift.
                                let permutation_cosets_device = domain
                                    .coeff_to_extended_part_many_device(
                                        pk_permutation_polys.iter().collect::<Vec<_>>(),
                                        current_extended_omega,
                                    )?;
                                (permutation_product_cosets_device, permutation_cosets_device)
                            };

                            let perm_prod_dev_ptrs: Vec<*const std::ffi::c_void> =
                                permutation_product_cosets_device
                                    .iter()
                                    .map(|b| b.as_raw_ptr())
                                    .collect();
                            let perm_coset_dev_ptrs: Vec<*const std::ffi::c_void> =
                                permutation_cosets_device
                                    .iter()
                                    .map(|b| b.as_raw_ptr())
                                    .collect();

                            gpu_compute.add_permutation_constraints(
                                &perm_prod_dev_ptrs,
                                &perm_coset_dev_ptrs,
                                &column_values_device_ptrs,
                                last_rotation.0,
                                rot_scale,
                                isize,
                                chunk_len,
                                C::Scalar::DELTA,
                                delta_start,
                                current_extended_omega,
                            )?;
                        }
                    }

                    {
                        crate::perf_section!("lookups");
                        let d_buf = {
                            for (n, lookup) in lookups.iter().enumerate() {
                                {
                                    crate::perf_section!("table_values");
                                    compress_expressions_with_runtime_constants_device::<C>(
                                        &evaluator.lookups[n],
                                        theta,
                                        beta,
                                        gamma,
                                        y,
                                        size,
                                        rot_scale,
                                        &part_pool,
                                        challenges,
                                        &mut d_table_values,
                                    )?;
                                }

                                crate::perf_section!("gpu_quotient_lookups");
                                // The `MaybeDevice` carrier may arrive host-resident; in that case
                                // its values are staged to a temporary device buffer here.
                                // TODO: refactor in future PR, remove the MaybeDevice so this won't be necessary
                                let permuted_input_staging: Option<DeviceBuffer<C::Scalar>> =
                                    match &lookup.permuted_input_expression {
                                        crate::poly::MaybeDevice::Device(_) => None,
                                        crate::poly::MaybeDevice::Host(p) => {
                                            Some(p.values().to_device_on(&HALO2_GPU_CTX).unwrap())
                                        }
                                    };
                                let permuted_table_staging: Option<DeviceBuffer<C::Scalar>> =
                                    match &lookup.permuted_table_expression {
                                        crate::poly::MaybeDevice::Device(_) => None,
                                        crate::poly::MaybeDevice::Host(p) => {
                                            Some(p.values().to_device_on(&HALO2_GPU_CTX).unwrap())
                                        }
                                    };
                                let d_permuted_input: &DeviceBuffer<C::Scalar> =
                                    match &lookup.permuted_input_expression {
                                        crate::poly::MaybeDevice::Device(p) => p.device_buf(),
                                        crate::poly::MaybeDevice::Host(_) => {
                                            permuted_input_staging.as_ref().unwrap()
                                        }
                                    };
                                let d_permuted_table: &DeviceBuffer<C::Scalar> =
                                    match &lookup.permuted_table_expression {
                                        crate::poly::MaybeDevice::Device(p) => p.device_buf(),
                                        crate::poly::MaybeDevice::Host(_) => {
                                            permuted_table_staging.as_ref().unwrap()
                                        }
                                    };
                                let d_product_poly: &DeviceBuffer<C::Scalar> =
                                    lookup.product_poly.device_buf();
                                gpu_compute.calculate_constraints_full_device(
                                    &d_table_values,
                                    d_product_poly,
                                    d_permuted_input,
                                    d_permuted_table,
                                    (*domain.g_coset()) * current_extended_omega,
                                )?;
                            }
                            crate::perf_section!("take_values_device_for_assembly");
                            gpu_compute.take_values_device()
                        };

                        acc = Some(d_buf);
                    }

                    current_extended_omega *= extended_omega;
                }
                match acc {
                    Some(d_buf) => Ok(crate::poly::MaybeDevice::Device(Polynomial::<
                        C::ScalarExt,
                        LagrangeCoeff,
                        crate::poly::Device,
                    >::from_device(
                        d_buf
                    ))),
                    None => {
                        let zeros = DeviceBuffer::with_capacity_on(size, &HALO2_GPU_CTX);
                        zeros.fill_zero_on(&HALO2_GPU_CTX)?;
                        Ok(crate::poly::MaybeDevice::Device(Polynomial::from_device(zeros)))
                    }
                }
            },
        )
        .collect::<Result<Vec<_>, _>>()?;

    domain.extended_from_lagrange_vec_device(value_parts)
}

fn get_poly_ffi<F: Field>(poly: &[F], row_start: usize) -> FFITraitObject {
    assert!(poly.len() > row_start);
    FFITraitObject::from_slice(&poly[row_start..])
}

#[cfg(test)]
fn get_rule_ffi(rules: &[CalcRule]) -> *const u64 {
    &rules[0].0 as *const u128 as *const u64
}

#[cfg(test)]
fn get_rule_ffi_u64(rules: &[u64]) -> *const u64 {
    &rules[0] as *const u64
}

extern "C" {
    pub fn _halo2_evaluate_h_max_rows(
        rules: *const libc::c_ulong,
        num_rules: u64,
        num_vp_rules: u64,
        num_quotient: u64,
        num_fixed: u64,
        num_instance: u64,
        num_advice: u64,
        num_challenges: u64,
        num_constants: u64,
        rotation_scale: u64,
        row_start: u64,
        row_end: u64,
        free_bytes: u64,
    ) -> u64;

    pub fn _halo2_quotient_workspace_size(
        rules: *const libc::c_ulong,
        num_rules: u64,
        num_vp_rules: u64,
        num_quotient: u64,
        num_fixed: u64,
        num_instance: u64,
        num_advice: u64,
        num_challenges: u64,
        num_constants: u64,
        rotation_scale: u64,
        row_start: u64,
        row_end: u64,
    ) -> u64;

    pub fn _halo2_quotient(
        fixed: *const FFITraitObject,
        num_fixed: u64,
        instance: *const FFITraitObject,
        num_instance: u64,
        advices: *const FFITraitObject,
        num_advice: u64,
        challenges: FFITraitObject,
        num_challenges: u64,
        constants: FFITraitObject,
        num_constants: u64,
        expr_constants: FFITraitObject,
        rules: *const libc::c_ulong,
        num_rules: u64,
        value_part_rules: *const libc::c_ulong,
        num_vp_rules: u64,
        quotient_poly: FFITraitObject,
        num_quotient: u64,
        rotation_scale: u64,
        row_start: u64,
        row_end: u64,
        scratch: *mut libc::c_void,
        scratch_bytes: u64,
        stream: *mut libc::c_void,
    ) -> CudaStatus;

    // Device-pointer column input AND device-pointer output sibling
    // of `_halo2_quotient_device_columns`. The output is written via
    // D2D into a caller-provided device buffer; this FFI does not D2H
    // the quotient poly back to the host.
    //
    // D2H classification: not applicable — input and output are both
    // device-resident.
    pub fn _halo2_quotient_device_columns_device_out(
        fixed_d_ptrs: *const *const libc::c_void,
        num_fixed: u64,
        instance_d_ptrs: *const *const libc::c_void,
        num_instance: u64,
        advice_d_ptrs: *const *const libc::c_void,
        num_advice: u64,
        challenges: FFITraitObject,
        num_challenges: u64,
        constants: FFITraitObject,
        num_constants: u64,
        expr_constants: FFITraitObject,
        rules: *const libc::c_ulong,
        num_rules: u64,
        value_part_rules: *const libc::c_ulong,
        num_vp_rules: u64,
        quotient_poly_device_ptr: *mut libc::c_void,
        num_quotient: u64,
        prev_values_device_ptr: *const libc::c_void,
        rotation_scale: u64,
        row_start: u64,
        row_end: u64,
        scratch: *mut libc::c_void,
        scratch_bytes: u64,
        stream: *mut libc::c_void,
    ) -> CudaStatus;

    // Device-pointer column input variant of `_halo2_quotient`. The
    // caller passes flat arrays of device pointers for fixed /
    // instance / advice columns; the FFI does D2D copies from those
    // device buffers into the kernel's internal layout. The
    // `*const c_void` array shape (rather than `FFITraitObject`) makes
    // the device-pointer contract explicit at every call site.
    //
    // The `quotient_poly` slot is a host buffer; the kernel ends with a
    // D2H of the quotient poly. Downstream consumers
    // (`compress_expressions_device` -> `permute_expression_pair`) are
    // CPU-only, so a device-output variant
    // (`_halo2_quotient_device_columns_device_out`) is only useful once that
    // chain can stay on device.
    pub fn _halo2_quotient_device_columns(
        fixed_d_ptrs: *const *const libc::c_void,
        num_fixed: u64,
        instance_d_ptrs: *const *const libc::c_void,
        num_instance: u64,
        advice_d_ptrs: *const *const libc::c_void,
        num_advice: u64,
        challenges: FFITraitObject,
        num_challenges: u64,
        constants: FFITraitObject,
        num_constants: u64,
        expr_constants: FFITraitObject,
        rules: *const libc::c_ulong,
        num_rules: u64,
        value_part_rules: *const libc::c_ulong,
        num_vp_rules: u64,
        quotient_poly: FFITraitObject,
        num_quotient: u64,
        rotation_scale: u64,
        row_start: u64,
        row_end: u64,
        scratch: *mut libc::c_void,
        scratch_bytes: u64,
        stream: *mut libc::c_void,
    ) -> CudaStatus;
}

// TODO: this is used in lookup prover and tests, remove in future PR and get rid of D2H round trip
//
// Device-side `compress_expressions` for `lookup.commit_permuted`.
// Consumes device-resident columns via the
// `_halo2_quotient_device_columns` FFI.
//
// `pool` must be initialized (the caller calls `pool.try_init(...)`
// once before the lookups loop). On `pool.is_initialized() == false`
// the caller is expected to fall back to the CPU
// `compress_expressions` closure.
//
// The output is a host `Vec<F>` because the downstream consumer
// (`permute_expression_pair`) is CPU-only; this entry pays one D2H of the
// quotient poly per chunk. Device-resident output requires a device variant
// of `permute_expression_pair`.
#[allow(clippy::too_many_arguments)]
pub fn compress_expressions_device<C: CurveAffine>(
    expressions: &[GpuExpression<C::ScalarExt>],
    theta: C::ScalarExt,
    size: usize,
    rot_scale: i32,
    pool: &ColumnPool<C::ScalarExt>,
    challenges: &[C::ScalarExt],
) -> Result<Vec<C::ScalarExt>, HaloGpuError>
where
    C::ScalarExt: WithSmallOrderMulGroup<3>,
{
    crate::perf_section!("lookup.compress_expressions_device");
    debug_assert!(pool.is_initialized());
    // Fence the current device id here; the FFI launcher does not
    // call `cudaSetDevice` itself.
    ensure_current_device_matches_ctx()?;

    let graph = GraphEvaluator::<C>::for_compress(expressions);
    debug_assert!(!graph.calculations.is_empty());

    let (intermediate_rules, value_part_rules) = graph.encode_for_device();

    let mut values = vec![C::ScalarExt::ZERO; size];
    let num_quotient = values.len();
    let num_fixed = pool.num_fixed() as u64;
    let num_advice = pool.num_advice() as u64;
    let num_instance = pool.num_instance() as u64;

    let batch_size = unsafe {
        _halo2_evaluate_h_max_rows(
            intermediate_rules.as_ptr() as *const u64,
            intermediate_rules.len() as u64,
            value_part_rules.len() as u64,
            num_quotient as u64,
            num_fixed,
            num_instance,
            num_advice,
            challenges.len() as u64,
            graph.constants.len() as u64,
            rot_scale as u64,
            0_u64,
            values.len() as u64,
            query_device_free_bytes_for_chunking() as u64,
        ) as usize
    };
    log::debug!("compress_expressions_device batch_size: {}", batch_size);

    for (idx, values_chunk) in values.chunks_mut(batch_size).enumerate() {
        let batch_row_start = idx * batch_size;
        let batch_row_end = (batch_row_start + values_chunk.len()) as u64;

        let challenges_ffi = FFITraitObject::new(challenges.as_ptr() as usize);
        let constant_ffi = get_poly_ffi(&graph.constants, 0);
        let quotient_poly_ffi = FFITraitObject::from_slice(values_chunk);
        // Slot 4 is the kernel's hard-coded Horner factor; place `theta`
        // there. Slots 0..3 retain [0, 1, -1, 2] per kernel-side
        // `c1/c2` semantics; do NOT mutate.
        let expr_constant = vec![
            C::ScalarExt::ZERO,
            C::ScalarExt::ONE,
            -C::ScalarExt::ONE,
            C::ScalarExt::from(2),
            theta,
        ];
        let expr_constant_ffi = get_poly_ffi(&expr_constant, 0);
        let intermediate_rule_ffi = intermediate_rules.as_ptr() as *const u64;
        debug_assert!(!value_part_rules.is_empty());
        let value_part_rules_ffi = value_part_rules.as_ptr();

        let scratch_bytes = unsafe {
            _halo2_quotient_workspace_size(
                intermediate_rule_ffi,
                intermediate_rules.len() as u64,
                value_part_rules.len() as u64,
                num_quotient as u64,
                num_fixed,
                num_instance,
                num_advice,
                challenges.len() as u64,
                graph.constants.len() as u64,
                rot_scale as u64,
                batch_row_start as u64,
                batch_row_end,
            )
        } as usize;
        let scratch = DeviceBuffer::<u8>::with_capacity_on(scratch_bytes, &HALO2_GPU_CTX);
        // D2H of the quotient-poly result for this chunk.
        crate::perf_d2h!(
            "compress_expressions_device.result",
            std::mem::size_of_val(values_chunk) as u64
        );
        let err = unsafe {
            _halo2_quotient_device_columns(
                pool.fixed_ptrs(),
                num_fixed,
                pool.instance_ptrs(),
                num_instance,
                pool.advice_ptrs(),
                num_advice,
                challenges_ffi,
                challenges.len() as u64,
                constant_ffi,
                graph.constants.len() as u64,
                expr_constant_ffi,
                intermediate_rule_ffi,
                intermediate_rules.len() as u64,
                value_part_rules_ffi,
                value_part_rules.len() as u64,
                quotient_poly_ffi,
                num_quotient as u64,
                rot_scale as u64,
                batch_row_start as u64,
                batch_row_end,
                scratch.as_mut_raw_ptr(),
                scratch_bytes as u64,
                HALO2_GPU_CTX.stream.as_raw(),
            )
        };
        if err.code != 0 {
            return Err(err.into());
        }
    }
    Ok(values)
}

// TODO: this is used in lookup prover and tests, remove in future PR and get rid of D2H round trip
//
// Writes the
// metadata-evaluator output into a caller-provided device buffer, so
// no per-call D2H is performed. This lets the lookup-evaluator hand
// device pointers downstream and skip
// `calculate_constraints`'s internal table-values H2D.
//
// The expression list is provided indirectly via a pre-built
// `GraphEvaluator<C>`. The lookup-evaluator graph nests `Add`, `Mul`,
// `Beta`, and `Gamma` calculations on top of two compressed-lc
// sub-Horners; reusing the pre-built graph avoids rebuilding metadata
// on every call and matches the existing `Evaluator::lookups[n]`
// layout.
#[allow(clippy::too_many_arguments)]
pub fn compress_expressions_in_place_device<C: CurveAffine>(
    graph: &GraphEvaluator<C>,
    expr_constants: &[C::ScalarExt],
    size: usize,
    rot_scale: i32,
    pool: &ColumnPool<C::ScalarExt>,
    challenges: &[C::ScalarExt],
    d_out: &mut DeviceBuffer<C::ScalarExt>,
) -> Result<(), HaloGpuError>
where
    C::ScalarExt: WithSmallOrderMulGroup<3>,
{
    crate::perf_section!("compress_expressions_in_place_device");
    debug_assert!(pool.is_initialized());
    debug_assert_eq!(d_out.len(), size);
    debug_assert!(!graph.calculations.is_empty());
    // Fence the current device id here; the FFI launcher does not
    // call `cudaSetDevice` itself.
    ensure_current_device_matches_ctx()?;

    let (intermediate_rules, value_part_rules) = graph.encode_for_device();

    let num_quotient = size;
    let num_fixed = pool.num_fixed() as u64;
    let num_advice = pool.num_advice() as u64;
    let num_instance = pool.num_instance() as u64;

    let batch_size = unsafe {
        _halo2_evaluate_h_max_rows(
            intermediate_rules.as_ptr() as *const u64,
            intermediate_rules.len() as u64,
            value_part_rules.len() as u64,
            num_quotient as u64,
            num_fixed,
            num_instance,
            num_advice,
            challenges.len() as u64,
            graph.constants.len() as u64,
            rot_scale as u64,
            0_u64,
            num_quotient as u64,
            query_device_free_bytes_for_chunking() as u64,
        ) as usize
    };

    // Walk chunks identically to `compress_expressions_device`, but advance a
    // device-pointer cursor instead of a host slice.
    let mut row_cursor = 0usize;
    while row_cursor < num_quotient {
        let chunk_len = (num_quotient - row_cursor).min(batch_size);
        let batch_row_start = row_cursor as u64;
        let batch_row_end = (row_cursor + chunk_len) as u64;

        let challenges_ffi = FFITraitObject::new(challenges.as_ptr() as usize);
        let constant_ffi = get_poly_ffi(&graph.constants, 0);
        let expr_constant_ffi = get_poly_ffi(expr_constants, 0);
        let intermediate_rule_ffi = intermediate_rules.as_ptr() as *const u64;
        debug_assert!(!value_part_rules.is_empty());
        let value_part_rules_ffi = value_part_rules.as_ptr();

        let scratch_bytes = unsafe {
            _halo2_quotient_workspace_size(
                intermediate_rule_ffi,
                intermediate_rules.len() as u64,
                value_part_rules.len() as u64,
                num_quotient as u64,
                num_fixed,
                num_instance,
                num_advice,
                challenges.len() as u64,
                graph.constants.len() as u64,
                rot_scale as u64,
                batch_row_start,
                batch_row_end,
            )
        } as usize;
        let scratch = DeviceBuffer::<u8>::with_capacity_on(scratch_bytes, &HALO2_GPU_CTX);

        // Device-output pointer = d_out + row_cursor (raw scalar pointer).
        let elem_bytes = std::mem::size_of::<C::ScalarExt>();
        let d_out_chunk_ptr =
            unsafe { (d_out.as_mut_raw_ptr() as *mut u8).add(row_cursor * elem_bytes) }
                as *mut libc::c_void;

        let err = unsafe {
            _halo2_quotient_device_columns_device_out(
                pool.fixed_ptrs(),
                num_fixed,
                pool.instance_ptrs(),
                num_instance,
                pool.advice_ptrs(),
                num_advice,
                challenges_ffi,
                challenges.len() as u64,
                constant_ffi,
                graph.constants.len() as u64,
                expr_constant_ffi,
                intermediate_rule_ffi,
                intermediate_rules.len() as u64,
                value_part_rules_ffi,
                value_part_rules.len() as u64,
                d_out_chunk_ptr,
                num_quotient as u64,
                std::ptr::null(),
                rot_scale as u64,
                batch_row_start,
                batch_row_end,
                scratch.as_mut_raw_ptr(),
                scratch_bytes as u64,
                HALO2_GPU_CTX.stream.as_raw(),
            )
        };
        if err.code != 0 {
            return Err(err.into());
        }
        row_cursor += chunk_len;
    }
    Ok(())
}

// for `GraphEvaluator`s that reference `Beta`/`Gamma`/`Theta`/`Y` or
// contain internal `Horner` calculations (e.g. `Evaluator::lookups[n]`,
// whose terminal is `Mul(Add(_, Beta), Add(_, Gamma))`).
//
// Flattens the graph via
// [`GraphEvaluator::encode_for_device_with_runtime_constants`] and
// passes the extended `constants` buffer (= `graph.constants` plus
// trailing `[theta, beta, gamma, y]` slots) to the kernel. Per-call
// scalars are read from the same `d_constants` slot as ordinary
// constants; the `expr_constants` slot keeps its kernel-side
// `[0, 1, -1, 2, _]` convention and the value-part Horner factor is
// unused because the synthetic single-part wrap yields
// `val = 0 * y + term`.
#[allow(clippy::too_many_arguments)]
pub fn compress_expressions_with_runtime_constants_device<C: CurveAffine>(
    graph: &GraphEvaluator<C>,
    theta: C::ScalarExt,
    beta: C::ScalarExt,
    gamma: C::ScalarExt,
    y: C::ScalarExt,
    size: usize,
    rot_scale: i32,
    pool: &ColumnPool<C::ScalarExt>,
    challenges: &[C::ScalarExt],
    d_out: &mut DeviceBuffer<C::ScalarExt>,
) -> Result<(), HaloGpuError>
where
    C::ScalarExt: WithSmallOrderMulGroup<3>,
{
    crate::perf_section!("compress_expressions_with_runtime_constants_device");
    debug_assert!(pool.is_initialized());
    debug_assert_eq!(d_out.len(), size);
    debug_assert!(!graph.calculations.is_empty());
    ensure_current_device_matches_ctx()?;

    let (extended_constants, intermediate_rules, value_part_rules) =
        graph.encode_for_device_with_runtime_constants(theta, beta, gamma, y);

    let expr_constants = vec![
        C::ScalarExt::ZERO,
        C::ScalarExt::ONE,
        -C::ScalarExt::ONE,
        C::ScalarExt::from(2),
        C::ScalarExt::ZERO,
    ];

    let num_quotient = size;
    let num_fixed = pool.num_fixed() as u64;
    let num_advice = pool.num_advice() as u64;
    let num_instance = pool.num_instance() as u64;
    let num_constants = extended_constants.len() as u64;

    let batch_size = unsafe {
        _halo2_evaluate_h_max_rows(
            intermediate_rules.as_ptr() as *const u64,
            intermediate_rules.len() as u64,
            value_part_rules.len() as u64,
            num_quotient as u64,
            num_fixed,
            num_instance,
            num_advice,
            challenges.len() as u64,
            num_constants,
            rot_scale as u64,
            0_u64,
            num_quotient as u64,
            query_device_free_bytes_for_chunking() as u64,
        ) as usize
    };

    let mut row_cursor = 0usize;
    while row_cursor < num_quotient {
        let chunk_len = (num_quotient - row_cursor).min(batch_size);
        let batch_row_start = row_cursor as u64;
        let batch_row_end = (row_cursor + chunk_len) as u64;

        let challenges_ffi = FFITraitObject::new(challenges.as_ptr() as usize);
        let constant_ffi = get_poly_ffi(&extended_constants, 0);
        let expr_constant_ffi = get_poly_ffi(&expr_constants, 0);
        let intermediate_rule_ffi = intermediate_rules.as_ptr() as *const u64;
        debug_assert!(!value_part_rules.is_empty());
        let value_part_rules_ffi = value_part_rules.as_ptr();

        let scratch_bytes = unsafe {
            _halo2_quotient_workspace_size(
                intermediate_rule_ffi,
                intermediate_rules.len() as u64,
                value_part_rules.len() as u64,
                num_quotient as u64,
                num_fixed,
                num_instance,
                num_advice,
                challenges.len() as u64,
                num_constants,
                rot_scale as u64,
                batch_row_start,
                batch_row_end,
            )
        } as usize;
        let scratch = DeviceBuffer::<u8>::with_capacity_on(scratch_bytes, &HALO2_GPU_CTX);

        let elem_bytes = std::mem::size_of::<C::ScalarExt>();
        let d_out_chunk_ptr =
            unsafe { (d_out.as_mut_raw_ptr() as *mut u8).add(row_cursor * elem_bytes) }
                as *mut libc::c_void;

        let err = unsafe {
            let scratch_ptr = scratch.as_mut_raw_ptr();
            _halo2_quotient_device_columns_device_out(
                pool.fixed_ptrs(),
                num_fixed,
                pool.instance_ptrs(),
                num_instance,
                pool.advice_ptrs(),
                num_advice,
                challenges_ffi,
                challenges.len() as u64,
                constant_ffi,
                num_constants,
                expr_constant_ffi,
                intermediate_rule_ffi,
                intermediate_rules.len() as u64,
                value_part_rules_ffi,
                value_part_rules.len() as u64,
                d_out_chunk_ptr,
                num_quotient as u64,
                std::ptr::null(),
                rot_scale as u64,
                batch_row_start,
                batch_row_end,
                scratch_ptr,
                scratch_bytes as u64,
                HALO2_GPU_CTX.stream.as_raw(),
            )
        };
        if err.code != 0 {
            return Err(err.into());
        }
        row_cursor += chunk_len;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plonk::ConstraintSystem;

    /// Pure-Rust mirror of the device-side `decode_value` (see
    /// `cuda/quotient/quotient.cu`). Unpacks the 40-bit packed format
    /// emitted by `ValueSource::encode`.
    fn decode_value_source(rule: u64) -> (u32, u32, i32) {
        let src = (rule & 0x0f) as u32;
        let idx = ((rule >> 4) & 0xf_ffff) as u32;
        let rot_abs = ((rule >> 24) & 0x7fff) as u32;
        let rot_sign = ((rule >> 39) & 0x1) as u32;
        let rotation = if rot_sign == 0 {
            rot_abs as i32
        } else {
            -(rot_abs as i32)
        };
        (src, idx, rotation)
    }

    /// Split a 128-bit packed `CalcRule` into its (a, c1, d, b, c2) fields
    /// using the same layout as `Calculation::encode`. `a` and `b` are the
    /// raw 40-bit `ValueSource`-encoded halves; decode each via
    /// [`decode_value_source`].
    fn split_calc_rule(rule: u128) -> (u64, u32, u32, u64, u32) {
        let lo = rule as u64;
        let hi = (rule >> 64) as u64;
        let a = lo & 0xff_ffff_ffff;
        let c1 = ((lo >> 40) & 0xf) as u32;
        let d = ((lo >> 44) & 0xf) as u32;
        let b = hi & 0xff_ffff_ffff;
        let c2 = ((hi >> 40) & 0xf) as u32;
        (a, c1, d, b, c2)
    }

    #[test]
    fn test_encode_for_device_roundtrip() {
        use halo2curves::bn256::{Fr, G1Affine};

        // Source-code tags must match `ValueSource::encode` in this file.
        const SRC_FIXED: u32 = 0;
        const SRC_INSTANCE: u32 = 1;
        const SRC_ADVICE: u32 = 2;
        const SRC_INTERMEDIATE: u32 = 3;
        // CombineType tags must match the `combines` array order in
        // `Calculation::encode` (Zero, One, NegOne, Two).
        const C_ZERO: u32 = 0;
        const C_ONE: u32 = 1;
        const C_NEG_ONE: u32 = 2;
        // CalcDegree tags must match the `degrees` array order in
        // `Calculation::encode` (One, Two).
        const D_ONE: u32 = 0;
        const D_TWO: u32 = 1;

        // -------- Lookup-compress shape: Horner(Constant(0), parts, Y())
        //
        // Mirror what `for_compress` produces for a three-expression
        // lookup with mixed column kinds and rotation signs.
        let mut graph_lookup: GraphEvaluator<G1Affine> = GraphEvaluator::default();
        let rot_zero = graph_lookup.add_rotation(&Rotation(0));
        let rot_plus = graph_lookup.add_rotation(&Rotation(1));
        let rot_minus = graph_lookup.add_rotation(&Rotation(-1));
        let p0 = graph_lookup.add_calculation(Calculation::Store(ValueSource::Fixed(7, rot_zero)));
        let p1 = graph_lookup.add_calculation(Calculation::Store(ValueSource::Advice(3, rot_plus)));
        let p2 =
            graph_lookup.add_calculation(Calculation::Store(ValueSource::Instance(2, rot_minus)));
        graph_lookup.add_calculation(Calculation::Horner(
            ValueSource::Constant(0),
            vec![p0, p1, p2],
            ValueSource::Y(),
        ));

        let (rules_lookup, vp_lookup) = graph_lookup.encode_for_device();
        assert_eq!(rules_lookup.len(), 3);
        assert_eq!(vp_lookup.len(), 3);

        // Each intermediate is a Store: (c1, c2, d) = (One, Zero, One).
        let expected_operands: [(u32, u32, i32); 3] =
            [(SRC_FIXED, 7, 0), (SRC_ADVICE, 3, 1), (SRC_INSTANCE, 2, -1)];
        for (rule, &(want_src, want_idx, want_rot)) in
            rules_lookup.iter().zip(expected_operands.iter())
        {
            let (a, c1, d, _b, c2) = split_calc_rule(rule.0);
            assert_eq!(decode_value_source(a), (want_src, want_idx, want_rot));
            assert_eq!((c1, c2, d), (C_ONE, C_ZERO, D_ONE));
        }
        // Horner parts reference the three Stores above as Intermediates 0..2.
        for (i, &vp) in vp_lookup.iter().enumerate() {
            let (src, idx, rot) = decode_value_source(vp);
            assert_eq!(src, SRC_INTERMEDIATE);
            assert_eq!(idx as usize, i);
            assert_eq!(rot, 0);
        }

        // -------- Custom-gate shape: Horner(PreviousValue(), parts, Y())
        //
        // Exercise Add / Negate / Mul operator codes and confirm the
        // packed (c1, c2, d) triples match the encode tables.
        let mut graph_gate: GraphEvaluator<G1Affine> = GraphEvaluator::default();
        let rot_g = graph_gate.add_rotation(&Rotation(0));
        let const_42 = graph_gate.add_constant(&Fr::from(42));
        let fixed_ref =
            graph_gate.add_calculation(Calculation::Store(ValueSource::Fixed(0, rot_g)));
        let adv_ref = graph_gate.add_calculation(Calculation::Store(ValueSource::Advice(1, rot_g)));
        let sum = graph_gate.add_calculation(Calculation::Add(fixed_ref, adv_ref));
        let neg = graph_gate.add_calculation(Calculation::Negate(sum));
        let prod = graph_gate.add_calculation(Calculation::Mul(neg, const_42));
        graph_gate.add_calculation(Calculation::Horner(
            ValueSource::PreviousValue(),
            vec![prod],
            ValueSource::Y(),
        ));

        let (rules_gate, vp_gate) = graph_gate.encode_for_device();
        assert_eq!(rules_gate.len(), 5);
        assert_eq!(vp_gate.len(), 1);

        // Add: (One, One, One)
        let (_, c1_a, d_a, _, c2_a) = split_calc_rule(rules_gate[2].0);
        assert_eq!((c1_a, c2_a, d_a), (C_ONE, C_ONE, D_ONE));
        // Negate: (NegOne, Zero, One)
        let (_, c1_n, d_n, _, c2_n) = split_calc_rule(rules_gate[3].0);
        assert_eq!((c1_n, c2_n, d_n), (C_NEG_ONE, C_ZERO, D_ONE));
        // Mul: (One, Zero, Two)
        let (_, c1_m, d_m, _, c2_m) = split_calc_rule(rules_gate[4].0);
        assert_eq!((c1_m, c2_m, d_m), (C_ONE, C_ZERO, D_TWO));

        // The Mul's output (Intermediate(4)) is the only Horner part.
        let (vs, vi, vr) = decode_value_source(vp_gate[0]);
        assert_eq!((vs, vi, vr), (SRC_INTERMEDIATE, 4, 0));

        // -------- Evaluator::encode delegates to encode_for_device on
        // the custom_gates graph; verify byte-identical output.
        let mut cs = ConstraintSystem::<Fr>::default();
        let a = cs.advice_column();
        let b = cs.advice_column();
        cs.create_gate("ab", |meta| {
            let qa = meta.query_advice(a, Rotation::cur());
            let qb = meta.query_advice(b, Rotation::next());
            vec![qa * qb]
        });
        let gpu_cs = GpuConstraintSystem::from(&cs);
        let ev = Evaluator::<G1Affine>::new(&gpu_cs);
        let (rules_ev, vp_ev) = ev.encode();
        let (rules_direct, vp_direct) = ev.custom_gates.encode_for_device();
        assert_eq!(vp_ev, vp_direct);
        assert_eq!(rules_ev.len(), rules_direct.len());
        for (lhs, rhs) in rules_ev.iter().zip(rules_direct.iter()) {
            assert_eq!(lhs.0, rhs.0);
        }
    }

    /// CPU mirror of the kernel's per-row evaluation tape. Walks
    /// `intermediate_rules` then the `value_part_rules` Horner exactly
    /// as `cuda/include/kernel/quotient.h::evaluate` does, so a CPU
    /// reference value can be compared to what the device would compute
    /// from the same `(constants, intermediate_rules, value_part_rules)`
    /// triple.
    #[allow(clippy::too_many_arguments)]
    fn cpu_evaluate_from_encoding<F: Field>(
        _rotations: &[i32],
        constants: &[F],
        intermediate_rules: &[CalcRule],
        value_part_rules: &[u64],
        expr_constants_horner_factor: F,
        fixed: &[Vec<F>],
        instance: &[Vec<F>],
        advice: &[Vec<F>],
        challenges: &[F],
        row: i32,
        isize: i32,
    ) -> F {
        let row_lookup = |row: i32| -> usize {
            let mut r = row;
            while r < 0 {
                r += isize;
            }
            while r >= isize {
                r -= isize;
            }
            r as usize
        };
        let read_vs = |encoded: u64, intermediates: &[F]| -> F {
            // The kernel reads both operands unconditionally and lets
            // the c1/c2=Zero combine path discard the unused side. The
            // tape encodes a placeholder `dummy_var = 1` (decodes as
            // Instance/idx=0/rot=0) in those unused slots, so the
            // mirror has to tolerate out-of-bounds column reads as
            // ZERO rather than panic.
            let (src, idx, rotation) = decode_value_source(encoded);
            let oob = F::ZERO;
            match src {
                0 => fixed
                    .get(idx as usize)
                    .map(|col| col[row_lookup(row + rotation)])
                    .unwrap_or(oob),
                1 => instance
                    .get(idx as usize)
                    .map(|col| col[row_lookup(row + rotation)])
                    .unwrap_or(oob),
                2 => advice
                    .get(idx as usize)
                    .map(|col| col[row_lookup(row + rotation)])
                    .unwrap_or(oob),
                3 => intermediates[idx as usize],
                4 => constants[idx as usize],
                5 => challenges.get(idx as usize).copied().unwrap_or(oob),
                6 => F::ZERO,
                _ => unreachable!(),
            }
        };
        // Combine tables mirror `Calculation::encode`'s `combines` /
        // `degrees` ordering.
        let c_zero = F::ZERO;
        let c_one = F::ONE;
        let c_neg_one = -F::ONE;
        let c_two = F::ONE + F::ONE;
        let combines = [c_zero, c_one, c_neg_one, c_two];

        let mut intermediates: Vec<F> = vec![F::ZERO; intermediate_rules.len()];
        for (i, rule) in intermediate_rules.iter().enumerate() {
            let (a_enc, c1, d, b_enc, c2) = split_calc_rule(rule.0);
            let a = read_vs(a_enc, &intermediates);
            let b = read_vs(b_enc, &intermediates);
            let c1_val = combines[c1 as usize];
            let c2_val = combines[c2 as usize];
            let factor_b = if d == 1 { b } else { F::ONE };
            intermediates[i] = (c1_val * a + c2_val * b) * factor_b;
        }
        let mut val = F::ZERO;
        for &vp in value_part_rules.iter() {
            let part = {
                let (src, idx, _) = decode_value_source(vp);
                if src == 4 {
                    constants[idx as usize]
                } else {
                    intermediates[idx as usize]
                }
            };
            val = val * expr_constants_horner_factor + part;
        }
        val
    }

    #[test]
    fn test_encode_for_device_with_runtime_constants_lookup_shape() {
        use halo2curves::bn256::{Fr, G1Affine};

        // Mirror Evaluator::new's lookup-graph shape: two compressed-lc
        // sub-Horners over theta, then terminal
        // Mul(Add(input, Beta), Add(table, Gamma)).
        let mut graph: GraphEvaluator<G1Affine> = GraphEvaluator::default();
        let rot_zero = graph.add_rotation(&Rotation(0));
        let in0 = graph.add_calculation(Calculation::Store(ValueSource::Advice(0, rot_zero)));
        let in1 = graph.add_calculation(Calculation::Store(ValueSource::Advice(1, rot_zero)));
        let in2 = graph.add_calculation(Calculation::Store(ValueSource::Advice(2, rot_zero)));
        let t0 = graph.add_calculation(Calculation::Store(ValueSource::Fixed(0, rot_zero)));
        let t1 = graph.add_calculation(Calculation::Store(ValueSource::Fixed(1, rot_zero)));
        let t2 = graph.add_calculation(Calculation::Store(ValueSource::Fixed(2, rot_zero)));
        let compressed_input = graph.add_calculation(Calculation::Horner(
            ValueSource::Constant(0),
            vec![in0, in1, in2],
            ValueSource::Theta(),
        ));
        let compressed_table = graph.add_calculation(Calculation::Horner(
            ValueSource::Constant(0),
            vec![t0, t1, t2],
            ValueSource::Theta(),
        ));
        let lc = graph.add_calculation(Calculation::Add(compressed_input, ValueSource::Beta()));
        let rhs = graph.add_calculation(Calculation::Add(compressed_table, ValueSource::Gamma()));
        graph.add_calculation(Calculation::Mul(lc, rhs));

        let theta = Fr::from(7);
        let beta = Fr::from(11);
        let gamma = Fr::from(13);
        let y = Fr::from(17);

        let (extended_constants, rules, vp_rules) =
            graph.encode_for_device_with_runtime_constants(theta, beta, gamma, y);

        // The first len(self.constants) entries must equal the original
        // pool; the last four entries are theta/beta/gamma/y.
        assert_eq!(
            &extended_constants[..graph.constants.len()],
            &graph.constants[..]
        );
        assert_eq!(
            &extended_constants[graph.constants.len()..],
            &[theta, beta, gamma, y][..]
        );
        // No CalcRule may carry a Beta/Gamma/Theta/Y src tag (encoded
        // form would have panicked); each operand's src must be in
        // {Fixed, Instance, Advice, Intermediate, Constant, Challenge}.
        for rule in &rules {
            let (a, _, _, b, _) = split_calc_rule(rule.0);
            let (src_a, _, _) = decode_value_source(a);
            let (src_b, _, _) = decode_value_source(b);
            assert!(src_a <= 6);
            assert!(src_b <= 6);
        }
        // All value-part refs must resolve to Intermediate or Constant.
        assert_eq!(vp_rules.len(), 1);
        let (src, _, _) = decode_value_source(vp_rules[0]);
        assert!(src == 3 || src == 4);

        // Semantic equivalence: build per-row inputs and check that the
        // CPU-mirror evaluation of `(constants, rules, vp_rules)`
        // matches `Calculation::evaluate` on the original graph.
        let isize_n = 16i32;
        let advice_cols: Vec<Vec<Fr>> = (0..3)
            .map(|c| {
                (0..isize_n)
                    .map(|r| Fr::from((c * 100 + r) as u64))
                    .collect()
            })
            .collect();
        let fixed_cols: Vec<Vec<Fr>> = (0..3)
            .map(|c| {
                (0..isize_n)
                    .map(|r| Fr::from((c * 1000 + r) as u64 + 1))
                    .collect()
            })
            .collect();

        let fixed_polys: Vec<crate::poly::Polynomial<Fr, crate::poly::LagrangeCoeff>> = fixed_cols
            .iter()
            .map(|v| crate::poly::Polynomial::<Fr, crate::poly::LagrangeCoeff>::new(v.clone()))
            .collect();
        let advice_polys: Vec<crate::poly::Polynomial<Fr, crate::poly::LagrangeCoeff>> =
            advice_cols
                .iter()
                .map(|v| crate::poly::Polynomial::<Fr, crate::poly::LagrangeCoeff>::new(v.clone()))
                .collect();

        for row in 0..isize_n {
            // CPU mirror via decoded rules (kernel-equivalent path).
            let cpu_decoded = cpu_evaluate_from_encoding::<Fr>(
                &graph.rotations,
                &extended_constants,
                &rules,
                &vp_rules,
                /* expr_constants_horner_factor = */ Fr::ZERO,
                &fixed_cols,
                &[],
                &advice_cols,
                &[],
                row,
                isize_n,
            );
            let mut data = graph.instance();
            for (idx, rot) in graph.rotations.iter().enumerate() {
                let r = (row + rot).rem_euclid(isize_n);
                data.rotations[idx] = r as usize;
            }
            let mut last = Fr::ZERO;
            for info in &graph.calculations {
                last = info.calculation.evaluate(
                    &data.rotations,
                    &graph.constants,
                    &data.intermediates,
                    &fixed_polys,
                    &advice_polys,
                    &[],
                    &[],
                    &beta,
                    &gamma,
                    &theta,
                    &y,
                    &Fr::ZERO,
                );
                data.intermediates[info.target] = last;
            }
            assert_eq!(cpu_decoded, last, "row {} mismatch", row);
        }
        let _ = G1Affine::generator();
    }

    fn get_polys_ffi<F: Field>(polys: &[Vec<F>], row_start: usize) -> Vec<FFITraitObject> {
        polys
            .iter()
            .map(|poly| {
                assert!(row_start < poly.len());
                FFITraitObject::from_slice(&poly[row_start..])
            })
            .collect::<Vec<_>>()
    }

    #[test]
    fn test_eval() {
        use halo2curves::bn256::{Fr, G1Affine};

        let mut cs = ConstraintSystem::<Fr>::default();

        let a = cs.advice_column();
        let b = cs.advice_column();
        let c = cs.instance_column();

        cs.create_gate("sum_and_square", |cs| {
            let qa = cs.query_advice(a, Rotation::cur());
            let qb = cs.query_advice(b, Rotation::cur());
            let qc = cs.query_instance(c, Rotation::next());

            let sum_qa_qb = qa + qb;
            let qc_squared = qc.clone() * qc.clone();
            let qc_cubed = qc_squared * qc;
            let ret = sum_qa_qb.clone() * sum_qa_qb + qc_cubed;
            vec![ret]
        });

        let gpu_cs = GpuConstraintSystem::from(&cs);
        let evaluator = Evaluator::<G1Affine>::new(&gpu_cs);
        println!("{:?}", evaluator);
    }

    #[test]
    fn test_evaluate_h() {
        use std::time::Instant;

        use halo2curves::bn256::{Fr, G1Affine};

        use crate::{
            circuit::{Layouter, SimpleFloorPlanner, Value},
            plonk::{Advice, Circuit, Column, Error, Fixed, TableColumn},
        };
        #[derive(Clone, Debug)]
        struct TestCircuitConfig {
            advice_cols: Vec<Column<Advice>>,
            coeff_cols: Vec<Column<Fixed>>,
            coeff_next_col: Column<Fixed>,

            byte_selector: Column<Fixed>,
            byte_table: TableColumn,
        }

        struct TestCircuit {
            // an array of uint128
            data: Vec<Option<u128>>,
        }

        impl<F: PrimeField> Circuit<F> for TestCircuit {
            type Config = TestCircuitConfig;
            type FloorPlanner = SimpleFloorPlanner;
            #[cfg(feature = "circuit-params")]
            type Params = ();

            fn without_witnesses(&self) -> Self {
                todo!()
            }

            fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
                let n = 32;
                let advice_cols = (0..n)
                    .map(|_| cs.advice_column())
                    .collect::<Vec<Column<Advice>>>();
                let coeff_cols = (0..n)
                    .map(|_| cs.fixed_column())
                    .collect::<Vec<Column<Fixed>>>();
                let coeff_next_col = cs.fixed_column();
                let byte_selector = cs.fixed_column();
                let byte_table = cs.lookup_table_column();

                cs.create_gate("linear combine", |cells| {
                    let q_advices = advice_cols
                        .iter()
                        .map(|adv| cells.query_advice(*adv, Rotation::cur()))
                        .collect::<Vec<_>>();

                    let q_coeffs = coeff_cols
                        .iter()
                        .map(|fixed| cells.query_fixed(*fixed, Rotation::cur()))
                        .collect::<Vec<_>>();
                    let q_coeff_next = cells.query_fixed(coeff_next_col, Rotation::cur());

                    let q_advice_next = cells.query_advice(advice_cols[n - 1], Rotation::next());

                    let expr = q_advice_next * q_coeff_next;
                    let expr = q_advices
                        .iter()
                        .zip(q_coeffs.iter().skip(1))
                        .fold(expr, |acc, (adv, coeff)| acc + adv.clone() * coeff.clone());

                    vec![expr]
                });

                cs.lookup("byte table lookup", |cells| {
                    let q_byte = cells.query_fixed(byte_selector, Rotation::cur());

                    let advice_lookups = advice_cols
                        .iter()
                        .map(|adv| {
                            let q_adv = cells.query_advice(*adv, Rotation::cur());

                            (q_byte.clone() * q_adv, byte_table)
                        })
                        .collect::<Vec<_>>();

                    advice_lookups
                });

                TestCircuitConfig {
                    advice_cols,
                    coeff_cols,
                    coeff_next_col,
                    byte_selector,
                    byte_table,
                }
            }

            fn synthesize(
                &self,
                config: Self::Config,
                mut layouter: impl Layouter<F>,
            ) -> Result<(), Error> {
                layouter.assign_table(
                    || "table for uint8",
                    |mut table| {
                        for i in 0..256 {
                            table.assign_cell(
                                || format!("entry {}", i),
                                config.byte_table,
                                i,
                                || Value::known(F::from(i as u64)),
                            )?;
                        }
                        Ok(())
                    },
                )?;

                layouter.assign_region(
                    || "uint256 assign",
                    |mut region| {
                        let mut offset = 0;
                        let n = 32;
                        let powers = (0..n)
                            .scan(F::ONE, |acc, _| {
                                let b = F::from(256_u64);
                                *acc = *acc * b;

                                Some(*acc)
                            })
                            .collect::<Vec<F>>();

                        for val in self.data.iter() {
                            let (le_bytes, a) = match val {
                                None => (vec![None; n], None),
                                Some(x) => {
                                    let low = *x;
                                    let high = 0_u128;
                                    let le_bytes = low
                                        .to_le_bytes()
                                        .iter()
                                        .chain(high.to_le_bytes().iter())
                                        .map(|byte| Some(F::from(*byte as u64)))
                                        .collect::<Vec<Option<F>>>();
                                    let mask = 1_u128 << (64 - 1_u128);
                                    let a = F::from((low >> 64) as u64);
                                    let b = F::from((low & mask) as u64);
                                    let v = a * powers[8] + b;

                                    (le_bytes, Some(v))
                                }
                            };

                            // a uint256 is split into two rows
                            // a0 | a1 | .... | a255
                            //  0 |  0 | .... |  a
                            region.assign_fixed(config.byte_selector, offset, F::ONE);
                            for i in 0..n {
                                region.assign_advice(
                                    config.advice_cols[i],
                                    offset,
                                    Value::known(le_bytes[i].unwrap()),
                                );

                                region.assign_fixed(config.coeff_cols[i], offset, powers[i]);
                            }

                            region.assign_fixed(config.byte_selector, offset + 1, F::ZERO);
                            region.assign_advice(
                                config.advice_cols[n - 1],
                                offset + 1,
                                Value::known(a.unwrap()),
                            );
                            region.assign_fixed(config.coeff_next_col, offset + 1, -F::ONE);
                            offset += 2;
                        }
                        Ok(())
                    },
                )?;

                Ok(())
            }
        }

        let mut cs = ConstraintSystem::<Fr>::default();
        // For now, TestCircuit is only used to create a new CS such that we can extract
        // a evaluator from it.
        let test_config = TestCircuit::configure(&mut cs);

        let gpu_cs = GpuConstraintSystem::from(&cs);
        let mut evaluator = Evaluator::<G1Affine>::new(&gpu_cs);

        let vp = evaluator.custom_gates.calculations.last_mut().unwrap();
        match &mut (vp.calculation) {
            Calculation::Horner(_value, parts, _y) => {
                parts.push(ValueSource::Intermediate(120));
            }
            _ => unreachable!("{:?}", vp.calculation),
        }

        let (intermediate_rules, value_part_rules) = evaluator.encode();

        // Generate deterministic rows for comparing GPU and CPU h evaluation.
        let gen_rand_rows = |num_rows: usize, num_cols: usize| {
            (0..num_cols)
                .map(|i| {
                    (0..num_rows)
                        .map(|j| {
                            let s = num_rows * i + j;
                            Fr::from(s as u64)
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        };

        let num_rows = 1 << 20;
        let num_fixed = 3 + test_config.coeff_cols.len();
        let num_advices = test_config.advice_cols.len();

        let fixed = gen_rand_rows(num_rows, num_fixed);
        let instance = gen_rand_rows(num_rows, 0);
        let advices = gen_rand_rows(num_rows, num_advices);

        let fixed_polys = fixed
            .clone()
            .into_iter()
            .map(Polynomial::<Fr, LagrangeCoeff>::new)
            .collect::<Vec<_>>();
        let instance_polys = instance
            .clone()
            .into_iter()
            .map(Polynomial::<Fr, LagrangeCoeff>::new)
            .collect::<Vec<_>>();
        let advice_polys = advices
            .clone()
            .into_iter()
            .map(Polynomial::<Fr, LagrangeCoeff>::new)
            .collect::<Vec<_>>();
        let challenges = vec![Fr::from(1u64), Fr::from(2u64), Fr::from(3u64)];
        println!(
            "# intermediate rule: {}, # value_part rule = {}",
            intermediate_rules.len(),
            value_part_rules.len()
        );
        println!(
            "value_part: {:?}",
            evaluator.custom_gates.calculations.last().unwrap()
        );

        let gen_test = |row_start: usize, row_end: usize| {
            println!("test row: [{}, {}]", row_start, row_end);
            let custom_gates = &evaluator.custom_gates;
            let fixed_ffi = get_polys_ffi(&fixed, 0);
            let instance_ffi = get_polys_ffi(&instance, 0);
            let advice_ffi = get_polys_ffi(&advices, 0);
            let challenges_ffi = FFITraitObject::from_slice(&challenges);
            let constant_ffi = get_poly_ffi(&custom_gates.constants, 0);

            // [0, 1, -1, beta, theta, gamma, y]
            let beta = Fr::from(2);
            let theta = Fr::from(3);
            let gamma = Fr::from(4);
            let expr_constant = vec![Fr::ZERO, Fr::ONE, -Fr::ONE, Fr::from(2), Fr::from(1)];
            let expr_constant_ffi = get_poly_ffi(&expr_constant, 0);

            // intermediate rule
            let intermediate_rule_ffi = get_rule_ffi(&intermediate_rules);

            // value part rule
            let value_part_rules_ffi = get_rule_ffi_u64(&value_part_rules);

            // quotient polynomial h
            let quotient_poly = vec![Fr::ZERO; num_rows];
            let quotient_poly_ffi = get_poly_ffi(&quotient_poly, row_start);
            let quotient_lenght = quotient_poly.len();

            let gpu_time = Instant::now();
            let scratch_bytes = unsafe {
                _halo2_quotient_workspace_size(
                    intermediate_rule_ffi,
                    intermediate_rules.len() as u64,
                    value_part_rules.len() as u64,
                    quotient_lenght as u64,
                    fixed.len() as u64,
                    instance.len() as u64,
                    advices.len() as u64,
                    challenges.len() as u64,
                    custom_gates.constants.len() as u64,
                    1_u64,
                    row_start as u64,
                    row_end as u64,
                )
            } as usize;
            let scratch = DeviceBuffer::<u8>::with_capacity_on(scratch_bytes, &HALO2_GPU_CTX);
            let err = unsafe {
                _halo2_quotient(
                    fixed_ffi.as_ptr(),
                    fixed.len() as u64,
                    instance_ffi.as_ptr(),
                    instance.len() as u64,
                    advice_ffi.as_ptr(),
                    advices.len() as u64,
                    challenges_ffi,
                    challenges.len() as u64,
                    constant_ffi,
                    custom_gates.constants.len() as u64,
                    expr_constant_ffi,
                    intermediate_rule_ffi,
                    intermediate_rules.len() as u64,
                    value_part_rules_ffi,
                    value_part_rules.len() as u64,
                    quotient_poly_ffi,
                    quotient_lenght as u64,
                    1_u64,
                    row_start as u64,
                    row_end as u64,
                    scratch.as_mut_raw_ptr(),
                    scratch_bytes as u64,
                    HALO2_GPU_CTX.stream.as_raw(),
                )
            };
            if err.code != 0 {
                println!("gpu[{}]: _halo2_quotient error\n", 0);
                panic!("{}", String::from(err));
            }

            let gpu_time = gpu_time.elapsed();
            println!("    GPU h took {:?}", gpu_time);

            // compute h using CPU
            // this is the correct h
            let isize = num_rows as i32;
            let rot_scale = 1;
            {
                crate::perf_section!("cpu_evaluate_h");
                for (i, value_gpu) in quotient_poly
                    .iter()
                    .enumerate()
                    .take(row_end)
                    .skip(row_start)
                {
                    let mut eval_data = custom_gates.instance();
                    let mut value = Fr::ZERO;
                    value = custom_gates.evaluate(
                        &mut eval_data,
                        &fixed_polys,
                        &advice_polys,
                        &instance_polys,
                        &challenges,
                        &beta,
                        &gamma,
                        &theta,
                        &expr_constant[4],
                        &value,
                        i,
                        rot_scale,
                        isize,
                    );

                    assert_eq!(value, *value_gpu, "i = {}", i);
                }
            }
        };

        let intermediate_rule_ffi = get_rule_ffi(&intermediate_rules);
        let chunk_size = unsafe {
            let custom_gates = &evaluator.custom_gates;
            _halo2_evaluate_h_max_rows(
                intermediate_rule_ffi,
                intermediate_rules.len() as u64,
                value_part_rules.len() as u64,
                num_rows as u64,
                fixed.len() as u64,
                instance.len() as u64,
                advices.len() as u64,
                challenges.len() as u64,
                custom_gates.constants.len() as u64,
                1,
                0_u64,
                (num_rows / 4) as u64,
                query_device_free_bytes_for_chunking() as u64,
            )
        };
        let num_chunks = ((num_rows as f64) / (chunk_size as f64)).ceil() as usize;
        println!(
            "num_rows={}, max_chunk_size = {}, num_chunks = {}",
            num_rows, chunk_size, num_chunks
        );
        for i in 0..num_chunks {
            let row_start = i as u64 * chunk_size;
            let row_end = std::cmp::min((i + 1) as u64 * chunk_size, (num_rows) as u64);
            gen_test(row_start as usize, row_end as usize);
        }
    }
}
