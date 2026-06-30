//! Implementation of permutation argument.

use super::circuit::{GpuAny, GpuColumn};
use crate::arithmetic::CurveAffine;

pub(crate) mod keygen;
pub(crate) mod prover;
pub(crate) mod verifier;

pub use keygen::Assembly;

/// A permutation argument.
#[derive(Debug, Clone)]
pub struct Argument {
    /// A sequence of columns involved in the argument.
    pub columns: Vec<GpuColumn<GpuAny>>,
}

impl Argument {
    pub(crate) fn new() -> Self {
        Argument { columns: vec![] }
    }

    /// Returns the minimum circuit degree required by the permutation argument.
    /// The argument may use larger degree gates depending on the actual
    /// circuit's degree and how many columns are involved in the permutation.
    pub(crate) fn required_degree(&self) -> usize {
        // degree 2:
        // l_0(X) * (1 - z(X)) = 0
        //
        // We will fit as many polynomials p_i(X) as possible
        // into the required degree of the circuit, so the
        // following will not affect the required degree of
        // this middleware.
        //
        // (1 - (l_last(X) + l_blind(X))) * (
        //   z(\omega X) \prod (p(X) + \beta s_i(X) + \gamma)
        // - z(X) \prod (p(X) + \delta^i \beta X + \gamma)
        // )
        //
        // On the first sets of columns, except the first
        // set, we will do
        //
        // l_0(X) * (z(X) - z'(\omega^(last) X)) = 0
        //
        // where z'(X) is the permutation for the previous set
        // of columns.
        //
        // On the final set of columns, we will do
        //
        // degree 3:
        // l_last(X) * (z'(X)^2 - z'(X)) = 0
        //
        // which will allow the last value to be zero to
        // ensure the argument is perfectly complete.

        // There are constraints of degree 3 regardless of the
        // number of columns involved.
        3
    }

    pub(crate) fn add_column(&mut self, column: GpuColumn<GpuAny>) {
        if !self.columns.contains(&column) {
            self.columns.push(column);
        }
    }

    /// Returns columns that participate on the permutation argument.
    pub fn get_columns(&self) -> Vec<GpuColumn<GpuAny>> {
        self.columns.clone()
    }
}

/// Rebuilds the GPU-crate permutation `Argument` from the canonical halo2-axiom one.
impl From<&halo2_axiom::plonk::permutation::Argument> for Argument {
    fn from(a: &halo2_axiom::plonk::permutation::Argument) -> Self {
        Argument { columns: a.get_columns().iter().map(GpuColumn::from).collect() }
    }
}

/// The verifying key for a single permutation argument.
#[derive(Debug, Clone)]
pub struct VerifyingKey<C: CurveAffine> {
    pub commitments: Vec<C>,
}

impl<C: CurveAffine> VerifyingKey<C> {
    /// Returns commitments of sigma polynomials
    pub fn commitments(&self) -> &Vec<C> {
        &self.commitments
    }
}
