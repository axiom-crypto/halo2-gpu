//! Implementation of permutation argument.

use super::circuit::{Any, Column};
use crate::{
    arithmetic::CurveAffine,
    helpers::{
        polynomial_slice_byte_length, read_polynomial_vec, write_polynomial_slice,
        SerdeCurveAffine, SerdePrimeField,
    },
    poly::{Coeff, LagrangeCoeff, Polynomial},
    SerdeFormat,
};
use once_cell::sync::OnceCell;

pub(crate) mod keygen;
pub(crate) mod prover;
pub(crate) mod verifier;

pub use keygen::Assembly;

use std::io;

/// A permutation argument.
#[derive(Debug, Clone)]
pub struct Argument {
    /// A sequence of columns involved in the argument.
    pub columns: Vec<Column<Any>>,
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

    pub(crate) fn add_column(&mut self, column: Column<Any>) {
        if !self.columns.contains(&column) {
            self.columns.push(column);
        }
    }

    /// Returns columns that participate on the permutation argument.
    pub fn get_columns(&self) -> Vec<Column<Any>> {
        self.columns.clone()
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

    pub(crate) fn write<W: io::Write>(&self, writer: &mut W, format: SerdeFormat) -> io::Result<()>
    where
        C: SerdeCurveAffine,
    {
        for commitment in &self.commitments {
            commitment.write(writer, format)?;
        }
        Ok(())
    }

    pub(crate) fn read<R: io::Read>(
        reader: &mut R,
        argument: &Argument,
        format: SerdeFormat,
    ) -> io::Result<Self>
    where
        C: SerdeCurveAffine,
    {
        let commitments = (0..argument.columns.len())
            .map(|_| C::read(reader, format))
            .collect::<io::Result<Vec<_>>>()?;
        Ok(VerifyingKey { commitments })
    }

    pub(crate) fn bytes_length(&self) -> usize {
        self.commitments.len() * C::default().to_bytes().as_ref().len()
    }
}

/// The proving key for a single permutation argument.
#[derive(Debug)]
pub(crate) struct ProvingKey<C: CurveAffine> {
    pub(super) permutations: Vec<Polynomial<C::Scalar, LagrangeCoeff>>,
    pub(super) polys: Vec<Polynomial<C::Scalar, Coeff>>,
    /// Lazy device mirror of `permutations` (the Lagrange σ-columns
    /// referenced per-column by `permutation_product_device`).
    /// Populated on first `permutations_device()` call; empty after Clone.
    /// Mirrors the lifecycle of `pk.fixed_values_device` /
    /// `pk.permutation_polys_device` in the top-level `plonk::ProvingKey`.
    permutations_device: OnceCell<Vec<Polynomial<C::Scalar, LagrangeCoeff, crate::poly::Device>>>,
}

impl<C: CurveAffine> Clone for ProvingKey<C> {
    /// Empties the device `OnceCell` mirror, matching the `ParamsKZG` /
    /// top-level `ProvingKey` clone contract. A clone regenerates its
    /// device mirror lazily on first use.
    fn clone(&self) -> Self {
        Self {
            permutations: self.permutations.clone(),
            polys: self.polys.clone(),
            permutations_device: OnceCell::new(),
        }
    }
}

impl<C: SerdeCurveAffine> ProvingKey<C>
where
    C::Scalar: SerdePrimeField,
{
    /// Reads proving key for a single permutation argument from buffer using `Polynomial::read`.
    pub(super) fn read<R: io::Read>(reader: &mut R, format: SerdeFormat) -> Self {
        let permutations = read_polynomial_vec(reader, format);
        let polys = read_polynomial_vec(reader, format);
        ProvingKey {
            permutations,
            polys,
            permutations_device: OnceCell::new(),
        }
    }

    /// Writes proving key for a single permutation argument to buffer using `Polynomial::write`.
    pub(super) fn write<W: io::Write>(&self, writer: &mut W, format: SerdeFormat) {
        write_polynomial_slice(&self.permutations, writer, format);
        write_polynomial_slice(&self.polys, writer, format);
    }
}

impl<C: CurveAffine> ProvingKey<C> {
    /// Gets the total number of bytes in the serialization of `self`
    pub(super) fn bytes_length(&self) -> usize {
        polynomial_slice_byte_length(&self.permutations) + polynomial_slice_byte_length(&self.polys)
    }

    /// Read-only accessor for the Coeff form sigma polys. Used by
    /// `plonk::ProvingKey::permutation_polys_device` to build the PK
    /// Device-resident mirror.
    pub(crate) fn polys(&self) -> &[Polynomial<C::Scalar, Coeff>] {
        &self.polys
    }

    /// Lazy device mirror of `permutations` (Lagrange σ-columns). Returns
    /// `None` if VRAM-gated out. Used by `permutation::Argument::commit` to
    /// feed device σ pointers into `permutation_product_device`.
    pub(crate) fn permutations_device(
        &self,
    ) -> Option<&[Polynomial<C::Scalar, LagrangeCoeff, crate::poly::Device>]> {
        if let Some(v) = self.permutations_device.get() {
            return Some(v.as_slice());
        }
        super::try_init_pk_device_mirror::<C, LagrangeCoeff>(
            &self.permutations,
            "pk.permutation.permutations_device.init",
            &self.permutations_device,
        )
        .map(|v| v.as_slice())
    }
}
