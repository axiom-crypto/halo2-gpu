use ff::{Field, PrimeField};
use group::Curve;
use rayon::prelude::*;

// Keygen consumes the CANONICAL constraint system, so the permutation argument
// (and its columns) are the canonical halo2-axiom types — not the GPU fork
// `permutation::Argument`. `Any`/`Column` likewise resolve to the canonical
// frontend re-exports; `Error` is the GPU crate's own error enum.
use crate::arithmetic::CurveAffine;
use crate::cpu::arithmetic::parallelize;
use crate::plonk::{Any, Column, GpuError};
use crate::poly::{
    commitment::{Blind, Params},
    EvaluationDomain, LagrangeCoeff, Polynomial,
};
use halo2_axiom::plonk::permutation::Argument;

/// Struct that accumulates all the necessary data in order to construct the permutation argument.
///
/// The permutation proving/verifying keys are now produced by halo2-axiom's keygen
/// (the canonical pk/vk), so the GPU crate keeps only the copy-constraint assembly
/// used by `dev::MockProver`; the former `build_pk`/`build_vk` helpers were removed
/// along with the GPU `permutation::ProvingKey`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Assembly {
    /// Columns that participate on the copy permutation argument.
    columns: Vec<Column<Any>>,
    /// Mapping of the actual copies done.
    mapping: Vec<Vec<(usize, usize)>>,
    /// Some aux data used to swap positions directly when sorting.
    aux: Vec<Vec<(usize, usize)>>,
    /// More aux data
    sizes: Vec<Vec<usize>>,
}

impl Assembly {
    pub(crate) fn new(n: usize, p: &Argument) -> Self {
        // Canonical permutation argument: `columns` field is private upstream, so
        // pull the (owned) column list via the public `get_columns()` accessor.
        let perm_columns = p.get_columns();
        let num_columns = perm_columns.len();

        // Initialize the copy vector to keep track of copy constraints in all
        // the permutation arguments.
        let mut mapping = vec![];
        for i in 0..num_columns {
            // Computes [(i, 0), (i, 1), ..., (i, n - 1)]
            mapping.push((0..n).map(|j| (i, j)).collect());
        }

        // Before any equality constraints are applied, every cell in the permutation is
        // in a 1-cycle; therefore mapping and aux are identical, because every cell is
        // its own distinguished element.
        Assembly {
            columns: perm_columns,
            aux: mapping.clone(),
            mapping,
            sizes: vec![vec![1usize; n]; num_columns],
        }
    }

    pub(crate) fn copy(
        &mut self,
        left_column: Column<Any>,
        left_row: usize,
        right_column: Column<Any>,
        right_row: usize,
    ) -> Result<(), GpuError> {
        let left_column = self
            .columns
            .iter()
            .position(|c| c == &left_column)
            .ok_or(GpuError::ColumnNotInPermutation(left_column))?;
        let right_column = self
            .columns
            .iter()
            .position(|c| c == &right_column)
            .ok_or(GpuError::ColumnNotInPermutation(right_column))?;

        // Check bounds
        if left_row >= self.mapping[left_column].len()
            || right_row >= self.mapping[right_column].len()
        {
            return Err(GpuError::BoundsFailure);
        }

        // See book/src/design/permutation.md for a description of this algorithm.

        let mut left_cycle = self.aux[left_column][left_row];
        let mut right_cycle = self.aux[right_column][right_row];

        // If left and right are in the same cycle, do nothing.
        if left_cycle == right_cycle {
            return Ok(());
        }

        if self.sizes[left_cycle.0][left_cycle.1] < self.sizes[right_cycle.0][right_cycle.1] {
            std::mem::swap(&mut left_cycle, &mut right_cycle);
        }

        // Merge the right cycle into the left one.
        self.sizes[left_cycle.0][left_cycle.1] += self.sizes[right_cycle.0][right_cycle.1];
        let mut i = right_cycle;
        loop {
            self.aux[i.0][i.1] = left_cycle;
            i = self.mapping[i.0][i.1];
            if i == right_cycle {
                break;
            }
        }

        let tmp = self.mapping[left_column][left_row];
        self.mapping[left_column][left_row] = self.mapping[right_column][right_row];
        self.mapping[right_column][right_row] = tmp;

        Ok(())
    }

    /// Returns columns that participate in the permutation argument.
    pub fn columns(&self) -> &[Column<Any>] {
        &self.columns
    }

    /// Builds the **canonical** permutation [`VerifyingKey`] (σ-column
    /// commitments) via GPU MSM. Consumes `self` for symmetry with `build_pk`.
    pub(crate) fn build_vk<'params, C: CurveAffine, P: Params<'params, C>>(
        self,
        params: &P,
        domain: &EvaluationDomain<C::Scalar>,
        p: &Argument,
    ) -> halo2_axiom::plonk::permutation::VerifyingKey<C> {
        build_vk(params, domain, p, |i, j| self.mapping[i][j])
    }

    /// Builds the **canonical** permutation [`ProvingKey`] (σ-polys in Lagrange
    /// + Coeff form) via GPU iFFT.
    pub(crate) fn build_pk<'params, C: CurveAffine, P: Params<'params, C>>(
        self,
        params: &P,
        domain: &EvaluationDomain<C::Scalar>,
        p: &Argument,
    ) -> Result<halo2_axiom::plonk::permutation::ProvingKey<C>, GpuError> {
        build_pk(params, domain, p, |i, j| self.mapping[i][j])
    }

    /// Returns mappings of the copies.
    pub fn mapping(
        &self,
    ) -> impl Iterator<Item = impl IndexedParallelIterator<Item = (usize, usize)> + '_> {
        self.mapping.iter().map(|c| c.par_iter().copied())
    }
}

/// Computes the σ-permutation polynomials (Lagrange basis) from the copy
/// `mapping`, then GPU-iFFTs them to Coeff form, returning the **canonical**
/// permutation `ProvingKey` via `from_parts`. Port of the b222e4d GPU keygen
/// (GPU `lagrange_to_coeff_many`; no CPU-MSM).
pub(crate) fn build_pk<'params, C: CurveAffine, P: Params<'params, C>>(
    params: &P,
    domain: &EvaluationDomain<C::Scalar>,
    p: &Argument,
    mapping: impl Fn(usize, usize) -> (usize, usize) + Sync,
) -> Result<halo2_axiom::plonk::permutation::ProvingKey<C>, GpuError> {
    let permutations = permutation_lagrange_polys::<C, P>(params, domain, p, mapping);
    // GPU iFFT. NOTE: do not interleave parallelize() with GPU fft() — risks GPU OOM.
    let polys = domain.lagrange_to_coeff_many(&permutations)?;
    Ok(halo2_axiom::plonk::permutation::ProvingKey::from_parts(
        permutations,
        polys,
    ))
}

/// Computes the σ-permutation polynomials and their GPU-MSM commitments,
/// returning the **canonical** permutation `VerifyingKey` via `from_commitments`.
pub(crate) fn build_vk<'params, C: CurveAffine, P: Params<'params, C>>(
    params: &P,
    domain: &EvaluationDomain<C::Scalar>,
    p: &Argument,
    mapping: impl Fn(usize, usize) -> (usize, usize) + Sync,
) -> halo2_axiom::plonk::permutation::VerifyingKey<C> {
    let permutations = permutation_lagrange_polys::<C, P>(params, domain, p, mapping);
    // GPU MSM commitment per σ-column.
    let commitments = permutations
        .iter()
        .map(|permutation| {
            params
                .commit_lagrange(permutation, Blind::default())
                .to_affine()
        })
        .collect();
    halo2_axiom::plonk::permutation::VerifyingKey::from_commitments(commitments)
}

/// Shared σ-polynomial construction: `permutation_poly[col][row] =
/// δ^{permuted_col} · ω^{permuted_row}` per the copy `mapping`.
fn permutation_lagrange_polys<'params, C: CurveAffine, P: Params<'params, C>>(
    params: &P,
    domain: &EvaluationDomain<C::Scalar>,
    p: &Argument,
    mapping: impl Fn(usize, usize) -> (usize, usize) + Sync,
) -> Vec<Polynomial<C::Scalar, LagrangeCoeff>> {
    let num_columns = p.get_columns().len();

    // Compute [omega^0, omega^1, ..., omega^{params.n - 1}]
    let mut omega_powers = vec![C::Scalar::ZERO; params.n() as usize];
    {
        let omega = domain.get_omega();
        parallelize(&mut omega_powers, |o, start| {
            let mut cur = omega.pow_vartime([start as u64]);
            for v in o.iter_mut() {
                *v = cur;
                cur *= &omega;
            }
        })
    }

    // Compute [omega_powers * \delta^0, omega_powers * \delta^1, ..., omega_powers * \delta^m]
    let mut deltaomega = vec![omega_powers; num_columns];
    {
        parallelize(&mut deltaomega, |o, start| {
            let mut cur = C::Scalar::DELTA.pow_vartime([start as u64]);
            for omega_powers in o.iter_mut() {
                for v in omega_powers {
                    *v *= &cur;
                }
                cur *= &C::Scalar::DELTA;
            }
        });
    }

    // Compute permutation polynomials.
    let mut permutations = vec![domain.empty_lagrange(); num_columns];
    {
        parallelize(&mut permutations, |o, start| {
            for (x, permutation_poly) in o.iter_mut().enumerate() {
                let i = start + x;
                for (j, p) in permutation_poly.iter_mut().enumerate() {
                    let (permuted_i, permuted_j) = mapping(i, j);
                    *p = deltaomega[permuted_i][permuted_j];
                }
            }
        });
    }

    permutations
}
