use rayon::prelude::*;

// Keygen consumes the CANONICAL constraint system, so the permutation argument
// (and its columns) are the canonical halo2-axiom types — not the GPU fork
// `permutation::Argument`. `Any`/`Column` likewise resolve to the canonical
// frontend re-exports; `Error` is the GPU crate's own error enum.
use crate::plonk::{Any, Column, Error};
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
    ) -> Result<(), Error> {
        let left_column = self
            .columns
            .iter()
            .position(|c| c == &left_column)
            .ok_or(Error::ColumnNotInPermutation(left_column))?;
        let right_column = self
            .columns
            .iter()
            .position(|c| c == &right_column)
            .ok_or(Error::ColumnNotInPermutation(right_column))?;

        // Check bounds
        if left_row >= self.mapping[left_column].len()
            || right_row >= self.mapping[right_column].len()
        {
            return Err(Error::BoundsFailure);
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

    /// Returns mappings of the copies.
    pub fn mapping(
        &self,
    ) -> impl Iterator<Item = impl IndexedParallelIterator<Item = (usize, usize)> + '_> {
        self.mapping.iter().map(|c| c.par_iter().copied())
    }
}
