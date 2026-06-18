use ff::Field;
use rustc_hash::FxHashMap;

use std::fmt;
use std::marker::PhantomData;

use crate::circuit::layouter::SyncDeps;
use crate::circuit::table_layouter::compute_table_lengths;
use crate::circuit::SimpleTableLayouter;
use crate::{
    circuit::{
        layouter::{RegionColumn, RegionLayouter, TableLayouter},
        AssignedCell, Cell, Layouter, Region, RegionIndex, Table, Value,
    },
    plonk::{
        Advice, Any, Assigned, Assignment, Challenge, Circuit, Column, Error, Fixed, FloorPlanner,
        Instance, Selector, TableColumn,
    },
};

/// A simple [`FloorPlanner`] that performs minimal optimizations.
///
/// This floor planner is suitable for debugging circuits. It aims to reflect the circuit
/// "business logic" in the circuit layout as closely as possible. It uses a single-pass
/// layouter that does not reorder regions for optimal packing.
#[derive(Debug)]
pub struct SimpleFloorPlanner;

impl FloorPlanner for SimpleFloorPlanner {
    fn synthesize<F: Field, CS: Assignment<F> + SyncDeps, C: Circuit<F>>(
        cs: &mut CS,
        circuit: &C,
        config: C::Config,
        constants: Vec<Column<Fixed>>,
    ) -> Result<(), Error> {
        let layouter = SingleChipLayouter::new(cs, constants)?;
        circuit.synthesize(config, layouter)
    }
}

/// A [`Layouter`] for a single-chip circuit.
pub struct SingleChipLayouter<'a, F: Field, CS: Assignment<F> + 'a> {
    cs: &'a mut CS,
    constants: Vec<Column<Fixed>>,
    /// Stores the first empty row for each column.
    columns: FxHashMap<RegionColumn, usize>,
    /// Stores the table fixed columns.
    table_columns: Vec<TableColumn>,
    _marker: PhantomData<F>,
}

impl<'a, F: Field, CS: Assignment<F> + 'a> fmt::Debug for SingleChipLayouter<'a, F, CS> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SingleChipLayouter")
            .field("columns", &self.columns)
            .finish()
    }
}

impl<'a, F: Field, CS: Assignment<F>> SingleChipLayouter<'a, F, CS> {
    /// Creates a new single-chip layouter.
    pub fn new(cs: &'a mut CS, constants: Vec<Column<Fixed>>) -> Result<Self, Error> {
        let ret = SingleChipLayouter {
            cs,
            constants,
            columns: FxHashMap::default(),
            table_columns: vec![],
            _marker: PhantomData,
        };
        Ok(ret)
    }
}

impl<'a, F: Field, CS: Assignment<F> + 'a + SyncDeps> Layouter<F>
    for SingleChipLayouter<'a, F, CS>
{
    type Root = Self;

    fn assign_region<A, AR, N, NR>(&mut self, name: N, assignment: A) -> Result<AR, Error>
    where
        A: FnOnce(Region<'_, F>) -> Result<AR, Error>,
        N: Fn() -> NR,
        NR: Into<String>,
    {
        // Assign region cells.
        self.cs.enter_region(name);
        let mut region = SingleChipLayouterRegion::new(self, 0.into());
        let result = {
            let region: &mut dyn RegionLayouter<F> = &mut region;
            assignment(region.into())
        }?;
        let constants_to_assign = region.constants;
        self.cs.exit_region();

        // Assign constants. For the simple floor planner, we assign constants in order in
        // the first `constants` column.
        if self.constants.is_empty() {
            if !constants_to_assign.is_empty() {
                return Err(Error::NotEnoughColumnsForConstants);
            }
        } else {
            let constants_column = self.constants[0];
            let next_constant_row = self
                .columns
                .entry(Column::<Any>::from(constants_column).into())
                .or_default();
            for (constant, advice) in constants_to_assign {
                self.cs
                    .assign_fixed(constants_column, *next_constant_row, constant);
                self.cs.copy(
                    constants_column.into(),
                    *next_constant_row,
                    advice.column,
                    advice.row_offset,
                );
                *next_constant_row += 1;
            }
        }

        Ok(result)
    }

    fn assign_table<A, N, NR>(&mut self, name: N, mut assignment: A) -> Result<(), Error>
    where
        A: FnMut(Table<'_, F>) -> Result<(), Error>,
        N: Fn() -> NR,
        NR: Into<String>,
    {
        // Maintenance hazard: there is near-duplicate code in `v1::AssignmentPass::assign_table`.
        // Assign table cells.
        self.cs.enter_region(name);
        let mut table = SimpleTableLayouter::new(self.cs, &self.table_columns);
        {
            let table: &mut dyn TableLayouter<F> = &mut table;
            assignment(table.into())
        }?;
        let default_and_assigned = table.default_and_assigned;
        self.cs.exit_region();

        // Check that all table columns have the same length `first_unused`,
        // and all cells up to that length are assigned.
        let first_unused = compute_table_lengths(&default_and_assigned)?;

        // Record these columns so that we can prevent them from being used again.
        for column in default_and_assigned.keys() {
            self.table_columns.push(*column);
        }

        for (col, (default_val, _)) in default_and_assigned {
            // default_val must be Some because we must have assigned
            // at least one cell in each column, and in that case we checked
            // that all cells up to first_unused were assigned.
            self.cs
                .fill_from_row(col.inner(), first_unused, default_val.unwrap())?;
        }

        Ok(())
    }

    fn constrain_instance(&mut self, cell: Cell, instance: Column<Instance>, row: usize) {
        self.cs
            .copy(cell.column, cell.row_offset, instance.into(), row);
    }

    fn get_challenge(&self, challenge: Challenge) -> Value<F> {
        self.cs.get_challenge(challenge)
    }

    fn next_phase(&mut self) {
        self.cs.next_phase();
    }

    fn get_root(&mut self) -> &mut Self::Root {
        self
    }

    fn push_namespace<NR, N>(&mut self, name_fn: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
        self.cs.push_namespace(name_fn)
    }

    fn pop_namespace(&mut self, gadget_name: Option<String>) {
        self.cs.pop_namespace(gadget_name)
    }
}

struct SingleChipLayouterRegion<'r, 'a, F: Field, CS: Assignment<F> + 'a> {
    layouter: &'r mut SingleChipLayouter<'a, F, CS>,
    region_index: RegionIndex,
    /// Stores the constants to be assigned, and the cells to which they are copied.
    constants: Vec<(Assigned<F>, Cell)>,
}

impl<'r, 'a, F: Field, CS: Assignment<F> + 'a> fmt::Debug
    for SingleChipLayouterRegion<'r, 'a, F, CS>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SingleChipLayouterRegion")
            .field("layouter", &self.layouter)
            .field("region_index", &self.region_index)
            .finish()
    }
}

impl<'r, 'a, F: Field, CS: Assignment<F> + 'a> SingleChipLayouterRegion<'r, 'a, F, CS> {
    fn new(layouter: &'r mut SingleChipLayouter<'a, F, CS>, region_index: RegionIndex) -> Self {
        SingleChipLayouterRegion {
            layouter,
            region_index,
            constants: vec![],
        }
    }
}

impl<'r, 'a, F: Field, CS: Assignment<F> + 'a + SyncDeps> RegionLayouter<F>
    for SingleChipLayouterRegion<'r, 'a, F, CS>
{
    fn enable_selector<'v>(
        &'v mut self,
        annotation: &'v (dyn Fn() -> String + 'v),
        selector: &Selector,
        offset: usize,
    ) -> Result<(), Error> {
        self.layouter
            .cs
            .enable_selector(annotation, selector, offset)
    }

    fn name_column<'v>(
        &'v mut self,
        annotation: &'v (dyn Fn() -> String + 'v),
        column: Column<Any>,
    ) {
        self.layouter.cs.annotate_column(annotation, column);
    }

    fn assign_advice<'v>(
        &mut self,
        column: Column<Advice>,
        offset: usize,
        to: Value<Assigned<F>>,
    ) -> AssignedCell<&'v Assigned<F>, F> {
        let value = self.layouter.cs.assign_advice(column, offset, to);

        AssignedCell {
            value,
            cell: Cell {
                row_offset: offset,
                column: column.into(),
            },
            _marker: PhantomData,
        }
    }

    fn assign_advice_from_constant<'v>(
        &'v mut self,
        _annotation: &'v (dyn Fn() -> String + 'v),
        column: Column<Advice>,
        offset: usize,
        constant: Assigned<F>,
    ) -> Result<Cell, Error> {
        let advice = self
            .assign_advice(column, offset, Value::known(constant))
            .cell;
        self.constrain_constant(advice, constant)?;

        Ok(advice)
    }

    fn assign_advice_from_instance<'v>(
        &mut self,
        _annotation: &'v (dyn Fn() -> String + 'v),
        instance: Column<Instance>,
        row: usize,
        advice: Column<Advice>,
        offset: usize,
    ) -> Result<(Cell, Value<F>), Error> {
        let value = self.layouter.cs.query_instance(instance, row)?;

        let cell = self
            .assign_advice(advice, offset, value.map(|v| Assigned::Trivial(v)))
            .cell;

        self.layouter
            .cs
            .copy(cell.column, cell.row_offset, instance.into(), row);

        Ok((cell, value))
    }

    fn instance_value(
        &mut self,
        instance: Column<Instance>,
        row: usize,
    ) -> Result<Value<F>, Error> {
        self.layouter.cs.query_instance(instance, row)
    }

    fn assign_fixed(&mut self, column: Column<Fixed>, offset: usize, to: Assigned<F>) -> Cell {
        self.layouter.cs.assign_fixed(column, offset, to);

        Cell {
            row_offset: offset,
            column: column.into(),
        }
    }

    fn constrain_constant(&mut self, cell: Cell, constant: Assigned<F>) -> Result<(), Error> {
        self.constants.push((constant, cell));
        Ok(())
    }

    fn constrain_equal(&mut self, left: Cell, right: Cell) {
        self.layouter
            .cs
            .copy(left.column, left.row_offset, right.column, right.row_offset);
    }

    fn get_challenge(&self, challenge: Challenge) -> Value<F> {
        self.layouter.cs.get_challenge(challenge)
    }

    fn next_phase(&mut self) {
        self.layouter.cs.next_phase();
    }
}
