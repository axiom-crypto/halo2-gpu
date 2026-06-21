use std::error;
use std::fmt;
use std::io;

use openvm_cuda_common::error::CudaError;

use crate::cuda::HaloGpuError;

use super::TableColumn;
use super::{Any, Column};

/// This is an error that could occur during proving or circuit synthesis.
#[derive(Debug)]
pub enum GpuError {
    /// This is an error that can occur during synthesis of the circuit, for
    /// example, when the witness is not present.
    Synthesis,
    /// The provided instances do not match the circuit parameters.
    InvalidInstances,
    /// The constraint system is not satisfied.
    ConstraintSystemFailure,
    /// Out of bounds index passed to a backend
    BoundsFailure,
    /// Opening error
    Opening,
    /// Transcript error
    Transcript(io::Error),
    /// `k` is too small for the given circuit.
    NotEnoughRowsAvailable {
        /// The current value of `k` being used.
        current_k: u32,
    },
    /// Instance provided exceeds number of available rows
    InstanceTooLarge,
    /// Circuit synthesis requires global constants, but circuit configuration did not
    /// call [`ConstraintSystem::enable_constant`] on fixed columns with sufficient space.
    ///
    /// [`ConstraintSystem::enable_constant`]: crate::plonk::ConstraintSystem::enable_constant
    NotEnoughColumnsForConstants,
    /// The instance sets up a copy constraint involving a column that has not been
    /// included in the permutation.
    ColumnNotInPermutation(Column<Any>),
    /// An error relating to a lookup table.
    TableError(TableError),
    /// A CUDA FFI call returned a non-zero status. Surfaces the typed
    /// `openvm_cuda_common::error::CudaError`.
    Cuda(CudaError),
    /// A halo2-gpu-internal error: synthetic preflight rejection or wrapped
    /// CUDA error from a fn that can produce both. Distinct from `Cuda`
    /// because the variants distinguish synthetic-vs-driver origin at the
    /// type level.
    HaloGpu(HaloGpuError),
}

impl From<io::Error> for GpuError {
    fn from(error: io::Error) -> Self {
        // The only place we can get io::Error from is the transcript.
        GpuError::Transcript(error)
    }
}

impl From<CudaError> for GpuError {
    fn from(err: CudaError) -> Self {
        GpuError::Cuda(err)
    }
}

impl From<HaloGpuError> for GpuError {
    fn from(err: HaloGpuError) -> Self {
        GpuError::HaloGpu(err)
    }
}

/// Converts a canonical halo2-axiom synthesis error into the GPU crate's `Error`.
/// The GPU keygen drives the canonical `Circuit::synthesize` (whose `Assignment`
/// methods return `halo2_axiom::plonk::Error`); this bridge lets the keygen
/// entry points propagate those via `?` while returning the GPU `Error`.
impl From<halo2_axiom::plonk::Error> for GpuError {
    fn from(e: halo2_axiom::plonk::Error) -> Self {
        use halo2_axiom::plonk::Error as Canonical;
        match e {
            Canonical::NotEnoughRowsAvailable { current_k } => {
                GpuError::NotEnoughRowsAvailable { current_k }
            }
            Canonical::BoundsFailure => GpuError::BoundsFailure,
            Canonical::InvalidInstances => GpuError::InvalidInstances,
            Canonical::ConstraintSystemFailure => GpuError::ConstraintSystemFailure,
            Canonical::InstanceTooLarge => GpuError::InstanceTooLarge,
            Canonical::NotEnoughColumnsForConstants => GpuError::NotEnoughColumnsForConstants,
            Canonical::Opening => GpuError::Opening,
            // These three are losslessly mappable: after the unified-pk ESCALATE
            // the payloads are canonical re-exports — `io::Error` is `std`,
            // `Column<Any>` is the canonical type `GpuError::ColumnNotInPermutation`
            // already holds, and canonical `TableError` converts variant-for-variant
            // (its `TableColumn` is the canonical re-export too).
            Canonical::Transcript(e) => GpuError::Transcript(e),
            Canonical::ColumnNotInPermutation(column) => GpuError::ColumnNotInPermutation(column),
            Canonical::TableError(e) => GpuError::TableError(e.into()),
            // `Synthesis` carries no payload; it is the true generic fallback.
            _ => GpuError::Synthesis,
        }
    }
}

/// Converts a canonical halo2-axiom `TableError` into the GPU crate's local
/// `TableError`. The payloads (`TableColumn`) are canonical re-exports, so this
/// is a lossless, variant-for-variant mapping.
impl From<halo2_axiom::plonk::TableError> for TableError {
    fn from(e: halo2_axiom::plonk::TableError) -> Self {
        use halo2_axiom::plonk::TableError as Canonical;
        match e {
            Canonical::ColumnNotAssigned(col) => TableError::ColumnNotAssigned(col),
            Canonical::UnevenColumnLengths(a, b) => TableError::UnevenColumnLengths(a, b),
            Canonical::UsedColumn(col) => TableError::UsedColumn(col),
            Canonical::OverwriteDefault(col, default, val) => {
                TableError::OverwriteDefault(col, default, val)
            }
        }
    }
}

impl GpuError {
    /// Constructs an `Error::NotEnoughRowsAvailable`.
    pub(crate) fn not_enough_rows_available(current_k: u32) -> Self {
        GpuError::NotEnoughRowsAvailable { current_k }
    }
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuError::Synthesis => write!(f, "General synthesis error"),
            GpuError::InvalidInstances => write!(f, "Provided instances do not match the circuit"),
            GpuError::ConstraintSystemFailure => write!(f, "The constraint system is not satisfied"),
            GpuError::BoundsFailure => write!(f, "An out-of-bounds index was passed to the backend"),
            GpuError::Opening => write!(f, "Multi-opening proof was invalid"),
            GpuError::Transcript(e) => write!(f, "Transcript error: {}", e),
            GpuError::NotEnoughRowsAvailable { current_k } => write!(
                f,
                "k = {} is too small for the given circuit. Try using a larger value of k",
                current_k,
            ),
            GpuError::InstanceTooLarge => write!(f, "Instance vectors are larger than the circuit"),
            GpuError::NotEnoughColumnsForConstants => {
                write!(
                    f,
                    "Too few fixed columns are enabled for global constants usage"
                )
            }
            GpuError::ColumnNotInPermutation(column) => write!(
                f,
                "Column {:?} must be included in the permutation. Help: try applying `meta.enable_equalty` on the column",
                column
            ),
            GpuError::TableError(error) => write!(f, "{}", error),
            GpuError::Cuda(e) => write!(f, "CUDA error: {}", e),
            GpuError::HaloGpu(e) => write!(f, "halo2-gpu error: {}", e),
        }
    }
}

impl error::Error for GpuError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            GpuError::Transcript(e) => Some(e),
            GpuError::Cuda(e) => Some(e),
            GpuError::HaloGpu(e) => Some(e),
            _ => None,
        }
    }
}

/// This is an error that could occur during table synthesis.
#[derive(Debug)]
pub enum TableError {
    /// A `TableColumn` has not been assigned.
    ColumnNotAssigned(TableColumn),
    /// A Table has columns of uneven lengths.
    UnevenColumnLengths((TableColumn, usize), (TableColumn, usize)),
    /// Attempt to assign a used `TableColumn`
    UsedColumn(TableColumn),
    /// Attempt to overwrite a default value
    OverwriteDefault(TableColumn, String, String),
}

impl fmt::Display for TableError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TableError::ColumnNotAssigned(col) => {
                write!(
                    f,
                    "{:?} not fully assigned. Help: assign a value at offset 0.",
                    col
                )
            }
            TableError::UnevenColumnLengths((col, col_len), (table, table_len)) => write!(
                f,
                "{:?} has length {} while {:?} has length {}",
                col, col_len, table, table_len
            ),
            TableError::UsedColumn(col) => {
                write!(f, "{:?} has already been used", col)
            }
            TableError::OverwriteDefault(col, default, val) => {
                write!(
                    f,
                    "Attempted to overwrite default value {} with {} in {:?}",
                    default, val, col
                )
            }
        }
    }
}
