use ff::PrimeField;
use halo2curves::{serde::SerdeObject, CurveAffine};

use std::io;

// The serde format is the halo2-axiom `SerdeFormat`, re-exported so that
// consumers calling `ProvingKey`/`VerifyingKey` read/write (e.g.
// `snark-verifier-sdk`) pass the type the key methods expect. The GPU crate's
// own serde (`ParamsKZG`, the `Serde*` traits below) matches on the same three
// variants.
pub use halo2_axiom::SerdeFormat;

// Keep this trait for compatibility with IPA serialization
pub(crate) trait CurveRead: CurveAffine {
    /// Reads a compressed element from the buffer and attempts to parse it
    /// using `from_bytes`.
    fn read<R: io::Read>(reader: &mut R) -> io::Result<Self> {
        let mut compressed = Self::Repr::default();
        reader.read_exact(compressed.as_mut())?;
        Option::from(Self::from_bytes(&compressed))
            .ok_or_else(|| io::Error::other("Invalid point encoding in proof"))
    }
}
impl<C: CurveAffine> CurveRead for C {}

pub trait SerdeCurveAffine: CurveAffine + SerdeObject {
    /// Reads an element from the buffer and parses it according to the `format`:
    /// - `Processed`: Reads a compressed curve element and decompress it
    /// - `RawBytes`: Reads an uncompressed curve element with coordinates in Montgomery form.
    ///   Checks that field elements are less than modulus, and then checks that the point is on the curve.
    /// - `RawBytesUnchecked`: Reads an uncompressed curve element with coordinates in Montgomery form;
    ///   does not perform any checks
    fn read<R: io::Read>(reader: &mut R, format: SerdeFormat) -> io::Result<Self> {
        match format {
            SerdeFormat::Processed => <Self as CurveRead>::read(reader),
            SerdeFormat::RawBytes => <Self as SerdeObject>::read_raw(reader),
            SerdeFormat::RawBytesUnchecked => Ok(<Self as SerdeObject>::read_raw_unchecked(reader)),
        }
    }
    /// Writes a curve element according to `format`:
    /// - `Processed`: Writes a compressed curve element
    /// - Otherwise: Writes an uncompressed curve element with coordinates in Montgomery form
    fn write<W: io::Write>(&self, writer: &mut W, format: SerdeFormat) -> io::Result<()> {
        match format {
            SerdeFormat::Processed => writer.write_all(self.to_bytes().as_ref()),
            _ => self.write_raw(writer),
        }
    }
}
impl<C: CurveAffine + SerdeObject> SerdeCurveAffine for C {}

pub trait SerdePrimeField: PrimeField + SerdeObject {
    /// Reads a field element as bytes from the buffer according to the `format`:
    /// - `Processed`: Reads a field element in standard form, with endianness specified by the
    ///   `PrimeField` implementation, and checks that the element is less than the modulus.
    /// - `RawBytes`: Reads a field element from raw bytes in its internal Montgomery representations,
    ///   and checks that the element is less than the modulus.
    /// - `RawBytesUnchecked`: Reads a field element in Montgomery form and performs no checks.
    fn read<R: io::Read>(reader: &mut R, format: SerdeFormat) -> io::Result<Self> {
        match format {
            SerdeFormat::Processed => {
                let mut compressed = Self::Repr::default();
                reader.read_exact(compressed.as_mut()).unwrap();
                Option::from(Self::from_repr(compressed))
                    .ok_or_else(|| io::Error::other("Invalid prime field point encoding"))
            }
            SerdeFormat::RawBytes => <Self as SerdeObject>::read_raw(reader),
            SerdeFormat::RawBytesUnchecked => Ok(<Self as SerdeObject>::read_raw_unchecked(reader)),
        }
    }

    /// Writes a field element as bytes to the buffer according to the `format`:
    /// - `Processed`: Writes a field element in standard form, with endianness specified by the
    ///   `PrimeField` implementation.
    /// - Otherwise: Writes a field element into raw bytes in its internal Montgomery representation,
    ///   WITHOUT performing the expensive Montgomery reduction.
    fn write<W: io::Write>(&self, writer: &mut W, format: SerdeFormat) -> io::Result<()> {
        match format {
            SerdeFormat::Processed => writer.write_all(self.to_repr().as_ref()),
            _ => self.write_raw(writer),
        }
    }
}
impl<F: PrimeField + SerdeObject> SerdePrimeField for F {}
// NOTE: the polynomial-slice serde helpers (`pack`/`unpack`/`read_polynomial_vec`/
// `write_polynomial_slice`/`polynomial_slice_byte_length`) plus the `PolyIo` trait
// were removed: pk/vk serialization is now the canonical halo2-axiom path
// (`GpuProvingKey::write` delegates to `inner`), so these gpu-side helpers had no
// remaining callers.
