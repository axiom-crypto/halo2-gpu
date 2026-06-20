use std::any::TypeId;
use std::mem;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use group::ff::Field;

/// A value assigned to a cell within a circuit.
///
/// Stored as a fraction, so the backend can use batch inversion.
///
/// A denominator of zero maps to an assigned value of zero.
///
/// `#[repr(C, u8)]` pins the in-memory layout so the GPU
/// `_halo2_decode_assigned` kernel can read raw `GpuAssigned<F>` bytes
/// directly off device without a host-side enum-decode pass. The layout
/// is: a `u8` discriminant at offset 0 (Zero=0, Trivial=1, Rational=2)
/// followed by a `#[repr(C)]` union of per-variant structs starting at
/// `align_of::<F>()`. The kernel reads `num` at offset `align_of::<F>()`
/// and `denom` at offset `align_of::<F>() + size_of::<F>()`.
/// `poly::batch_invert_assigned_device` runs a probe self-check
/// against this layout before launching the kernel.
#[repr(C, u8)]
#[derive(Clone, Copy, Debug)]
pub enum GpuAssigned<F> {
    /// The field element zero.
    Zero,
    /// A value that does not require inversion to evaluate.
    Trivial(F),
    /// A value stored as a fraction to enable batch inversion.
    Rational(F, F),
}

impl<F: Field> From<&GpuAssigned<F>> for GpuAssigned<F> {
    fn from(val: &GpuAssigned<F>) -> Self {
        *val
    }
}

/// Synthesis→device boundary conversion: the canonical `halo2-axiom`
/// `Assigned<F>` (produced by the canonical `Assignment` trait the GPU
/// `WitnessCollection`/keygen `Assembly` implement) is reinterpreted into the
/// GPU `#[repr(C, u8)]` `GpuAssigned<F>` just before the raw-byte device upload
/// (`batch_invert_assigned_device`). Canonical `Assigned` is *not* `repr(C,u8)`,
/// so it cannot feed the `_halo2_decode_assigned` kernel directly — this `From`
/// is the (host-side, per-cell) bridge. Variant-for-variant; no field math.
impl<F> From<halo2_axiom::plonk::Assigned<F>> for GpuAssigned<F> {
    fn from(val: halo2_axiom::plonk::Assigned<F>) -> Self {
        match val {
            halo2_axiom::plonk::Assigned::Zero => GpuAssigned::Zero,
            halo2_axiom::plonk::Assigned::Trivial(num) => GpuAssigned::Trivial(num),
            halo2_axiom::plonk::Assigned::Rational(num, denom) => {
                GpuAssigned::Rational(num, denom)
            }
        }
    }
}

impl<F: Field> From<&F> for GpuAssigned<F> {
    fn from(numerator: &F) -> Self {
        GpuAssigned::Trivial(*numerator)
    }
}

impl<F: Field> From<F> for GpuAssigned<F> {
    fn from(numerator: F) -> Self {
        GpuAssigned::Trivial(numerator)
    }
}

impl<F: Field> From<(F, F)> for GpuAssigned<F> {
    fn from((numerator, denominator): (F, F)) -> Self {
        GpuAssigned::Rational(numerator, denominator)
    }
}

impl<F> AsRef<GpuAssigned<F>> for GpuAssigned<F> {
    fn as_ref(&self) -> &GpuAssigned<F> {
        self
    }
}

impl<F: Field> PartialEq for GpuAssigned<F> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            // At least one side is directly zero.
            (Self::Zero, Self::Zero) => true,
            (Self::Zero, x) | (x, Self::Zero) => x.is_zero_vartime(),

            // One side is x/0 which maps to zero.
            (Self::Rational(_, denominator), x) | (x, Self::Rational(_, denominator))
                if denominator.is_zero_vartime() =>
            {
                x.is_zero_vartime()
            }

            // Okay, we need to do some actual math...
            (Self::Trivial(lhs), Self::Trivial(rhs)) => lhs == rhs,
            (Self::Trivial(x), Self::Rational(numerator, denominator))
            | (Self::Rational(numerator, denominator), Self::Trivial(x)) => {
                &(*x * denominator) == numerator
            }
            (
                Self::Rational(lhs_numerator, lhs_denominator),
                Self::Rational(rhs_numerator, rhs_denominator),
            ) => *lhs_numerator * rhs_denominator == *lhs_denominator * rhs_numerator,
        }
    }
}

impl<F: Field> Eq for GpuAssigned<F> {}

impl<F: Field> Neg for GpuAssigned<F> {
    type Output = GpuAssigned<F>;
    fn neg(self) -> Self::Output {
        match self {
            Self::Zero => Self::Zero,
            Self::Trivial(numerator) => Self::Trivial(-numerator),
            Self::Rational(numerator, denominator) => Self::Rational(-numerator, denominator),
        }
    }
}

impl<F: Field> Neg for &GpuAssigned<F> {
    type Output = GpuAssigned<F>;
    fn neg(self) -> Self::Output {
        -*self
    }
}

impl<F: Field> Add for GpuAssigned<F> {
    type Output = GpuAssigned<F>;
    fn add(self, rhs: GpuAssigned<F>) -> GpuAssigned<F> {
        match (self, rhs) {
            // One side is directly zero.
            (Self::Zero, _) => rhs,
            (_, Self::Zero) => self,

            // One side is x/0 which maps to zero.
            (Self::Rational(_, denominator), other) | (other, Self::Rational(_, denominator))
                if denominator.is_zero_vartime() =>
            {
                other
            }

            // Okay, we need to do some actual math...
            (Self::Trivial(lhs), Self::Trivial(rhs)) => Self::Trivial(lhs + rhs),
            (Self::Rational(numerator, denominator), Self::Trivial(other))
            | (Self::Trivial(other), Self::Rational(numerator, denominator)) => {
                Self::Rational(numerator + denominator * other, denominator)
            }
            (
                Self::Rational(lhs_numerator, lhs_denominator),
                Self::Rational(rhs_numerator, rhs_denominator),
            ) => Self::Rational(
                lhs_numerator * rhs_denominator + lhs_denominator * rhs_numerator,
                lhs_denominator * rhs_denominator,
            ),
        }
    }
}

impl<F: Field> Add<F> for GpuAssigned<F> {
    type Output = GpuAssigned<F>;
    fn add(self, rhs: F) -> GpuAssigned<F> {
        self + Self::Trivial(rhs)
    }
}

impl<F: Field> Add<F> for &GpuAssigned<F> {
    type Output = GpuAssigned<F>;
    fn add(self, rhs: F) -> GpuAssigned<F> {
        *self + rhs
    }
}

impl<F: Field> Add<&GpuAssigned<F>> for GpuAssigned<F> {
    type Output = GpuAssigned<F>;
    fn add(self, rhs: &Self) -> GpuAssigned<F> {
        self + *rhs
    }
}

impl<F: Field> Add<GpuAssigned<F>> for &GpuAssigned<F> {
    type Output = GpuAssigned<F>;
    fn add(self, rhs: GpuAssigned<F>) -> GpuAssigned<F> {
        *self + rhs
    }
}

impl<F: Field> Add<&GpuAssigned<F>> for &GpuAssigned<F> {
    type Output = GpuAssigned<F>;
    fn add(self, rhs: &GpuAssigned<F>) -> GpuAssigned<F> {
        *self + *rhs
    }
}

impl<F: Field> AddAssign for GpuAssigned<F> {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<F: Field> AddAssign<&GpuAssigned<F>> for GpuAssigned<F> {
    fn add_assign(&mut self, rhs: &Self) {
        *self = *self + rhs;
    }
}

impl<F: Field> Sub for GpuAssigned<F> {
    type Output = GpuAssigned<F>;
    fn sub(self, rhs: GpuAssigned<F>) -> GpuAssigned<F> {
        self + (-rhs)
    }
}

impl<F: Field> Sub<F> for GpuAssigned<F> {
    type Output = GpuAssigned<F>;
    fn sub(self, rhs: F) -> GpuAssigned<F> {
        self + (-rhs)
    }
}

impl<F: Field> Sub<F> for &GpuAssigned<F> {
    type Output = GpuAssigned<F>;
    fn sub(self, rhs: F) -> GpuAssigned<F> {
        *self - rhs
    }
}

impl<F: Field> Sub<&GpuAssigned<F>> for GpuAssigned<F> {
    type Output = GpuAssigned<F>;
    fn sub(self, rhs: &Self) -> GpuAssigned<F> {
        self - *rhs
    }
}

impl<F: Field> Sub<GpuAssigned<F>> for &GpuAssigned<F> {
    type Output = GpuAssigned<F>;
    fn sub(self, rhs: GpuAssigned<F>) -> GpuAssigned<F> {
        *self - rhs
    }
}

impl<F: Field> Sub<&GpuAssigned<F>> for &GpuAssigned<F> {
    type Output = GpuAssigned<F>;
    fn sub(self, rhs: &GpuAssigned<F>) -> GpuAssigned<F> {
        *self - *rhs
    }
}

impl<F: Field> SubAssign for GpuAssigned<F> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<F: Field> SubAssign<&GpuAssigned<F>> for GpuAssigned<F> {
    fn sub_assign(&mut self, rhs: &Self) {
        *self = *self - rhs;
    }
}

impl<F: Field> Mul for GpuAssigned<F> {
    type Output = GpuAssigned<F>;
    fn mul(self, rhs: GpuAssigned<F>) -> GpuAssigned<F> {
        match (self, rhs) {
            (Self::Zero, _) | (_, Self::Zero) => Self::Zero,
            (Self::Trivial(lhs), Self::Trivial(rhs)) => Self::Trivial(lhs * rhs),
            (Self::Rational(numerator, denominator), Self::Trivial(other))
            | (Self::Trivial(other), Self::Rational(numerator, denominator)) => {
                Self::Rational(numerator * other, denominator)
            }
            (
                Self::Rational(lhs_numerator, lhs_denominator),
                Self::Rational(rhs_numerator, rhs_denominator),
            ) => Self::Rational(
                lhs_numerator * rhs_numerator,
                lhs_denominator * rhs_denominator,
            ),
        }
    }
}

impl<F: Field> Mul<F> for GpuAssigned<F> {
    type Output = GpuAssigned<F>;
    fn mul(self, rhs: F) -> GpuAssigned<F> {
        self * Self::Trivial(rhs)
    }
}

impl<F: Field> Mul<F> for &GpuAssigned<F> {
    type Output = GpuAssigned<F>;
    fn mul(self, rhs: F) -> GpuAssigned<F> {
        *self * rhs
    }
}

impl<F: Field> Mul<&GpuAssigned<F>> for GpuAssigned<F> {
    type Output = GpuAssigned<F>;
    fn mul(self, rhs: &GpuAssigned<F>) -> GpuAssigned<F> {
        self * *rhs
    }
}

impl<F: Field> MulAssign for GpuAssigned<F> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<F: Field> MulAssign<&GpuAssigned<F>> for GpuAssigned<F> {
    fn mul_assign(&mut self, rhs: &Self) {
        *self = *self * rhs;
    }
}

impl<F: Field> GpuAssigned<F> {
    /// Returns the numerator.
    pub fn numerator(&self) -> F {
        match self {
            Self::Zero => F::ZERO,
            Self::Trivial(x) => *x,
            Self::Rational(numerator, _) => *numerator,
        }
    }

    /// Returns the denominator, if non-trivial.
    pub fn denominator(&self) -> Option<F> {
        match self {
            Self::Zero => None,
            Self::Trivial(_) => None,
            Self::Rational(_, denominator) => Some(*denominator),
        }
    }

    /// Returns true iff this element is zero.
    pub fn is_zero_vartime(&self) -> bool {
        match self {
            Self::Zero => true,
            Self::Trivial(x) => x.is_zero_vartime(),
            // GpuAssigned maps x/0 -> 0.
            Self::Rational(numerator, denominator) => {
                numerator.is_zero_vartime() || denominator.is_zero_vartime()
            }
        }
    }

    /// Doubles this element.
    #[must_use]
    pub fn double(&self) -> Self {
        match self {
            Self::Zero => Self::Zero,
            Self::Trivial(x) => Self::Trivial(x.double()),
            Self::Rational(numerator, denominator) => {
                Self::Rational(numerator.double(), *denominator)
            }
        }
    }

    /// Squares this element.
    #[must_use]
    pub fn square(&self) -> Self {
        match self {
            Self::Zero => Self::Zero,
            Self::Trivial(x) => Self::Trivial(x.square()),
            Self::Rational(numerator, denominator) => {
                Self::Rational(numerator.square(), denominator.square())
            }
        }
    }

    /// Cubes this element.
    #[must_use]
    pub fn cube(&self) -> Self {
        self.square() * self
    }

    /// Inverts this assigned value (taking the inverse of zero to be zero).
    pub fn invert(&self) -> Self {
        match self {
            Self::Zero => Self::Zero,
            Self::Trivial(x) => Self::Rational(F::ONE, *x),
            Self::Rational(numerator, denominator) => Self::Rational(*denominator, *numerator),
        }
    }

    /// Evaluates this assigned value directly, performing an unbatched inversion if
    /// necessary.
    ///
    /// If the denominator is zero, this returns zero.
    pub fn evaluate(self) -> F {
        match self {
            Self::Zero => F::ZERO,
            Self::Trivial(x) => x,
            Self::Rational(numerator, denominator) => {
                if denominator == F::ONE {
                    numerator
                } else {
                    numerator * denominator.invert().unwrap_or(F::ZERO)
                }
            }
        }
    }
}

/// Byte offsets and per-element stride of `GpuAssigned<F>` under the
/// `#[repr(C, u8)]` layout pinned on the enum in this module.
///
/// Layout: tag `u8` at offset 0, then a `#[repr(C)]` union of variant
/// structs starting at `align_of::<F>()`. So the first `F` payload
/// (`Trivial`'s value or `Rational`'s numerator) lives at offset
/// `align_of::<F>()`, and `Rational`'s denominator at
/// `align_of::<F>() + size_of::<F>()`.
///
/// Returns `(stride_bytes, num_offset, denom_offset)`.
pub(crate) const fn assigned_layout_offsets<F: Field>() -> (u32, u32, u32) {
    let num_offset = mem::align_of::<F>() as u32;
    let denom_offset = num_offset + mem::size_of::<F>() as u32;
    let stride_bytes = mem::size_of::<GpuAssigned<F>>() as u32;
    (stride_bytes, num_offset, denom_offset)
}

/// Self-check on the `GpuAssigned<F>` byte layout the kernel reads.
///
/// `#[repr(C, u8)]` pins the layout per the Rust reference, but a
/// compiler regression or a generic `F` with unusual alignment could
/// shift the payload offsets. Probe values are constructed and inspected as raw
/// bytes to confirm: the discriminant byte (0/1/2 for Zero/Trivial/Rational),
/// the size matches `assigned_layout_offsets`, and the F payload sits
/// at the computed offsets. Mismatch panics with a clear message before
/// any kernel launch.
pub(crate) fn verify_assigned_layout<F: Field>() {
    let (stride, num_off, denom_off) = assigned_layout_offsets::<F>();
    let actual_stride = mem::size_of::<GpuAssigned<F>>() as u32;
    assert_eq!(
        stride, actual_stride,
        "GpuAssigned<F> stride mismatch: expected {stride}, got {actual_stride}"
    );

    // Construct distinguishable F probes from Field-only ops (no
    // `From<u64>` bound). `denom` is `num.double()`, which differs from
    // `num` for every non-trivial F we care about (char != 2).
    let num = F::ONE + F::ONE;
    let denom = num.double();
    let rational = GpuAssigned::<F>::Rational(num, denom);
    let bytes = unsafe {
        std::slice::from_raw_parts(
            &rational as *const GpuAssigned<F> as *const u8,
            stride as usize,
        )
    };
    assert_eq!(
        bytes[0], 2,
        "GpuAssigned::Rational discriminant expected 2, got {}",
        bytes[0]
    );
    let probe_num =
        unsafe { std::ptr::read_unaligned(bytes.as_ptr().add(num_off as usize) as *const F) };
    let probe_denom =
        unsafe { std::ptr::read_unaligned(bytes.as_ptr().add(denom_off as usize) as *const F) };
    assert!(
        probe_num == num,
        "GpuAssigned<F> numerator offset mismatch at byte {num_off}"
    );
    assert!(
        probe_denom == denom,
        "GpuAssigned<F> denominator offset mismatch at byte {denom_off}"
    );

    let trivial = GpuAssigned::<F>::Trivial(num);
    let tb = unsafe {
        std::slice::from_raw_parts(&trivial as *const GpuAssigned<F> as *const u8, stride as usize)
    };
    assert_eq!(
        tb[0], 1,
        "GpuAssigned::Trivial discriminant expected 1, got {}",
        tb[0]
    );
    let probe_trivial =
        unsafe { std::ptr::read_unaligned(tb.as_ptr().add(num_off as usize) as *const F) };
    assert!(
        probe_trivial == num,
        "GpuAssigned<F> Trivial payload offset mismatch at byte {num_off}"
    );

    let zero = GpuAssigned::<F>::Zero;
    let zb = unsafe {
        std::slice::from_raw_parts(&zero as *const GpuAssigned<F> as *const u8, stride as usize)
    };
    assert_eq!(
        zb[0], 0,
        "GpuAssigned::Zero discriminant expected 0, got {}",
        zb[0]
    );
}

/// Hard guard: the CUDA decoder is hardwired to `fr_t` (bn256 Fr) in
/// `halo2_proofs/cuda/include/kernel/decode_assigned.h`. Any non-Fr `F`
/// would land bn256 Fr bytes into a `DeviceBuffer<F>` and corrupt
/// downstream device arithmetic. Field-type compatibility cannot be
/// verified by `verify_assigned_layout::<F>()` (which only probes enum
/// byte layout), so it is asserted explicitly here before the FFI
/// launch.
pub(crate) fn assert_assigned_kernel_field_is_bn256_fr<F: Field>() {
    assert!(
        TypeId::of::<F>() == TypeId::of::<halo2curves::bn256::Fr>(),
        "decode_assigned GPU path is hardwired to bn256::Fr in \
         `cuda/include/kernel/decode_assigned.h`; non-Fr F={} would \
         decode Fr bytes into the wrong field representation",
        std::any::type_name::<F>(),
    );
}

#[cfg(test)]
mod tests {
    use halo2curves::pasta::Fp;

    use super::GpuAssigned;
    // We use (numerator, denominator) in the comments below to denote a rational.
    #[test]
    fn add_trivial_to_inv0_rational() {
        // a = 2
        // b = (1,0)
        let a = GpuAssigned::Trivial(Fp::from(2));
        let b = GpuAssigned::Rational(Fp::one(), Fp::zero());

        // 2 + (1,0) = 2 + 0 = 2
        // This fails if addition is implemented using normal rules for rationals.
        assert_eq!((a + b).evaluate(), a.evaluate());
        assert_eq!((b + a).evaluate(), a.evaluate());
    }

    #[test]
    fn add_rational_to_inv0_rational() {
        // a = (1,2)
        // b = (1,0)
        let a = GpuAssigned::Rational(Fp::one(), Fp::from(2));
        let b = GpuAssigned::Rational(Fp::one(), Fp::zero());

        // (1,2) + (1,0) = (1,2) + 0 = (1,2)
        // This fails if addition is implemented using normal rules for rationals.
        assert_eq!((a + b).evaluate(), a.evaluate());
        assert_eq!((b + a).evaluate(), a.evaluate());
    }

    #[test]
    fn sub_trivial_from_inv0_rational() {
        // a = 2
        // b = (1,0)
        let a = GpuAssigned::Trivial(Fp::from(2));
        let b = GpuAssigned::Rational(Fp::one(), Fp::zero());

        // (1,0) - 2 = 0 - 2 = -2
        // This fails if subtraction is implemented using normal rules for rationals.
        assert_eq!((b - a).evaluate(), (-a).evaluate());

        // 2 - (1,0) = 2 - 0 = 2
        assert_eq!((a - b).evaluate(), a.evaluate());
    }

    #[test]
    fn sub_rational_from_inv0_rational() {
        // a = (1,2)
        // b = (1,0)
        let a = GpuAssigned::Rational(Fp::one(), Fp::from(2));
        let b = GpuAssigned::Rational(Fp::one(), Fp::zero());

        // (1,0) - (1,2) = 0 - (1,2) = -(1,2)
        // This fails if subtraction is implemented using normal rules for rationals.
        assert_eq!((b - a).evaluate(), (-a).evaluate());

        // (1,2) - (1,0) = (1,2) - 0 = (1,2)
        assert_eq!((a - b).evaluate(), a.evaluate());
    }

    #[test]
    fn mul_rational_by_inv0_rational() {
        // a = (1,2)
        // b = (1,0)
        let a = GpuAssigned::Rational(Fp::one(), Fp::from(2));
        let b = GpuAssigned::Rational(Fp::one(), Fp::zero());

        // (1,2) * (1,0) = (1,2) * 0 = 0
        assert_eq!((a * b).evaluate(), Fp::zero());

        // (1,0) * (1,2) = 0 * (1,2) = 0
        assert_eq!((b * a).evaluate(), Fp::zero());
    }
}

#[cfg(test)]
mod proptests {
    use std::{
        cmp,
        ops::{Add, Mul, Neg, Sub},
    };

    use group::ff::Field;
    use halo2curves::pasta::Fp;
    use proptest::{collection::vec, prelude::*, sample::select};

    use super::GpuAssigned;

    trait UnaryOperand: Neg<Output = Self> {
        fn double(&self) -> Self;
        fn square(&self) -> Self;
        fn cube(&self) -> Self;
        fn inv0(&self) -> Self;
    }

    impl<F: Field> UnaryOperand for F {
        fn double(&self) -> Self {
            self.double()
        }

        fn square(&self) -> Self {
            self.square()
        }

        fn cube(&self) -> Self {
            self.cube()
        }

        fn inv0(&self) -> Self {
            self.invert().unwrap_or(F::ZERO)
        }
    }

    impl<F: Field> UnaryOperand for GpuAssigned<F> {
        fn double(&self) -> Self {
            self.double()
        }

        fn square(&self) -> Self {
            self.square()
        }

        fn cube(&self) -> Self {
            self.cube()
        }

        fn inv0(&self) -> Self {
            self.invert()
        }
    }

    #[derive(Clone, Debug)]
    enum UnaryOperator {
        Neg,
        Double,
        Square,
        Cube,
        Inv0,
    }

    const UNARY_OPERATORS: &[UnaryOperator] = &[
        UnaryOperator::Neg,
        UnaryOperator::Double,
        UnaryOperator::Square,
        UnaryOperator::Cube,
        UnaryOperator::Inv0,
    ];

    impl UnaryOperator {
        fn apply<F: UnaryOperand>(&self, a: F) -> F {
            match self {
                Self::Neg => -a,
                Self::Double => a.double(),
                Self::Square => a.square(),
                Self::Cube => a.cube(),
                Self::Inv0 => a.inv0(),
            }
        }
    }

    trait BinaryOperand: Sized + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> {}
    impl<F: Field> BinaryOperand for F {}
    impl<F: Field> BinaryOperand for GpuAssigned<F> {}

    #[derive(Clone, Debug)]
    enum BinaryOperator {
        Add,
        Sub,
        Mul,
    }

    const BINARY_OPERATORS: &[BinaryOperator] = &[
        BinaryOperator::Add,
        BinaryOperator::Sub,
        BinaryOperator::Mul,
    ];

    impl BinaryOperator {
        fn apply<F: BinaryOperand>(&self, a: F, b: F) -> F {
            match self {
                Self::Add => a + b,
                Self::Sub => a - b,
                Self::Mul => a * b,
            }
        }
    }

    #[derive(Clone, Debug)]
    enum Operator {
        Unary(UnaryOperator),
        Binary(BinaryOperator),
    }

    prop_compose! {
        /// Use narrow that can be easily reduced.
        fn arb_element()(val in any::<u64>()) -> Fp {
            Fp::from(val)
        }
    }

    prop_compose! {
        fn arb_trivial()(element in arb_element()) -> GpuAssigned<Fp> {
            GpuAssigned::Trivial(element)
        }
    }

    prop_compose! {
        /// Generates half of the denominators as zero to represent a deferred inversion.
        fn arb_rational()(
            numerator in arb_element(),
            denominator in prop_oneof![
                1 => Just(Fp::zero()),
                2 => arb_element(),
            ],
        ) -> GpuAssigned<Fp> {
            GpuAssigned::Rational(numerator, denominator)
        }
    }

    prop_compose! {
        fn arb_operators(num_unary: usize, num_binary: usize)(
            unary in vec(select(UNARY_OPERATORS), num_unary),
            binary in vec(select(BINARY_OPERATORS), num_binary),
        ) -> Vec<Operator> {
            unary.into_iter()
                .map(Operator::Unary)
                .chain(binary.into_iter().map(Operator::Binary))
                .collect()
        }
    }

    prop_compose! {
        fn arb_testcase()(
            num_unary in 0usize..5,
            num_binary in 0usize..5,
        )(
            values in vec(
                prop_oneof![
                    1 => Just(GpuAssigned::Zero),
                    2 => arb_trivial(),
                    2 => arb_rational(),
                ],
                // Ensure that:
                // - we have at least one value to apply unary operators to.
                // - we can apply every binary operator pairwise sequentially.
                cmp::max(usize::from(num_unary > 0), num_binary + 1)),
            operations in arb_operators(num_unary, num_binary).prop_shuffle(),
        ) -> (Vec<GpuAssigned<Fp>>, Vec<Operator>) {
            (values, operations)
        }
    }

    proptest! {
        #[test]
        fn operation_commutativity((values, operations) in arb_testcase()) {
            // Evaluate the values at the start.
            let elements: Vec<_> = values.iter().cloned().map(|v| v.evaluate()).collect();

            // Apply the operations to both the deferred and evaluated values.
            fn evaluate<F: UnaryOperand + BinaryOperand>(
                items: Vec<F>,
                operators: &[Operator],
            ) -> F {
                let mut ops = operators.iter();

                // Process all binary operators. We are guaranteed to have exactly as many
                // binary operators as we need calls to the reduction closure.
                let mut res = items.into_iter().reduce(|mut a, b| loop {
                    match ops.next() {
                        Some(Operator::Unary(op)) => a = op.apply(a),
                        Some(Operator::Binary(op)) => break op.apply(a, b),
                        None => unreachable!(),
                    }
                }).unwrap();

                // Process any unary operators that weren't handled in the reduce() call
                // above (either if we only had one item, or there were unary operators
                // after the last binary operator). We are guaranteed to have no binary
                // operators remaining at this point.
                loop {
                    match ops.next() {
                        Some(Operator::Unary(op)) => res = op.apply(res),
                        Some(Operator::Binary(_)) => unreachable!(),
                        None => break res,
                    }
                }
            }
            let deferred_result = evaluate(values, &operations);
            let evaluated_result = evaluate(elements, &operations);

            // The two should be equal, i.e. deferred inversion should commute with the
            // list of operations.
            assert_eq!(deferred_result.evaluate(), evaluated_result);
        }
    }
}
