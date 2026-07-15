use std::fmt::Debug;

use super::commitment::MSM;
use crate::{
    arithmetic::eval_polynomial,
    cuda::funcs::{batch_eval_polynomial_d2h, eval_polynomial_device},
    poly::{Coeff, Device, DevicePolyExt, Host, Polynomial},
};
use halo2curves::CurveAffine;

pub trait Query<F>: Sized + Clone + Send + Sync {
    type Commitment: PartialEq + Copy + Send + Sync;
    type Eval: Clone + Default + Debug;

    fn get_point(&self) -> F;
    fn get_eval(&self) -> Self::Eval;
    fn get_commitment(&self) -> Self::Commitment;

    /// Evaluate every query, returning evals in query order. Default: per-query
    /// `get_eval`; device backends may override to batch the evals.
    fn batch_get_evals(queries: &[Self]) -> Vec<Self::Eval> {
        queries.iter().map(|q| q.get_eval()).collect()
    }
}

/// A residency-tagged borrow of a `Polynomial<F, Coeff, _>`.
///
/// Used by [`ProverQuery`] / [`PolynomialPointer`] to carry either a
/// host-resident or device-resident polynomial through the multiopen
/// pipeline. The variant is part of the type, so each consumer must
/// handle host and device residency explicitly at the call site.
#[derive(Debug)]
pub enum PolyRef<'a, F> {
    Host(&'a Polynomial<F, Coeff, Host>),
    Device(&'a Polynomial<F, Coeff, Device>),
}

impl<'a, F> Clone for PolyRef<'a, F> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, F> Copy for PolyRef<'a, F> {}

impl<'a, F> PolyRef<'a, F> {
    pub fn len(&self) -> usize {
        match self {
            PolyRef::Host(p) => p.len(),
            PolyRef::Device(p) => p.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<'a, F> From<&'a Polynomial<F, Coeff, Host>> for PolyRef<'a, F> {
    fn from(p: &'a Polynomial<F, Coeff, Host>) -> Self {
        PolyRef::Host(p)
    }
}

impl<'a, F> From<&'a Polynomial<F, Coeff, Device>> for PolyRef<'a, F> {
    fn from(p: &'a Polynomial<F, Coeff, Device>) -> Self {
        PolyRef::Device(p)
    }
}

/// A polynomial query at a point
#[derive(Debug, Clone)]
pub struct ProverQuery<'com, C: CurveAffine> {
    /// point at which polynomial is queried
    pub(crate) point: C::Scalar,
    /// coefficients of polynomial (residency-tagged borrow)
    pub(crate) poly: PolyRef<'com, C::Scalar>,
}

#[doc(hidden)]
#[derive(Copy, Clone)]
pub struct PolynomialPointer<'com, C: CurveAffine> {
    pub(crate) poly: PolyRef<'com, C::Scalar>,
}

impl<'com, C: CurveAffine> PartialEq for PolynomialPointer<'com, C> {
    fn eq(&self, other: &Self) -> bool {
        match (&self.poly, &other.poly) {
            (PolyRef::Host(a), PolyRef::Host(b)) => std::ptr::eq(*a, *b),
            (PolyRef::Device(a), PolyRef::Device(b)) => std::ptr::eq(*a, *b),
            _ => false,
        }
    }
}

impl<'com, C: CurveAffine> Query<C::Scalar> for ProverQuery<'com, C> {
    type Commitment = PolynomialPointer<'com, C>;
    type Eval = C::Scalar;

    fn get_point(&self) -> C::Scalar {
        self.point
    }
    fn get_eval(&self) -> Self::Eval {
        match self.poly {
            PolyRef::Host(p) => eval_polynomial(p.values(), self.get_point()),
            PolyRef::Device(p) => eval_polynomial_device(p.device_buf(), self.get_point())
                .expect("eval_polynomial_device failed in ProverQuery::get_eval"),
        }
    }
    fn get_commitment(&self) -> Self::Commitment {
        PolynomialPointer { poly: self.poly }
    }

    /// Device-resident polys are evaluated in one batched pass; host polys use
    /// the per-query CPU eval. Results are in query order.
    fn batch_get_evals(queries: &[Self]) -> Vec<Self::Eval> {
        let mut evals = vec![C::Scalar::default(); queries.len()];
        let mut d_polys = Vec::new();
        let mut d_points = Vec::new();
        let mut d_slots = Vec::new();
        for (i, q) in queries.iter().enumerate() {
            match q.poly {
                PolyRef::Device(p) => {
                    d_polys.push(p.device_buf());
                    d_points.push(q.point);
                    d_slots.push(i);
                }
                PolyRef::Host(p) => {
                    evals[i] = eval_polynomial(p.values(), q.point);
                }
            }
        }
        if !d_polys.is_empty() {
            let mut d_evals = vec![C::Scalar::default(); d_polys.len()];
            batch_eval_polynomial_d2h(&d_polys, &d_points, &mut d_evals)
                .expect("batch_eval_polynomial_d2h failed in ProverQuery::batch_get_evals");
            for (k, &slot) in d_slots.iter().enumerate() {
                evals[slot] = d_evals[k];
            }
        }
        evals
    }
}

impl<'com, C: CurveAffine, M: MSM<C>> VerifierQuery<'com, C, M> {
    /// Create a new verifier query based on a commitment
    pub fn new_commitment(commitment: &'com C, point: C::Scalar, eval: C::Scalar) -> Self {
        VerifierQuery {
            point,
            eval,
            commitment: CommitmentReference::Commitment(commitment),
        }
    }

    /// Create a new verifier query based on a linear combination of commitments
    pub fn new_msm(msm: &'com M, point: C::Scalar, eval: C::Scalar) -> VerifierQuery<'com, C, M> {
        VerifierQuery {
            point,
            eval,
            commitment: CommitmentReference::MSM(msm),
        }
    }
}

/// A polynomial query at a point
#[derive(Debug)]
pub struct VerifierQuery<'com, C: CurveAffine, M: MSM<C>> {
    /// point at which polynomial is queried
    pub(crate) point: C::Scalar,
    /// commitment to polynomial
    pub(crate) commitment: CommitmentReference<'com, C, M>,
    /// evaluation of polynomial at query point
    pub(crate) eval: C::Scalar,
}

impl<'com, C: CurveAffine, M: MSM<C>> Clone for VerifierQuery<'com, C, M> {
    fn clone(&self) -> Self {
        Self {
            point: self.point,
            commitment: self.commitment,
            eval: self.eval,
        }
    }
}

#[allow(clippy::upper_case_acronyms)]
#[derive(Clone, Debug)]
pub enum CommitmentReference<'r, C: CurveAffine, M: MSM<C>> {
    Commitment(&'r C),
    MSM(&'r M),
}

impl<'r, C: CurveAffine, M: MSM<C>> Copy for CommitmentReference<'r, C, M> {}

impl<'r, C: CurveAffine, M: MSM<C>> PartialEq for CommitmentReference<'r, C, M> {
    #![allow(ambiguous_wide_pointer_comparisons)]
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (&CommitmentReference::Commitment(a), &CommitmentReference::Commitment(b)) => {
                std::ptr::eq(a, b)
            }
            (&CommitmentReference::MSM(a), &CommitmentReference::MSM(b)) => std::ptr::eq(a, b),
            _ => false,
        }
    }
}

impl<'com, C: CurveAffine, M: MSM<C>> Query<C::Scalar> for VerifierQuery<'com, C, M> {
    type Eval = C::Scalar;
    type Commitment = CommitmentReference<'com, C, M>;

    fn get_point(&self) -> C::Scalar {
        self.point
    }
    fn get_eval(&self) -> C::Scalar {
        self.eval
    }
    fn get_commitment(&self) -> Self::Commitment {
        self.commitment
    }
}
