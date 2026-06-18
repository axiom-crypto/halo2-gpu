//! CPU counterparts of operations defined in `crate::poly`.

use group::ff::{BatchInvert, Field};
use itertools::Itertools;
use rayon::prelude::*;

use crate::plonk::Assigned;
use crate::poly::{LagrangeCoeff, Polynomial};

#[cfg(test)]
use crate::cuda::funcs::batch_invert_gpu;
#[cfg(test)]
use crate::cuda::HaloGpuError;

pub(crate) mod domain;

pub(crate) fn batch_invert_assigned<F: Field, PR>(
    assigned: impl AsRef<[PR]>,
) -> Vec<Polynomial<F, LagrangeCoeff>>
where
    PR: AsRef<[Assigned<F>]> + Send + Sync,
{
    batch_invert_assigned_par(assigned)
}

pub(crate) fn batch_invert_assigned_par<F: Field, PR>(
    assigned: impl AsRef<[PR]>,
) -> Vec<Polynomial<F, LagrangeCoeff>>
where
    PR: AsRef<[Assigned<F>]> + Send + Sync,
{
    let assigned = assigned.as_ref();
    if assigned.is_empty() {
        return vec![];
    }
    let n = assigned[0].as_ref().len();
    // 1d vector better for memory allocation
    let mut assigned_denominators: Vec<Option<_>> = assigned
        .par_iter()
        .flat_map(|f| f.as_ref().par_iter().map(|value| value.denominator()))
        .collect();

    let mut_denominators = assigned_denominators
        .par_iter_mut()
        // If the denominator is trivial, we can skip it, reducing the
        // size of the batch inversion.
        .filter_map(|d| d.as_mut())
        .collect::<Vec<_>>();

    if !mut_denominators.is_empty() {
        let num_threads = rayon::current_num_threads();
        let chunk_size = mut_denominators.len().div_ceil(num_threads);
        rayon::scope(|scope| {
            for chunk in mut_denominators.into_iter().chunks(chunk_size).into_iter() {
                let chunk = chunk.collect_vec();
                scope.spawn(move |_| {
                    chunk.batch_invert();
                });
            }
        });
    }

    assigned
        .par_iter()
        .zip(assigned_denominators.par_chunks(n))
        .map(|(poly, inv_denoms)| {
            let poly = poly.as_ref();
            debug_assert_eq!(inv_denoms.len(), poly.len());
            let values: Vec<F> = poly
                .par_iter()
                .zip(inv_denoms.par_iter())
                .map(|(a, inv_den)| {
                    if let Some(inv_den) = inv_den {
                        a.numerator() * inv_den
                    } else {
                        a.numerator()
                    }
                })
                .collect();
            Polynomial::new(values)
        })
        .collect()
}

// currently, the host overhead for processing Assigned<F> is huge
// the e2e time of batch_invert_gpu() is a very small percentage of the total time
// benchmarking found it's better to just not process the slice of `Assigned<F>` and just use gpu to invert it ALL
//
// Host-output sibling retained under `#[cfg(test)]` only: production callers
// use the device-output sibling `batch_invert_assigned_device`. The
// equivalence tests pair the two siblings.
#[cfg(test)]
pub(crate) fn batch_invert_assigned_gpu<F: Field, PR>(
    assigned: impl AsRef<[PR]>,
) -> Result<Vec<Polynomial<F, LagrangeCoeff>>, HaloGpuError>
where
    PR: AsRef<[Assigned<F>]> + Send + Sync,
{
    #[cfg(feature = "profile")]
    let time = std::time::Instant::now();
    let assigned = assigned.as_ref();
    if assigned.is_empty() {
        return Ok(vec![]);
    }
    let n = assigned[0].as_ref().len();
    // 1d vector better for memory allocation
    let mut assigned_denominators: Vec<_> = assigned
        .par_iter()
        .flat_map(|f| {
            f.as_ref()
                .par_iter()
                .map(|value| value.denominator().unwrap_or(F::ONE))
        })
        .collect();

    #[cfg(feature = "profile")]
    let gpu_time = std::time::Instant::now();
    batch_invert_gpu(&mut assigned_denominators)?;
    #[cfg(feature = "profile")]
    println!("    gpu_time = {:?}", gpu_time.elapsed());
    #[allow(clippy::let_and_return)]
    let res = assigned
        .par_iter()
        .zip(assigned_denominators.par_chunks_exact(n))
        .map(|(poly, inv_denoms)| {
            let poly = poly.as_ref();
            debug_assert_eq!(inv_denoms.len(), poly.len());
            let values: Vec<F> = poly
                .par_iter()
                .zip(inv_denoms.par_iter())
                .map(|(a, inv_den)| {
                    if a.denominator().is_some() {
                        a.numerator() * inv_den
                    } else {
                        a.numerator()
                    }
                })
                .collect();
            Polynomial::new(values)
        })
        .collect();
    #[cfg(feature = "profile")]
    println!(" . batch_invert_assigned_gpu_time = {:?}", time.elapsed());
    Ok(res)
}
