//! CPU counterparts of operations defined in `crate::poly::domain`.
//!
//! Hosts the host-resident batch iFFT entry point (small-input fallback /
//! trait-dispatch target) plus the host-arm body that the production path
//! invokes under the `vram-fallback` feature when a gather input is not fully
//! device-resident.
//!
//! The `#[cfg(feature = "vram-fallback")]` gate stays at the dispatch site in
//! `poly/domain.rs`; the helper in this file mirrors the gate so it compiles
//! only when the feature is enabled.

use ff::WithSmallOrderMulGroup;

use crate::plonk::GpuError;
use crate::poly::{Coeff, EvaluationDomain, LagrangeCoeff, Polynomial};

#[cfg(feature = "vram-fallback")]
use crate::poly::{DevicePolyExt, ExtendedLagrangeCoeff, Host, MaybeDevice};

#[allow(clippy::uninit_vec)]
pub(crate) fn lagrange_to_coeff_many_host<F: WithSmallOrderMulGroup<3>>(
    domain: &EvaluationDomain<F>,
    in_many: &[Polynomial<F, LagrangeCoeff>],
) -> Result<Vec<Polynomial<F, Coeff>>, GpuError> {
    crate::perf_section!("lagrange_to_coeff_many");
    log::info!("using lagrange_to_coeff_many: vec_num[{}]", in_many.len());
    if in_many.is_empty() {
        return Ok(vec![]);
    }
    let mut out_many: Vec<Polynomial<F, Coeff>> = Vec::with_capacity(in_many.len());
    for _ in 0..in_many.len() {
        let mut out: Vec<F> = Vec::with_capacity(domain.extended_len());
        unsafe {
            out.set_len(1 << domain.k());
        }
        out_many.push(Polynomial::new(out));
    }

    domain.ifft_many(in_many, &mut out_many)?;
    Ok(out_many)
}

/// VRAM-fallback host arm for `extended_from_lagrange_vec_device`: at least one
/// part is host-resident, so materialize the remaining device parts and route
/// through the host path.
#[cfg(feature = "vram-fallback")]
pub(crate) fn extended_from_lagrange_vec_not_all_device<F: WithSmallOrderMulGroup<3>>(
    domain: &EvaluationDomain<F>,
    values: Vec<MaybeDevice<F, LagrangeCoeff>>,
) -> Result<MaybeDevice<F, ExtendedLagrangeCoeff>, GpuError> {
    tracing::warn!(
        target: "halo2_vram_fallback",
        site = "extended_from_lagrange_vec_device.not_all_device",
        "VRAM fallback fired: not all input parts are Device-resident; materializing to Host"
    );
    let host_values: Vec<Polynomial<F, LagrangeCoeff, Host>> = values
        .into_iter()
        .map(|m| match m {
            MaybeDevice::Host(p) => p,
            MaybeDevice::Device(p) => p.materialize_host(),
        })
        .collect();
    Ok(MaybeDevice::Host(
        domain.extended_from_lagrange_vec(host_values),
    ))
}
