//! CPU counterparts of operations defined in `crate::poly::domain`.
//!
//! Hosts the host-resident batch iFFT entry point (small-input fallback /
//! trait-dispatch target) plus the host-arm bodies that the production paths
//! invoke under the `vram-fallback` feature when device memory is tight.
//!
//! The `#[cfg(feature = "vram-fallback")]` gate stays at the dispatch site in
//! `poly/domain.rs`; the helpers in this file mirror the gate so they compile
//! only when the feature is enabled.

use ff::WithSmallOrderMulGroup;

use crate::plonk::GpuError;
use crate::poly::{Coeff, EvaluationDomain, LagrangeCoeff, Polynomial};

#[cfg(feature = "vram-fallback")]
use openvm_cuda_common::copy::MemCopyH2D;

#[cfg(feature = "vram-fallback")]
use crate::cuda::utils::HALO2_GPU_CTX;
#[cfg(feature = "vram-fallback")]
use crate::cuda::HaloGpuError;
#[cfg(feature = "vram-fallback")]
use crate::poly::{Device, DevicePolyExt, ExtendedLagrangeCoeff, Host, MaybeDevice};

#[allow(clippy::uninit_vec)]
pub(crate) fn lagrange_to_coeff_many_host<F: WithSmallOrderMulGroup<3>>(
    domain: &EvaluationDomain<'_, F>,
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
    domain: &EvaluationDomain<'_, F>,
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
    Ok(MaybeDevice::Host(domain.extended_from_lagrange_vec(host_values)))
}

/// VRAM-fallback host arm for `extended_from_lagrange_vec_device`: VRAM is
/// tight, so materialize the device parts and run the host gather.
#[cfg(feature = "vram-fallback")]
pub(crate) fn extended_from_lagrange_vec_vram_tight<F: WithSmallOrderMulGroup<3>>(
    domain: &EvaluationDomain<'_, F>,
    values: Vec<MaybeDevice<F, LagrangeCoeff>>,
    free_bytes: usize,
    extended_bytes: usize,
) -> Result<MaybeDevice<F, ExtendedLagrangeCoeff>, GpuError> {
    tracing::warn!(
        target: "halo2_vram_fallback",
        site = "extended_from_lagrange_vec_device.vram_tight",
        free_bytes,
        needed_bytes = extended_bytes,
        "VRAM fallback fired: insufficient VRAM for Device-output; materializing to Host"
    );
    let host_values: Vec<Polynomial<F, LagrangeCoeff, Host>> = values
        .into_iter()
        .map(|m| match m {
            MaybeDevice::Host(p) => p,
            MaybeDevice::Device(p) => p.materialize_host(),
        })
        .collect();
    Ok(MaybeDevice::Host(domain.extended_from_lagrange_vec(host_values)))
}

/// VRAM-fallback host arm for `lagrange_to_coeff_many_device_inputs`: D2H the
/// inputs, run the host batch iFFT, then H2D the outputs to keep the producer
/// site's contract of device-resident outputs.
#[cfg(feature = "vram-fallback")]
pub(crate) fn lagrange_to_coeff_many_device_inputs_host_arm<F: WithSmallOrderMulGroup<3>>(
    domain: &EvaluationDomain<'_, F>,
    in_many: &[Polynomial<F, LagrangeCoeff, Device>],
    free_bytes: usize,
    total_bytes: usize,
) -> Result<Vec<Polynomial<F, Coeff, Device>>, GpuError> {
    tracing::warn!(
        target: "halo2_vram_fallback",
        site = "lagrange_to_coeff_many_device_inputs.vram_tight",
        free_bytes,
        needed_bytes = total_bytes,
        batch_len = in_many.len(),
        "VRAM fallback fired: insufficient VRAM for device-input batch iFFT; D2H inputs → host iFFT → H2D outputs"
    );
    let host_ins: Vec<Polynomial<F, LagrangeCoeff>> = in_many.iter().map(|p| p.to_host()).collect();
    let host_outs: Vec<Polynomial<F, Coeff>> = domain.lagrange_to_coeff_many(&host_ins)?;
    let device_outs = host_outs
        .into_iter()
        .map(|p| {
            let d_buf = p
                .values()
                .to_device_on(&HALO2_GPU_CTX)
                .map_err(HaloGpuError::from)
                .map_err(GpuError::from)?;
            Ok::<Polynomial<F, Coeff, Device>, GpuError>(Polynomial::from_device(d_buf))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(device_outs)
}

/// VRAM-fallback host arm for `lagrange_to_coeff_device`: run the host iFFT
/// then H2D the result so the device-output contract holds.
#[cfg(feature = "vram-fallback")]
pub(crate) fn lagrange_to_coeff_device_host_arm<F: WithSmallOrderMulGroup<3>>(
    domain: &EvaluationDomain<'_, F>,
    a: Polynomial<F, LagrangeCoeff>,
    free_bytes: usize,
    n_bytes: usize,
) -> Result<Polynomial<F, Coeff, Device>, GpuError> {
    tracing::warn!(
        target: "halo2_vram_fallback",
        site = "lagrange_to_coeff_device.vram_tight",
        free_bytes,
        needed_bytes = n_bytes,
        "VRAM fallback fired: insufficient VRAM for Device-output iFFT; falling back to host arm + H2D"
    );
    let host_out = domain.lagrange_to_coeff(a)?;
    let d_buf = host_out
        .values()
        .to_device_on(&HALO2_GPU_CTX)
        .map_err(HaloGpuError::from)
        .map_err(GpuError::from)?;
    Ok(Polynomial::<F, Coeff, Device>::from_device(d_buf))
}

/// VRAM-fallback host arm for `lagrange_to_coeff_device_input`: D2H the input,
/// run host iFFT, then H2D the output.
#[cfg(feature = "vram-fallback")]
pub(crate) fn lagrange_to_coeff_device_input_host_arm<F: WithSmallOrderMulGroup<3>>(
    domain: &EvaluationDomain<'_, F>,
    a: Polynomial<F, LagrangeCoeff, Device>,
    free_bytes: usize,
    n_bytes: usize,
) -> Result<Polynomial<F, Coeff, Device>, GpuError> {
    tracing::warn!(
        target: "halo2_vram_fallback",
        site = "lagrange_to_coeff_device_input.vram_tight",
        free_bytes,
        needed_bytes = n_bytes,
        "VRAM fallback fired: insufficient VRAM for device-input iFFT; D2H input → host iFFT → H2D output"
    );
    let host_in = a.to_host();
    let host_out = domain.lagrange_to_coeff(host_in)?;
    let d_buf = host_out
        .values()
        .to_device_on(&HALO2_GPU_CTX)
        .map_err(HaloGpuError::from)
        .map_err(GpuError::from)?;
    Ok(Polynomial::<F, Coeff, Device>::from_device(d_buf))
}

/// VRAM-fallback host arm for `lagrange_to_coeff_many_device`: run the host
/// batch iFFT then H2D the outputs.
#[cfg(feature = "vram-fallback")]
pub(crate) fn lagrange_to_coeff_many_device_host_arm<F: WithSmallOrderMulGroup<3>>(
    domain: &EvaluationDomain<'_, F>,
    in_many: &[Polynomial<F, LagrangeCoeff>],
    free_bytes: usize,
    total_bytes: usize,
) -> Result<Vec<Polynomial<F, Coeff, Device>>, GpuError> {
    tracing::warn!(
        target: "halo2_vram_fallback",
        site = "lagrange_to_coeff_many_device.vram_tight",
        free_bytes,
        needed_bytes = total_bytes,
        batch_len = in_many.len(),
        "VRAM fallback fired: insufficient VRAM for Device-output batch iFFT; falling back to host arm + H2D"
    );
    let host_outs: Vec<Polynomial<F, Coeff>> = domain.lagrange_to_coeff_many(in_many)?;
    let device_outs = host_outs
        .into_iter()
        .map(|p| {
            let d_buf = p
                .values()
                .to_device_on(&HALO2_GPU_CTX)
                .map_err(HaloGpuError::from)
                .map_err(GpuError::from)?;
            Ok::<Polynomial<F, Coeff, Device>, GpuError>(Polynomial::from_device(d_buf))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(device_outs)
}
