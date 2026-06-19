//! Contains utilities for performing polynomial arithmetic over an evaluation
//! domain that is of a suitable size for the application.
use std::collections::HashMap;

use std::ops::{Deref, DerefMut};
use std::panic;

use crate::cuda::culib::_halo2_fft_normal_check_memory;
use crate::cuda::funcs::{
    cosetfft_gpu, cosetfft_gpu_many, cosetfft_many_device, cosetfft_many_h2d,
    distribute_powers_zeta_device, divide_by_vanishing_poly_device,
    extended_from_lagrange_vec_device, fft_gpu, fft_gpu_many, fft_normal_device, ifft_gpu_many,
    ifft_many_device, ifft_many_h2d, split_radix_fft_gpu, split_radix_fft_inout_gpu,
};
use crate::cuda::modules::ifft_cosetfftpart_gpu;
use crate::cuda::utils::query_device_free_bytes_for_chunking;
use crate::cuda::utils::{FFITraitObject, HALO2_GPU_CTX};
use crate::cuda::HaloGpuError;
use crate::{
    cpu::arithmetic::parallelize,
    fft::recursive::FFTData,
    plonk::{Assigned, Error},
};
use openvm_cuda_common::copy::MemCopyH2D;
use openvm_cuda_common::d_buffer::DeviceBuffer;

use super::{
    Coeff, Device, DevicePolyExt, ExtendedLagrangeCoeff, Host, LagrangeCoeff, Polynomial, Rotation,
};

/// Helper trait dispatching [`EvaluationDomain::lagrange_to_coeff_many`] per
/// input residency. Impl'd for `Polynomial<F, LagrangeCoeff, Host>` and
/// `Polynomial<F, LagrangeCoeff, Device>`; the output residency matches the
/// input.
pub trait LagrangeToCoeffManyInput<F: Field>: Sized {
    type Output;
    fn lagrange_to_coeff_many_impl(
        domain: &EvaluationDomain<F>,
        in_many: &[Self],
    ) -> Result<Vec<Self::Output>, Error>;
}

impl<F: WithSmallOrderMulGroup<3>> LagrangeToCoeffManyInput<F>
    for Polynomial<F, LagrangeCoeff, Host>
{
    type Output = Polynomial<F, Coeff, Host>;
    fn lagrange_to_coeff_many_impl(
        domain: &EvaluationDomain<F>,
        in_many: &[Self],
    ) -> Result<Vec<Self::Output>, Error> {
        crate::cpu::poly::domain::lagrange_to_coeff_many_host(domain, in_many)
    }
}

impl<F: WithSmallOrderMulGroup<3>> LagrangeToCoeffManyInput<F>
    for Polynomial<F, LagrangeCoeff, Device>
{
    type Output = Polynomial<F, Coeff, Device>;
    fn lagrange_to_coeff_many_impl(
        domain: &EvaluationDomain<F>,
        in_many: &[Self],
    ) -> Result<Vec<Self::Output>, Error> {
        domain.lagrange_to_coeff_many_device_inputs(in_many)
    }
}

/// Helper trait dispatching [`EvaluationDomain::coeff_to_extended_part_many_device`]
/// per input residency. Both Host and Device inputs produce
/// `Vec<DeviceBuffer<F>>` outputs (the kernel always writes to device memory).
pub trait CoeffToExtendedPartManyDeviceInput<F: Field>: Sized {
    fn coeff_to_extended_part_many_device_impl(
        domain: &EvaluationDomain<F>,
        in_many: Vec<&Self>,
        extended_omega_factor: F,
    ) -> Result<Vec<DeviceBuffer<F>>, Error>;
}

impl<F: WithSmallOrderMulGroup<3>> CoeffToExtendedPartManyDeviceInput<F>
    for Polynomial<F, Coeff, Host>
{
    fn coeff_to_extended_part_many_device_impl(
        domain: &EvaluationDomain<F>,
        in_many: Vec<&Self>,
        extended_omega_factor: F,
    ) -> Result<Vec<DeviceBuffer<F>>, Error> {
        domain.coeff_to_extended_part_many_device_host_inputs(in_many, extended_omega_factor)
    }
}

impl<F: WithSmallOrderMulGroup<3>> CoeffToExtendedPartManyDeviceInput<F>
    for Polynomial<F, Coeff, Device>
{
    fn coeff_to_extended_part_many_device_impl(
        domain: &EvaluationDomain<F>,
        in_many: Vec<&Self>,
        extended_omega_factor: F,
    ) -> Result<Vec<DeviceBuffer<F>>, Error> {
        domain.coeff_to_extended_part_many_device_device_inputs(in_many, extended_omega_factor)
    }
}

use ff::{BatchInvert, Field, WithSmallOrderMulGroup};

#[allow(non_camel_case_types)]
#[derive(Copy, Clone, Debug)]
pub(crate) enum NttType {
    FFT = 1,
    iFFT = 2,
    CosetFFT = 3,
    iCosetFFT = 4,
    // kernel-selector enum: discriminant pinned to match the C++ NTT_TYPE enum
    // at cuda/src/ntt.cu (iFFT_cosetFFT = 5). No Rust caller dispatches this
    // case today, but removing the variant would renumber CosetFFT_Part = 6 and
    // break every `NttType::CosetFFT_Part as u32` site that talks to the FFI.
    #[allow(dead_code)]
    iFFT_CosetFFT = 5,
    CosetFFT_Part = 6,
}

impl From<NttType> for u32 {
    fn from(val: NttType) -> Self {
        val as u32
    }
}

/// This structure contains precomputed constants and other details needed for
/// performing operations on an evaluation domain of size $2^k$ and an extended
/// domain of size $2^{k} * j$ with $j \neq 0$.
#[derive(Clone, Debug)]
pub struct EvaluationDomain<F: Field> {
    n: u64,
    k: u32,
    extended_k: u32,
    pub omega: F,
    pub omega_inv: F,
    extended_omega: F,
    extended_omega_inv: F,
    pub g_coset: F,
    g_coset_inv: F,
    quotient_poly_degree: u64,
    pub ifft_divisor: F,
    extended_ifft_divisor: F,
    t_evaluations: Vec<F>,
    barycentric_weight: F,

    // Recursive stuff
    fft_data: HashMap<usize, FFTData<F>>,
}

impl<F: WithSmallOrderMulGroup<3>> EvaluationDomain<F> {
    /// This constructs a new evaluation domain object based on the provided
    /// values $j, k$.
    pub fn new(j: u32, k: u32) -> Self {
        // quotient_poly_degree * params.n - 1 is the degree of the quotient polynomial
        let quotient_poly_degree = (j - 1) as u64;

        // n = 2^k
        let n = 1u64 << k;

        // We need to work within an extended domain, not params.k but params.k + i
        // for some integer i such that 2^(params.k + i) is sufficiently large to
        // describe the quotient polynomial.
        let mut extended_k = k;
        while (1 << extended_k) < (n * quotient_poly_degree) {
            extended_k += 1;
        }
        #[cfg(feature = "profile")]
        log::debug!("k: {}, extended_k: {}", k, extended_k);

        let mut extended_omega = F::ROOT_OF_UNITY;

        // Get extended_omega, the 2^{extended_k}'th root of unity
        // The loop computes extended_omega = omega^{2 ^ (S - extended_k)}
        // Notice that extended_omega ^ {2 ^ extended_k} = omega ^ {2^S} = 1.
        for _ in extended_k..F::S {
            extended_omega = extended_omega.square();
        }
        let extended_omega = extended_omega;

        // Get omega, the 2^{k}'th root of unity (i.e. n'th root of unity)
        // The loop computes omega = extended_omega ^ {2 ^ (extended_k - k)}
        //           = (omega^{2 ^ (S - extended_k)})  ^ {2 ^ (extended_k - k)}
        //           = omega ^ {2 ^ (S - k)}.
        // Notice that omega ^ {2^k} = omega ^ {2^S} = 1.
        let mut omegas = Vec::with_capacity((extended_k - k + 1) as usize);
        let mut omega = extended_omega;
        omegas.push(omega);
        for _ in k..extended_k {
            omega = omega.square();
            omegas.push(omega);
        }
        let omega = omega;
        omegas.reverse();
        let mut omegas_inv = omegas.clone(); // Inversion computed later

        // We use zeta here because we know it generates a coset, and it's available
        // already.
        // The coset evaluation domain is:
        // zeta {1, extended_omega, extended_omega^2, ..., extended_omega^{(2^extended_k) - 1}}
        let g_coset = F::ZETA;
        let g_coset_inv = g_coset.square();

        let mut t_evaluations = Vec::with_capacity(1 << (extended_k - k));
        {
            // Compute the evaluations of t(X) = X^n - 1 in the coset evaluation domain.
            // We don't have to compute all of them, because it will repeat.
            let orig = F::ZETA.pow_vartime([n]);
            let step = extended_omega.pow_vartime([n]);
            let mut cur = orig;
            loop {
                t_evaluations.push(cur);
                cur *= &step;
                if cur == orig {
                    break;
                }
            }
            assert_eq!(t_evaluations.len(), 1 << (extended_k - k));

            // Subtract 1 from each to give us t_evaluations[i] = t(zeta * extended_omega^i)
            for coeff in &mut t_evaluations {
                *coeff -= &F::ONE;
            }

            // Invert, because we're dividing by this polynomial.
            // We invert in a batch, below.
        }

        let mut ifft_divisor = F::from(1 << k); // Inversion computed later
        let mut extended_ifft_divisor = F::from(1 << extended_k); // Inversion computed later

        // The barycentric weight of 1 over the evaluation domain
        // 1 / \prod_{i != 0} (1 - omega^i)
        let mut barycentric_weight = F::from(n); // Inversion computed later

        // Compute batch inversion
        t_evaluations
            .iter_mut()
            .chain(Some(&mut ifft_divisor))
            .chain(Some(&mut extended_ifft_divisor))
            .chain(Some(&mut barycentric_weight))
            .chain(&mut omegas_inv)
            .batch_invert();

        let omega_inv = omegas_inv[0];
        let extended_omega_inv = *omegas_inv.last().unwrap();
        let mut fft_data = HashMap::new();
        for (i, (omega, omega_inv)) in omegas.into_iter().zip(omegas_inv).enumerate() {
            let intermediate_k = k as usize + i;
            let len = 1usize << intermediate_k;
            fft_data.insert(len, FFTData::<F>::new(len, omega, omega_inv));
        }

        EvaluationDomain {
            n,
            k,
            extended_k,
            omega,
            omega_inv,
            extended_omega,
            extended_omega_inv,
            g_coset,
            g_coset_inv,
            quotient_poly_degree,
            ifft_divisor,
            extended_ifft_divisor,
            t_evaluations,
            barycentric_weight,
            fft_data,
        }
    }

    /// Obtains a polynomial in Lagrange form when given a vector of Lagrange
    /// coefficients of size `n`; panics if the provided vector is the wrong
    /// length.
    pub fn lagrange_from_vec(&self, values: Vec<F>) -> Polynomial<F, LagrangeCoeff> {
        assert_eq!(values.len(), self.n as usize);

        Polynomial::new(values)
    }

    pub fn lagrange_assigned_from_vec(
        &self,
        values: Vec<Assigned<F>>,
    ) -> Polynomial<Assigned<F>, LagrangeCoeff> {
        assert_eq!(values.len(), self.n as usize);

        Polynomial::new(values)
    }

    /// Obtains a polynomial in coefficient form when given a vector of
    /// coefficients of size `n`; panics if the provided vector is the wrong
    /// length.
    pub fn coeff_from_vec(&self, values: Vec<F>) -> Polynomial<F, Coeff> {
        assert_eq!(values.len(), self.n as usize);

        Polynomial::new(values)
    }

    /// Obtains a polynomial in ExtendedLagrange form when given a vector of
    /// Lagrange polynomials with total size `extended_n`; panics if the
    /// provided vector is the wrong length.
    pub fn extended_from_lagrange_vec(
        &self,
        values: Vec<Polynomial<F, LagrangeCoeff>>,
    ) -> Polynomial<F, ExtendedLagrangeCoeff> {
        assert_eq!(values.len(), self.extended_len() >> self.k);
        assert_eq!(values[0].len(), self.n as usize);

        // transpose the values in parallel
        let mut transposed = vec![vec![F::ZERO; values.len()]; self.n as usize];
        values.into_iter().enumerate().for_each(|(i, p)| {
            parallelize(&mut transposed, |transposed, start| {
                for (transposed, p) in transposed.iter_mut().zip(p.values()[start..].iter()) {
                    transposed[i] = *p;
                }
            });
        });

        Polynomial::new(transposed.into_iter().flatten().collect())
    }

    /// Device-side variant of `extended_from_lagrange_vec`.
    ///
    /// If every part is device-resident and there is enough free VRAM for the
    /// output, this gathers the parts on the GPU and returns a device-resident
    /// polynomial. Otherwise, it materializes the inputs and uses the host path.
    pub fn extended_from_lagrange_vec_device(
        &self,
        values: Vec<super::MaybeDevice<F, LagrangeCoeff>>,
    ) -> Result<super::MaybeDevice<F, ExtendedLagrangeCoeff>, Error> {
        assert_eq!(values.len(), self.extended_len() >> self.k);
        assert_eq!(values[0].len(), self.n as usize);

        let all_device = values.iter().all(|p| p.is_device());
        #[cfg(not(feature = "vram-fallback"))]
        {
            debug_assert!(
                all_device,
                "extended_from_lagrange_vec_device requires all-Device input parts; \
                 caller may opt into the host-arm materialize fallback with the \
                 `vram-fallback` feature"
            );
            if !all_device {
                return Err(Error::HaloGpu(HaloGpuError::InsufficientGpuMemory {
                    context: "extended_from_lagrange_vec_device.not_all_device",
                    magnitude: values.len() as u64,
                    free_bytes: 0,
                }));
            }
        }
        #[cfg(feature = "vram-fallback")]
        if !all_device {
            return crate::cpu::poly::domain::extended_from_lagrange_vec_not_all_device(
                self, values,
            );
        }

        let extended_bytes = self.extended_len() * std::mem::size_of::<F>();
        let free_bytes = query_device_free_bytes_for_chunking();
        if free_bytes < extended_bytes {
            #[cfg(not(feature = "vram-fallback"))]
            {
                tracing::error!(
                    target: "halo2_vram_fallback",
                    site = "extended_from_lagrange_vec_device.vram_tight",
                    free_bytes,
                    needed_bytes = extended_bytes,
                    "VRAM-tight: returning InsufficientGpuMemory (vram-fallback feature disabled)"
                );
                return Err(Error::HaloGpu(HaloGpuError::InsufficientGpuMemory {
                    context: "extended_from_lagrange_vec_device.vram_tight",
                    magnitude: extended_bytes as u64,
                    free_bytes: free_bytes as u64,
                }));
            }
            #[cfg(feature = "vram-fallback")]
            {
                return crate::cpu::poly::domain::extended_from_lagrange_vec_vram_tight(
                    self,
                    values,
                    free_bytes,
                    extended_bytes,
                );
            }
        }

        // All values are Device-resident.
        let device_values: Vec<Polynomial<F, LagrangeCoeff, Device>> = values
            .into_iter()
            .map(|m| match m {
                super::MaybeDevice::Device(p) => p,
                super::MaybeDevice::Host(_) => unreachable!("all_device checked above"),
            })
            .collect();
        let d_out: DeviceBuffer<F> =
            DeviceBuffer::<F>::with_capacity_on(self.extended_len(), &HALO2_GPU_CTX);
        let part_refs: Vec<&DeviceBuffer<F>> =
            device_values.iter().map(|p| p.device_buf()).collect();
        extended_from_lagrange_vec_device(&d_out, &part_refs, self.n).map_err(Error::from)?;
        drop(device_values);
        Ok(super::MaybeDevice::Device(Polynomial::from_device(d_out)))
    }

    /// Returns an empty (zero) polynomial in the coefficient basis
    pub fn empty_coeff(&self) -> Polynomial<F, Coeff> {
        Polynomial::new(vec![F::ZERO; self.n as usize])
    }

    pub unsafe fn empty_coeff_unsafe(&self) -> Polynomial<F, Coeff> {
        let mut values = Vec::with_capacity(self.n as usize);
        values.set_len(self.n as usize);

        Polynomial::new(values)
    }

    /// Returns an empty (zero) polynomial in the Lagrange coefficient basis
    pub fn empty_lagrange(&self) -> Polynomial<F, LagrangeCoeff> {
        Polynomial::new(vec![F::ZERO; self.n as usize])
    }

    /// Returns an empty (zero) polynomial in the Lagrange coefficient basis, with
    /// deferred inversions.
    pub(crate) fn empty_lagrange_assigned(&self) -> Polynomial<Assigned<F>, LagrangeCoeff> {
        Polynomial::new(vec![F::ZERO.into(); self.n as usize])
    }

    /// Returns a constant polynomial in the Lagrange coefficient basis
    pub fn constant_lagrange(&self, scalar: F) -> Polynomial<F, LagrangeCoeff> {
        Polynomial::new(vec![scalar; self.n as usize])
    }

    /// Returns an empty (zero) polynomial in the extended Lagrange coefficient
    /// basis
    pub fn empty_extended(&self) -> Polynomial<F, ExtendedLagrangeCoeff> {
        Polynomial::new(vec![F::ZERO; self.extended_len()])
    }

    pub unsafe fn empty_extended_unsafe(&self) -> Polynomial<F, ExtendedLagrangeCoeff> {
        let mut values = Vec::with_capacity(self.extended_len());
        values.set_len(self.extended_len());

        Polynomial::new(values)
    }

    /// Returns a constant polynomial in the extended Lagrange coefficient
    /// basis
    pub fn constant_extended(&self, scalar: F) -> Polynomial<F, ExtendedLagrangeCoeff> {
        Polynomial::new(vec![scalar; self.extended_len()])
    }

    /// This takes us from an n-length vector into the coefficient form.
    ///
    /// This function will panic if the provided vector is not the correct
    /// length.
    pub fn lagrange_to_coeff(
        &self,
        mut a: Polynomial<F, LagrangeCoeff>,
    ) -> Result<Polynomial<F, Coeff>, Error> {
        crate::perf_section!("lagrange_to_coeff");
        assert_eq!(a.len(), 1 << self.k);

        // Perform inverse FFT to obtain the polynomial in coefficient form
        self.ifft(a.values_mut(), self.omega_inv, self.k, self.ifft_divisor)?;

        Ok(Polynomial::new(a.into_values()))
    }

    #[allow(clippy::uninit_vec)]
    pub fn lagrange_to_extend_part(
        &self,
        a: &Polynomial<F, LagrangeCoeff>,
        omega_extend_part: F,
    ) -> Result<Polynomial<F, ExtendedLagrangeCoeff>, Error> {
        assert_eq!(a.len(), 1 << self.k);

        let log_n = self.k;
        let mut b: Vec<F> = Vec::with_capacity(1 << log_n);
        unsafe {
            b.set_len(1 << self.k());
        };
        let mut b = Polynomial::new(b);

        self.ifft_cosetfft_part(
            a.values(),
            b.values_mut(),
            log_n,
            self.omega_inv,
            self.ifft_divisor,
            self.omega,
            self.g_coset * omega_extend_part,
        )?;
        Ok(b)
    }

    /// Batch iFFT from Lagrange basis to coefficient basis. Generic over input
    /// residency via [`LagrangeToCoeffManyInput`]: Host inputs produce Host
    /// outputs through the existing CPU/GPU iFFT path; Device inputs stage
    /// through host memory and re-upload the result to keep outputs
    /// device-resident.
    pub fn lagrange_to_coeff_many<P>(
        &self,
        in_many: &[P],
    ) -> Result<Vec<<P as LagrangeToCoeffManyInput<F>>::Output>, Error>
    where
        P: LagrangeToCoeffManyInput<F>,
    {
        P::lagrange_to_coeff_many_impl(self, in_many)
    }

    /// Device-input batch iFFT: `Vec<Polynomial<F, LagrangeCoeff, Device>>`
    /// → `Vec<Polynomial<F, Coeff, Device>>`. Both endpoints device-resident,
    /// no PCIe traffic. Dispatches `ifft_many_device`.
    ///
    /// VRAM gating: per-batch check on aggregate device bytes
    /// (`in_many.len() * n_bytes`); on tight VRAM either returns
    /// `HaloGpuError::InsufficientGpuMemory` (default) or, under the
    /// `vram-fallback` feature, D2H's inputs → host iFFT → H2D's outputs.
    pub(crate) fn lagrange_to_coeff_many_device_inputs(
        &self,
        in_many: &[Polynomial<F, LagrangeCoeff, Device>],
    ) -> Result<Vec<Polynomial<F, Coeff, Device>>, Error> {
        crate::perf_section!("domain.lagrange_to_coeff_many_device_inputs");
        log::info!(
            "using lagrange_to_coeff_many_device_inputs: vec_num[{}]",
            in_many.len()
        );
        if in_many.is_empty() {
            return Ok(vec![]);
        }

        // VRAM gating: mirror the sibling `lagrange_to_coeff_many_device`.
        // Aggregate output bytes drive the budget; on tight VRAM either
        // return `InsufficientGpuMemory` or fall back to the host arm
        // (D2H inputs → host iFFT → H2D outputs) so producer sites never
        // OOM.
        let n_bytes = (1usize << self.k) * std::mem::size_of::<F>();
        let total_bytes = in_many.len() * n_bytes;
        let free_bytes = query_device_free_bytes_for_chunking();
        if free_bytes < total_bytes {
            #[cfg(not(feature = "vram-fallback"))]
            {
                tracing::error!(
                    target: "halo2_vram_fallback",
                    site = "lagrange_to_coeff_many_device_inputs.vram_tight",
                    free_bytes,
                    needed_bytes = total_bytes,
                    batch_len = in_many.len(),
                    "VRAM-tight: returning InsufficientGpuMemory (vram-fallback feature disabled)"
                );
                return Err(Error::HaloGpu(HaloGpuError::InsufficientGpuMemory {
                    context: "lagrange_to_coeff_many_device_inputs.vram_tight",
                    magnitude: total_bytes as u64,
                    free_bytes: free_bytes as u64,
                }));
            }
            #[cfg(feature = "vram-fallback")]
            {
                return crate::cpu::poly::domain::lagrange_to_coeff_many_device_inputs_host_arm(
                    self,
                    in_many,
                    free_bytes,
                    total_bytes,
                );
            }
        }

        let in_objs: Vec<FFITraitObject> = in_many
            .iter()
            .map(|p| FFITraitObject::new(p.device_buf().as_raw_ptr() as usize))
            .collect();
        let out_bufs = ifft_many_device::<F>(in_objs, self.k, self.omega_inv, self.ifft_divisor)
            .map_err(Error::HaloGpu)?;
        Ok(out_bufs.into_iter().map(Polynomial::from_device).collect())
    }

    /// Device-output variant of `lagrange_to_coeff`.
    ///
    /// Returns a `Polynomial<F, Coeff>::Device` (a Device-resident
    /// coefficient-form polynomial). The input is consumed (matches the
    /// signature of `lagrange_to_coeff`). Internally dispatches the
    /// `_halo2_fft_many_to_device` FFI (Host-input, Device-output batch
    /// iFFT) with `num_many = 1`.
    ///
    /// VRAM gating: if the per-poly Device output would push the GPU
    /// memory budget past `query_device_free_bytes_for_chunking()`, the
    /// implementation falls back to the existing host arm
    /// (`lagrange_to_coeff`) so producer sites never OOM.
    pub fn lagrange_to_coeff_device(
        &self,
        a: Polynomial<F, LagrangeCoeff>,
    ) -> Result<Polynomial<F, Coeff, Device>, Error> {
        crate::perf_section!("domain.lagrange_to_coeff_device");
        assert_eq!(a.len(), 1 << self.k);

        let n_bytes = (1usize << self.k) * std::mem::size_of::<F>();
        let free_bytes = query_device_free_bytes_for_chunking();
        if free_bytes < n_bytes {
            #[cfg(not(feature = "vram-fallback"))]
            {
                tracing::error!(
                    target: "halo2_vram_fallback",
                    site = "lagrange_to_coeff_device.vram_tight",
                    free_bytes,
                    needed_bytes = n_bytes,
                    "VRAM-tight: returning InsufficientGpuMemory (vram-fallback feature disabled)"
                );
                return Err(Error::HaloGpu(HaloGpuError::InsufficientGpuMemory {
                    context: "lagrange_to_coeff_device.vram_tight",
                    magnitude: n_bytes as u64,
                    free_bytes: free_bytes as u64,
                }));
            }
            #[cfg(feature = "vram-fallback")]
            {
                return crate::cpu::poly::domain::lagrange_to_coeff_device_host_arm(
                    self, a, free_bytes, n_bytes,
                );
            }
        }

        let outs = ifft_many_h2d::<F>(&[a], self.k, self.omega_inv, self.ifft_divisor)
            .map_err(Error::HaloGpu)?;
        let mut outs = outs;
        let out = outs.pop().expect("ifft_many_h2d returned empty vec");
        Ok(out)
    }

    /// Device-input single-poly variant of `lagrange_to_coeff_device`.
    ///
    /// Consumes a `Polynomial<F, LagrangeCoeff, Device>` and returns a
    /// `Polynomial<F, Coeff, Device>` — both endpoints device-resident, no
    /// PCIe traffic. Dispatches the device-input batch iFFT
    /// (`ifft_many_device`) with `num_many = 1`, mirroring
    /// the batch arm `lagrange_to_coeff_many_device`.
    ///
    /// VRAM gating: if the per-poly Device output would push the GPU
    /// memory budget past `query_device_free_bytes_for_chunking()`, the
    /// implementation either returns `HaloGpuError::InsufficientGpuMemory`
    /// (default) or, under the `vram-fallback` feature, D2H's the input,
    /// runs the host iFFT (`lagrange_to_coeff`), then H2D's the output —
    /// mirroring the sibling `lagrange_to_coeff_device`.
    pub fn lagrange_to_coeff_device_input(
        &self,
        a: Polynomial<F, LagrangeCoeff, Device>,
    ) -> Result<Polynomial<F, Coeff, Device>, Error> {
        crate::perf_section!("domain.lagrange_to_coeff_device_input");
        assert_eq!(a.len(), 1 << self.k);

        let n_bytes = (1usize << self.k) * std::mem::size_of::<F>();
        let free_bytes = query_device_free_bytes_for_chunking();
        if free_bytes < n_bytes {
            #[cfg(not(feature = "vram-fallback"))]
            {
                tracing::error!(
                    target: "halo2_vram_fallback",
                    site = "lagrange_to_coeff_device_input.vram_tight",
                    free_bytes,
                    needed_bytes = n_bytes,
                    "VRAM-tight: returning InsufficientGpuMemory (vram-fallback feature disabled)"
                );
                return Err(Error::HaloGpu(HaloGpuError::InsufficientGpuMemory {
                    context: "lagrange_to_coeff_device_input.vram_tight",
                    magnitude: n_bytes as u64,
                    free_bytes: free_bytes as u64,
                }));
            }
            #[cfg(feature = "vram-fallback")]
            {
                return crate::cpu::poly::domain::lagrange_to_coeff_device_input_host_arm(
                    self, a, free_bytes, n_bytes,
                );
            }
        }

        let in_objs = vec![FFITraitObject::new(a.device_buf().as_raw_ptr() as usize)];
        let out_bufs = ifft_many_device::<F>(in_objs, self.k, self.omega_inv, self.ifft_divisor)
            .map_err(Error::HaloGpu)?;
        let mut out_bufs = out_bufs;
        let out = out_bufs.pop().expect("ifft_many_device returned empty vec");
        Ok(Polynomial::from_device(out))
    }

    /// Device-output variant of `lagrange_to_coeff_many`.
    ///
    /// Returns a `Vec<Polynomial<F, Coeff>>` where each entry is a
    /// Device-resident polynomial. Inputs are consumed.
    ///
    /// VRAM gating: per-batch check on aggregate device bytes
    /// (`in_many.len() * n_bytes`); falls back to the host arm
    /// (`lagrange_to_coeff_many`) when device memory is tight, so the
    /// producer sites never OOM. Producer sites that prefer per-poly
    /// gating can chunk their input batches manually.
    pub fn lagrange_to_coeff_many_device(
        &self,
        in_many: &[Polynomial<F, LagrangeCoeff>],
    ) -> Result<Vec<Polynomial<F, Coeff, Device>>, Error> {
        crate::perf_section!("domain.lagrange_to_coeff_many_device");
        log::info!(
            "using lagrange_to_coeff_many_device: vec_num[{}]",
            in_many.len()
        );
        if in_many.is_empty() {
            return Ok(vec![]);
        }
        let n_bytes = (1usize << self.k) * std::mem::size_of::<F>();
        let total_bytes = in_many.len() * n_bytes;
        let free_bytes = query_device_free_bytes_for_chunking();
        if free_bytes < total_bytes {
            #[cfg(not(feature = "vram-fallback"))]
            {
                tracing::error!(
                    target: "halo2_vram_fallback",
                    site = "lagrange_to_coeff_many_device.vram_tight",
                    free_bytes,
                    needed_bytes = total_bytes,
                    batch_len = in_many.len(),
                    "VRAM-tight: returning InsufficientGpuMemory (vram-fallback feature disabled)"
                );
                return Err(Error::HaloGpu(HaloGpuError::InsufficientGpuMemory {
                    context: "lagrange_to_coeff_many_device.vram_tight",
                    magnitude: total_bytes as u64,
                    free_bytes: free_bytes as u64,
                }));
            }
            #[cfg(feature = "vram-fallback")]
            {
                return crate::cpu::poly::domain::lagrange_to_coeff_many_device_host_arm(
                    self,
                    in_many,
                    free_bytes,
                    total_bytes,
                );
            }
        }
        let outs = ifft_many_h2d::<F>(in_many, self.k, self.omega_inv, self.ifft_divisor)
            .map_err(Error::HaloGpu)?;
        Ok(outs)
    }

    /// This takes us from an n-length coefficient vector into a coset of the extended
    /// evaluation domain, rotating by `rotation` if desired.
    // Todo: use cosetfft
    pub fn coeff_to_extended(
        &self,
        a: &Polynomial<F, Coeff>,
    ) -> Result<Polynomial<F, ExtendedLagrangeCoeff>, Error> {
        crate::perf_section!("coeff_to_extended");
        assert_eq!(a.len(), 1 << self.k);
        let mut b: Vec<F> = Vec::with_capacity(self.extended_len());
        unsafe {
            b.set_len(self.extended_len());
        }

        self.cosetfft(
            a.values(),
            &mut b,
            self.extended_omega,
            self.k,
            self.extended_k,
        )?;

        Ok(Polynomial::new(b))
    }

    /// This takes us from an n-length coefficient vector into parts of the
    /// extended evaluation domain. For example, for a polynomial with size n,
    /// and an extended domain of size mn, we can compute all parts
    /// independently, which are
    ///     `FFT(f(zeta * X), n)`
    ///     `FFT(f(zeta * extended_omega * X), n)`
    ///     ...
    ///     `FFT(f(zeta * extended_omega^{m-1} * X), n)`
    pub fn coeff_to_extended_parts(
        &self,
        a: &Polynomial<F, Coeff>,
    ) -> Result<Vec<Polynomial<F, LagrangeCoeff>>, Error> {
        assert_eq!(a.len(), 1 << self.k);

        let num_parts = self.extended_len() >> self.k;
        let mut extended_omega_factor = F::ONE;
        (0..num_parts)
            .map(|_| {
                let part = self.coeff_to_extended_part(a.clone(), extended_omega_factor);
                extended_omega_factor *= self.extended_omega;
                part
            })
            .collect()
    }

    /// This takes us from several n-length coefficient vectors each into parts
    /// of the extended evaluation domain. For example, for a polynomial with
    /// size n, and an extended domain of size mn, we can compute all parts
    /// independently, which are
    ///     `FFT(f(zeta * X), n)`
    ///     `FFT(f(zeta * extended_omega * X), n)`
    ///     ...
    ///     `FFT(f(zeta * extended_omega^{m-1} * X), n)`
    pub fn batched_coeff_to_extended_parts(
        &self,
        a: &[Polynomial<F, Coeff>],
    ) -> Result<Vec<Vec<Polynomial<F, LagrangeCoeff>>>, Error> {
        assert_eq!(a[0].len(), 1 << self.k);

        let mut extended_omega_factor = F::ONE;
        let num_parts = self.extended_len() >> self.k;
        (0..num_parts)
            .map(|_| {
                let a_lagrange = a
                    .iter()
                    .map(|poly| self.coeff_to_extended_part(poly.clone(), extended_omega_factor))
                    .collect::<Result<Vec<_>, _>>();
                extended_omega_factor *= self.extended_omega;
                a_lagrange
            })
            .collect()
    }

    /// This takes us from an n-length coefficient vector into a part of the
    /// extended evaluation domain. For example, for a polynomial with size n,
    /// and an extended domain of size mn, we can compute one of the m parts
    /// separately, which is
    ///     `FFT(f(zeta * extended_omega_factor * X), n)`
    /// where `extended_omega_factor` is `extended_omega^i` with `i` in `[0, m)`.
    pub fn coeff_to_extended_part(
        &self,
        mut a: Polynomial<F, Coeff>,
        extended_omega_factor: F,
    ) -> Result<Polynomial<F, LagrangeCoeff>, Error> {
        crate::perf_section!("coeff_to_extended_part");
        assert_eq!(a.len(), 1 << self.k);

        self.distribute_powers(a.values_mut(), self.g_coset * extended_omega_factor);
        fft_gpu(
            NttType::FFT as u32,
            a.values_mut(),
            self.k,
            self.omega,
            F::ONE,
        )?;

        Ok(Polynomial::new(a.into_values()))
    }

    /// Device-output variant of `coeff_to_extended_part_many`. Same
    /// FFT, but the result stays on device and is returned as
    /// `DeviceBuffer<F>` per polynomial.
    /// Caller can pipe the device pointers into a downstream GPU kernel
    /// (e.g. `_halo2_quotient_permutation`) without paying a redundant
    /// D→H + H→D round trip.
    ///
    /// Each returned `DeviceBuffer<F>` holds `1 << self.k` field elements
    /// of FFT output. The caller must keep the `Vec<DeviceBuffer<F>>`
    /// alive for the duration of the downstream kernel that reads from
    /// these pointers.
    /// Batch CosetFFT_Part producing device-resident outputs. Generic over
    /// input residency via [`CoeffToExtendedPartManyDeviceInput`].
    pub fn coeff_to_extended_part_many_device<P>(
        &self,
        in_many: Vec<&P>,
        extended_omega_factor: F,
    ) -> Result<Vec<DeviceBuffer<F>>, Error>
    where
        P: CoeffToExtendedPartManyDeviceInput<F>,
    {
        P::coeff_to_extended_part_many_device_impl(self, in_many, extended_omega_factor)
    }

    fn coeff_to_extended_part_many_device_host_inputs(
        &self,
        in_many: Vec<&Polynomial<F, Coeff>>,
        extended_omega_factor: F,
    ) -> Result<Vec<DeviceBuffer<F>>, Error> {
        crate::perf_section!("coeff_to_extended_part_many_device");
        if in_many.is_empty() {
            return Ok(vec![]);
        }

        let in_objs: Vec<FFITraitObject> = in_many
            .iter()
            .map(|p| FFITraitObject::from_slice(p.values()))
            .collect();

        // Swapping `omega` and `divisor` in the FFI is a verified footgun —
        // both slots feed CosetFFT_Part's internal `mult_power_of_omega` shift
        // and confused values produced an earlier SNARK verification failure
        // during development. `extend_log_n` is unused by CosetFFT_Part but
        // mismatching it changes the internal sizing.
        Ok(cosetfft_many_h2d::<F>(
            crate::poly::NttType::CosetFFT_Part as u32,
            in_objs,
            self.k,
            self.k,
            self.omega,
            self.g_coset * extended_omega_factor,
        )?)
    }

    pub(crate) fn coeff_to_extended_part_many_device_device_inputs(
        &self,
        in_many: Vec<&Polynomial<F, Coeff, Device>>,
        extended_omega_factor: F,
    ) -> Result<Vec<DeviceBuffer<F>>, Error> {
        crate::perf_section!("domain.coeff_to_extended_part_many_device_device_inputs");
        if in_many.is_empty() {
            return Ok(vec![]);
        }

        let in_objs: Vec<FFITraitObject> = in_many
            .iter()
            .map(|p| FFITraitObject::new(p.device_buf().as_raw_ptr() as usize))
            .collect();

        Ok(cosetfft_many_device::<F>(
            crate::poly::NttType::CosetFFT_Part as u32,
            in_objs,
            self.k,
            self.k,
            self.omega,
            self.g_coset * extended_omega_factor,
        )?)
    }

    pub fn coeff_to_extended_part_many(
        &self,
        in_many: Vec<&Polynomial<F, Coeff>>,
        extended_omega_factor: F,
    ) -> Result<Vec<Polynomial<F, LagrangeCoeff>>, Error> {
        crate::perf_section!("coeff_to_extended_part_many");
        log::info!(
            "using coeff_to_extended_part_many: vec_num[{}]",
            in_many.len()
        );
        if in_many.is_empty() {
            return Ok(vec![]);
        }

        use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
        let mut inout_many: Vec<Polynomial<F, LagrangeCoeff>> = in_many
            .par_iter()
            .map(|poly| Polynomial::new(poly.values().to_vec()))
            .collect();

        let ntt_type = NttType::CosetFFT_Part as u32;
        self.fft_many(
            ntt_type,
            &mut inout_many,
            self.g_coset * extended_omega_factor,
        )?;
        Ok(inout_many)
    }

    pub fn coeff_to_extended_many<PR>(
        &self,
        in_many: &[PR],
    ) -> Result<Vec<Polynomial<F, ExtendedLagrangeCoeff>>, Error>
    where
        PR: Deref<Target = [F]> + Send + Sync,
    {
        crate::perf_section!("coeff_to_extended_many");
        log::info!("using coeff_to_extended_many: vec_num[{}]", in_many.len());
        if in_many.is_empty() {
            return Ok(vec![]);
        }

        let mut out_many: Vec<Polynomial<F, ExtendedLagrangeCoeff>> =
            Vec::with_capacity(in_many.len());
        for _ in 0..in_many.len() {
            let mut out: Vec<F> = Vec::with_capacity(self.extended_len());
            unsafe {
                out.set_len(self.extended_len());
            }
            out_many.push(Polynomial::new(out));
        }

        self.cosetfft_many(in_many, &mut out_many)?;
        Ok(out_many)
    }

    /// Rotate the extended domain polynomial over the original domain.
    pub fn rotate_extended(
        &self,
        poly: &Polynomial<F, ExtendedLagrangeCoeff>,
        rotation: Rotation,
    ) -> Polynomial<F, ExtendedLagrangeCoeff> {
        let new_rotation = ((1 << (self.extended_k - self.k)) * rotation.0.abs()) as usize;

        let mut poly = poly.clone();

        if rotation.0 >= 0 {
            poly.values_mut().rotate_left(new_rotation);
        } else {
            poly.values_mut().rotate_right(new_rotation);
        }

        poly
    }

    /// This takes us from the extended evaluation domain and gets us the
    /// quotient polynomial coefficients.
    ///
    /// This function will panic if the provided vector is not the correct
    /// length.
    ///
    /// Host-residency inverse FFT + coset adjustment producing a host
    /// coefficient polynomial.
    pub fn extended_to_coeff(
        &self,
        mut a: Polynomial<F, ExtendedLagrangeCoeff, Host>,
    ) -> Result<Polynomial<F, Coeff, Host>, Error> {
        crate::perf_section!("extended_to_coeff");
        assert_eq!(a.len(), self.extended_len());

        let target_len = (self.n * self.quotient_poly_degree) as usize;

        self.ifft(
            a.values_mut(),
            self.extended_omega_inv,
            self.extended_k,
            self.extended_ifft_divisor,
        )?;
        // Distribute powers to move from coset; opposite from the
        // transformation we performed earlier.
        self.distribute_powers_zeta(a.values_mut(), false);
        let mut out = a.into_values();
        out.truncate(target_len);
        Ok(Polynomial::<F, Coeff, Host>::new(out))
    }

    /// Device-residency inverse FFT + coset adjustment producing a
    /// device-resident coefficient polynomial.
    pub fn extended_to_coeff_device(
        &self,
        a: Polynomial<F, ExtendedLagrangeCoeff, Device>,
    ) -> Result<Polynomial<F, Coeff, Device>, Error> {
        crate::perf_section!("domain.extended_to_coeff_device");
        assert_eq!(a.len(), self.extended_len());

        let target_len = (self.n * self.quotient_poly_degree) as usize;
        let extended_len = self.extended_len();

        let d_buf = a.into_device_buf();
        fft_normal_device(
            NttType::iFFT as u32,
            self.extended_k,
            &d_buf,
            &d_buf,
            self.extended_omega_inv,
            self.extended_ifft_divisor,
        )?;
        let coset_powers = [self.g_coset_inv, self.g_coset];
        distribute_powers_zeta_device(&d_buf, &coset_powers).map_err(Error::HaloGpu)?;

        if target_len == extended_len {
            Ok(Polynomial::<F, Coeff, Device>::from_device(d_buf))
        } else {
            use openvm_cuda_common::copy::cuda_memcpy_on;
            let dst: DeviceBuffer<F> =
                DeviceBuffer::<F>::with_capacity_on(target_len, &HALO2_GPU_CTX);
            let bytes = target_len * std::mem::size_of::<F>();
            unsafe {
                cuda_memcpy_on::<true, true>(
                    dst.as_mut_raw_ptr(),
                    d_buf.as_raw_ptr(),
                    bytes,
                    &HALO2_GPU_CTX,
                )
                .map_err(HaloGpuError::from)
                .map_err(Error::from)?;
            }
            Ok(Polynomial::<F, Coeff, Device>::from_device(dst))
        }
    }

    /// Host-residency divide by the vanishing polynomial of the $2^k$ domain.
    pub fn divide_by_vanishing_poly(
        &self,
        mut a: Polynomial<F, ExtendedLagrangeCoeff, Host>,
    ) -> Result<Polynomial<F, ExtendedLagrangeCoeff, Host>, Error> {
        crate::perf_section!("divide_by_vanishing_poly");
        assert_eq!(a.len(), self.extended_len());

        parallelize(a.values_mut(), |h, mut index| {
            for h in h {
                *h *= &self.t_evaluations[index % self.t_evaluations.len()];
                index += 1;
            }
        });
        Ok(Polynomial::new(a.into_values()))
    }

    /// Device-residency divide by the vanishing polynomial of the $2^k$ domain.
    pub fn divide_by_vanishing_poly_device(
        &self,
        a: Polynomial<F, ExtendedLagrangeCoeff, Device>,
    ) -> Result<Polynomial<F, ExtendedLagrangeCoeff, Device>, Error> {
        crate::perf_section!("domain.divide_by_vanishing_poly_device");
        assert_eq!(a.len(), self.extended_len());

        let t_dev = self
            .t_evaluations
            .as_slice()
            .to_device_on(&HALO2_GPU_CTX)
            .map_err(HaloGpuError::from)
            .map_err(Error::from)?;
        let d_buf = a.into_device_buf();
        divide_by_vanishing_poly_device::<F>(&d_buf, &t_dev).map_err(Error::from)?;
        Ok(Polynomial::<F, ExtendedLagrangeCoeff, Device>::from_device(
            d_buf,
        ))
    }

    /// Given a slice of group elements `[a_0, a_1, a_2, ...]`, this returns
    /// `[a_0, [zeta]a_1, [zeta^2]a_2, a_3, [zeta]a_4, [zeta^2]a_5, a_6, ...]`,
    /// where zeta is a cube root of unity in the multiplicative subgroup with
    /// order (p - 1), i.e. zeta^3 = 1.
    ///
    /// `into_coset` should be set to `true` when moving into the coset,
    /// and `false` when moving out. This toggles the choice of `zeta`.
    fn distribute_powers_zeta(&self, a: &mut [F], into_coset: bool) {
        let coset_powers = if into_coset {
            [self.g_coset, self.g_coset_inv]
        } else {
            [self.g_coset_inv, self.g_coset]
        };
        parallelize(a, |a, mut index| {
            for a in a {
                // Distribute powers to move into/from coset
                let i = index % (coset_powers.len() + 1);
                if i != 0 {
                    *a *= &coset_powers[i - 1];
                }
                index += 1;
            }
        });
    }

    fn ifft_cosetfft_part(
        &self,
        a: &[F],
        b: &mut [F],
        log_n: u32,
        omega_inv: F,
        divisor: F,
        omega: F,
        omega_extend_part: F,
    ) -> Result<(), HaloGpuError> {
        ifft_cosetfftpart_gpu(
            a,
            b,
            log_n,
            log_n,
            omega_inv,
            divisor,
            omega,
            omega_extend_part,
        )?;
        Ok(())
    }

    fn coset_fft_single_gpu(
        &self,
        a: &[F],
        b: &mut [F],
        omega: F,
        log_n: u32,
        extend_log_n: u32,
    ) -> Result<(), HaloGpuError> {
        let ntt_type = NttType::CosetFFT.into();
        let is_memory_enough = unsafe {
            _halo2_fft_normal_check_memory(
                ntt_type,
                a.as_ptr() as *const libc::c_void,
                log_n,
                extend_log_n,
            )
        };
        if is_memory_enough {
            cosetfft_gpu(ntt_type, a, b, log_n, extend_log_n, omega, F::ONE)?;
        } else {
            let mut c = a.to_vec();
            c.resize(self.extended_len(), F::ZERO);
            b.clone_from_slice(&c);
            self.distribute_powers_zeta(b, true);
            split_radix_fft_gpu(ntt_type, b, log_n, extend_log_n, omega, F::ONE)?;
        }
        Ok(())
    }

    fn cosetfft(
        &self,
        a: &[F],
        b: &mut [F],
        omega: F,
        log_n: u32,
        extend_log_n: u32,
    ) -> Result<(), HaloGpuError> {
        Self::coset_fft_single_gpu(self, a, b, omega, log_n, extend_log_n)?;
        Ok(())
    }

    fn cosetfft_many<PR, PM>(&self, a_many: &[PR], b_many: &mut [PM]) -> Result<(), HaloGpuError>
    where
        PR: Deref<Target = [F]> + Send + Sync,
        PM: DerefMut<Target = [F]> + Send + Sync,
    {
        let get_slice_polys_ffi_in = |polys: &[PR]| {
            polys
                .iter()
                .map(|poly| FFITraitObject::from_slice(poly))
                .collect::<Vec<FFITraitObject>>()
        };

        let get_slice_polys_ffi_out = |polys: &mut [PM]| {
            polys
                .iter_mut()
                .map(|poly| FFITraitObject::from_slice(poly))
                .collect::<Vec<FFITraitObject>>()
        };

        // Single-stream GPU prover: run the whole batch on gpu 0.
        let ntt_type = NttType::CosetFFT.into();
        let is_memory_enough = unsafe {
            // if normal() memory is enough, then it's enough for many()
            _halo2_fft_normal_check_memory(ntt_type, std::ptr::null(), self.k, self.extended_k)
        };
        if is_memory_enough {
            let batch_a_ffi = get_slice_polys_ffi_in(a_many);
            let batch_b_ffi = get_slice_polys_ffi_out(b_many);
            cosetfft_gpu_many::<F>(
                ntt_type,
                batch_a_ffi,
                batch_b_ffi,
                self.k,
                self.extended_k,
                self.extended_omega,
                F::ONE,
            )?;
        } else {
            for (a, b) in a_many.iter().zip(b_many.iter_mut()) {
                self.coset_fft_single_gpu(a, b, self.extended_omega, self.k, self.extended_k)?;
            }
        }
        Ok(())
    }

    /// Given a slice of group elements `[a_0, a_1, a_2, ...]`, this returns
    /// `[a_0, [c]a_1, [c^2]a_2, [c^3]a_3, [c^4]a_4, ...]`.
    fn distribute_powers(&self, a: &mut [F], c: F) {
        parallelize(a, |a, index| {
            let mut c_power = c.pow_vartime([index as u64]);
            for a in a {
                a.mul_assign(&c_power);
                c_power = c_power * c;
            }
        });
    }

    fn ifft(&self, a: &mut [F], omega_inv: F, log_n: u32, divisor: F) -> Result<(), HaloGpuError> {
        let ntt_type = NttType::iFFT.into();
        fft_gpu(ntt_type, a, log_n, omega_inv, divisor)?;
        Ok(())
    }

    // batched in-place FFT over many polynomials
    fn fft_many(
        &self,
        ntt_type: u32,
        b_many: &mut [Polynomial<F, LagrangeCoeff>],
        part_power: F,
    ) -> Result<(), HaloGpuError> {
        if ntt_type != u32::from(NttType::FFT) && ntt_type != u32::from(NttType::CosetFFT_Part) {
            panic!("ntt_type should be CosetFFT_Part / FFT");
        }
        let get_slice_polys_ffi_out = |polys: &mut [Polynomial<F, LagrangeCoeff>]| {
            polys
                .iter()
                .map(|poly| FFITraitObject::from_slice(poly.values()))
                .collect::<Vec<FFITraitObject>>()
        };

        // Single-stream GPU prover: run the whole batch on gpu 0.
        let is_memory_enough = unsafe {
            _halo2_fft_normal_check_memory(ntt_type, std::ptr::null(), self.k, self.extended_k)
        };
        if is_memory_enough {
            let batch_b_ffi = get_slice_polys_ffi_out(b_many);
            fft_gpu_many::<F>(
                ntt_type,
                batch_b_ffi,
                self.k,
                self.omega,
                part_power, /*borrow this param slot*/
            )?;
        } else {
            for b in b_many.iter_mut() {
                split_radix_fft_gpu(
                    ntt_type, b, self.k, self.k, self.omega,
                    part_power, /*borrow this param slot*/
                )?;
            }
        }
        Ok(())
    }

    // batched out-of-place iFFT over many polynomials
    pub(crate) fn ifft_many(
        &self,
        a_many: &[Polynomial<F, LagrangeCoeff>],
        b_many: &mut [Polynomial<F, Coeff>],
    ) -> Result<(), HaloGpuError> {
        let get_slice_polys_ffi_in = |polys: &[Polynomial<F, LagrangeCoeff>]| {
            polys
                .iter()
                .map(|poly| FFITraitObject::from_slice(poly.values()))
                .collect::<Vec<FFITraitObject>>()
        };

        let get_slice_polys_ffi_out = |polys: &mut [Polynomial<F, Coeff>]| {
            polys
                .iter()
                .map(|poly| FFITraitObject::from_slice(poly.values()))
                .collect::<Vec<FFITraitObject>>()
        };

        // Single-stream GPU prover: run the whole batch on gpu 0.
        let ntt_type = NttType::iFFT.into();
        let is_memory_enough = unsafe {
            _halo2_fft_normal_check_memory(ntt_type, std::ptr::null(), self.k, self.extended_k)
        };
        if is_memory_enough {
            let batch_a_ffi = get_slice_polys_ffi_in(a_many);
            let batch_b_ffi = get_slice_polys_ffi_out(b_many);
            ifft_gpu_many::<F>(
                ntt_type,
                batch_a_ffi,
                batch_b_ffi,
                self.k,
                self.omega_inv,
                self.ifft_divisor,
            )?;
        } else {
            for (a, b) in a_many.iter().zip(b_many.iter_mut()) {
                split_radix_fft_inout_gpu(
                    ntt_type,
                    a,
                    b,
                    self.k,
                    self.k,
                    self.omega_inv,
                    self.ifft_divisor,
                )?;
            }
        }
        Ok(())
    }

    #[allow(dead_code)]
    fn fft_inner(&self, a: &mut [F], omega: F, log_n: u32) -> Result<(), HaloGpuError> {
        assert_eq!(a.len(), 1 << log_n);

        let ntt_type = NttType::FFT.into(); // FFT
        fft_gpu(ntt_type, a, log_n, omega, F::ONE)?;
        Ok(())
    }

    /// Get the size of the domain
    pub fn k(&self) -> u32 {
        self.k
    }

    /// Get the size of the extended domain
    pub fn extended_k(&self) -> u32 {
        self.extended_k
    }

    /// Get the size of the extended domain
    pub fn extended_len(&self) -> usize {
        1 << self.extended_k
    }

    /// Get $\omega$, the generator of the $2^k$ order multiplicative subgroup.
    pub fn get_omega(&self) -> F {
        self.omega
    }

    /// Get $\omega^{-1}$, the inverse of the generator of the $2^k$ order
    /// multiplicative subgroup.
    pub fn get_omega_inv(&self) -> F {
        self.omega_inv
    }

    /// Get the generator of the extended domain's multiplicative subgroup.
    pub fn get_extended_omega(&self) -> F {
        self.extended_omega
    }

    /// Multiplies a value by some power of $\omega$, essentially rotating over
    /// the domain.
    pub fn rotate_omega(&self, value: F, rotation: Rotation) -> F {
        let mut point = value;
        if rotation.0 >= 0 {
            point *= &self.get_omega().pow_vartime([rotation.0 as u64]);
        } else {
            point *= &self
                .get_omega_inv()
                .pow_vartime([(rotation.0 as i64).unsigned_abs()]);
        }
        point
    }

    /// Computes evaluations (at the point `x`, where `xn = x^n`) of Lagrange
    /// basis polynomials `l_i(X)` defined such that `l_i(omega^i) = 1` and
    /// `l_i(omega^j) = 0` for all `j != i` at each provided rotation `i`.
    ///
    /// # Implementation
    ///
    /// The polynomial
    ///     $$\prod_{j=0,j \neq i}^{n - 1} (X - \omega^j)$$
    /// has a root at all points in the domain except $\omega^i$, where it evaluates to
    ///     $$\prod_{j=0,j \neq i}^{n - 1} (\omega^i - \omega^j)$$
    /// and so we divide that polynomial by this value to obtain $l_i(X)$. Since
    ///     $$\prod_{j=0,j \neq i}^{n - 1} (X - \omega^j)
    ///       = \frac{X^n - 1}{X - \omega^i}$$
    /// then $l_i(x)$ for some $x$ is evaluated as
    ///     $$\left(\frac{x^n - 1}{x - \omega^i}\right)
    ///       \cdot \left(\frac{1}{\prod_{j=0,j \neq i}^{n - 1} (\omega^i - \omega^j)}\right).$$
    /// We refer to
    ///     $$1 \over \prod_{j=0,j \neq i}^{n - 1} (\omega^i - \omega^j)$$
    /// as the barycentric weight of $\omega^i$.
    ///
    /// We know that for $i = 0$
    ///     $$\frac{1}{\prod_{j=0,j \neq i}^{n - 1} (\omega^i - \omega^j)} = \frac{1}{n}.$$
    ///
    /// If we multiply $(1 / n)$ by $\omega^i$ then we obtain
    ///     $$\frac{1}{\prod_{j=0,j \neq 0}^{n - 1} (\omega^i - \omega^j)}
    ///       = \frac{1}{\prod_{j=0,j \neq i}^{n - 1} (\omega^i - \omega^j)}$$
    /// which is the barycentric weight of $\omega^i$.
    pub fn l_i_range<I: IntoIterator<Item = i32> + Clone>(
        &self,
        x: F,
        xn: F,
        rotations: I,
    ) -> Vec<F> {
        let mut results;
        {
            let rotations = rotations.clone().into_iter();
            results = Vec::with_capacity(rotations.size_hint().1.unwrap_or(0));
            for rotation in rotations {
                let rotation = Rotation(rotation);
                let result = x - self.rotate_omega(F::ONE, rotation);
                results.push(result);
            }
            results.iter_mut().batch_invert();
        }

        let common = (xn - F::ONE) * self.barycentric_weight;
        for (rotation, result) in rotations.into_iter().zip(results.iter_mut()) {
            let rotation = Rotation(rotation);
            *result = self.rotate_omega(*result * common, rotation);
        }

        results
    }

    /// Gets the quotient polynomial's degree (as a multiple of n)
    pub fn get_quotient_poly_degree(&self) -> usize {
        self.quotient_poly_degree as usize
    }

    /// Obtain a pinned version of this evaluation domain; a structure with the
    /// minimal parameters needed to determine the rest of the evaluation
    /// domain.
    pub fn pinned(&self) -> PinnedEvaluationDomain<'_, F> {
        PinnedEvaluationDomain {
            k: &self.k,
            extended_k: &self.extended_k,
            omega: &self.omega,
        }
    }

    /// Get the private field `n`
    pub fn get_n(&self) -> u64 {
        self.n
    }

    /// Get the private `fft_data`
    pub fn get_fft_data(&self, l: usize) -> &FFTData<F> {
        self.fft_data
            .get(&l)
            .expect("log_2(l) must be in k..=extended_k")
    }
}

/// Represents the minimal parameters that determine an `EvaluationDomain`.
// Load-bearing: referenced as `domain: PinnedEvaluationDomain<'a, ...>` inside
// `PinnedVerificationKey`, whose Debug output is hashed into `vk.transcript_repr`
// via Blake2b. Removing fields would change the VK transcript representation.
#[allow(dead_code)]
#[derive(Debug)]
pub struct PinnedEvaluationDomain<'a, F: Field> {
    k: &'a u32,
    extended_k: &'a u32,
    omega: &'a F,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    // Test-only helper: only callers are `test_fft`'s round-trip below.
    // Kept inside `mod tests` so the `#[cfg(test)]` gating stays implicit.
    fn icosetfft<F: WithSmallOrderMulGroup<3>>(
        domain: &EvaluationDomain<F>,
        a: &mut [F],
        omega_inv: F,
        log_n: u32,
        divisor: F,
    ) -> Result<(), HaloGpuError> {
        let ntt_type = NttType::iCosetFFT.into();
        let is_memory_enough = unsafe {
            _halo2_fft_normal_check_memory(
                ntt_type,
                a.as_ptr() as *const libc::c_void,
                log_n,
                log_n,
            )
        };
        if is_memory_enough {
            fft_gpu(ntt_type, a, log_n, omega_inv, divisor)?;
        } else {
            split_radix_fft_gpu(ntt_type, a, log_n, log_n, omega_inv, divisor)?;
            domain.distribute_powers_zeta(a, false);
        }
        Ok(())
    }

    #[test]
    fn test_rotate() {
        use rand_core::OsRng;

        use crate::arithmetic::eval_polynomial;
        use halo2curves::bn256::Fr;

        let domain = EvaluationDomain::<Fr>::new(1, 3);
        let rng = OsRng;

        let mut poly = domain.empty_lagrange();
        assert_eq!(poly.len(), 8);
        for value in poly.iter_mut() {
            *value = Fr::random(rng);
        }

        let poly_rotated_cur = poly.rotate(Rotation::cur());
        let poly_rotated_next = poly.rotate(Rotation::next());
        let poly_rotated_prev = poly.rotate(Rotation::prev());

        let poly = domain.lagrange_to_coeff(poly).unwrap();
        let poly_rotated_cur = domain.lagrange_to_coeff(poly_rotated_cur).unwrap();
        let poly_rotated_next = domain.lagrange_to_coeff(poly_rotated_next).unwrap();
        let poly_rotated_prev = domain.lagrange_to_coeff(poly_rotated_prev).unwrap();

        let x = Fr::random(rng);

        assert_eq!(
            eval_polynomial(&poly[..], x),
            eval_polynomial(&poly_rotated_cur[..], x)
        );
        assert_eq!(
            eval_polynomial(&poly[..], x * domain.omega),
            eval_polynomial(&poly_rotated_next[..], x)
        );
        assert_eq!(
            eval_polynomial(&poly[..], x * domain.omega_inv),
            eval_polynomial(&poly_rotated_prev[..], x)
        );
    }

    #[test]
    fn test_l_i() {
        use rand_core::OsRng;

        use crate::arithmetic::{eval_polynomial, lagrange_interpolate};
        use halo2curves::pasta::pallas::Scalar;
        let domain = EvaluationDomain::<Scalar>::new(1, 3);

        let mut l = vec![];
        let mut points = vec![];
        for i in 0..8 {
            points.push(domain.omega.pow([i, 0, 0, 0]));
        }
        for i in 0..8 {
            let mut l_i = vec![Scalar::zero(); 8];
            l_i[i] = Scalar::ONE;
            let l_i = lagrange_interpolate(&points[..], &l_i[..]);
            l.push(l_i);
        }

        let x = Scalar::random(OsRng);
        let xn = x.pow([8, 0, 0, 0]);

        let evaluations = domain.l_i_range(x, xn, -7..=7);
        for i in 0..8 {
            assert_eq!(eval_polynomial(&l[i][..], x), evaluations[7 + i]);
            assert_eq!(eval_polynomial(&l[(8 - i) % 8][..], x), evaluations[7 - i]);
        }
    }

    #[test]
    fn test_power_of_omega() {
        use halo2curves::bn256::Fr as Scalar;
        use rand::thread_rng;

        use crate::cuda::funcs::power_of_omega_gpu;
        let log_n = 31;
        let mut rng = thread_rng();
        let omega = Scalar::random(&mut rng);
        let pow = u32::MAX; //34895;
        println!("omega = {:?}", omega);
        let pow_of_omega_cpu = omega.pow_vartime([pow as u64, 0, 0, 0]);
        let mut omega_lut_cpu = vec![Scalar::one(); (log_n + 1) as usize];
        omega_lut_cpu[1..]
            .iter_mut()
            .enumerate()
            .for_each(|(i, v)| {
                *v = omega.pow_vartime([(1 << i) as u64, 0, 0, 0]);
            });

        let mut omega_lut_gpu = vec![Scalar::zero(); (log_n + 1) as usize];
        let pow_of_omega_gpu = power_of_omega_gpu(omega, &mut omega_lut_gpu, log_n, pow).unwrap();
        assert_eq!(omega_lut_cpu, omega_lut_gpu);
        assert_eq!(pow_of_omega_cpu, pow_of_omega_gpu);

        let _time = Instant::now();
        let _pow_of_omega_gpu = power_of_omega_gpu(omega, &mut omega_lut_gpu, log_n, pow).unwrap();
        println!("gpu_time: {:?}", _time.elapsed());
    }

    #[test]
    fn test_omega_powers_generation() {
        // cargo test --release --package halo2_proofs --lib domain::test_omega_powers_generation -- --nocapture
        use halo2curves::bn256::Fr as Scalar;

        use crate::cuda::funcs::generate_omega_powers_gpu;
        let max_log_n = 26;
        let min_log_n = 10;
        let cutoff_num = 13; // cut off the last 13 elements of omega_powers
        for log_n in min_log_n..=max_log_n {
            let domain = EvaluationDomain::<Scalar>::new(1, log_n);
            let omega = domain.omega;

            let mut omega_powers_cpu = vec![Scalar::zero(); (1 << log_n) as usize];
            parallelize(&mut omega_powers_cpu, |o, start| {
                let mut cur = omega.pow_vartime([start as u64]);
                for v in o.iter_mut() {
                    *v = cur;
                    cur *= &omega;
                }
            });

            let output_num = (1 << log_n) as usize - cutoff_num;
            let mut omega_powers_gpu = vec![Scalar::zero(); output_num];
            generate_omega_powers_gpu(&mut omega_powers_gpu, omega, log_n, output_num as u64)
                .unwrap();
            assert_eq!(
                omega_powers_cpu[0..output_num],
                omega_powers_gpu[0..output_num]
            );

            let cpu_time = Instant::now();
            parallelize(&mut omega_powers_cpu, |o, start| {
                let mut cur = omega.pow_vartime([start as u64]);
                for v in o.iter_mut() {
                    *v = cur;
                    cur *= &omega;
                }
            });
            let cpu_time = cpu_time.elapsed();
            let cpu_micros = f64::from(cpu_time.as_micros() as u32);

            let gpu_time = Instant::now();
            generate_omega_powers_gpu(&mut omega_powers_gpu, omega, log_n, output_num as u64)
                .unwrap();
            let gpu_time = gpu_time.elapsed();
            let gpu_micros = f64::from(gpu_time.as_micros() as u32);

            // NOTE: memory copy from device to host take most of the time (>90%)
            //       this e2e test is not a good benchmark for gpu
            //       just used for correctness check
            println!(
                "  [log_n = {}] cpu_time: {:?}, gpu_time: {:?}, speedup: {}",
                log_n,
                cpu_time,
                gpu_time,
                cpu_micros / gpu_micros
            );
        }
    }

    #[test]
    fn test_omega_lookup_table_generation() {
        // high and low degree lookup table
        use halo2curves::bn256::Fr as Scalar;

        use crate::{
            arithmetic::DENSE_POWER_DEGREE, cpu::arithmetic::generate_omega_lut_cpu,
            cuda::funcs::generate_omega_lut_gpu,
        };

        let max_log_n = 28;
        let min_log_n = 10;

        for log_n in min_log_n..=max_log_n {
            let domain = EvaluationDomain::<Scalar>::new(1, log_n);
            let omega = domain.omega;

            // correctness and warmup
            let cpu_result = generate_omega_lut_cpu(omega, log_n, DENSE_POWER_DEGREE);
            let gpu_result = generate_omega_lut_gpu(omega, log_n).unwrap();
            assert_eq!(cpu_result, gpu_result);

            let cpu_time = Instant::now();
            let _cpu_result = generate_omega_lut_cpu(omega, log_n, DENSE_POWER_DEGREE);
            let cpu_time = cpu_time.elapsed();
            let cpu_micros = f64::from(cpu_time.as_micros() as u32);

            let gpu_time = Instant::now();
            let _gpu_result = generate_omega_lut_gpu(omega, log_n).unwrap();
            let gpu_time = gpu_time.elapsed();
            let gpu_micros = f64::from(gpu_time.as_micros() as u32);

            println!(
                "  [log_n = {}] cpu_time: {:?}, gpu_time: {:?}, speedup: {}",
                log_n,
                cpu_time,
                gpu_time,
                cpu_micros / gpu_micros
            );
        }
    }

    #[test]
    #[ignore = "expensive"]
    fn test_fft() {
        use halo2curves::bn256::Fr as Scalar;
        use rand::thread_rng;

        use crate::arithmetic::best_fft;

        let max_log_n = 25; // for the sake of ci speed, not set to 26
        let min_log_n = 10;
        let mut rng = thread_rng();
        let a = (0..(1 << max_log_n))
            .map(|_i| Scalar::random(&mut rng))
            .collect::<Vec<_>>();
        println!("----------test FFT---------");
        let ntt_type = NttType::FFT.into();
        for log_n in min_log_n..=max_log_n {
            let domain = EvaluationDomain::<Scalar>::new(1, log_n);
            let mut a0 = a[0..(1 << log_n)].to_vec();
            let mut a1 = a0.clone();

            // warm up & correct test & init gpu twiddle
            fft_gpu(ntt_type, &mut a1, log_n, domain.omega, Scalar::one()).unwrap();
            let data = domain.get_fft_data(a0.len());
            best_fft(&mut a0, domain.omega, log_n, data, false);
            assert_eq!(a0, a1);

            let gpu_time = Instant::now();
            fft_gpu(ntt_type, &mut a1, log_n, domain.omega, Scalar::one()).unwrap();
            let gpu_time = gpu_time.elapsed();
            let gpu_micros = f64::from(gpu_time.as_micros() as u32);

            let cpu_time = Instant::now();
            let data = domain.get_fft_data(a0.len());
            best_fft(&mut a0, domain.omega, log_n, data, false);
            let cpu_time = cpu_time.elapsed();
            let cpu_micros = f64::from(cpu_time.as_micros() as u32);

            println!(
                "  [log_n = {}] cpu_time: {:?}, gpu_time: {:?}, speedup: {}",
                log_n,
                cpu_time,
                gpu_time,
                cpu_micros / gpu_micros
            );
        }

        println!("----------test iFFT---------");
        let _ntt_type: u32 = NttType::iFFT.into();
        for log_n in min_log_n..=max_log_n {
            let domain = EvaluationDomain::<Scalar>::new(1, log_n);
            let mut a0 = a[0..(1 << log_n)].to_vec();
            let mut a1 = a0.clone();

            // warm up & correct test & init gpu iFFT twiddle
            let data = domain.get_fft_data(a1.len());
            best_fft(&mut a1, domain.omega_inv, log_n, data, true);
            parallelize(&mut a1, |a, _| {
                for a in a {
                    *a *= &domain.ifft_divisor;
                }
            });
            domain
                .ifft(&mut a0, domain.omega_inv, log_n, domain.ifft_divisor)
                .unwrap();
            assert_eq!(a1, a0);

            let cpu_time = Instant::now();
            let data = domain.get_fft_data(a1.len());
            best_fft(&mut a1, domain.omega_inv, log_n, data, true);
            parallelize(&mut a1, |a, _| {
                for a in a {
                    *a *= &domain.ifft_divisor;
                }
            });
            let cpu_time = cpu_time.elapsed();
            let cpu_micros = f64::from(cpu_time.as_micros() as u32);

            let gpu_time = Instant::now();
            domain
                .ifft(&mut a0, domain.omega_inv, log_n, domain.ifft_divisor)
                .unwrap();
            let gpu_time = gpu_time.elapsed();
            let gpu_micros = f64::from(gpu_time.as_micros() as u32);

            println!(
                "  [log_n = {}] cpu_time: {:?}, gpu_time: {:?}, speedup: {}",
                log_n,
                cpu_time,
                gpu_time,
                cpu_micros / gpu_micros
            );
        }

        let max_log_n = 25; // for the sake of ci speed, not set to 26
        println!("----------test cosetFFT---------");
        let _ntt_type: u32 = NttType::CosetFFT.into();
        for log_n in min_log_n..=max_log_n {
            let domain = EvaluationDomain::<Scalar>::new(5, log_n);
            let a0 = a[0..(1 << log_n)].to_vec();
            let a1 = a0.clone();
            let mut b = a0.clone();
            b.resize(domain.extended_len(), Scalar::zero());

            // warm up & correct test & init gpu cosetFFT twiddle
            let mut c = a1.to_vec();
            c.resize(domain.extended_len(), Scalar::zero());
            domain.distribute_powers_zeta(&mut c, true);
            let data = domain.get_fft_data(c.len());
            best_fft(
                &mut c,
                domain.extended_omega,
                domain.extended_k,
                data,
                false,
            );
            EvaluationDomain::cosetfft(
                &domain,
                &a0,
                &mut b,
                domain.extended_omega,
                domain.k,
                domain.extended_k,
            )
            .unwrap();
            assert_eq!(c, b);

            let cpu_time = Instant::now();
            let mut c = a1.to_vec();
            c.resize(domain.extended_len(), Scalar::zero());
            domain.distribute_powers_zeta(&mut c, true);
            let data = domain.get_fft_data(c.len());
            best_fft(
                &mut c,
                domain.extended_omega,
                domain.extended_k,
                data,
                false,
            );
            let cpu_time = cpu_time.elapsed();
            let cpu_micros = f64::from(cpu_time.as_micros() as u32);

            let gpu_time = Instant::now();
            EvaluationDomain::cosetfft(
                &domain,
                &a0,
                &mut b,
                domain.extended_omega,
                domain.k,
                domain.extended_k,
            )
            .unwrap();
            let gpu_time = gpu_time.elapsed();
            let gpu_micros = f64::from(gpu_time.as_micros() as u32);

            println!(
                "  [extended_log_n = {}] cpu_time: {:?}, gpu_time: {:?}, speedup: {}",
                domain.extended_k,
                cpu_time,
                gpu_time,
                cpu_micros / gpu_micros
            );
        }

        println!("----------test icosetFFT---------");
        let _ntt_type: u32 = NttType::iCosetFFT.into(); // icosetFFT
        for log_n in min_log_n..=max_log_n {
            let domain = EvaluationDomain::<Scalar>::new(5, log_n);
            let a0 = a[0..(1 << log_n)].to_vec();
            let a1 = a0.clone();

            // warm up & correct test & init gpu icosetFFT twiddle
            let mut c = a1.to_vec();
            c.resize(domain.extended_len(), Scalar::zero());
            let data = domain.get_fft_data(c.len());
            best_fft(
                &mut c,
                domain.extended_omega_inv,
                domain.extended_k,
                data,
                true,
            );
            parallelize(&mut c, |c, _| {
                for c in c {
                    *c *= &domain.extended_ifft_divisor;
                }
            });
            domain.distribute_powers_zeta(&mut c, false);
            let mut b = a0.clone();
            b.resize(domain.extended_len(), Scalar::zero());
            icosetfft(
                &domain,
                &mut b,
                domain.extended_omega_inv,
                domain.extended_k,
                domain.extended_ifft_divisor,
            )
            .unwrap();
            assert_eq!(c, b);

            let cpu_time = Instant::now();
            let mut c = a1.to_vec();
            c.resize(domain.extended_len(), Scalar::zero());
            let data = domain.get_fft_data(c.len());
            best_fft(
                &mut c,
                domain.extended_omega_inv,
                domain.extended_k,
                data,
                true,
            );
            parallelize(&mut c, |c, _| {
                for c in c {
                    *c *= &domain.extended_ifft_divisor;
                }
            });
            domain.distribute_powers_zeta(&mut c, false);
            let cpu_time = cpu_time.elapsed();
            let cpu_micros = f64::from(cpu_time.as_micros() as u32);

            let gpu_time = Instant::now();
            let mut b = a0.clone();
            b.resize(domain.extended_len(), Scalar::zero());
            icosetfft(
                &domain,
                &mut b,
                domain.extended_omega_inv,
                domain.extended_k,
                domain.extended_ifft_divisor,
            )
            .unwrap();
            let gpu_time = gpu_time.elapsed();
            let gpu_micros = f64::from(gpu_time.as_micros() as u32);

            println!(
                "  [extended_log_n = {}] cpu_time: {:?}, gpu_time: {:?}, speedup: {}",
                domain.extended_k,
                cpu_time,
                gpu_time,
                cpu_micros / gpu_micros
            );
        }
    }

    #[test]
    #[ignore = "expensive"]
    fn test_ifft_many() {
        use halo2curves::bn256::Fr as Scalar;
        use rand::thread_rng;

        let min_log_n = 10;
        let max_log_n = 26;
        let mut rng = thread_rng();
        let a = (0..(1 << max_log_n))
            .map(|_| Scalar::random(&mut rng))
            .collect::<Vec<_>>();

        println!("----------test iFFT---------\n");
        let ntt_type = NttType::iFFT.into();
        for log_n in min_log_n..=max_log_n {
            let a0 = a[0..(1 << log_n)].to_vec();
            let domain = EvaluationDomain::<Scalar>::new(1, log_n);

            // many data
            const TASK_NUM: usize = 15;
            let mut a0_many: Vec<Polynomial<Scalar, LagrangeCoeff>> = Vec::with_capacity(TASK_NUM);
            for _ in 0..TASK_NUM {
                a0_many.push(Polynomial::new(a0.clone()));
            }
            let mut result_many: Vec<Polynomial<Scalar, Coeff>> = Vec::with_capacity(TASK_NUM);
            for _ in 0..TASK_NUM {
                let mut result: Vec<Scalar> = Vec::with_capacity(domain.extended_len());
                unsafe {
                    result.set_len(domain.extended_len());
                }
                result_many.push(Polynomial::new(result));
            }

            // base
            let mut result_base = a0.clone();
            fft_gpu(
                ntt_type,
                &mut result_base,
                log_n,
                domain.omega_inv,
                domain.ifft_divisor,
            )
            .unwrap();
            // many
            EvaluationDomain::ifft_many(&domain, &a0_many, &mut result_many).unwrap();
            // check result
            for result in result_many.iter() {
                assert_eq!(result.values(), result_base);
            }

            //single iFFT task: use gpu_0
            for i in 0..TASK_NUM {
                let _a1 = a0.clone();
                let single_gpu_time = Instant::now();
                fft_gpu(
                    ntt_type,
                    &mut result_base,
                    log_n,
                    domain.omega_inv,
                    domain.ifft_divisor,
                )
                .unwrap();
                let single_gpu_time = single_gpu_time.elapsed();
                println!("single GPU ifft[{}] elapsed time: {:?}", i, single_gpu_time);
            }

            // batched iFFT task
            let batched_time = Instant::now();
            EvaluationDomain::ifft_many(&domain, &a0_many, &mut result_many).unwrap();
            let batched_time = batched_time.elapsed();
            println!(
                "[log_n = {}] batched iFFTx[{}] time: {:?}",
                log_n, TASK_NUM, batched_time
            );
        }
    }

    #[test]
    fn test_cosetfft_many() {
        // param
        let log_n = 23;
        use halo2curves::bn256::Fr as Scalar;
        use rand::thread_rng;

        use crate::arithmetic::best_fft;
        let _ntt_type: u32 = NttType::CosetFFT.into();
        let domain = EvaluationDomain::<Scalar>::new(5, log_n);

        println!("----------test cosetFFT many---------\n");
        println!(
            "[log_n = {}] domain.k = {} domain.extended_k = {}",
            log_n, domain.k, domain.extended_k
        );

        // data
        let mut rng = thread_rng();
        let a0 = (0..(1 << domain.k))
            .map(|_i| Scalar::random(&mut rng))
            .collect::<Vec<_>>();

        // many data
        const TASK_NUM: usize = 15;
        let mut a0_many: Vec<Polynomial<Scalar, Coeff>> = Vec::with_capacity(TASK_NUM);
        for _ in 0..TASK_NUM {
            a0_many.push(Polynomial::new(a0.clone()));
        }
        let mut result_many: Vec<Polynomial<Scalar, ExtendedLagrangeCoeff>> =
            Vec::with_capacity(TASK_NUM);
        for _ in 0..TASK_NUM {
            let mut result: Vec<Scalar> = Vec::with_capacity(domain.extended_len());
            unsafe {
                result.set_len(domain.extended_len());
            }
            result_many.push(Polynomial::new(result));
        }

        // cpu result
        let mut cpu_result = vec![Scalar::zero(); domain.extended_len()];
        let mut a_extended = a0.to_vec();
        a_extended.resize(domain.extended_len(), Scalar::zero());
        cpu_result.clone_from_slice(&a_extended);
        domain.distribute_powers_zeta(&mut cpu_result, true);
        // EvaluationDomain::fft(&mut cpu_result, domain.extended_omega, domain.extended_k);
        let data = domain.get_fft_data(cpu_result.len());
        best_fft(
            &mut cpu_result,
            domain.extended_omega,
            domain.extended_k,
            data,
            false,
        );

        println!("get cpu results ... done");

        // result_many
        EvaluationDomain::cosetfft_many(&domain, &a0_many, &mut result_many).unwrap();
        // check result
        for result in result_many.iter() {
            assert_eq!(result.values(), cpu_result);
        }
        println!("assert batched results ... done");

        //single iFFT task
        let mut single_result = vec![Scalar::zero(); domain.extended_len()];
        for i in 0..TASK_NUM {
            let single_gpu_time = Instant::now();
            EvaluationDomain::cosetfft(
                &domain,
                &a0,
                &mut single_result,
                domain.extended_omega,
                domain.k,
                domain.extended_k,
            )
            .unwrap(); // note: use gpu_0 internal
            let single_gpu_time = single_gpu_time.elapsed();
            println!(
                "single GPU cosetfft[{}] elapsed time: {:?}",
                i, single_gpu_time
            );
        }
        assert_eq!(single_result, cpu_result);

        // batched cosetFFT task
        let batched_time = Instant::now();
        EvaluationDomain::cosetfft_many(&domain, &a0_many, &mut result_many).unwrap();
        let batched_time = batched_time.elapsed();
        println!("batched cosetFFTx[{}] time: {:?}", TASK_NUM, batched_time);
    }

    #[test]
    fn test_coeff_to_extended_part() {
        use halo2curves::bn256::Fr as Scalar;
        use rand_core::OsRng;

        for k in 5..20 {
            let domain = EvaluationDomain::<Scalar>::new(3, k);

            let mut poly = domain.empty_coeff();
            for value in poly.iter_mut() {
                *value = Scalar::random(&mut OsRng);
            }

            let got = {
                let parts = domain.coeff_to_extended_parts(&poly).unwrap();
                domain.extended_from_lagrange_vec(parts)
            };
            let expected = domain.coeff_to_extended(&poly).unwrap();
            assert_eq!(expected.values(), got.values());
        }
    }

    #[test]
    fn test_coeff_to_extended_part_many() {
        use halo2curves::bn256::Fr as Scalar;
        use rand::thread_rng;

        let mut rng = thread_rng();
        let batch_size = 8;
        let k = 20;
        let domain = EvaluationDomain::<Scalar>::new(5, k);

        let mut polys = vec![];
        let mut poly = domain.empty_coeff();
        for value in poly.iter_mut() {
            *value = Scalar::random(&mut rng);
        }
        for _ in 0..batch_size {
            polys.push(poly.clone());
        }

        let mut extended_omega_factor = Scalar::one();
        let num_parts = domain.extended_len() >> domain.k;
        for part_idx in 0..num_parts {
            let batched_time = Instant::now();
            let part_many = domain
                .coeff_to_extended_part_many(polys.iter().collect(), extended_omega_factor)
                .unwrap();
            let batched_time = batched_time.elapsed();

            let sequential_time = Instant::now();
            let parts = (0..batch_size)
                .map(|i| {
                    domain
                        .coeff_to_extended_part(polys[i].clone(), extended_omega_factor)
                        .unwrap()
                })
                .collect::<Vec<_>>();
            let sequential_time = sequential_time.elapsed();
            for i in 0..batch_size {
                assert_eq!(parts[i].values(), part_many[i].values());
            }
            extended_omega_factor *= domain.extended_omega;

            println!(
                "part[{}], batched time[{:?}], sequential time[{:?}], speedup: {}",
                part_idx,
                batched_time,
                sequential_time,
                sequential_time.as_secs_f64() / batched_time.as_secs_f64()
            );
        }
    }

    #[test]
    fn bench_coeff_to_extended_parts() {
        use halo2curves::bn256::Fr as Scalar;
        use rand::thread_rng;
        use std::time::Instant;

        let k = 20;
        let mut rng = thread_rng();
        let domain = EvaluationDomain::<Scalar>::new(3, k);

        let mut poly = domain.empty_coeff();
        for value in poly.iter_mut() {
            *value = Scalar::random(&mut rng);
        }

        let coeff_to_extended_timer = Instant::now();
        let expected = domain.coeff_to_extended(&poly).unwrap();
        println!(
            "domain.coeff_to_extended(k = {}) time: {}s",
            k,
            coeff_to_extended_timer.elapsed().as_secs_f64()
        );

        let coeff_to_extended_parts_timer = Instant::now();
        let parts = domain.coeff_to_extended_parts(&poly).unwrap();
        let got = domain.extended_from_lagrange_vec(parts);
        println!(
            "domain.coeff_to_extended_parts(k = {}) time: {}s",
            k,
            coeff_to_extended_parts_timer.elapsed().as_secs_f64()
        );

        assert_eq!(got.values(), expected.values());
    }

    #[test]
    fn test_modular_fft() {
        use halo2curves::bn256::Fr as Scalar;

        // Exercises `lagrange_to_extend_part` (iFFT + CosetFFT_Part fused)
        // against the unfused two-step path.

        let max_log_n = 25;
        let min_log_n = 20;
        let a = (0..(1 << max_log_n))
            .map(|i| Scalar::from(i as u64))
            .collect::<Vec<_>>();

        println!("----------test iFFT + cosetFFT_part---------");
        for log_n in min_log_n..=max_log_n {
            let domain = EvaluationDomain::<Scalar>::new(1, log_n);
            let a0 = Polynomial::<Scalar, LagrangeCoeff>::new(a[0..(1 << log_n)].to_vec());

            let mut rng = rand::thread_rng();
            let extended_omega_factor = Scalar::random(&mut rng);

            // normal: ifft >>> coset_fft_part (two-step CPU+GPU)
            let a1 = Polynomial::<Scalar, LagrangeCoeff>::new(a0.values().to_vec());
            let b0 = domain.lagrange_to_coeff(a1).unwrap();
            let b0 = domain
                .coeff_to_extended_part(b0, extended_omega_factor)
                .unwrap();

            // fused: ifft_cosetfft_part on dense input
            let b1 = domain
                .lagrange_to_extend_part(&a0, extended_omega_factor)
                .unwrap();
            assert_eq!(b0.values(), b1.values());

            // benchmark
            let normal_time = Instant::now();
            let a1 = Polynomial::<Scalar, LagrangeCoeff>::new(a0.values().to_vec());
            let b0 = domain.lagrange_to_coeff(a1).unwrap();
            let _ = domain
                .coeff_to_extended_part(b0, extended_omega_factor)
                .unwrap();
            let normal_time = normal_time.elapsed();
            let normal_micros = f64::from(normal_time.as_micros() as u32);

            let fused_time = Instant::now();
            let _ = domain
                .lagrange_to_extend_part(&a0, extended_omega_factor)
                .unwrap();
            let fused_time = fused_time.elapsed();
            let fused_micros = f64::from(fused_time.as_micros() as u32);

            println!(
                "[log_n = {}] normal_time: {:?}, fused_time: {:?}, speedup: 1 / {:.3}",
                log_n,
                normal_time,
                fused_time,
                normal_micros / fused_micros
            );
        }
    }
}
