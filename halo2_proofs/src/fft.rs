//! halo2-axiom-gpu re-exports the canonical FFT modules from halo2-axiom.
//! The local fork carried verbatim copies of the dispatcher and submodules;
//! consolidation lets the GPU build share the upstream implementations.
pub use halo2_axiom::fft::{baseline, fft, parallel, recursive};

#[cfg(test)]
mod tests {
    use ark_std::{end_timer, start_timer};
    use ff::Field;
    use halo2curves::bn256::Fr as Scalar;
    use rand_core::OsRng;

    use crate::{arithmetic::best_fft, fft, multicore, poly::EvaluationDomain};
    use halo2_axiom::poly::EvaluationDomain as EvaluationDomainCPU;

    #[test]
    fn test_fft_recursive() {
        let k = 22;

        let domain = EvaluationDomainCPU::<Scalar>::new(1, k);
        let domain = EvaluationDomain::from_host_domain(&domain);
        let n = domain.get_n() as usize;

        let input = vec![Scalar::random(OsRng); n];

        let num_threads = multicore::current_num_threads();

        let mut a = input.clone();
        let l_a = a.len();
        let start = start_timer!(|| format!("best fft {} ({})", a.len(), num_threads));
        fft::baseline::fft(
            &mut a,
            domain.get_omega(),
            k,
            domain.get_fft_data(l_a),
            false,
        );
        end_timer!(start);

        let mut b = input;
        let l_b = b.len();
        let start = start_timer!(|| format!("recursive fft {} ({})", a.len(), num_threads));
        fft::recursive::fft(
            &mut b,
            domain.get_omega(),
            k,
            domain.get_fft_data(l_b),
            false,
        );
        end_timer!(start);

        for i in 0..n {
            assert_eq!(a[i], b[i]);
        }
    }

    #[test]
    fn test_ifft_recursive() {
        let k = 22;

        let domain = EvaluationDomainCPU::<Scalar>::new(1, k);
        let domain = EvaluationDomain::from_host_domain(&domain);
        let n = domain.get_n() as usize;

        let input = vec![Scalar::random(OsRng); n];

        let mut a = input.clone();
        let l_a = a.len();
        fft::recursive::fft(
            &mut a,
            domain.get_omega(),
            k,
            domain.get_fft_data(l_a),
            false,
        );
        fft::recursive::fft(
            &mut a,
            domain.get_omega_inv(), // doesn't actually do anything
            k,
            domain.get_fft_data(l_a),
            true,
        );
        let ifft_divisor = Scalar::from(n as u64).invert().unwrap();

        for i in 0..n {
            assert_eq!(input[i], a[i] * ifft_divisor);
        }
    }

    #[test]
    fn test_mem_leak() {
        let k = 3;
        let domain = EvaluationDomainCPU::<Scalar>::new(1, k);
        let domain = EvaluationDomain::from_host_domain(&domain);
        let omega = domain.get_omega();
        let l = 1 << k;
        let data = domain.get_fft_data(l);
        let mut a = (0..(1 << k))
            .map(|_| Scalar::random(OsRng))
            .collect::<Vec<_>>();

        best_fft(&mut a, omega, k, data, false);
    }
}
