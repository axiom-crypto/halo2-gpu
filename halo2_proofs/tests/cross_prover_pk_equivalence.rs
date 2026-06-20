//! Cross-prover proving-key equivalence (the runtime lock).
//!
//! A proving key produced by `halo2-axiom`'s CPU `keygen_pk` is the canonical,
//! serde-source-of-truth `ProvingKey`. This test proves, end to end, that such
//! a key — serialized to bytes by the CPU side — is consumable by the GPU
//! prover after a `GpuProvingKey::from_host` rebuild, and that the resulting
//! proof verifies.
//!
//! Both circuits share ONE SRS: the GPU `ParamsKZG` is generated once, written
//! to bytes, and read back into a `halo2-axiom` `ParamsKZG` for CPU keygen. A
//! fresh `ParamsKZG::new` per side would draw a different toxic `s` and break
//! the shared-SRS requirement, so the bytes round-trip is load-bearing.
//!
//! The circuit is a single-region multiplication gate `q * (a * b - c)` with a
//! public input that ties advice `a` to the instance column (a copy constraint
//! that activates the permutation argument). It implements both
//! `halo2_axiom::plonk::Circuit<Fr>` (for CPU keygen) and
//! `halo2_axiom_gpu::plonk::Circuit<Fr>` (for the GPU prover), with byte-
//! identical configure/synthesize bodies re-typed against each crate.

use halo2_axiom_gpu::circuit::{Layouter, SimpleFloorPlanner, Value};
use halo2_axiom_gpu::plonk::{
    create_proof, verify_proof, Advice, Circuit, Column, ConstraintSystem, Error, Fixed,
    GpuProvingKey, Instance, VerifyingKey,
};
use halo2_axiom_gpu::poly::commitment::{ParamsProver, Verifier};
use halo2_axiom_gpu::poly::kzg::commitment::{KZGCommitmentScheme, ParamsKZG};
use halo2_axiom_gpu::poly::kzg::multiopen::{ProverSHPLONK, VerifierSHPLONK};
use halo2_axiom_gpu::poly::kzg::strategy::AccumulatorStrategy;
use halo2_axiom_gpu::poly::{Rotation, VerificationStrategy};
use halo2_axiom_gpu::transcript::{
    Blake2bRead, Blake2bWrite, Challenge255, TranscriptReadBuffer, TranscriptWriterBuffer,
};
use halo2curves::bn256::{Bn256, Fr, G1Affine};
use rand_core::OsRng;

/// `1 << K` rows. K >= 14 clears `GPU_MSM_THRESHOLD` so the real GPU MSM path
/// runs (rather than the small-circuit CPU-multiexp fallback).
const K: u32 = 14;

/// GPU-side single-region multiplication circuit (drives `create_proof`).
#[derive(Clone)]
struct MulCircuit {
    /// Public input; tied to advice `a` at row 0.
    public: Value<Fr>,
    /// Private multiplicand.
    b: Value<Fr>,
}

#[derive(Clone)]
struct MulConfig {
    a: Column<Advice>,
    b: Column<Advice>,
    c: Column<Advice>,
    q: Column<Fixed>,
    instance: Column<Instance>,
}

impl Circuit<Fr> for MulCircuit {
    type Config = MulConfig;
    type FloorPlanner = SimpleFloorPlanner;
    #[cfg(feature = "circuit-params")]
    type Params = ();

    fn without_witnesses(&self) -> Self {
        Self {
            public: Value::unknown(),
            b: Value::unknown(),
        }
    }

    fn configure(meta: &mut ConstraintSystem<Fr>) -> MulConfig {
        let a = meta.advice_column();
        let b = meta.advice_column();
        let c = meta.advice_column();
        let q = meta.fixed_column();
        let instance = meta.instance_column();

        meta.enable_equality(a);
        meta.enable_equality(instance);

        meta.create_gate("mul", |meta| {
            let a = meta.query_advice(a, Rotation::cur());
            let b = meta.query_advice(b, Rotation::cur());
            let c = meta.query_advice(c, Rotation::cur());
            let q = meta.query_fixed(q, Rotation::cur());
            // Degree 3 (q * a * b), so `cs.degree() >= 3`.
            vec![q * (a * b - c)]
        });

        MulConfig {
            a,
            b,
            c,
            q,
            instance,
        }
    }

    fn synthesize(&self, config: MulConfig, mut layouter: impl Layouter<Fr>) -> Result<(), Error> {
        layouter.assign_region(
            || "mul",
            |mut region| {
                // Tie advice `a[0]` to public `instance[0]`: a copy constraint
                // that populates the permutation argument.
                region.assign_advice_from_instance(
                    || "a = public",
                    config.instance,
                    0,
                    config.a,
                    0,
                )?;
                region.assign_advice(config.b, 0, self.b);
                let c_val = self.public.zip(self.b).map(|(a, b)| a * b);
                region.assign_advice(config.c, 0, c_val);
                region.assign_fixed(config.q, 0, Fr::from(1));
                Ok(())
            },
        )
    }
}

/// CPU (halo2-axiom) circuit variant + canonical keygen. Byte-identical to the
/// GPU `MulCircuit` above, re-typed against `halo2_axiom::{plonk, circuit}` so
/// `halo2_axiom::keygen_pk` (which needs a `halo2_axiom::Circuit`) resolves.
mod cpu {
    use halo2_axiom::circuit::{Layouter, SimpleFloorPlanner, Value};
    use halo2_axiom::plonk::{
        keygen_pk, keygen_vk, Advice, Circuit, Column, ConstraintSystem, Error, Fixed, Instance,
        ProvingKey,
    };
    use halo2_axiom::poly::kzg::commitment::ParamsKZG;
    use halo2_axiom::poly::Rotation;
    use halo2_axiom::SerdeFormat;
    use halo2curves::bn256::{Bn256, Fr, G1Affine};

    #[derive(Clone)]
    pub struct MulCircuit {
        pub public: Value<Fr>,
        pub b: Value<Fr>,
    }

    #[derive(Clone)]
    pub struct MulConfig {
        a: Column<Advice>,
        b: Column<Advice>,
        c: Column<Advice>,
        q: Column<Fixed>,
        instance: Column<Instance>,
    }

    impl Circuit<Fr> for MulCircuit {
        type Config = MulConfig;
        type FloorPlanner = SimpleFloorPlanner;
        #[cfg(feature = "circuit-params")]
        type Params = ();

        fn without_witnesses(&self) -> Self {
            Self {
                public: Value::unknown(),
                b: Value::unknown(),
            }
        }

        fn configure(meta: &mut ConstraintSystem<Fr>) -> MulConfig {
            let a = meta.advice_column();
            let b = meta.advice_column();
            let c = meta.advice_column();
            let q = meta.fixed_column();
            let instance = meta.instance_column();

            meta.enable_equality(a);
            meta.enable_equality(instance);

            meta.create_gate("mul", |meta| {
                let a = meta.query_advice(a, Rotation::cur());
                let b = meta.query_advice(b, Rotation::cur());
                let c = meta.query_advice(c, Rotation::cur());
                let q = meta.query_fixed(q, Rotation::cur());
                vec![q * (a * b - c)]
            });

            MulConfig {
                a,
                b,
                c,
                q,
                instance,
            }
        }

        fn synthesize(
            &self,
            config: MulConfig,
            mut layouter: impl Layouter<Fr>,
        ) -> Result<(), Error> {
            layouter.assign_region(
                || "mul",
                |mut region| {
                    region.assign_advice_from_instance(
                        || "a = public",
                        config.instance,
                        0,
                        config.a,
                        0,
                    )?;
                    region.assign_advice(config.b, 0, self.b);
                    let c_val = self.public.zip(self.b).map(|(a, b)| a * b);
                    region.assign_advice(config.c, 0, c_val);
                    region.assign_fixed(config.q, 0, Fr::from(1));
                    Ok(())
                },
            )
        }
    }

    /// Reads the SHARED SRS bytes into a halo2-axiom `ParamsKZG`, then runs the
    /// canonical CPU `keygen_vk`/`keygen_pk` on an empty (witness-free) circuit.
    pub fn build_pk(srs_bytes: &[u8]) -> ProvingKey<G1Affine> {
        let params =
            ParamsKZG::<Bn256>::read_custom(&mut &srs_bytes[..], SerdeFormat::RawBytesUnchecked)
                .expect("read shared SRS into halo2-axiom ParamsKZG");
        let circuit = MulCircuit {
            public: Value::unknown(),
            b: Value::unknown(),
        };
        let vk = keygen_vk(&params, &circuit).expect("cpu keygen_vk");
        keygen_pk(&params, vk, &circuit).expect("cpu keygen_pk")
    }
}

/// Runs the GPU verifier and finalizes the strategy. Standalone-generic over
/// the `Verifier`/`VerificationStrategy` pair so `finalize` can resolve the
/// verifier impl (`AccumulatorStrategy` implements `VerificationStrategy` for
/// every verifier, so the choice must be pinned at the call site).
fn gpu_verify<'params, V, Strategy>(
    params: &'params ParamsKZG<Bn256>,
    vk: &VerifyingKey<G1Affine>,
    instances: &[&[&[Fr]]],
    proof: &[u8],
) -> bool
where
    V: Verifier<'params, KZGCommitmentScheme<Bn256>>,
    Strategy: VerificationStrategy<'params, KZGCommitmentScheme<Bn256>, V, Output = Strategy>,
{
    let strategy = Strategy::new(params);
    let mut transcript = Blake2bRead::<_, _, Challenge255<_>>::init(proof);
    let strategy = verify_proof::<KZGCommitmentScheme<Bn256>, V, _, _, _>(
        params,
        vk,
        strategy,
        instances,
        &mut transcript,
    )
    .expect("gpu verify_proof");
    strategy.finalize()
}

#[test]
fn cross_prover_pk_bytes_equivalence() {
    // 1. ONE SRS, generated once on the GPU side and shared with CPU keygen via
    //    a byte round-trip (identical serde layout across the two forks).
    let gpu_params = ParamsKZG::<Bn256>::setup(K, OsRng);
    let mut srs_bytes = Vec::new();
    gpu_params
        .write_custom(
            &mut srs_bytes,
            halo2_axiom_gpu::SerdeFormat::RawBytesUnchecked,
        )
        .expect("write shared SRS");

    // 2. Canonical CPU keygen on the shared SRS, then serialize the pk.
    let cpu_pk = cpu::build_pk(&srs_bytes);
    let fmt = halo2_axiom::SerdeFormat::RawBytes;
    let bytes = cpu_pk.to_bytes(fmt);

    // 3. Serde-identity guard: wrapping the CPU pk in a GpuProvingKey serializes
    //    to byte-identical output (the serde delegates to the canonical pk).
    let gpk_guard = GpuProvingKey::<G1Affine>::from_host(cpu_pk.clone());
    assert_eq!(
        gpk_guard.to_bytes(fmt),
        bytes,
        "GpuProvingKey serialization must be byte-identical to the canonical pk"
    );

    // 4. Substantive lock: read the CPU-serialized pk back into a canonical
    //    ProvingKey, wrap it for the GPU prover, then prove + verify.
    let inner = {
        #[cfg(feature = "circuit-params")]
        let pk = halo2_axiom::plonk::ProvingKey::<G1Affine>::read::<_, cpu::MulCircuit>(
            &mut &bytes[..],
            fmt,
            (),
        );
        #[cfg(not(feature = "circuit-params"))]
        let pk = halo2_axiom::plonk::ProvingKey::<G1Affine>::read::<_, cpu::MulCircuit>(
            &mut &bytes[..],
            fmt,
        );
        pk.expect("read canonical ProvingKey from CPU-serialized bytes")
    };
    // GPU prove with concrete witnesses: public = 7, b = 3, so c = a*b = 21.
    let public = Fr::from(7);
    let b = Fr::from(3);
    let circuit = MulCircuit {
        public: Value::known(public),
        b: Value::known(b),
    };
    let pubinputs = [public];
    let instances: &[&[&[Fr]]] = &[&[&pubinputs[..]]];

    let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
    create_proof::<KZGCommitmentScheme<Bn256>, ProverSHPLONK<_>, _, _, _, _>(
        &gpu_params,
        &inner,
        std::slice::from_ref(&circuit),
        instances,
        OsRng,
        &mut transcript,
    )
    .expect("gpu create_proof");
    let proof = transcript.finalize();

    // GPU verify with the canonical vk (the verifier rebuilds GpuVerifyingKey).
    let verifier_params = gpu_params.verifier_params();
    assert!(
        gpu_verify::<VerifierSHPLONK<_>, AccumulatorStrategy<_>>(
            verifier_params,
            inner.get_vk(),
            instances,
            &proof[..],
        ),
        "GPU proof from a CPU-serialized proving key must verify"
    );
}
