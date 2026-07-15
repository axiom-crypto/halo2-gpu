//! End-to-end GPU prove/verify over a circuit that drives the FULL phase-5
//! eval path: instance/advice/fixed evals, the permutation argument (copy
//! constraints + global constant), AND a lookup argument.
//!
//! This is the end-to-end guard for the M2 phase-5 device-out batch-eval
//! rewiring. `verify_proof` reads the phase-5 evaluations in a fixed,
//! protocol-defined order; if the prover's device-out batching wrote any eval
//! into the wrong slot (wrong point, wrong poly, or wrong order), the verifier
//! reads mismatched scalars and the polynomial-identity checks FAIL. So a
//! passing prove→verify roundtrip proves the batched eval order is preserved.
//!
//! The workspace otherwise lacks a lookup prove/verify test (`plonk_api` is
//! `#[ignore]`d), so this specifically covers the lookup `evaluate()` rewiring;
//! `cross_prover_pk_equivalence` already covers the permutation rewiring.

use halo2_axiom_gpu::circuit::{Layouter, SimpleFloorPlanner, Value};
use halo2_axiom_gpu::plonk::{
    create_proof, keygen_pk, keygen_vk, verify_proof, Advice, Circuit, Column, ConstraintSystem,
    Error, Fixed, Instance, Selector, TableColumn, VerifyingKey,
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
use rand_chacha::ChaCha20Rng;
use rand_core::{OsRng, SeedableRng};

/// `1 << K` rows. K >= 14 clears `GPU_MSM_THRESHOLD` so the real GPU device
/// paths run (not the CPU fallback).
const K: u32 = 14;

/// Size of the lookup table assigned during synthesis.
const TABLE_SIZE: usize = 8;

/// A circuit exercising a selector-gated custom gate, a permutation (copy from
/// the instance column + a global constant), and a lookup argument — so a
/// single prove/verify drives every phase-5 eval site.
#[derive(Clone)]
struct RichCircuit {
    /// Public input; tied to advice `a` at row 0 (copy constraint).
    public: Value<Fr>,
    /// Private multiplicand.
    b: Value<Fr>,
}

#[derive(Clone)]
struct RichConfig {
    a: Column<Advice>,
    b: Column<Advice>,
    c: Column<Advice>,
    d: Column<Advice>,
    s: Selector,
    ql: Column<Fixed>,
    instance: Column<Instance>,
    table: TableColumn,
}

impl Circuit<Fr> for RichCircuit {
    type Config = RichConfig;
    type FloorPlanner = SimpleFloorPlanner;
    #[cfg(feature = "circuit-params")]
    type Params = ();

    fn without_witnesses(&self) -> Self {
        Self {
            public: Value::unknown(),
            b: Value::unknown(),
        }
    }

    fn configure(meta: &mut ConstraintSystem<Fr>) -> RichConfig {
        let a = meta.advice_column();
        let b = meta.advice_column();
        let c = meta.advice_column();
        let d = meta.advice_column();
        let s = meta.selector();
        let ql = meta.fixed_column();
        let constant = meta.fixed_column();
        let instance = meta.instance_column();
        let table = meta.lookup_table_column();

        meta.enable_equality(a);
        meta.enable_equality(c);
        meta.enable_equality(d);
        meta.enable_equality(instance);
        meta.enable_constant(constant);

        meta.create_gate("mul", |meta| {
            let a = meta.query_advice(a, Rotation::cur());
            let b = meta.query_advice(b, Rotation::cur());
            let c = meta.query_advice(c, Rotation::cur());
            let s = meta.query_selector(s);
            vec![s * (a * b - c)]
        });

        meta.lookup("range", |meta| {
            let a = meta.query_advice(a, Rotation::cur());
            let ql = meta.query_fixed(ql, Rotation::cur());
            vec![(ql * a, table)]
        });

        RichConfig {
            a,
            b,
            c,
            d,
            s,
            ql,
            instance,
            table,
        }
    }

    fn synthesize(&self, config: RichConfig, mut layouter: impl Layouter<Fr>) -> Result<(), Error> {
        layouter.assign_table(
            || "range table",
            |mut table| {
                for i in 0..TABLE_SIZE {
                    table.assign_cell(
                        || "table cell",
                        config.table,
                        i,
                        || Value::known(Fr::from(i as u64)),
                    )?;
                }
                Ok(())
            },
        )?;

        layouter.assign_region(
            || "main",
            |mut region| {
                config.s.enable(&mut region, 0)?;
                region.assign_fixed(config.ql, 0, Fr::from(1));
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
                region.assign_advice_from_constant(|| "d const", config.d, 0, Fr::from(5))?;
                Ok(())
            },
        )
    }
}

/// Runs the GPU verifier and finalizes the strategy. Generic over the
/// `Verifier`/`VerificationStrategy` pair (pinned at the call site, since
/// `AccumulatorStrategy` implements `VerificationStrategy` for every verifier).
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
fn phase5_prove_verify_lookup_permutation() {
    let params = ParamsKZG::<Bn256>::setup(K, OsRng);

    // Keygen on the witness-free circuit.
    let circuit = RichCircuit {
        public: Value::unknown(),
        b: Value::unknown(),
    };
    let vk: VerifyingKey<G1Affine> = keygen_vk(&params, &circuit).expect("gpu keygen_vk");
    let pk = keygen_pk(&params, vk, &circuit).expect("gpu keygen_pk");

    // Prove with concrete, constraint-satisfying witnesses:
    //   public = 3 (must be in the range table [0, 8)), b = 2, c = a*b = 6.
    let public = Fr::from(3);
    let circuit = RichCircuit {
        public: Value::known(public),
        b: Value::known(Fr::from(2)),
    };
    let pubinputs = [public];
    let instances: &[&[&[Fr]]] = &[&[&pubinputs[..]]];

    let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
    create_proof::<KZGCommitmentScheme<Bn256>, ProverSHPLONK<_>, _, _, _, _>(
        &params,
        &pk,
        std::slice::from_ref(&circuit),
        instances,
        OsRng,
        &mut transcript,
    )
    .expect("gpu create_proof");
    let proof = transcript.finalize();

    // Verify: a wrong phase-5 eval order would make this fail.
    let verifier_params = params.verifier_params();
    assert!(
        gpu_verify::<VerifierSHPLONK<_>, AccumulatorStrategy<_>>(
            verifier_params,
            pk.get_vk(),
            instances,
            &proof,
        ),
        "phase-5 lookup+permutation proof failed to verify (eval batching must \
         preserve the write_scalar order the verifier reads)"
    );
}

/// Byte-identity guard for the phase-1 scoped-worker pk/SRS mirror warm-up.
///
/// The worker eagerly fills the witness-independent pk/SRS `OnceCell`s that the
/// lazy paths would otherwise fill on first touch, so the proof must not depend
/// on eager-vs-lazy. Two identically-seeded runs are fully deterministic
/// (Fiat-Shamir + seeded blinding; GPU MSM/NTT are exact) and the shared params
/// go cold→warm between them, so byte-identical proofs prove warm-up neutrality;
/// a race, wrong-device bind, or torn read would break it here.
#[test]
fn create_proof_pk_warmup_deterministic_byte_identity() {
    let params = ParamsKZG::<Bn256>::setup(K, OsRng);

    let circuit_kg = RichCircuit {
        public: Value::unknown(),
        b: Value::unknown(),
    };
    let vk: VerifyingKey<G1Affine> = keygen_vk(&params, &circuit_kg).expect("gpu keygen_vk");
    let pk = keygen_pk(&params, vk, &circuit_kg).expect("gpu keygen_pk");

    let public = Fr::from(3);
    let circuit = RichCircuit {
        public: Value::known(public),
        b: Value::known(Fr::from(2)),
    };
    let pubinputs = [public];
    let instances: &[&[&[Fr]]] = &[&[&pubinputs[..]]];

    // Two proofs with fresh, identically-seeded RNGs (the worker never touches
    // the RNG, so consumption matches).
    let prove_once = || {
        let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
        create_proof::<KZGCommitmentScheme<Bn256>, ProverSHPLONK<_>, _, _, _, _>(
            &params,
            &pk,
            std::slice::from_ref(&circuit),
            instances,
            ChaCha20Rng::seed_from_u64(0),
            &mut transcript,
        )
        .expect("gpu create_proof");
        transcript.finalize()
    };
    let proof_a = prove_once();
    let proof_b = prove_once();

    assert_eq!(
        proof_a, proof_b,
        "pk device-mirror warm-up must be proof-neutral: identically-seeded \
         create_proof runs must yield byte-identical proofs regardless of \
         whether the phase-1 worker or the main thread populated each mirror"
    );

    // The deterministic warm-up proof must still verify.
    let verifier_params = params.verifier_params();
    assert!(
        gpu_verify::<VerifierSHPLONK<_>, AccumulatorStrategy<_>>(
            verifier_params,
            pk.get_vk(),
            instances,
            &proof_a,
        ),
        "deterministic pk-warm-up proof failed to verify"
    );
}
