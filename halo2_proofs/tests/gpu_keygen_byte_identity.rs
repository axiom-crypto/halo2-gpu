//! GPU-keygen byte-identity lock.
//!
//! Pins the GPU keygen body
//! (`halo2_axiom_gpu::plonk::{keygen_vk_custom, keygen_pk, keygen_pk2}`): runs
//! CPU and GPU keygen against the SAME SRS and circuit and asserts the resulting
//! verifying/proving keys are byte-identical. The GPU path diverges from CPU
//! only in the per-column fixed/selector MSM commitments and the fixed-column
//! iFFT, so any differing byte localizes to that GPU code.
//!
//! ONE SRS feeds both sides: a fresh `setup`/`new` per side would draw a
//! different toxic `s`, so the GPU `ParamsKZG` is serialized and read back into
//! a `halo2-axiom` `ParamsKZG` — the bytes round-trip is load-bearing.
//!
//! Covered canonical->GPU bridge paths: a `Selector`-gated custom gate (so
//! `compress_selectors` has a selector to compress), a permutation/copy
//! constraint, a lookup argument, and a global constant.
//!
//! Run for both `compress_selectors = false` and `true`.

use halo2_axiom_gpu::circuit::{Layouter, SimpleFloorPlanner, Value};
use halo2_axiom_gpu::plonk::{
    Advice, Circuit, Column, ConstraintSystem, Error, Fixed, Instance, Selector, TableColumn,
};
use halo2_axiom_gpu::poly::Rotation;
use halo2curves::bn256::{Bn256, Fr};
use rand_core::OsRng;

/// `1 << K` rows. K >= 14 clears `GPU_MSM_THRESHOLD` so keygen's fixed/selector
/// commitments run on the real GPU device-MSM path (not the CPU fallback).
const K: u32 = 14;

/// Size of the lookup table assigned during synthesis.
const TABLE_SIZE: usize = 8;

/// A circuit exercising the four canonical->GPU keygen bridge paths at once: a
/// `Selector`-gated custom gate, a permutation copy from the instance column, a
/// lookup argument, and a global constant.
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
    /// Advice cell pinned to a global constant.
    d: Column<Advice>,
    /// Simple selector gating the multiplication gate (compressible).
    s: Selector,
    /// Fixed "selector" for the lookup; a plain fixed column, since `lookup`
    /// rejects inputs containing a simple selector.
    ql: Column<Fixed>,
    instance: Column<Instance>,
    /// Lookup table column.
    table: TableColumn,
}

impl Circuit<Fr> for RichCircuit {
    type Config = RichConfig;
    type FloorPlanner = SimpleFloorPlanner;
    #[cfg(feature = "circuit-params")]
    type Params = ();

    fn without_witnesses(&self) -> Self {
        Self { public: Value::unknown(), b: Value::unknown() }
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
        // Adds `constant` to the permutation and marks it the constants column.
        meta.enable_constant(constant);

        // Custom gate, degree 3: exercises selector compression.
        meta.create_gate("mul", |meta| {
            let a = meta.query_advice(a, Rotation::cur());
            let b = meta.query_advice(b, Rotation::cur());
            let c = meta.query_advice(c, Rotation::cur());
            let s = meta.query_selector(s);
            vec![s * (a * b - c)]
        });

        // Lookup argument: `ql * a` must appear in `table`.
        meta.lookup("range", |meta| {
            let a = meta.query_advice(a, Rotation::cur());
            let ql = meta.query_fixed(ql, Rotation::cur());
            vec![(ql * a, table)]
        });

        RichConfig { a, b, c, d, s, ql, instance, table }
    }

    fn synthesize(&self, config: RichConfig, mut layouter: impl Layouter<Fr>) -> Result<(), Error> {
        // Lookup table fixed column (committed in the vk).
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

                // Copy constraint: advice `a[0]` <- public `instance[0]`.
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

                // Global constant: pin advice `d[0]` to a fixed constant,
                // assigning the constants fixed column and a permutation copy.
                region.assign_advice_from_constant(|| "d const", config.d, 0, Fr::from(5))?;
                Ok(())
            },
        )
    }
}

/// Asserts CPU and GPU keygen produce byte-identical verifying and proving keys
/// for `circuit` under `compress_selectors`, across all three entry points:
/// `keygen_vk_custom`, `keygen_pk`, and `keygen_pk2`.
fn assert_keygen_byte_identity(
    gpu_params: &halo2_axiom_gpu::poly::kzg::commitment::ParamsKZG<Bn256>,
    cpu_params: &halo2_axiom::poly::kzg::commitment::ParamsKZG<Bn256>,
    circuit: &RichCircuit,
    compress_selectors: bool,
) {
    let fmt = halo2_axiom::SerdeFormat::RawBytes;

    let gpu_vk = halo2_axiom_gpu::plonk::keygen_vk_custom(gpu_params, circuit, compress_selectors)
        .expect("gpu keygen_vk_custom");
    let cpu_vk = halo2_axiom::plonk::keygen_vk_custom(cpu_params, circuit, compress_selectors)
        .expect("cpu keygen_vk_custom");
    assert_eq!(
        gpu_vk.to_bytes(fmt),
        cpu_vk.to_bytes(fmt),
        "GPU keygen_vk_custom must be byte-identical to CPU (compress_selectors = {compress_selectors})"
    );

    let gpu_pk =
        halo2_axiom_gpu::plonk::keygen_pk(gpu_params, gpu_vk, circuit).expect("gpu keygen_pk");
    let cpu_pk = halo2_axiom::plonk::keygen_pk(cpu_params, cpu_vk, circuit).expect("cpu keygen_pk");
    assert_eq!(
        gpu_pk.to_bytes(fmt),
        cpu_pk.to_bytes(fmt),
        "GPU keygen_pk must be byte-identical to CPU (compress_selectors = {compress_selectors})"
    );

    let gpu_pk2 = halo2_axiom_gpu::plonk::keygen_pk2(gpu_params, circuit, compress_selectors)
        .expect("gpu keygen_pk2");
    let cpu_pk2 = halo2_axiom::plonk::keygen_pk2(cpu_params, circuit, compress_selectors)
        .expect("cpu keygen_pk2");
    assert_eq!(
        gpu_pk2.to_bytes(fmt),
        cpu_pk2.to_bytes(fmt),
        "GPU keygen_pk2 must be byte-identical to CPU (compress_selectors = {compress_selectors})"
    );
    assert_eq!(
        gpu_pk2.to_bytes(fmt),
        gpu_pk.to_bytes(fmt),
        "GPU keygen_pk2 must match GPU keygen_pk (compress_selectors = {compress_selectors})"
    );
}

#[test]
fn gpu_keygen_bytes_match_cpu() {
    // ONE SRS: generate on the GPU side, serialize, read back into halo2-axiom.
    let gpu_params = halo2_axiom_gpu::poly::kzg::commitment::ParamsKZG::<Bn256>::setup(K, OsRng);
    let mut srs_bytes = Vec::new();
    gpu_params
        .write_custom(&mut srs_bytes, halo2_axiom_gpu::SerdeFormat::RawBytesUnchecked)
        .expect("write shared SRS");
    let cpu_params = halo2_axiom::poly::kzg::commitment::ParamsKZG::<Bn256>::read_custom(
        &mut &srs_bytes[..],
        halo2_axiom::SerdeFormat::RawBytesUnchecked,
    )
    .expect("read shared SRS into halo2-axiom ParamsKZG");

    // Witness-free circuit: keygen ignores advice values.
    let circuit = RichCircuit { public: Value::unknown(), b: Value::unknown() };

    for compress_selectors in [false, true] {
        assert_keygen_byte_identity(&gpu_params, &cpu_params, &circuit, compress_selectors);
    }
}
