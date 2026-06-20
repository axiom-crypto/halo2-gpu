#![allow(clippy::many_single_char_names)]
#![allow(clippy::op_ref)]
#![allow(unused_macros)]
#![allow(dead_code)]
#![allow(unused_variables)]

// use assert_matches::assert_matches;
use ff::{FromUniformBytes, WithSmallOrderMulGroup};
use halo2_axiom_gpu::arithmetic::Field;
use halo2_axiom_gpu::circuit::{Cell, Layouter, SimpleFloorPlanner, Value};
use halo2_axiom_gpu::dev::MockProver;
use halo2_axiom_gpu::plonk::{
    create_proof as create_plonk_proof, verify_proof as verify_plonk_proof, Advice, Assigned,
    Circuit, Column, ConstraintSystem, Error, Fixed, ProvingKey, TableColumn, VerifyingKey,
};
use halo2_axiom_gpu::poly::commitment::{CommitmentScheme, Params, ParamsProver, Prover, Verifier};
use halo2_axiom_gpu::poly::kzg::commitment::ParamsKZG;
use halo2_axiom_gpu::poly::Rotation;
use halo2_axiom_gpu::poly::VerificationStrategy;
use halo2_axiom_gpu::transcript::{
    Blake2bRead, Blake2bWrite, Challenge255, EncodedChallenge, TranscriptReadBuffer,
    TranscriptWriterBuffer,
};
use halo2curves::bn256::{Bn256, Fr, G1Affine};
use log::info;
use rand_core::{OsRng, RngCore};
use std::fs::File;
use std::hash::Hash;
use std::io::{BufReader, BufWriter};
use std::marker::PhantomData;

/// CPU (halo2-axiom) circuit variant that drives the canonical `keygen_vk`/
/// `keygen_pk` (which are halo2-axiom's). The resulting host `ProvingKey` is
/// passed directly to `create_proof`, which borrows it and builds the GPU view
/// internally; the GPU prove path keeps the gpu `MyCircuit` below. The circuit
/// logic is byte-identical to that
/// `MyCircuit` (the gpu Circuit/Region API is a fork of halo2-axiom's with the
/// same shape), only re-typed against halo2-axiom's `plonk`/`circuit` modules —
/// this is the dual-impl the maintainer asked for, isolated to a module so the
/// gpu prove/verify path is untouched.
///
/// NOTE: this is exercised for COMPILE only — `plonk_api` is `#[ignore]`d. For a
/// real run, the keygen params (built here) and the gpu prove params must share
/// the same SRS; the Phase-C cross-prover equivalence test locks the
/// CPU-serialized-pk → GPU-prove path properly.
mod cpu_keygen {
    use ff::Field;
    use halo2_axiom::circuit::{Cell, Layouter, SimpleFloorPlanner, Value};
    use halo2_axiom::plonk::{
        keygen_pk, keygen_vk, Advice, Assigned, Circuit, Column, ConstraintSystem, Error, Fixed,
        ProvingKey, TableColumn,
    };
    use halo2_axiom::poly::commitment::ParamsProver;
    use halo2_axiom::poly::kzg::commitment::ParamsKZG;
    use halo2_axiom::poly::Rotation;
    use halo2curves::bn256::{Bn256, Fr, G1Affine};
    use std::marker::PhantomData;

    #[derive(Clone)]
    struct PlonkConfig {
        a: Column<Advice>,
        b: Column<Advice>,
        c: Column<Advice>,
        d: Column<Advice>,
        e: Column<Advice>,
        sa: Column<Fixed>,
        sb: Column<Fixed>,
        sc: Column<Fixed>,
        sm: Column<Fixed>,
        sp: Column<Fixed>,
        sl: TableColumn,
    }

    #[allow(clippy::type_complexity)]
    trait StandardCs<FF: Field> {
        fn raw_multiply<F>(
            &self,
            layouter: &mut impl Layouter<FF>,
            f: F,
        ) -> Result<(Cell, Cell, Cell), Error>
        where
            F: FnMut() -> Value<(Assigned<FF>, Assigned<FF>, Assigned<FF>)>;
        fn raw_add<F>(
            &self,
            layouter: &mut impl Layouter<FF>,
            f: F,
        ) -> Result<(Cell, Cell, Cell), Error>
        where
            F: FnMut() -> Value<(Assigned<FF>, Assigned<FF>, Assigned<FF>)>;
        fn copy(&self, layouter: &mut impl Layouter<FF>, a: Cell, b: Cell) -> Result<(), Error>;
        fn public_input<F>(&self, layouter: &mut impl Layouter<FF>, f: F) -> Result<Cell, Error>
        where
            F: FnMut() -> Value<FF>;
        fn lookup_table(
            &self,
            layouter: &mut impl Layouter<FF>,
            values: &[FF],
        ) -> Result<(), Error>;
    }

    #[derive(Clone)]
    struct MyCircuit<F: Field> {
        a: Value<F>,
        lookup_table: Vec<F>,
    }

    struct StandardPlonk<F: Field> {
        config: PlonkConfig,
        _marker: PhantomData<F>,
    }

    impl<FF: Field> StandardPlonk<FF> {
        fn new(config: PlonkConfig) -> Self {
            StandardPlonk {
                config,
                _marker: PhantomData,
            }
        }
    }

    impl<FF: Field> StandardCs<FF> for StandardPlonk<FF> {
        fn raw_multiply<F>(
            &self,
            layouter: &mut impl Layouter<FF>,
            mut f: F,
        ) -> Result<(Cell, Cell, Cell), Error>
        where
            F: FnMut() -> Value<(Assigned<FF>, Assigned<FF>, Assigned<FF>)>,
        {
            layouter.assign_region(
                || "raw_multiply",
                |mut region| {
                    let value;
                    let lhs = region.assign_advice(self.config.a, 0, {
                        value = Some(f());
                        value.unwrap().map(|v| v.0)
                    });
                    region.assign_advice(
                        self.config.d,
                        0,
                        value.unwrap().map(|v| v.0).square().square(),
                    );
                    let rhs = region.assign_advice(self.config.b, 0, value.unwrap().map(|v| v.1));
                    region.assign_advice(
                        self.config.e,
                        0,
                        value.unwrap().map(|v| v.1).square().square(),
                    );
                    let out = region.assign_advice(self.config.c, 0, value.unwrap().map(|v| v.2));

                    region.assign_fixed(self.config.sa, 0, FF::ZERO);
                    region.assign_fixed(self.config.sb, 0, FF::ZERO);
                    region.assign_fixed(self.config.sc, 0, FF::ONE);
                    region.assign_fixed(self.config.sm, 0, FF::ONE);
                    Ok((lhs.cell(), rhs.cell(), out.cell()))
                },
            )
        }
        fn raw_add<F>(
            &self,
            layouter: &mut impl Layouter<FF>,
            mut f: F,
        ) -> Result<(Cell, Cell, Cell), Error>
        where
            F: FnMut() -> Value<(Assigned<FF>, Assigned<FF>, Assigned<FF>)>,
        {
            layouter.assign_region(
                || "raw_add",
                |mut region| {
                    let value;
                    let lhs = region.assign_advice(self.config.a, 0, {
                        value = Some(f());
                        value.unwrap().map(|v| v.0)
                    });
                    region.assign_advice(
                        self.config.d,
                        0,
                        value.unwrap().map(|v| v.0).square().square(),
                    );
                    let rhs = region.assign_advice(self.config.b, 0, value.unwrap().map(|v| v.1));
                    region.assign_advice(
                        self.config.e,
                        0,
                        value.unwrap().map(|v| v.1).square().square(),
                    );
                    let out = region.assign_advice(self.config.c, 0, value.unwrap().map(|v| v.2));

                    region.assign_fixed(self.config.sa, 0, FF::ONE);
                    region.assign_fixed(self.config.sb, 0, FF::ONE);
                    region.assign_fixed(self.config.sc, 0, FF::ONE);
                    region.assign_fixed(self.config.sm, 0, FF::ZERO);
                    Ok((lhs.cell(), rhs.cell(), out.cell()))
                },
            )
        }
        fn copy(
            &self,
            layouter: &mut impl Layouter<FF>,
            left: Cell,
            right: Cell,
        ) -> Result<(), Error> {
            layouter.assign_region(
                || "copy",
                |mut region| {
                    region.constrain_equal(left, right);
                    region.constrain_equal(left, right);
                    Ok(())
                },
            )
        }
        fn public_input<F>(&self, layouter: &mut impl Layouter<FF>, mut f: F) -> Result<Cell, Error>
        where
            F: FnMut() -> Value<FF>,
        {
            layouter.assign_region(
                || "public_input",
                |mut region| {
                    let value = region.assign_advice(self.config.a, 0, f());
                    region.assign_fixed(self.config.sp, 0, FF::ONE);

                    Ok(value.cell())
                },
            )
        }
        fn lookup_table(
            &self,
            layouter: &mut impl Layouter<FF>,
            values: &[FF],
        ) -> Result<(), Error> {
            layouter.assign_table(
                || "",
                |mut table| {
                    for (index, &value) in values.iter().enumerate() {
                        table.assign_cell(
                            || "table col",
                            self.config.sl,
                            index,
                            || Value::known(value),
                        )?;
                    }
                    Ok(())
                },
            )?;
            Ok(())
        }
    }

    impl<F: Field> Circuit<F> for MyCircuit<F> {
        type Config = PlonkConfig;
        type FloorPlanner = SimpleFloorPlanner;
        #[cfg(feature = "circuit-params")]
        type Params = ();

        fn without_witnesses(&self) -> Self {
            Self {
                a: Value::unknown(),
                lookup_table: self.lookup_table.clone(),
            }
        }

        fn configure(meta: &mut ConstraintSystem<F>) -> PlonkConfig {
            let e = meta.advice_column();
            let a = meta.advice_column();
            let b = meta.advice_column();
            let sf = meta.fixed_column();
            let c = meta.advice_column();
            let d = meta.advice_column();
            let p = meta.instance_column();

            meta.enable_equality(a);
            meta.enable_equality(b);
            meta.enable_equality(c);

            let sm = meta.fixed_column();
            let sa = meta.fixed_column();
            let sb = meta.fixed_column();
            let sc = meta.fixed_column();
            let sp = meta.fixed_column();
            let sl = meta.lookup_table_column();

            meta.lookup("lookup", |meta| {
                let a_ = meta.query_any(a, Rotation::cur());
                vec![(a_, sl)]
            });

            meta.create_gate("Combined add-mult", |meta| {
                let d = meta.query_advice(d, Rotation::next());
                let a = meta.query_advice(a, Rotation::cur());
                let sf = meta.query_fixed(sf, Rotation::cur());
                let e = meta.query_advice(e, Rotation::prev());
                let b = meta.query_advice(b, Rotation::cur());
                let c = meta.query_advice(c, Rotation::cur());

                let sa = meta.query_fixed(sa, Rotation::cur());
                let sb = meta.query_fixed(sb, Rotation::cur());
                let sc = meta.query_fixed(sc, Rotation::cur());
                let sm = meta.query_fixed(sm, Rotation::cur());

                vec![a.clone() * sa + b.clone() * sb + a * b * sm - (c * sc)]
            });

            meta.create_gate("Public input", |meta| {
                let a = meta.query_advice(a, Rotation::cur());
                let p = meta.query_instance(p, Rotation::cur());
                let sp = meta.query_fixed(sp, Rotation::cur());

                vec![sp * (a - p)]
            });

            meta.enable_equality(sf);
            meta.enable_equality(e);
            meta.enable_equality(d);
            meta.enable_equality(p);
            meta.enable_equality(sm);
            meta.enable_equality(sa);
            meta.enable_equality(sb);
            meta.enable_equality(sc);
            meta.enable_equality(sp);

            PlonkConfig {
                a,
                b,
                c,
                d,
                e,
                sa,
                sb,
                sc,
                sm,
                sp,
                sl,
            }
        }

        fn synthesize(
            &self,
            config: PlonkConfig,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            let cs = StandardPlonk::new(config);

            let _ = cs.public_input(&mut layouter, || Value::known(F::ONE + F::ONE))?;

            for _ in 0..10 {
                let a: Value<Assigned<_>> = self.a.into();
                let mut a_squared = Value::unknown();
                let (a0, _, c0) = cs.raw_multiply(&mut layouter, || {
                    a_squared = a.square();
                    a.zip(a_squared).map(|(a, a_squared)| (a, a, a_squared))
                })?;
                let (a1, b1, _) = cs.raw_add(&mut layouter, || {
                    let fin = a_squared + a;
                    a.zip(a_squared)
                        .zip(fin)
                        .map(|((a, a_squared), fin)| (a, a_squared, fin))
                })?;
                cs.copy(&mut layouter, a0, a1)?;
                cs.copy(&mut layouter, b1, c0)?;
            }

            cs.lookup_table(&mut layouter, &self.lookup_table)?;

            Ok(())
        }
    }

    /// Builds the canonical halo2-axiom `ProvingKey` from the CPU circuit variant.
    pub fn build_pk(k: u32, lookup_table: Vec<Fr>) -> ProvingKey<G1Affine> {
        let params = ParamsKZG::<Bn256>::new(k);
        let empty_circuit: MyCircuit<Fr> = MyCircuit {
            a: Value::unknown(),
            lookup_table,
        };
        let vk = keygen_vk(&params, &empty_circuit).expect("cpu keygen_vk should not fail");
        keygen_pk(&params, vk, &empty_circuit).expect("cpu keygen_pk should not fail")
    }
}

#[test]
#[ignore = "doesn't work because it uses multiple regions"]
fn plonk_api() {
    const K: u32 = 5;

    /// This represents an advice column at a certain row in the ConstraintSystem
    #[derive(Copy, Clone, Debug)]
    pub struct Variable(Column<Advice>, usize);

    #[derive(Clone)]
    struct PlonkConfig {
        a: Column<Advice>,
        b: Column<Advice>,
        c: Column<Advice>,
        d: Column<Advice>,
        e: Column<Advice>,
        sa: Column<Fixed>,
        sb: Column<Fixed>,
        sc: Column<Fixed>,
        sm: Column<Fixed>,
        sp: Column<Fixed>,
        sl: TableColumn,
    }

    #[allow(clippy::type_complexity)]
    trait StandardCs<FF: Field> {
        fn raw_multiply<F>(
            &self,
            layouter: &mut impl Layouter<FF>,
            f: F,
        ) -> Result<(Cell, Cell, Cell), Error>
        where
            F: FnMut() -> Value<(Assigned<FF>, Assigned<FF>, Assigned<FF>)>;
        fn raw_add<F>(
            &self,
            layouter: &mut impl Layouter<FF>,
            f: F,
        ) -> Result<(Cell, Cell, Cell), Error>
        where
            F: FnMut() -> Value<(Assigned<FF>, Assigned<FF>, Assigned<FF>)>;
        fn copy(&self, layouter: &mut impl Layouter<FF>, a: Cell, b: Cell) -> Result<(), Error>;
        fn public_input<F>(&self, layouter: &mut impl Layouter<FF>, f: F) -> Result<Cell, Error>
        where
            F: FnMut() -> Value<FF>;
        fn lookup_table(
            &self,
            layouter: &mut impl Layouter<FF>,
            values: &[FF],
        ) -> Result<(), Error>;
    }

    #[derive(Clone)]
    struct MyCircuit<F: Field> {
        a: Value<F>,
        lookup_table: Vec<F>,
    }

    struct StandardPlonk<F: Field> {
        config: PlonkConfig,
        _marker: PhantomData<F>,
    }

    impl<FF: Field> StandardPlonk<FF> {
        fn new(config: PlonkConfig) -> Self {
            StandardPlonk {
                config,
                _marker: PhantomData,
            }
        }
    }

    impl<FF: Field> StandardCs<FF> for StandardPlonk<FF> {
        fn raw_multiply<F>(
            &self,
            layouter: &mut impl Layouter<FF>,
            mut f: F,
        ) -> Result<(Cell, Cell, Cell), Error>
        where
            F: FnMut() -> Value<(Assigned<FF>, Assigned<FF>, Assigned<FF>)>,
        {
            layouter.assign_region(
                || "raw_multiply",
                |mut region| {
                    let value;
                    let lhs = region.assign_advice(self.config.a, 0, {
                        value = Some(f());
                        value.unwrap().map(|v| v.0)
                    });
                    region.assign_advice(
                        self.config.d,
                        0,
                        value.unwrap().map(|v| v.0).square().square(),
                    );
                    let rhs = region.assign_advice(self.config.b, 0, value.unwrap().map(|v| v.1));
                    region.assign_advice(
                        self.config.e,
                        0,
                        value.unwrap().map(|v| v.1).square().square(),
                    );
                    let out = region.assign_advice(self.config.c, 0, value.unwrap().map(|v| v.2));

                    region.assign_fixed(self.config.sa, 0, FF::ZERO);
                    region.assign_fixed(self.config.sb, 0, FF::ZERO);
                    region.assign_fixed(self.config.sc, 0, FF::ONE);
                    region.assign_fixed(self.config.sm, 0, FF::ONE);
                    Ok((lhs.cell(), rhs.cell(), out.cell()))
                },
            )
        }
        fn raw_add<F>(
            &self,
            layouter: &mut impl Layouter<FF>,
            mut f: F,
        ) -> Result<(Cell, Cell, Cell), Error>
        where
            F: FnMut() -> Value<(Assigned<FF>, Assigned<FF>, Assigned<FF>)>,
        {
            layouter.assign_region(
                || "raw_add",
                |mut region| {
                    let value;
                    let lhs = region.assign_advice(self.config.a, 0, {
                        value = Some(f());
                        value.unwrap().map(|v| v.0)
                    });
                    region.assign_advice(
                        self.config.d,
                        0,
                        value.unwrap().map(|v| v.0).square().square(),
                    );
                    let rhs = region.assign_advice(self.config.b, 0, value.unwrap().map(|v| v.1));
                    region.assign_advice(
                        self.config.e,
                        0,
                        value.unwrap().map(|v| v.1).square().square(),
                    );
                    let out = region.assign_advice(self.config.c, 0, value.unwrap().map(|v| v.2));

                    region.assign_fixed(self.config.sa, 0, FF::ONE);
                    region.assign_fixed(self.config.sb, 0, FF::ONE);
                    region.assign_fixed(self.config.sc, 0, FF::ONE);
                    region.assign_fixed(self.config.sm, 0, FF::ZERO);
                    Ok((lhs.cell(), rhs.cell(), out.cell()))
                },
            )
        }
        fn copy(
            &self,
            layouter: &mut impl Layouter<FF>,
            left: Cell,
            right: Cell,
        ) -> Result<(), Error> {
            layouter.assign_region(
                || "copy",
                |mut region| {
                    region.constrain_equal(left, right);
                    region.constrain_equal(left, right);
                    Ok(())
                },
            )
        }
        fn public_input<F>(&self, layouter: &mut impl Layouter<FF>, mut f: F) -> Result<Cell, Error>
        where
            F: FnMut() -> Value<FF>,
        {
            layouter.assign_region(
                || "public_input",
                |mut region| {
                    let value = region.assign_advice(self.config.a, 0, f());
                    region.assign_fixed(self.config.sp, 0, FF::ONE);

                    Ok(value.cell())
                },
            )
        }
        fn lookup_table(
            &self,
            layouter: &mut impl Layouter<FF>,
            values: &[FF],
        ) -> Result<(), Error> {
            layouter.assign_table(
                || "",
                |mut table| {
                    for (index, &value) in values.iter().enumerate() {
                        table.assign_cell(
                            || "table col",
                            self.config.sl,
                            index,
                            || Value::known(value),
                        )?;
                    }
                    Ok(())
                },
            )?;
            Ok(())
        }
    }

    impl<F: Field> Circuit<F> for MyCircuit<F> {
        type Config = PlonkConfig;
        type FloorPlanner = SimpleFloorPlanner;
        #[cfg(feature = "circuit-params")]
        type Params = ();

        fn without_witnesses(&self) -> Self {
            Self {
                a: Value::unknown(),
                lookup_table: self.lookup_table.clone(),
            }
        }

        fn configure(meta: &mut ConstraintSystem<F>) -> PlonkConfig {
            let e = meta.advice_column();
            let a = meta.advice_column();
            let b = meta.advice_column();
            let sf = meta.fixed_column();
            let c = meta.advice_column();
            let d = meta.advice_column();
            let p = meta.instance_column();

            meta.enable_equality(a);
            meta.enable_equality(b);
            meta.enable_equality(c);

            let sm = meta.fixed_column();
            let sa = meta.fixed_column();
            let sb = meta.fixed_column();
            let sc = meta.fixed_column();
            let sp = meta.fixed_column();
            let sl = meta.lookup_table_column();

            /*
             *   A         B      ...  sl
             * [
             *   instance  0      ...  0
             *   a         a      ...  0
             *   a         a^2    ...  0
             *   a         a      ...  0
             *   a         a^2    ...  0
             *   ...       ...    ...  ...
             *   ...       ...    ...  instance
             *   ...       ...    ...  a
             *   ...       ...    ...  a
             *   ...       ...    ...  0
             * ]
             */

            meta.lookup("lookup", |meta| {
                let a_ = meta.query_any(a, Rotation::cur());
                vec![(a_, sl)]
            });

            meta.create_gate("Combined add-mult", |meta| {
                let d = meta.query_advice(d, Rotation::next());
                let a = meta.query_advice(a, Rotation::cur());
                let sf = meta.query_fixed(sf, Rotation::cur());
                let e = meta.query_advice(e, Rotation::prev());
                let b = meta.query_advice(b, Rotation::cur());
                let c = meta.query_advice(c, Rotation::cur());

                let sa = meta.query_fixed(sa, Rotation::cur());
                let sb = meta.query_fixed(sb, Rotation::cur());
                let sc = meta.query_fixed(sc, Rotation::cur());
                let sm = meta.query_fixed(sm, Rotation::cur());

                vec![a.clone() * sa + b.clone() * sb + a * b * sm - (c * sc)]
            });

            meta.create_gate("Public input", |meta| {
                let a = meta.query_advice(a, Rotation::cur());
                let p = meta.query_instance(p, Rotation::cur());
                let sp = meta.query_fixed(sp, Rotation::cur());

                vec![sp * (a - p)]
            });

            meta.enable_equality(sf);
            meta.enable_equality(e);
            meta.enable_equality(d);
            meta.enable_equality(p);
            meta.enable_equality(sm);
            meta.enable_equality(sa);
            meta.enable_equality(sb);
            meta.enable_equality(sc);
            meta.enable_equality(sp);

            PlonkConfig {
                a,
                b,
                c,
                d,
                e,
                sa,
                sb,
                sc,
                sm,
                sp,
                sl,
            }
        }

        fn synthesize(
            &self,
            config: PlonkConfig,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            let cs = StandardPlonk::new(config);

            let _ = cs.public_input(&mut layouter, || Value::known(F::ONE + F::ONE))?;

            for _ in 0..10 {
                let a: Value<Assigned<_>> = self.a.into();
                let mut a_squared = Value::unknown();
                let (a0, _, c0) = cs.raw_multiply(&mut layouter, || {
                    a_squared = a.square();
                    a.zip(a_squared).map(|(a, a_squared)| (a, a, a_squared))
                })?;
                let (a1, b1, _) = cs.raw_add(&mut layouter, || {
                    let fin = a_squared + a;
                    a.zip(a_squared)
                        .zip(fin)
                        .map(|((a, a_squared), fin)| (a, a_squared, fin))
                })?;
                cs.copy(&mut layouter, a0, a1)?;
                cs.copy(&mut layouter, b1, c0)?;
            }

            cs.lookup_table(&mut layouter, &self.lookup_table)?;

            Ok(())
        }
    }

    macro_rules! common {
        ($scheme:ident) => {{
            let a = <$scheme as CommitmentScheme>::Scalar::from(2834758237)
                * <$scheme as CommitmentScheme>::Scalar::ZETA;
            let instance = <$scheme as CommitmentScheme>::Scalar::ONE
                + <$scheme as CommitmentScheme>::Scalar::ONE;
            let lookup_table = vec![instance, a, a, <$scheme as CommitmentScheme>::Scalar::ZERO];
            (a, instance, lookup_table)
        }};
    }

    fn keygen(params: &ParamsKZG<Bn256>) -> ProvingKey<G1Affine> {
        // Keygen runs on the CPU (halo2-axiom) circuit variant, producing the
        // canonical host `ProvingKey`; `create_proof` borrows it and builds the
        // GPU view internally. (The lookup table mirrors `common!`.)
        let a = Fr::from(2834758237) * Fr::ZETA;
        let instance = Fr::ONE + Fr::ONE;
        let lookup_table = vec![instance, a, a, Fr::ZERO];
        cpu_keygen::build_pk(params.k(), lookup_table)
    }

    fn create_proof<
        'params,
        Scheme: CommitmentScheme,
        P: Prover<'params, Scheme>,
        E: EncodedChallenge<Scheme::Curve>,
        R: RngCore + Send,
        T: TranscriptWriterBuffer<Vec<u8>, Scheme::Curve, E>,
    >(
        rng: R,
        params: &'params Scheme::ParamsProver,
        pk: &ProvingKey<Scheme::Curve>,
    ) -> Vec<u8>
    where
        Scheme::Scalar: Hash + Ord + WithSmallOrderMulGroup<3> + FromUniformBytes<64>,
        <Scheme as CommitmentScheme>::ParamsProver: Sync,
    {
        let (a, instance, lookup_table) = common!(Scheme);

        let circuit: MyCircuit<Scheme::Scalar> = MyCircuit {
            a: Value::known(a),
            lookup_table,
        };

        let mut transcript = T::init(vec![]);

        create_plonk_proof::<Scheme, P, _, _, _, _>(
            params,
            pk,
            std::slice::from_ref(&circuit),
            &[&[&[instance]]],
            rng,
            &mut transcript,
        )
        .expect("proof generation should not fail");

        // Check this circuit is satisfied.
        let prover = match MockProver::run(K, &circuit, vec![vec![instance]]) {
            Ok(prover) => prover,
            Err(e) => panic!("{:?}", e),
        };
        assert_eq!(prover.verify(), Ok(()));

        transcript.finalize()
    }

    fn verify_proof<
        'a,
        'params,
        Scheme: CommitmentScheme,
        V: Verifier<'params, Scheme>,
        E: EncodedChallenge<Scheme::Curve>,
        T: TranscriptReadBuffer<&'a [u8], Scheme::Curve, E>,
        Strategy: VerificationStrategy<'params, Scheme, V, Output = Strategy>,
    >(
        params_verifier: &'params Scheme::ParamsVerifier,
        vk: &VerifyingKey<Scheme::Curve>,
        proof: &'a [u8],
    ) where
        Scheme::Scalar: Ord + WithSmallOrderMulGroup<3> + FromUniformBytes<64>,
    {
        let (_, instance, _) = common!(Scheme);
        let pubinputs = [instance];

        let mut transcript = T::init(proof);

        let strategy = Strategy::new(params_verifier);
        let strategy = verify_plonk_proof(
            params_verifier,
            vk,
            strategy,
            // &[&[&pubinputs[..]], &[&pubinputs[..]]],
            &[&[&pubinputs[..]]],
            &mut transcript,
        )
        .unwrap();

        assert!(strategy.finalize());
    }

    fn create_or_load_params(degree: u32) -> ParamsKZG<Bn256> {
        let param_path_str = format!("./params_{}", degree);
        let param_path = std::path::Path::new(&param_path_str);
        if param_path.exists() {
            let file = File::open(param_path).expect("open param file failed");
            let param = ParamsKZG::read::<_>(&mut BufReader::new(file)).expect("read param failed");
            info!("load param of degree {} successfully", degree);
            param
        } else {
            let param = ParamsKZG::new(degree);
            let ofile = File::create(param_path).expect("create param file failed");
            param
                .write(&mut BufWriter::new(ofile))
                .expect("write param file failed");
            param
        }
    }

    fn test_plonk_api_gwc() {
        use halo2_axiom_gpu::poly::kzg::commitment::KZGCommitmentScheme;
        use halo2_axiom_gpu::poly::kzg::multiopen::{ProverGWC, VerifierGWC};
        use halo2_axiom_gpu::poly::kzg::strategy::AccumulatorStrategy;
        use halo2curves::bn256::Bn256;

        env_logger::init();
        type Scheme = KZGCommitmentScheme<Bn256>;
        //bad_keys!(Scheme);

        // let params = ParamsKZG::<Bn256>::new(K);
        let params = create_or_load_params(K);
        let rng = OsRng;

        let pk = keygen(&params);

        let proof = create_proof::<_, ProverGWC<_>, _, _, Blake2bWrite<_, _, Challenge255<_>>>(
            rng, &params, &pk,
        );

        let verifier_params = params.verifier_params();

        verify_proof::<
            _,
            VerifierGWC<_>,
            _,
            Blake2bRead<_, _, Challenge255<_>>,
            AccumulatorStrategy<_>,
        >(verifier_params, pk.get_vk(), &proof[..]);
    }

    fn test_plonk_api_shplonk() {
        use halo2_axiom_gpu::poly::kzg::commitment::{KZGCommitmentScheme, ParamsKZG};
        use halo2_axiom_gpu::poly::kzg::multiopen::{ProverSHPLONK, VerifierSHPLONK};
        use halo2_axiom_gpu::poly::kzg::strategy::AccumulatorStrategy;
        use halo2curves::bn256::Bn256;

        type Scheme = KZGCommitmentScheme<Bn256>;
        //bad_keys!(Scheme);

        let params = ParamsKZG::<Bn256>::new(K);
        let rng = OsRng;

        let pk = keygen(&params);

        let proof = create_proof::<_, ProverSHPLONK<_>, _, _, Blake2bWrite<_, _, Challenge255<_>>>(
            rng, &params, &pk,
        );

        let verifier_params = params.verifier_params();

        verify_proof::<
            _,
            VerifierSHPLONK<_>,
            _,
            Blake2bRead<_, _, Challenge255<_>>,
            AccumulatorStrategy<_>,
        >(verifier_params, pk.get_vk(), &proof[..]);
    }

    let _ = env_logger::try_init();
    test_plonk_api_shplonk();
    test_plonk_api_gwc();
}
