use benchmarks::metric_collection::run_with_metric_collection;
use eyre::Result;
use openvm::platform::memory::GUEST_MAX_MEM;
use openvm_sdk::{
    config::{AggregationSystemParams, AppConfig, DEFAULT_APP_L_SKIP},
    fs::{read_object_from_file, write_object_to_file},
    types::ExecutableFormat,
    Sdk, StdIn,
};
use openvm_stark_sdk::config::app_params_with_100_bits_security;
use openvm_transpiler::elf::Elf;
use std::path::Path;

const FIB_ELF: &[u8] = include_bytes!("../guest/fibonacci/elf/fibonacci.elf");

fn main() -> Result<()> {
    let n: u64 = 1000;
    let mut stdin = StdIn::default();
    stdin.write(&n);

    let elf = Elf::decode(FIB_ELF, GUEST_MAX_MEM as u32)?;

    let cache_dir = Path::new("cache");
    std::fs::create_dir_all(cache_dir)?;
    let root_proof_path = cache_dir.join("root_proof.bitcode");
    let app_pk_path = cache_dir.join("app_pk.bitcode");
    let agg_pk_path = cache_dir.join("agg_pk.bitcode");
    let root_pk_path = cache_dir.join("root_pk.bitcode");

    let n_stack = 19;
    let mut builder = Sdk::builder();

    if app_pk_path.exists() {
        eprintln!("reusing app pk");
        builder = builder.app_pk(read_object_from_file(&app_pk_path)?);
    } else {
        let app_params = app_params_with_100_bits_security(DEFAULT_APP_L_SKIP + n_stack);
        builder = builder.app_config(AppConfig::riscv32(app_params));
    }

    if agg_pk_path.exists() {
        eprintln!("reusing agg pk");
        builder = builder.agg_pk(read_object_from_file(&agg_pk_path)?);
    } else {
        let agg_params = AggregationSystemParams::default();
        builder = builder.agg_params(agg_params);
    }

    if root_pk_path.exists() {
        eprintln!("reusing root pk");
        builder = builder.root_pk(read_object_from_file(&root_pk_path)?);
    }

    let sdk = builder.build()?;
    let app_exe = sdk.convert_to_exe(ExecutableFormat::Elf(elf))?;

    let evm_proof = if root_proof_path.exists() {
        eprintln!("reusing root proof from {:?}", root_proof_path);
        let halo2_prover = sdk.halo2_prover();
        let root_proof = read_object_from_file(root_proof_path)?;
        run_with_metric_collection("OUTPUT_PATH", move || halo2_prover.prove_for_evm(&root_proof))
    } else {
        let mut evm_prover = sdk.evm_prover(app_exe).expect("evm_prover construction failed");

        let root_proof = evm_prover.prove_root(stdin, &[])?;

        write_object_to_file(root_proof_path, &root_proof)?;
        write_object_to_file(app_pk_path, sdk.app_pk())?;
        write_object_to_file(agg_pk_path, sdk.agg_pk())?;
        write_object_to_file(root_pk_path, sdk.root_pk())?;

        run_with_metric_collection("OUTPUT_PATH", move || {
            evm_prover.halo2_prover.as_ref().unwrap().prove_for_evm(&root_proof)
        })
    };

    let openvm_verifier =
        sdk.generate_halo2_verifier_solidity().expect("generate_halo2_verifier_solidity failed");
    Sdk::verify_evm_halo2_proof(&openvm_verifier, evm_proof, None)
        .expect("verify_evm_halo2_proof failed");

    Ok(())
}
