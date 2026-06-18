use std::process::exit;

use openvm_cuda_builder::{cuda_available, CudaBuilder};

fn main() {
    if !cuda_available() {
        eprintln!("cargo:warning=CUDA is not available");
        exit(1);
    }

    // libhalo2_gpu: the GPU backend for halo2-axiom-gpu. Canonical build shape
    // matching stark-backend/crates/cuda-backend/build.rs and openvm consumers.
    // The cuda-common include path comes first so canonical sppark headers
    // (mont_t.cuh, ff/alt_bn128.cuh, curve/{jacobian_t,xyzz_t}.hpp) resolve
    // there when the matching local copies are removed.
    let builder = CudaBuilder::new()
        .library_name("halo2_gpu")
        .include_from_dep("DEP_CUDA_COMMON_INCLUDE")
        .watch("cuda")
        .include("cuda/include")
        .files_from_glob("cuda/src/**/*.cu");

    builder.emit_link_directives();
    builder.build();
}
