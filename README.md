# halo2-gpu

This repository provides a Halo2 prover and verifier with CUDA acceleration on the prover path. The main crate is `halo2-axiom-gpu`.

## Requirements

### Required

- `git`
- `rustup` with the repo toolchain from [`rust-toolchain.toml`](./rust-toolchain.toml)
- An NVIDIA GPU with a working driver
- CUDA toolkit 12.8 or later available on `PATH`; verify with:

```bash
nvidia-smi
nvcc --version
```

This repository builds CUDA code as part of the main crate, so `cargo check`,
`cargo build`, and `cargo test` require a CUDA-capable environment.

### Benchmarking and Profiling

The benchmark harness may also need:

- `wget` to download KZG SRS files via [`scripts/trusted_setup_s3.sh`](./scripts/trusted_setup_s3.sh)
- `cargo openvm` for rebuilding benchmark guest programs
- `solc` 0.8.19 for EVM verification flows
- `samply` and Linux `perf` for Firefox profiler traces
- NVIDIA Nsight Systems, `nsys`, for GPU profiling

## Crates

- [`halo2-axiom-gpu`](./halo2_proofs): The Halo2 proving and verifying crate.
- [`benchmarks`](./benchmarks): End-to-end prove benchmark (fibonacci) with samply and nsys profiling harnesses.

## Running Tests

Tests require the same CUDA-capable environment as builds. The CI test command
uses `cargo nextest` and runs tests single-threaded to avoid oversubscribing GPU
memory:

```bash
cargo nextest run --workspace --no-fail-fast --test-threads=1
```

For a standard Cargo test run, use:

```bash
cargo test --workspace
```

## Benchmarking

### KZG Params

The benchmark and proving flows that need BN254 KZG SRS files use OpenVM's
default params directory, `~/.openvm/params`. Download the default range of
params with:

```bash
bash ./scripts/trusted_setup_s3.sh
```

To choose a custom directory or range:

```bash
bash ./scripts/trusted_setup_s3.sh --params-dir /path/to/params --min-k 5 --max-k 24
```

### Running the Benchmark

The benchmark harness runs the Fibonacci guest proving benchmark and writes
metrics to `metrics.json`:

```bash
./benchmarks/run.sh
```

To collect a Firefox profiler trace with `samply`:

```bash
./benchmarks/run.sh --samply
```

To collect an NVIDIA Nsight Systems profile:

```bash
./benchmarks/run.sh --nsys
```

The harness uses the checked-in Fibonacci guest ELF at
`benchmarks/guest/fibonacci/elf/fibonacci.elf`. Rebuild that guest with
`cargo openvm` when the benchmark guest or its OpenVM dependency state changes.

## License

Dual-licensed under [Apache-2.0](./LICENSE-APACHE) or [MIT](./LICENSE-MIT) at the user's option.

## Acknowledgements

This codebase builds on prior open-source work:

- [ZCash `halo2_proofs`](https://github.com/zcash/halo2): The original implementation that this fork chain descends from.
- [Privacy Scaling Explorations](https://github.com/privacy-scaling-explorations/halo2): The KZG-backed fork that this branch inherits from.
- [`halo2curves`](https://github.com/axiom-crypto/halo2curves): The elliptic-curve and field arithmetic used by this crate.
- [Supranational `sppark`](https://github.com/supranational/sppark): The Pippenger MSM kernels and several curve/field-arithmetic primitives derive from sppark.
