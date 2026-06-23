# halo2-axiom-gpu

This crate provides a Halo2 prover and verifier with CUDA acceleration on the prover path. The prover uses KZG commitments with SHPLONK multi-opening on BN254.

CUDA kernels live under [`cuda/`](./cuda); their Rust wrappers live under [`src/cuda/`](./src/cuda). CPU implementations used by tests and as the runtime fallback (`--features vram-fallback` and small-input shortcuts) live under [`src/cpu/`](./src/cpu).

`halo2-axiom` is the canonical source of truth: the proving/verifying keys, the synthesis frontend (`Circuit`/`ConstraintSystem`/`Expression`/…), serde, and the transcript are re-exported from it, so keys serialize byte-identically across CPU and GPU. The GPU-local `Gpu*` types (e.g. `GpuConstraintSystem`, `GpuProvingKey`) carry the device methods the canonical types can't, and are rebuilt from a canonical key via `from_host`.
