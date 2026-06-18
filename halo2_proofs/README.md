# halo2-axiom-gpu

This crate provides a Halo2 prover and verifier with CUDA acceleration on the prover path. The prover uses KZG commitments with SHPLONK multi-opening on BN254.

CUDA kernels live under [`cuda/`](./cuda); their Rust wrappers live under [`src/cuda/`](./src/cuda). CPU implementations used by tests and as the runtime fallback (`--features vram-fallback` and small-input shortcuts) live under [`src/cpu/`](./src/cpu).
