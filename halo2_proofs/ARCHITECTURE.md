# halo2-gpu architecture: canonical types vs. GPU forks

This crate (`halo2-axiom-gpu`) is a GPU prover/verifier that shares its
serialized data model with the CPU crate it depends on, `halo2-axiom`. This
document explains the one structural decision that the rest of the code keeps
coming back to: **which types are the canonical halo2-axiom types, which are
GPU-local forks, and why.**

## "Canonical" = halo2-axiom is the single source of truth

The serialized, equivalence-critical types are the **halo2-axiom** types, used
here by re-export, not by forking. In `src/plonk.rs` and `src/helpers.rs`:

- `ProvingKey`, `VerifyingKey`, `ConstraintSystem`, `Expression`, `Circuit`,
  `Assignment`, `Column`/`Any`, `Error`, `TableColumn`, … — re-exported from
  `halo2_axiom::plonk`.
- `SerdeFormat` and the Fiat-Shamir transcript types — re-exported from
  `halo2_axiom`.

Two consequences fall out of this:

1. **One serialization, byte-identical across CPU and GPU.** A key written by
   GPU keygen reads back into a CPU verifier (and vice versa) because both sides
   serialize the *same* struct. The `cross_prover_pk_equivalence` and
   `gpu_keygen_byte_identity` tests pin this.
2. **The frontend is canonical end-to-end.** Downstream consumers
   (`halo2-base`, `snark-verifier`, `openvm`) and this crate's own keygen drive
   the canonical `Circuit` / `configure(&mut ConstraintSystem)` / `Assignment`
   API. GPU keygen returns canonical `ProvingKey`/`VerifyingKey`, so it is a
   drop-in for the CPU `keygen_pk`/`keygen_vk` the consumer stack calls.

Throughout the code, "canonical" is shorthand for "the halo2-axiom type, as
defined here."

## Why the `Gpu*` forks exist

The canonical types cannot carry GPU device state or GPU inherent methods, so
the backend keeps a parallel family of working types — `GpuConstraintSystem`,
`GpuExpression`, `GpuGate`, `GpuColumn`, the GPU `permutation`/`lookup`
`Argument`s, and the GPU `EvaluationDomain` (the FFT/MSM engine). These hold the
device-facing behavior: the quotient `Evaluator`, the `commit`/`commit_permuted`
methods that launch MSM/FFT kernels, and the device-resident polynomial mirrors.

The forks are **structurally near-identical** to their canonical counterparts;
they are rebuilt from the canonical types rather than maintained independently.

## The `from_host` rebuild pattern

`GpuProvingKey` / `GpuVerifyingKey` are the bridge between a canonical key and
the GPU working types. `from_host` (and `from_host_ref`, the borrowing variant)
performs a **pure-host** rebuild — no device traffic, no kernel launches:

- `GpuConstraintSystem::from(inner.get_vk().cs())` — a field-copy via the
  `From<&ConstraintSystem>` bridge,
- `EvaluationDomain::new(j, k)` — reconstructed from the canonical domain's
  `(degree, k)`,
- `Evaluator::new(&cs)`.

`GpuProvingKey` holds the canonical key as `Cow<'a, ProvingKey>`, so the
per-proof hot path (`create_proof`) can *borrow* a consumer's `&ProvingKey` with
zero host-poly clones. Device-resident polynomial mirrors are `OnceCell`s that
populate lazily on first prove (and reset on `Clone`).

## Why some items are re-DEFINED rather than re-exported

A glob `pub use halo2_axiom::…::*` can only re-export items that are public
upstream. The following are `pub(crate)`/private in halo2-axiom (or have no
nameable canonical equivalent), so they are re-defined locally. These are the
re-definitions a reviewer is most likely to flag as "duplication" — they are
forced by upstream visibility, not by choice:

| Re-defined here | Where | Why it cannot be re-exported |
|---|---|---|
| `CurveRead`, `SerdeCurveAffine`, `SerdePrimeField` | `src/helpers.rs` | `CurveRead` is `pub(crate)` upstream; `SerdeCurveAffine`/`SerdePrimeField` are `pub` traits inside halo2-axiom's *private* `helpers` module, so they are externally unnameable and cannot be re-exported. |
| `read_n_points`, `read_n_scalars` | `src/transcript.rs` | `pub(crate)` batch-read helpers upstream; re-defined as thin wrappers over the re-exported `TranscriptRead`. |
| keygen `Assembly` | `src/plonk/keygen.rs` | Canonical `Assembly` is private upstream. The local one synthesizes the fixed columns (canonical `Assigned`), collects selectors/copy constraints, and drives the local permutation assembly, then feeds the GPU-accelerated keygen steps — which convert to `GpuAssigned` only at the batch-inversion boundary. Implements the canonical `Assignment` so it is still driven by the canonical `Circuit::synthesize`. |
| `GpuExpression` | `src/plonk/circuit.rs` | The substrate for the GPU evaluator/keygen AST. Built from the canonical `Expression` via `From`; removing it would require the `GraphEvaluator` to consume the canonical `Expression` directly (a large, separate refactor). |
| permutation `Argument` / `VerifyingKey` forks | `src/plonk/permutation.rs` | Carry GPU-specific commit behavior; built from the canonical permutation argument via `From`. |

## Error model

`GpuError` (`src/plonk/error.rs`) wraps the canonical error in a single
`Canonical(halo2_axiom::plonk::Error)` variant plus the GPU-native `Cuda` and
`HaloGpu` variants. Wrapping rather than re-enumerating keeps the variant set and
their `Display` in lockstep with upstream.
