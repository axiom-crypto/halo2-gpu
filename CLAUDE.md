# halo2-gpu Claude Instructions

## Code Review Rules

- Treat obvious performance pitfalls in changed code as review findings, even when they do not change correctness.
- For CUDA changes, check for avoidable global memory traffic, redundant host-device copies, unnecessary synchronizations, divergent branches in hot kernels, non-coalesced access patterns, repeated allocation in hot paths, and expensive recomputation that could be hoisted or cached.
- For Rust changes on GPU-facing paths, check for avoidable cloning, allocation, serialization, locking, or CPU/GPU synchronization in loops and prover hot paths.
- Report performance findings only when the optimization is local and concrete. Include the specific tweak, such as moving work out of a loop, reusing a buffer, batching a transfer, removing a synchronization point, or improving memory access locality.
- Do not report vague performance preferences. If the faster alternative depends on benchmarking, hardware assumptions, or a larger redesign, say nothing unless the PR already includes evidence.
- When reviewing a PR, inspect the full current PR diff and relevant surrounding code before deciding there are no findings.
