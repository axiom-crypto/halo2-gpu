#pragma once

#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include <cuda_runtime.h>

// Byte-span handed down from a Rust-allocated `DeviceBuffer<u8>` into C++
// kernels. C++ sub-allocates via `take()`, which advances the cursor and
// returns an aligned sub-slice. The underlying allocation (VPMM /
// `cudaMallocAsync`) is guaranteed 256-byte aligned by CUDA; inside the
// span we align each `take` to 32 bytes by default, sufficient for all
// current vectorized accesses (largest is `uint4` = 16 B in
// `kernel/quotient.h`).
//
// Sizing contract: every kernel that consumes a `ScratchSpan` is paired
// with a pure host preflight `_halo2_<kernel>_workspace_size(...)` that
// must return a byte count >= the sum of `align_up(sub_n, align)` for
// every `take(sub_n, align)` the kernel performs. Because Rust always
// hands us a pointer aligned to >= 32 (CUDA allocator gives 256-byte
// alignment), the preflight does not need to budget for leading pad on
// the very first `take`. Alignment must be a power of two; the API
// traps on violations rather than producing silent misalignment.
//
// Trap-on-violation: we use `__trap()` (device) / `std::abort()` (host)
// rather than `assert`, so guards fire in NDEBUG release builds — that
// is precisely when a wrong preflight would otherwise scribble OOB.
__host__ __device__ inline void scratch_span_trap()
{
#if defined(__CUDA_ARCH__)
    __trap();
#else
    std::abort();
#endif
}

struct ScratchSpan {
    uint8_t* ptr;
    size_t bytes;

    __host__ __device__ void* take(size_t n, size_t align = 32)
    {
        // Power-of-two alignment is required for the bitmask formula below.
        if (align == 0 || (align & (align - 1)) != 0) {
            scratch_span_trap();
        }
        uintptr_t cur = reinterpret_cast<uintptr_t>(ptr);
        // Guard the alignment-rounding addition itself against wrap.
        if (cur > UINTPTR_MAX - (align - 1)) {
            scratch_span_trap();
        }
        uintptr_t aligned = (cur + (align - 1)) & ~(uintptr_t)(align - 1);
        size_t pad = aligned - cur;
        // Subtraction form: equivalent to `pad + n > bytes` but cannot wrap.
        if (pad > bytes || n > bytes - pad) {
            scratch_span_trap();
        }
        void* out = reinterpret_cast<void*>(aligned);
        ptr = reinterpret_cast<uint8_t*>(aligned + n);
        bytes -= (pad + n);
        return out;
    }
};

// Round `x` up to a multiple of `a` (power-of-two only). Used inside
// `_halo2_<kernel>_workspace_size` to sum sub-buffer sizes with the same
// alignment that `ScratchSpan::take` applies at runtime. Traps on
// invalid alignment or on overflow of the rounding step.
__host__ __device__ inline uint64_t align_up(uint64_t x, uint64_t a)
{
    if (a == 0 || (a & (a - 1)) != 0) {
        scratch_span_trap();
    }
    if (x > UINT64_MAX - (a - 1)) {
        scratch_span_trap();
    }
    return (x + (a - 1)) & ~(a - 1);
}
