#pragma once

#include "field/alt_bn128.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

typedef fr_t scalar_t;

// Byte layout of one `Assigned<F>` element in the device-resident raw
// array: the per-element stride and the byte offsets of the numerator
// and denominator field payloads. Passed by value from the Rust caller.
struct assigned_layout_t {
    uint32_t stride_bytes;
    uint32_t num_offset;
    uint32_t denom_offset;
};

namespace zkpcuda {
namespace decode_assigned {

    // Decode a device-resident `[Assigned<F>]` raw-bytes array into
    // separate numerator and denominator device buffers (length `n`
    // each).
    //
    // Reads each `Assigned<F>` element at byte offset
    // `i * stride_bytes` from `d_raw`:
    //   tag byte at offset 0:
    //     0 (Zero)     -> num=0, denom=1
    //     1 (Trivial)  -> num=*((F*)(base + num_offset)),   denom=1
    //     2 (Rational) -> num=*((F*)(base + num_offset)),   denom=*((F*)(base + denom_offset))
    // and writes `d_nums[i]`, `d_denoms[i]`.
    //
    // The Rust caller pins the layout via `#[repr(C, u8)]` on
    // `Assigned<F>` and supplies the byte offsets, so the kernel does
    // not embed Rust-specific layout assumptions beyond the discriminant
    // values.
    //
    // Note: Rust's `Fr` is `align_of == 8` whereas CUDA's `scalar_t`
    // (sppark `mont_t`) is `__align__(16)`, so the F payload at
    // `base + num_offset` is only 8-byte aligned and would trip
    // `cudaErrorMisalignedAddress` under a 16-byte vector load. The
    // limb-wise `mont_t(const uint32_t*)` constructor copies element by
    // element (4-byte alignment is sufficient) and yields a 16-byte
    // aligned stack-local `scalar_t` for the subsequent aligned store
    // into `d_nums` / `d_denoms`.
    __global__ static void decode_assigned_kernel(
        scalar_t* __restrict__ d_nums,            // out: length n
        scalar_t* __restrict__ d_denoms,          // out: length n
        const uint8_t* __restrict__ d_raw,        // in : length n * layout.stride_bytes
        const uint64_t n,
        const assigned_layout_t layout)
    {
        const uint64_t stride = (uint64_t)gridDim.x * (uint64_t)blockDim.x;
        const uint64_t start  = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
        for (uint64_t i = start; i < n; i += stride) {
            const uint8_t* base = d_raw + i * (uint64_t)layout.stride_bytes;
            const uint8_t tag = base[0];
            scalar_t num;
            scalar_t denom = scalar_t::one();
            if (tag == 0) {
                // Assigned::Zero -> num=0, denom=1
                num.zero();
            } else if (tag == 1) {
                // Assigned::Trivial(x) -> num=x, denom=1
                num = scalar_t(reinterpret_cast<const uint32_t*>(base + layout.num_offset));
            } else {
                // Assigned::Rational(n, d) -> num=n, denom=d
                num   = scalar_t(reinterpret_cast<const uint32_t*>(base + layout.num_offset));
                denom = scalar_t(reinterpret_cast<const uint32_t*>(base + layout.denom_offset));
            }
            d_nums[i]   = num;
            d_denoms[i] = denom;
        }
    }

} // namespace decode_assigned
} // namespace zkpcuda
