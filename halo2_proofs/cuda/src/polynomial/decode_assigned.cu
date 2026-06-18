#include <stdint.h>

#include "common/exception.h"
#include "common/halo2_ffi.h"

#include "kernel/decode_assigned.h"

// Decode a device-resident `[Assigned<F>]` raw-bytes array into
// separate numerator and denominator device buffers.
//
// Layout (set by `#[repr(C, u8)]` on `Assigned<F>` in
// `halo2_proofs/src/plonk/assigned.rs`):
//
//   discriminant tag (u8) at offset 0
//     0 = Zero,  1 = Trivial,  2 = Rational
//   first  F payload at byte offset `num_offset`
//   second F payload at byte offset `denom_offset`  (Rational only)
//   total per-element size = `stride_bytes`
//
// The Rust caller (`decode_assigned_to_num_denom_device` in
// `halo2_proofs/src/cuda/funcs/polynomial_ops.rs`) derives these offsets
// from `size_of` / `align_of` and validates them with a runtime probe
// (`verify_assigned_layout`) before invoking the launcher; the kernel
// itself only encodes the variant-discriminant mapping.
//
// `n == 0` is a no-op (no kernel launch); both output buffers must have
// at least `n` scalars of capacity.
extern "C" RustError _halo2_decode_assigned(
    void* d_nums,                  // out: length n   scalars
    void* d_denoms,                // out: length n   scalars
    const void* d_raw,             // in : length n * layout.stride_bytes
    uint64_t n,
    assigned_layout_t layout,
    cudaStream_t stream)
{
    if (n == 0) {
        return cudaSuccess;
    }
    try {
        const uint32_t block_num = 512;
        const uint32_t tile_size = 256;
        zkpcuda::decode_assigned::decode_assigned_kernel
            <<<block_num, tile_size, 0, stream>>>(
                (scalar_t*)d_nums,
                (scalar_t*)d_denoms,
                (const uint8_t*)d_raw,
                n,
                layout);
    } catch (const cuda_error& error) {
        return RustError(error.code(), error.what());
    };
    return cudaSuccess;
}
