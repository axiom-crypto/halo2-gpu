#include <assert.h>
#include <chrono>
#include <map>
#include <string>
#include <vector>

#include "common/exception.h"
#include "common/halo2_ffi.h"
#include "common/scratch_span.h"

#include "kernel/prefix_scan.h"

using Scalar = utils::FFITraitObject;

// halo2_proofs/src/plonk/lookup/prover.rs
//     pub(in crate::plonk) fn commit_product
//         lookup Z(X) grand product
class GrandProductInfo {
public:
    static uint64_t get_required_memory_size(uint64_t poly_length)
    {
        const uint64_t field_size = Scalar::ELT_BYTES;
        const uint64_t poly_size = poly_length * field_size;
        uint64_t required_memory_size = 0;
        required_memory_size += poly_size; // data
        required_memory_size += field_size; // prefix
        return required_memory_size;
    }
};

extern "C" uint64_t _halo2_grand_product_max_len(
    uint64_t poly_length,
    uint64_t free_bytes)
{
    for (uint64_t _length = poly_length; _length > 0; _length = _length >> 1) {
        if (GrandProductInfo::get_required_memory_size(_length) < free_bytes) {
            return _length;
        }
    }
    return 0;
}

extern "C" uint64_t _halo2_grand_product_workspace_size(uint64_t poly_length)
{
    return align_up(GrandProductInfo::get_required_memory_size(poly_length), 32);
}

extern "C" RustError _halo2_grand_product(
    const Scalar* output, // output vec<Field>
    const Scalar* input, // input  vec<Field>
    const Scalar* prefix, // input  Field
    uint64_t poly_length,
    uint64_t poly_offset,
    void* scratch,
    uint64_t scratch_bytes,
    cudaStream_t stream)
{
    ScratchSpan span { (uint8_t*)scratch, (size_t)scratch_bytes };
    uint64_t field_size = Scalar::ELT_BYTES; // bytes
    uint64_t poly_size = poly_length * field_size; // bytes

    uint64_t required_memory_size = poly_size + field_size; // poly and prefix
    uint64_t* d_gpu_mem = (uint64_t*)span.take(required_memory_size);

    // host memory offset
    uint64_t* h_input = (uint64_t*)((char*)(input->ptr) + poly_offset * field_size);
    uint64_t* h_output = (uint64_t*)((char*)(output->ptr) + poly_offset * field_size);
    // gpu memory ptr
    uint64_t* d_input = d_gpu_mem;
    uint64_t* d_prefix = (uint64_t*)((char*)(d_input) + poly_size);

    // run
    try {
        CUDA_OK(cudaMemcpyAsync(d_input, h_input, poly_size, cudaMemcpyHostToDevice, stream));
        CUDA_OK(cudaMemcpyAsync(d_prefix, prefix->ptr, field_size, cudaMemcpyHostToDevice, stream));
        const uint32_t acc_per_thread = 4;
        const uint32_t tiles_per_block = 256;
        const uint32_t element_per_block = tiles_per_block * acc_per_thread;
        uint32_t block_num = (poly_length + element_per_block - 1) / element_per_block;
        // fist round
        uint32_t round_stride = 1;
        // printf(" block_num:%d  round_stride:%d\r\n", block_num, round_stride);
        zkpcuda::commit_product::prefix_scan_block<acc_per_thread, tiles_per_block>
            <<<block_num, tiles_per_block, 0, stream>>>(
                (scalar_t*)d_input,
                (scalar_t*)d_prefix,
                poly_length,
                round_stride);
        // subsequent rounds
        while (block_num > 1) {
            block_num = (block_num + element_per_block - 1) / element_per_block;
            round_stride = round_stride * element_per_block;
            zkpcuda::commit_product::prefix_scan_block<acc_per_thread, tiles_per_block>
                <<<block_num, tiles_per_block, 0, stream>>>(
                    (scalar_t*)d_input,
                    (scalar_t*)d_prefix,
                    poly_length,
                    round_stride);
            // printf(" block_num:%d round_stride:%d\r\n", block_num, round_stride);
        }
        // block downsweep
        while (round_stride > element_per_block) {
            uint64_t low_level_round_stride = round_stride / element_per_block;
            uint64_t node_num = (poly_length + low_level_round_stride - 1) / low_level_round_stride;
            uint64_t block_num = (node_num + 256 - 1) / 256;
            // printf("round_stride:%d node_num:%d block_num:%d \r\n", round_stride, node_num, block_num);
            zkpcuda::commit_product::prefix_scan_block_downsweep<<<block_num, 256, 0, stream>>>(
                (scalar_t*)d_input,
                poly_length,
                round_stride,
                element_per_block);
            round_stride = low_level_round_stride;
        }
        // epilogue
        uint32_t epilog_block_num = (poly_length + 256 - 1) / 256;
        zkpcuda::commit_product::prefix_scan_epilogue<<<epilog_block_num, 256, 0, stream>>>(
            (scalar_t*)d_input,
            poly_length,
            element_per_block);
    } catch (const cuda_error& error) {
        return RustError(error.code(), error.what());
    };

    // copy to host
    try {
        CUDA_OK(cudaMemcpyAsync(h_output, d_input, poly_size, cudaMemcpyDeviceToHost, stream));
        CUDA_OK(cudaStreamSynchronize(stream));
    } catch (const cuda_error& error) {
        return RustError(error.code(), error.what());
    };
    return cudaSuccess;
}

// Device-input variant of `_halo2_grand_product`. Computes the prefix-scan
// grand product in place over the chunk of `d_inout` at `poly_offset`;
// `d_prefix->ptr` is a device-resident 32-byte running prefix. The scan
// result is left on `d_inout`; copying it back to a host buffer is the
// Rust caller's responsibility.
extern "C" RustError _halo2_grand_product_device_inputs(
    const Scalar* d_inout, // inout vec<Field>, device pointer; scan mutates in place
    const Scalar* d_prefix, // input Field, device pointer (32 bytes)
    uint64_t poly_length,
    uint64_t poly_offset,
    cudaStream_t stream)
{
    uint64_t field_size = Scalar::ELT_BYTES; // bytes

    // device input + prefix (caller-owned)
    uint64_t* d_input = (uint64_t*)((char*)(d_inout->ptr) + poly_offset * field_size);
    uint64_t* d_prefix_ptr = (uint64_t*)(d_prefix->ptr);

    try {
        const uint32_t acc_per_thread = 4;
        const uint32_t tiles_per_block = 256;
        const uint32_t element_per_block = tiles_per_block * acc_per_thread;
        uint32_t block_num = (poly_length + element_per_block - 1) / element_per_block;
        uint32_t round_stride = 1;
        zkpcuda::commit_product::prefix_scan_block<acc_per_thread, tiles_per_block>
            <<<block_num, tiles_per_block, 0, stream>>>(
                (scalar_t*)d_input,
                (scalar_t*)d_prefix_ptr,
                poly_length,
                round_stride);
        while (block_num > 1) {
            block_num = (block_num + element_per_block - 1) / element_per_block;
            round_stride = round_stride * element_per_block;
            zkpcuda::commit_product::prefix_scan_block<acc_per_thread, tiles_per_block>
                <<<block_num, tiles_per_block, 0, stream>>>(
                    (scalar_t*)d_input,
                    (scalar_t*)d_prefix_ptr,
                    poly_length,
                    round_stride);
        }
        while (round_stride > element_per_block) {
            uint64_t low_level_round_stride = round_stride / element_per_block;
            uint64_t node_num = (poly_length + low_level_round_stride - 1) / low_level_round_stride;
            uint64_t down_block_num = (node_num + 256 - 1) / 256;
            zkpcuda::commit_product::prefix_scan_block_downsweep<<<down_block_num, 256, 0, stream>>>(
                (scalar_t*)d_input,
                poly_length,
                round_stride,
                element_per_block);
            round_stride = low_level_round_stride;
        }
        uint32_t epilog_block_num = (poly_length + 256 - 1) / 256;
        zkpcuda::commit_product::prefix_scan_epilogue<<<epilog_block_num, 256, 0, stream>>>(
            (scalar_t*)d_input,
            poly_length,
            element_per_block);
    } catch (const cuda_error& error) {
        return RustError(error.code(), error.what());
    };

    return cudaSuccess;
}
