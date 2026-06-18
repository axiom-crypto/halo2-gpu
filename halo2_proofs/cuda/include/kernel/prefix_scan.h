#pragma once

#include "field/alt_bn128.hpp"
#include <cuda.h>
#include <cuda_runtime.h>

typedef fr_t scalar_t;

namespace zkpcuda {
namespace commit_product {

    // Multiplicative prefix scan, Brent-Kung construction.
    // Reference: https://research.nvidia.com/sites/default/files/pubs/2016-03_Single-pass-Parallel-Prefix/nvr-2016-002.pdf (Fig. 1d).
    template <uint32_t ACC_PER_THREAD, uint32_t SHARED_DATA>
    __global__ void prefix_scan_block(
        scalar_t* d_inout,
        scalar_t* d_prefix,
        uint64_t length,
        uint64_t round_stride)
    {
        scalar_t acc_res = scalar_t::one();
        scalar_t acc_data[ACC_PER_THREAD];
        __shared__ scalar_t shared_mem[SHARED_DATA];

        uint32_t tile_idx = threadIdx.x;
        uint32_t shared_elem_per_block = blockDim.x;
        uint64_t index = (blockIdx.x * blockDim.x + threadIdx.x) * ACC_PER_THREAD;
        bool first_round = (round_stride == 1);

        // first thread in first round load prefix
        if (index == 0 && first_round) {
            acc_res = *d_prefix;
        }

#pragma unroll
        for (uint32_t i = 0; i < ACC_PER_THREAD; i++) {
            scalar_t data_in = scalar_t::one();
            uint64_t offset = first_round ? index + i : (index + i + 1) * round_stride - 1;
            if (offset < length) {
                data_in = d_inout[offset];
            }
            acc_res *= data_in;
            acc_data[i] = acc_res;
        }
        shared_mem[tile_idx] = acc_res;
        __syncthreads();

        // https://research.nvidia.com/sites/default/files/pubs/2016-03_Single-pass-Parallel-Prefix/nvr-2016-002.pdf
        // Brent-Kung construction in Fig.1d
        // upsweep
        uint32_t stride = 2;
        uint32_t src_offset = 0;
        uint32_t dst_offset = stride - 1;
        for (uint32_t idx = shared_elem_per_block >> 1; idx > 0; idx = idx >> 1) {
            if (tile_idx < idx) {
                scalar_t dst = shared_mem[tile_idx * stride + dst_offset];
                scalar_t src = shared_mem[tile_idx * stride + src_offset];
                dst *= src;
                shared_mem[tile_idx * stride + dst_offset] = dst;
            }
            src_offset = stride - 1;
            stride = stride << 1;
            dst_offset = stride - 1;
            __syncthreads();
        }

        // downsweep
        for (uint32_t stride = shared_elem_per_block; stride > 1; stride = stride >> 1) {
            src_offset = stride - 1;
            dst_offset = src_offset + (stride >> 1);
            uint32_t thread_in_round = shared_elem_per_block / stride - 1;
            if (tile_idx < thread_in_round) {
                scalar_t dst = shared_mem[tile_idx * stride + dst_offset];
                scalar_t src = shared_mem[tile_idx * stride + src_offset];
                dst *= src;
                shared_mem[tile_idx * stride + dst_offset] = dst;
            }
            __syncthreads();
        }

        // scan and write back
        scalar_t prefix_sum = scalar_t::one();
        if (tile_idx > 0)
            prefix_sum = shared_mem[tile_idx - 1];

#pragma unroll
        for (uint32_t i = 0; i < ACC_PER_THREAD; i++) {
            uint64_t offset = first_round ? index + i : (index + i + 1) * round_stride - 1;
            if (offset < length) {
                d_inout[offset] = prefix_sum * acc_data[i];
            }
        }
    }

    __global__ void prefix_scan_block_downsweep(
        scalar_t* d_inout,
        uint64_t length,
        uint64_t round_stride,
        uint64_t basic_level)
    {
        uint64_t low_level_round_stride = round_stride / basic_level;
        uint64_t dst_index = (blockIdx.x * blockDim.x + threadIdx.x) * low_level_round_stride - 1;
        uint64_t src_index = (dst_index / round_stride) * round_stride - 1; // last element in last "data block"
        bool is_last_elem = dst_index % round_stride == round_stride - 1; // bypass
        uint64_t level_offset = dst_index / round_stride;
        if (dst_index >= length || is_last_elem || level_offset == 0)
            return;

        scalar_t dst = d_inout[dst_index];
        scalar_t src = d_inout[src_index];
        d_inout[dst_index] = dst * src;
    }

    __global__ void prefix_scan_epilogue(
        scalar_t* d_inout,
        uint64_t length,
        uint64_t basic_level)
    {
        uint64_t dst_index = blockIdx.x * blockDim.x + threadIdx.x;
        uint64_t level_offset = dst_index / basic_level;
        uint64_t src_index = level_offset * basic_level - 1;
        bool is_last_elem_in_level = dst_index % basic_level == basic_level - 1;
        if (level_offset == 0 || dst_index >= length)
            return;
        if (is_last_elem_in_level)
            return;

        scalar_t dst = d_inout[dst_index];
        scalar_t prefix_sum = d_inout[src_index];
        d_inout[dst_index] = dst * prefix_sum;
    }

} // namespace commit_product
} // namespace zkpcuda
