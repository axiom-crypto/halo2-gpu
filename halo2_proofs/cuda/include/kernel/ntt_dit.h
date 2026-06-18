#pragma once

#include "field/alt_bn128.hpp"
#include "omega.h"
#include <cuda.h>
#include <cuda_runtime.h>

typedef fr_t scalar_t;

namespace zkpcuda {
namespace ntt {

#ifndef ZKPCUDA_OPT_SHARED_INPUTS
#define ZKPCUDA_OPT_SHARED_INPUTS
#endif

    // DIT NTT butterfly: bit-reversed input, natural-order output.
    __global__ void cuda_kernel_ntt_dit(
        const uint32_t k,
        const uint32_t block_k,
        const uint32_t combine_size,
        const uint32_t level0,
        const bool is_twiddle_dense,
        const scalar_t* d_twiddle,
        scalar_t* d_data)
    {
        const uint32_t block_idx = blockIdx.x;
        const uint32_t tile_idx = threadIdx.x;

        const uint32_t before_combine_slot_idx = tile_idx & ((1 << (block_k - 1)) - 1);
        const uint32_t before_combine_block_idx = block_idx * (1 << combine_size) + tile_idx / (1 << (block_k - 1));
        const uint32_t after_combine_block_idx = tile_idx / (1 << (block_k - 1));

        const uint32_t block_lower_mask = (1U << level0) - 1U;
        const uint32_t before_combine_block_lower = before_combine_block_idx & block_lower_mask;
        const uint32_t before_combine_block_offset = before_combine_block_lower | ((before_combine_block_idx & ~block_lower_mask) << block_k);

        const uint32_t read_slot_idx = tile_idx / (1 << combine_size);
        const uint32_t read_block_idx = block_idx * (1 << combine_size) + (tile_idx & ((1 << combine_size) - 1));
        const uint32_t after_read_block_idx = (tile_idx & ((1 << combine_size) - 1));
        const uint32_t read_block_lower = read_block_idx & block_lower_mask;
        const uint32_t read_block_offset = read_block_lower | ((read_block_idx & ~block_lower_mask) << block_k);

#ifdef ZKPCUDA_OPT_SHARED_INPUTS
        extern __shared__ scalar_t shm_data[];

        if (level0 != 0 && combine_size != 0) {
#pragma unroll
            for (uint32_t b = 0; b < 2; ++b) {
                const uint32_t read_ext_slot_idx = b << (block_k - 1) | read_slot_idx;
                shm_data[(after_read_block_idx << block_k) | read_ext_slot_idx] = d_data[(read_ext_slot_idx << level0 | read_block_offset)];
            }
        } else {
#pragma unroll
            for (uint32_t b = 0; b < 2; ++b) {
                const uint32_t ext_slot_idx = b << (block_k - 1) | before_combine_slot_idx;
                shm_data[(after_combine_block_idx << block_k) | ext_slot_idx] = d_data[(ext_slot_idx << level0 | before_combine_block_offset)];
            }
        }

        __syncthreads();
#endif

        for (uint32_t level = 0; level < block_k; ++level) {
            const uint32_t data_idx_mask = (1U << level) - 1U;

#ifdef ZKPCUDA_OPT_SHARED_INPUTS
            const uint32_t data_idx_0 = (((tile_idx & data_idx_mask) | (tile_idx & ~data_idx_mask) << 1));
            const uint32_t data_idx_1 = data_idx_0 | 1U << level;
#else
            const uint32_t data_idx_0 = ((((tile_idx & data_idx_mask) | (tile_idx & ~data_idx_mask) << 1))
                                            << level0)
                | block_offset;
            const uint32_t data_idx_1 = data_idx_0 | (1U << (level + level0));
#endif

            scalar_t a, b;
#ifdef ZKPCUDA_OPT_SHARED_INPUTS
            a = shm_data[data_idx_0];
            b = shm_data[data_idx_1];
#else
            a = d_data[data_idx_0];
            b = d_data[data_idx_1];
#endif

            // load twiddle
            scalar_t w;
            const uint32_t map_to_omega = 1 << (k - (level + level0) - 1);
            const uint32_t twiddle_idx = before_combine_block_lower | ((before_combine_slot_idx & data_idx_mask) << level0);
            if (is_twiddle_dense) {
                w = d_twiddle[twiddle_idx * map_to_omega];
            } else { // sparse twiddle
                scalar_t w1, w2;
                uint64_t low_degree_lut_len = 1 << DENSE_POWER_DEGREE;
                const scalar_t* twiddles_low = d_twiddle;
                const scalar_t* twiddles_high = d_twiddle + low_degree_lut_len;
                const uint32_t twiddle_idx_low = (twiddle_idx * map_to_omega) & (low_degree_lut_len - 1);
                const uint32_t twiddle_idx_high = (twiddle_idx * map_to_omega) >> DENSE_POWER_DEGREE;
                w1 = twiddles_low[twiddle_idx_low];
                w2 = twiddles_high[twiddle_idx_high];
                w = w1 * w2;
            }

            // radix-2 butterfly
            b = b * w;
            scalar_t a0 = a + b;
            scalar_t a1 = a - b;

#ifdef ZKPCUDA_OPT_SHARED_INPUTS
            shm_data[data_idx_0] = a0;
            shm_data[data_idx_1] = a1;
#else
            d_data[data_idx_0] = a0;
            d_data[data_idx_1] = a1;
#endif
            __syncthreads();
        }

#ifdef ZKPCUDA_OPT_SHARED_INPUTS
        if (level0 != 0 && combine_size != 0) {
            for (uint32_t b = 0; b < 2; ++b) {
                const uint32_t read_ext_slot_idx = b << (block_k - 1) | read_slot_idx;
                d_data[(read_ext_slot_idx << level0 | read_block_offset)] = shm_data[(after_read_block_idx << block_k) | read_ext_slot_idx];
            }
        } else {
            for (uint32_t b = 0; b < 2; ++b) {
                const uint32_t ext_slot_idx = b << (block_k - 1) | before_combine_slot_idx;
                d_data[(ext_slot_idx << level0 | before_combine_block_offset)] = shm_data[(after_combine_block_idx << block_k) | ext_slot_idx];
            }
        }
#endif
    }

    void dit_module(
        const uint32_t k,
        const uint32_t block_k,
        const uint32_t rest_block_k,
        const uint32_t combinate_size_1,
        const uint32_t combinate_size_2,
        bool is_twiddle_dense,
        const scalar_t* d_twiddle,
        scalar_t* d_data,
        cudaStream_t stream)
    {
        for (uint32_t level0 = 0; level0 + block_k <= k; level0 += block_k) {
            // TODO: griddim can exceed 65536
#ifdef ZKPCUDA_OPT_SHARED_INPUTS
            const uint32_t grid_dim = (1 << (k - block_k)) / (1 << combinate_size_1);
            const uint32_t thread_dim = (1 << (block_k - 1)) * (1 << combinate_size_1);
            const uint32_t shared_size = (scalar_t::nbytes << block_k) * (1 << combinate_size_1);
#else
            const uint32_t grid_dim = 1 << (k - block_k);
            const uint32_t thread_dim = 1 << (block_k - 1);
            const uint32_t shared_size = 0;
#endif
            cuda_kernel_ntt_dit<<<grid_dim, thread_dim, shared_size, stream>>>(
                k, block_k, combinate_size_1, level0, is_twiddle_dense, d_twiddle, d_data);
        }
        if (block_k > 1 && rest_block_k > 0) {
#ifdef ZKPCUDA_OPT_SHARED_INPUTS
            const uint32_t grid_dim = (1 << (k - rest_block_k)) / (1 << combinate_size_2);
            const uint32_t thread_dim = (1 << (rest_block_k - 1)) * (1 << combinate_size_2);
            const uint32_t shared_size = (scalar_t::nbytes << rest_block_k) * (1 << combinate_size_2);
#else
            const uint32_t grid_dim = 1 << (k - rest_block_k);
            const uint32_t thread_dim = 1 << (rest_block_k - 1);
            const uint32_t shared_size = 0;
#endif
            cuda_kernel_ntt_dit<<<grid_dim, thread_dim, shared_size, stream>>>(
                k, rest_block_k, combinate_size_2, k - rest_block_k, is_twiddle_dense, d_twiddle, d_data);
        }
    }

} // namespace ntt
} // namespace zkpcuda