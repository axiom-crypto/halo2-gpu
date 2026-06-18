#pragma once

#include "field/alt_bn128.hpp"
#include <cuda.h>
#include <cuda_runtime.h>

typedef fr_t scalar_t;

namespace zkpcuda {
namespace common {

    // =====================================
    // ============== scalar ===============
    // =====================================

    __global__ void cukernel_scale_by_constant(
        scalar_t* d_data,
        scalar_t* d_constant,
        const uint32_t k)
    {
        const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= (1 << k)) {
            return;
        }
        d_data[index] *= d_constant[0];
    }

    static inline void scale_by_constant(
        cudaStream_t& stream,
        scalar_t* d_data,
        scalar_t* d_constant,
        const uint32_t k)
    {
        const uint32_t block_degree = 6; // 1<<6=64
        if (k < block_degree) {
            cukernel_scale_by_constant<<<1, 1 << k, 0, stream>>>(d_data, d_constant, k);
        } else {
            cukernel_scale_by_constant<<<(1 << (k - block_degree + 1)), (1 << (block_degree - 1)), 0, stream>>>(
                d_data, d_constant, k);
        }
    }

    // =====================================
    // ============== revbin ===============
    // =====================================
    __device__ __forceinline__ uint32_t index_revbin(
        const uint32_t k,
        const uint32_t index)
    {
        uint32_t result;
        asm("brev.b32 %0, %1;"
            : "=r"(result)
            : "r"(index));
        return result >> (32 - k);
    }

    __global__ __launch_bounds__(1024) void cukernel_revbin(
        scalar_t* d_data,
        const uint32_t k)
    {
        const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
        const uint32_t rev_idx = index_revbin(k, index);
        if (index >= (1 << k)) {
            return;
        }

        if (index < rev_idx) {
            scalar_t a = d_data[index];
            scalar_t b = d_data[rev_idx];
            d_data[index] = b;
            d_data[rev_idx] = a;
        }
    }

    static inline void revbin(
        cudaStream_t stream,
        scalar_t* d_data,
        const uint32_t k)
    {
        const uint32_t num_thread = 64;
        const uint32_t num_block = ((1 << k) + num_thread - 1) / num_thread;
        cukernel_revbin<<<num_block, num_thread, 0, stream>>>(d_data, k);
    }

    // =====================================
    // ============== normalize ===============
    // =====================================
    __global__ void cukernel_normalize(
        scalar_t* d_data,
        const uint64_t length)
    {
        const uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= length)
            return;

        scalar_t zero;
        zero.zero();
        d_data[index] += zero;
    }

    static inline void normalize(cudaStream_t& stream, scalar_t* d_data, const uint64_t k)
    {
        const uint64_t block_degree = 6; // 1<<6=64
        const uint64_t length = 1 << k;
        if (k < block_degree) {
            cukernel_normalize<<<1, (1 << block_degree), 0, stream>>>(d_data, length);
        } else {
            cukernel_normalize<<<(1 << (k - block_degree + 1)), (1 << (block_degree - 1)), 0, stream>>>(d_data, length);
        }
    }

} // namespace common
} // namespace zkpcuda
