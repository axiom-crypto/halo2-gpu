#pragma once

#include "field/alt_bn128.hpp"
#include <cuda.h>
#include <cuda_runtime.h>

typedef fr_t scalar_t;

namespace zkpcuda {
namespace permutation_commit {

    __global__ void cuda_kernel_permutation_denominator(
        scalar_t* d_denominators,
        const scalar_t* d_permutations,
        const scalar_t* d_values,
        const scalar_t* d_beta,
        const scalar_t* d_gamma,
        const uint64_t length)
    {
        uint64_t offset = blockIdx.x * blockDim.x + threadIdx.x;
        uint64_t stride = gridDim.x * blockDim.x;

        scalar_t beta = *d_beta;
        scalar_t gamma = *d_gamma;
        for (uint64_t index = offset; index < length; index += stride) {
            if (index >= length)
                return;
            // element-wise load & computation
            scalar_t denominator = d_denominators[index];
            scalar_t permuted_value = d_permutations[index];
            scalar_t value = d_values[index];
            // *denominators *= &(*beta * permuted_value + &*gamma + value);
            scalar_t temp = beta * permuted_value + gamma + value;
            denominator *= temp;
            d_denominators[index] = denominator;
        }
    }

    __global__ void cuda_kernel_permutation_numerator(
        scalar_t* d_numerators,
        const scalar_t* d_values,
        const scalar_t* d_beta,
        const scalar_t* d_gamma,
        const scalar_t* d_omega_lut,
        const scalar_t* d_deltaomega,
        const uint64_t length)
    {
        uint64_t offset = blockIdx.x * blockDim.x + threadIdx.x;
        uint64_t stride = gridDim.x * blockDim.x;

        scalar_t beta = *d_beta;
        scalar_t gamma = *d_gamma;
        scalar_t deltaomega = *d_deltaomega;

        // init deltaomega
        uint32_t omega_idx_thread = threadIdx.x;
        uint32_t omega_idx_block = (blockIdx.x == 0) ? 0 : blockDim.x + (blockIdx.x - 1);
        scalar_t omega = d_omega_lut[omega_idx_thread]; // thread
        scalar_t temp = d_omega_lut[omega_idx_block]; // block
        omega *= temp;
        deltaomega *= omega;

        // omega (stride)
        uint32_t omega_idx_stride = (blockDim.x) + (gridDim.x - 1);
        omega = d_omega_lut[omega_idx_stride];

        for (uint64_t index = offset; index < length; index += stride) {
            if (index >= length)
                return;
            // element-wise load & computation
            scalar_t numerator = d_numerators[index];
            scalar_t value = d_values[index];
            // *numerators *= &(deltaomega * &*beta + &*gamma + value);
            temp = deltaomega * beta + gamma + value;
            numerator *= temp;
            d_numerators[index] = numerator;
            // _deltaomega = _deltaomega * omega.power(index);
            deltaomega *= omega;
        }
    }

    __global__ void cuda_kernel_permutation_numerator_set_one(
        scalar_t* d_numerators,
        const uint64_t length)
    {
        uint64_t offset = blockIdx.x * blockDim.x + threadIdx.x;
        uint64_t stride = gridDim.x * blockDim.x;

        for (uint64_t index = offset; index < length; index += stride) {
            if (index >= length)
                return;
            d_numerators[index] = scalar_t::one();
        }
    }

    template <uint32_t TILE_SIZE>
    __global__ void omega_lut_init(
        scalar_t* d_omega,
        scalar_t* d_omega_lut)
    {
        // e.g. TILE_SIZE = 128
        // init [0-127]
        scalar_t omega = *d_omega;
        scalar_t power = scalar_t::one();
        for (uint32_t j = 0; j < TILE_SIZE; ++j) {
            d_omega_lut[j] = power;
            power *= omega;
        }
        // and [128]
        d_omega_lut[TILE_SIZE] = power;
    }

    template <uint32_t TILE_SIZE>
    __global__ void omega_power_of_block(
        scalar_t* d_omega_lut,
        uint32_t block_num)
    {
        d_omega_lut += TILE_SIZE; // offset: TILE_SIZE
        scalar_t power_of_block = *d_omega_lut; // d_omega_lut[TILE_SIZE]

        // e.g. TILE_SIZE = 128, block_num = 4
        // we need [power_0, power_128, power_256, power_384]
        // power_0 and power_128 are already computed
        // begin with power_128, then power_256, power_384,
        // one more block: power_512 for stride multiplication
        scalar_t power = power_of_block;
        for (uint32_t j = 1; j < block_num; ++j) { //[1,2,3]: 128,256,384
            power *= power_of_block;
            d_omega_lut[j] = power;
        }
    }

    __global__ void cuda_kernel_permutation_multiply(
        scalar_t* d_denominators,
        const scalar_t* d_numerators,
        const uint64_t length)
    {
        uint64_t offset = blockIdx.x * blockDim.x + threadIdx.x;
        uint64_t stride = gridDim.x * blockDim.x;

        for (uint64_t index = offset; index < length; index += stride) {
            if (index >= length)
                return;
            scalar_t denominator = d_denominators[index];
            scalar_t numerator = d_numerators[index];
            denominator *= numerator;
            d_denominators[index] = denominator;
        }
    }

} // namespace permutation_commit
} // namespace zkpcuda
