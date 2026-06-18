#pragma once

#include "field/alt_bn128.hpp"
#include <cuda.h>
#include <cuda_runtime.h>

typedef fr_t scalar_t;

namespace zkpcuda {
namespace commit_product {

    __global__ void cuda_kernel_lookup_denominator(
        scalar_t* d_lookup_product,
        const scalar_t* d_permuted_input,
        const scalar_t* d_permuted_table,
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
            scalar_t permuted_input = d_permuted_input[index];
            scalar_t permuted_table = d_permuted_table[index];
            // *lookup_product = (*beta + permuted_input_value) * &(*gamma + permuted_table_value);
            permuted_input += beta;
            permuted_table += gamma;
            d_lookup_product[index] = permuted_input * permuted_table;
        }
    }

    __global__ void cuda_kernel_lookup_numerator(
        scalar_t* d_lookup_product,
        const scalar_t* d_compressed_input,
        const scalar_t* d_compressed_table,
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
            scalar_t compressed_input = d_compressed_input[index];
            scalar_t compressed_table = d_compressed_table[index];
            scalar_t lookup_product = d_lookup_product[index];
            // *product *= &(compressed_input[i] + &beta);
            // *product *= &(compressed_table[i] + &gamma);
            compressed_input += beta;
            compressed_table += gamma;
            lookup_product *= compressed_input;
            lookup_product *= compressed_table;
            d_lookup_product[index] = lookup_product;
        }
    }

} // namespace commit_product
} // namespace zkpcuda
