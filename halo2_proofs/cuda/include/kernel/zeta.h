#pragma once

#include "common.h"
#include "field/alt_bn128.hpp"
#include <cuda.h>
#include <cuda_runtime.h>

typedef fr_t scalar_t;

namespace zkpcuda {
namespace zeta {

    __global__ void cukernel_mul_by_coset(
        scalar_t* d_data,
        const uint32_t k,
        const bool neg,
        const bool revbin)
    {
        static const uint64_t ZETA_1[4] = {
            244305545194690131,
            8351807910065594880,
            14266533074055306532,
            404339206190769364
        };
        static const uint64_t ZETA_2[4] = {
            10657714497315350963,
            9029678389775483239,
            10080386412464207114,
            2070906320917503013
        };
        scalar_t zeta[3];
        zeta[0] = scalar_t::one();
        zeta[1] = scalar_t((const uint32_t*)ZETA_1); // zete^1
        zeta[2] = scalar_t((const uint32_t*)ZETA_2); // zeta^2

        const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
        uint32_t coset_idx = index;
        if (index >= (1 << k)) {
            return;
        }

        // for ifft+cosetFFT, optimize the bit_reverse step between ifft and cosetFFT
        if (revbin) {
            coset_idx = zkpcuda::common::index_revbin(k, coset_idx);
        }
        // a0, a1*zeta, a2*zeta^2, a3, a4*zeta, a5*zeta^2, ...
        coset_idx = coset_idx % 3;
        // for icosetFFT, a0, a1*(zeta^(-1)), a2*(zeta^(-2)), ...
        if (neg) {
            if (coset_idx != 0) {
                coset_idx = 3 - coset_idx;
            }
        }

        d_data[index] *= zeta[coset_idx];
    }

    // coset_factors = [1, zeta^1, zeta^2]
    // zeta^3 == 0
    void mul_by_zeta(
        const uint32_t k,
        const bool neg,
        const bool revbin,
        scalar_t* d_data,
        cudaStream_t stream)
    {
        const uint32_t block_degree = 6; // 1<<6=64
        if (k < block_degree) {
            cukernel_mul_by_coset<<<1, (1 << k), 0, stream>>>(
                d_data, k, neg, revbin);
        } else {
            cukernel_mul_by_coset<<<(1 << (k - block_degree + 1)), (1 << (block_degree - 1)), 0, stream>>>(
                d_data, k, neg, revbin);
        }
    }

} // namespace zeta
} // namespace zkpcuda
