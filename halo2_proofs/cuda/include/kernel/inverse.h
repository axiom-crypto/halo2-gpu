#pragma once

#include "field/alt_bn128.hpp"
#include <cuda.h>
#include <cuda_runtime.h>

namespace zkpcuda {
namespace operation {

    template <uint32_t inverse_per_thread>
    __global__ void cuda_kernel_batch_invert_fr(
        fr_t* d_inout,
        uint64_t length)
    {
        uint64_t inv_per_block = blockDim.x * inverse_per_thread;
        uint64_t index = blockIdx.x * inv_per_block + threadIdx.x * inverse_per_thread;
        if (index >= length)
            return;

        fr_t acc[inverse_per_thread];
        fr_t acc_gether[inverse_per_thread];
        acc_gether[0].zero(); // set zero

// forward gether
#pragma unroll
        for (uint32_t i = 0; i < inverse_per_thread; i++) {
            if (index + i >= length)
                acc[i].zero(); // set zero
            else
                acc[i] = d_inout[index + i];
        }
        uint32_t acc_cnt = 0;
        for (uint32_t i = 0; i < inverse_per_thread; i++) {
            if (acc[i].is_zero())
                continue;
            if (acc_cnt == 0)
                acc_gether[0] = acc[i];
            else
                acc_gether[acc_cnt] = acc_gether[acc_cnt - 1] * acc[i];
            ++acc_cnt;
        }

        if (acc_cnt == 0 || acc_gether[acc_cnt - 1].is_zero())
            return;
        // batch inv: (abcd) >>> (1/abcd)
        fr_t batch_inv = acc_gether[acc_cnt - 1].inverse();

        // backward scatter
        for (int32_t i = inverse_per_thread - 1; i >= 1; --i) {
            if (acc[i].is_zero())
                continue;
            --acc_cnt;
            d_inout[index + i] = (acc_cnt > 0)
                ? batch_inv * acc_gether[acc_cnt - 1] // (1/d) = (1/abcd)*(abc)
                : batch_inv;
            batch_inv = batch_inv * acc[i]; // (1/abc) = (1/abcd)*(d)
        }

        if (acc[0].is_zero())
            return;
        // result[0] = (1/ab)*(b) = (1/a) = batch_inv
        d_inout[index] = batch_inv;
    }

    // failed to use template<typename scalar_t>
    // so add a duplicate function for fp_t here
    template <uint32_t inverse_per_thread>
    __global__ void cuda_kernel_batch_invert_fp(
        fp_t* d_inout,
        uint64_t length)
    {
        uint64_t inv_per_block = blockDim.x * inverse_per_thread;
        uint64_t index = blockIdx.x * inv_per_block + threadIdx.x * inverse_per_thread;
        if (index >= length)
            return;

        fp_t acc[inverse_per_thread];
        fp_t acc_gether[inverse_per_thread];
        acc_gether[0].zero(); // set zero

// forward gether
#pragma unroll
        for (uint32_t i = 0; i < inverse_per_thread; i++) {
            if (index + i >= length)
                acc[i].zero(); // set zero
            else
                acc[i] = d_inout[index + i];
        }
        uint32_t acc_cnt = 0;
        for (uint32_t i = 0; i < inverse_per_thread; i++) {
            if (acc[i].is_zero())
                continue;
            if (acc_cnt == 0)
                acc_gether[0] = acc[i];
            else
                acc_gether[acc_cnt] = acc_gether[acc_cnt - 1] * acc[i];
            ++acc_cnt;
        }

        if (acc_cnt == 0 || acc_gether[acc_cnt - 1].is_zero())
            return;
        // batch inv: (abcd) >>> (1/abcd)
        fp_t batch_inv = acc_gether[acc_cnt - 1].inverse();

        // backward scatter
        for (int32_t i = inverse_per_thread - 1; i >= 1; --i) {
            if (acc[i].is_zero())
                continue;
            --acc_cnt;
            d_inout[index + i] = (acc_cnt > 0)
                ? batch_inv * acc_gether[acc_cnt - 1] // (1/d) = (1/abcd)*(abc)
                : batch_inv;
            batch_inv = batch_inv * acc[i]; // (1/abc) = (1/abcd)*(d)
        }

        if (acc[0].is_zero())
            return;
        // result[0] = (1/ab)*(b) = (1/a) = batch_inv
        d_inout[index] = batch_inv;
    }

    /* 0: Fr, 1: Fp */
    template <uint32_t field_type>
    void batch_invert(
        cudaStream_t& stream,
        uint64_t* d_scalars,
        uint64_t length)
    {
        const uint32_t thread_num = 64;
        const uint32_t inverse_per_thread = 8;
        const uint32_t inverse_per_block = thread_num * inverse_per_thread;
        const uint32_t block_num = (length + inverse_per_block - 1) / inverse_per_block;

        if (field_type == 0) { // Fr
            cuda_kernel_batch_invert_fr<inverse_per_thread>
                <<<block_num, thread_num, 0, stream>>>((fr_t*)d_scalars, length);
        } else if (field_type == 1) { // Fp
            cuda_kernel_batch_invert_fp<inverse_per_thread>
                <<<block_num, thread_num, 0, stream>>>((fp_t*)d_scalars, length);
        }
    }

} // namespace operation
} // namespace zkpcuda
