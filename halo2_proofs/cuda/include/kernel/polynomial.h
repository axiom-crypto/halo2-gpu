#pragma once

#include "field/alt_bn128.hpp"
#include <cuda.h>
#include <cuda_runtime.h>

typedef fr_t scalar_t;

namespace zkpcuda {
namespace polynomial {

    // result[i * num_parts + p] = parts[p][i]
    //
    // 1D strided gather. Each thread j computes (i, p) = (j / num_parts,
    // j % num_parts) and writes the corresponding element.
    __global__ static void extended_from_lagrange_vec_kernel(
        scalar_t* d_out,                       // length n * num_parts
        const scalar_t* const* d_parts,        // [num_parts] device pointers
        const uint32_t num_parts,
        const uint64_t n)
    {
        const uint64_t stride = (uint64_t)gridDim.x * (uint64_t)blockDim.x;
        const uint64_t start  = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
        const uint64_t n_total = n * (uint64_t)num_parts;
        for (uint64_t j = start; j < n_total; j += stride) {
            const uint64_t i = j / (uint64_t)num_parts;
            const uint32_t p = (uint32_t)(j - i * (uint64_t)num_parts);
            d_out[j] = d_parts[p][i];
        }
    }

    // a[i] *= t_evaluations[i % t_len]
    //
    // t_len is the quotient polynomial degree.
    __global__ static void divide_by_vanishing_poly_kernel(
        scalar_t* d_poly,            // in/out, length n
        const scalar_t* d_t_evals,   // length t_len
        const uint32_t t_len,
        const uint64_t n)
    {
        const uint32_t block_idx = blockIdx.x;
        const uint32_t tile_size = blockDim.x;
        const uint64_t tile_idx = threadIdx.x;
        const uint64_t stride = gridDim.x * tile_size;
        const uint64_t index = block_idx * tile_size + tile_idx;

        for (uint64_t idx = index; idx < n; idx += stride) {
            const uint32_t t_idx = (uint32_t)(idx % (uint64_t)t_len);
            scalar_t coeff = d_poly[idx];
            scalar_t t_eval = d_t_evals[t_idx];
            d_poly[idx] = coeff * t_eval;
        }
    }

    // a[i] *= coset_powers[i_mod - 1] when i_mod != 0; identity at i_mod == 0
    //
    // Uses modulus `coset_powers_len + 1`.
    __global__ static void distribute_powers_zeta_kernel(
        scalar_t* d_a,                       // in/out, length n
        const scalar_t* d_coset_powers,      // length coset_powers_len
        const uint32_t coset_powers_len,
        const uint64_t n)
    {
        const uint64_t stride = (uint64_t)gridDim.x * (uint64_t)blockDim.x;
        const uint64_t start  = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
        const uint32_t modulus = coset_powers_len + 1u;
        for (uint64_t i = start; i < n; i += stride) {
            const uint32_t i_mod = (uint32_t)(i % (uint64_t)modulus);
            if (i_mod != 0) {
                scalar_t a = d_a[i];
                a = a * d_coset_powers[i_mod - 1];
                d_a[i] = a;
            }
        }
    }

    // `d_out[i] = d_a[i] * d_b[i]` elementwise over `length` field elements.
    // Output aliasing with either input is safe (per-element kernel).
    __global__ static void poly_elementwise_multiply(
        scalar_t* d_out,
        const scalar_t* d_a,
        const scalar_t* d_b,
        const uint64_t length)
    {
        const uint64_t stride = (uint64_t)gridDim.x * (uint64_t)blockDim.x;
        const uint64_t start  = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
        for (uint64_t idx = start; idx < length; idx += stride) {
            d_out[idx] = d_a[idx] * d_b[idx];
        }
    }

    // FMA: `d_poly_sum[i] += scalar * d_poly_in[i]`; aliasing `d_poly_sum == d_poly_in` is safe.
    __global__ static void poly_multiply_add(
        scalar_t* d_poly_sum,
        const scalar_t* d_poly_in,
        const scalar_t* d_scalar,
        const uint64_t poly_length)
    {
        const uint32_t block_idx = blockIdx.x;
        const uint32_t tile_size = blockDim.x;
        const uint64_t tile_idx = threadIdx.x;
        const uint64_t stride = gridDim.x * tile_size;
        const uint64_t index = block_idx * tile_size + tile_idx;

        scalar_t scalar = *d_scalar;
        scalar_t poly_sum, poly_in;
        for (uint64_t idx = index; idx < poly_length; idx += stride) {
            if (idx >= poly_length)
                break;
            poly_in = d_poly_in[idx];
            poly_sum = d_poly_sum[idx];

            poly_sum += poly_in * scalar;
            d_poly_sum[idx] = poly_sum;
        }
    }

    // Broadcast-fill: `d_out[i] = *d_scalar` for i in [0, length). The fill
    // value is a device-resident scalar, so the caller uploads it once (one
    // small H2D) instead of staging a full length-sized host buffer.
    __global__ static void poly_fill_scalar(
        scalar_t* d_out,
        const scalar_t* d_scalar,
        const uint64_t length)
    {
        const uint64_t stride = (uint64_t)gridDim.x * (uint64_t)blockDim.x;
        const uint64_t start = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
        const scalar_t v = *d_scalar;
        for (uint64_t idx = start; idx < length; idx += stride) {
            d_out[idx] = v;
        }
    }

    // use 1 thread to init
    template <uint32_t TILE_SIZE>
    __global__ __launch_bounds__(1) void power_of_scalar_init(
        const scalar_t* d_point_in,
        scalar_t* d_point_lut)
    {
        // e.g. TILE_SIZE = 256
        // init [0-255]
        scalar_t power = scalar_t::one();
        scalar_t point = *d_point_in;

        for (int j = 0; j < TILE_SIZE; ++j) {
            d_point_lut[j] = power;
            power = power * point;
        }
        // and [256]
        d_point_lut[TILE_SIZE] = power;
    }

    // use 1 thread to init
    template <uint32_t TILE_SIZE>
    __global__ __launch_bounds__(1) void power_of_scalar_block(
        scalar_t* d_point_lut,
        const uint64_t block_num)
    {
        // offset
        scalar_t power_block;
        d_point_lut += TILE_SIZE;
        power_block = *d_point_lut;

        // e.g. TILE_SIZE = 256, block_num = 4
        // power_0 and power_256 are ready
        // begin from power_512, then power_768
        // one more block: power_1024 for power_stride
        scalar_t power = power_block;
        for (uint32_t j = 1; j < block_num; ++j) { //[1,2,3]: 512,768,1024
            power = power * power_block;
            d_point_lut[j] = power;
        }
    }

    template <uint32_t TILE_SIZE>
    __global__ void eval_polynomial_batch(
        const scalar_t* d_poly,
        const scalar_t* d_point_lut,
        scalar_t* d_batch_res,
        const uint64_t length)
    {
        const uint32_t block_idx = (blockIdx.x == 0) ? 0 : TILE_SIZE + (blockIdx.x - 1);
        const uint32_t tile_idx = threadIdx.x;
        const uint32_t stride_idx = TILE_SIZE + (gridDim.x - 1);

        scalar_t power, power_block, power_stride;
        power_block = d_point_lut[block_idx]; // load from block offset
        power = d_point_lut[tile_idx]; // load from tile range
        power = power * power_block; // current power in global range
        power_stride = d_point_lut[stride_idx]; // load from stride offset

        scalar_t coeff, acc;
        acc.zero();
        uint64_t offset = TILE_SIZE * blockIdx.x + tile_idx;
        uint64_t stride = TILE_SIZE * gridDim.x;
        for (uint64_t idx = offset; idx < length; idx += stride) {
            if (idx >= length)
                break;
            coeff = d_poly[idx];
            acc += coeff * power;
            power *= power_stride;
        }

        __shared__ scalar_t shared_acc[TILE_SIZE];
        shared_acc[tile_idx] = acc;
        __syncthreads();

        for (uint32_t m = 1; m < TILE_SIZE; m = m * 2) {
            uint32_t base = tile_idx * (m * 2);
            if ((base + m) < TILE_SIZE) {
                scalar_t a1 = shared_acc[base];
                scalar_t a2 = shared_acc[base + m];
                acc = a1 + a2;
                shared_acc[base] = acc;
            }
            __syncthreads();
        }

        if (tile_idx == 0) {
            d_batch_res[blockIdx.x] = shared_acc[0];
        }
    }

    __global__ __launch_bounds__(1) void eval_polynomial_epilogue(
        scalar_t* d_output,
        const scalar_t* d_batch_res,
        const uint64_t result_num)
    {
        scalar_t acc;
        acc.zero();
        for (uint64_t idx = 0; idx < result_num; idx += 1) {
            acc += d_batch_res[idx];
        }

        d_output[0] = acc;
    }

    // `d_acc[i] -= d_short[i]` for i in [0, short_len). Sparse-prefix
    // subtract: the long accumulator is only touched in its short prefix.
    __global__ static void poly_sub_short_inplace(
        scalar_t* d_acc,
        const scalar_t* d_short,
        const uint64_t short_len)
    {
        const uint64_t stride = (uint64_t)gridDim.x * (uint64_t)blockDim.x;
        const uint64_t start = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
        for (uint64_t idx = start; idx < short_len; idx += stride) {
            scalar_t a = d_acc[idx];
            scalar_t b = d_short[idx];
            d_acc[idx] = a - b;
        }
    }

    // Out-of-place dual-consumer sibling of `poly_sub_short_inplace`:
    //   d_out[i] = d_long[i] - d_short[i]   for i in [0, short_len)
    //   d_out[i] = d_long[i]                for i in [short_len, long_len)
    // Lets a caller produce a fresh `d_out` from `d_long` without an
    // intervening D2D clone, preserving `d_long` for a second consumer.
    __global__ static void poly_sub_short_out_of_place(
        scalar_t* d_out,
        const scalar_t* d_long,
        const scalar_t* d_short,
        const uint64_t short_len,
        const uint64_t long_len)
    {
        const uint64_t stride = (uint64_t)gridDim.x * (uint64_t)blockDim.x;
        const uint64_t start = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
        for (uint64_t idx = start; idx < long_len; idx += stride) {
            scalar_t a = d_long[idx];
            if (idx < short_len) {
                scalar_t b = d_short[idx];
                d_out[idx] = a - b;
            } else {
                d_out[idx] = a;
            }
        }
    }

    // `d_buf[0] -= *d_scalar`. Single index-0 subtract on device. One
    // thread writes; aliasing irrelevant (single element).
    __global__ __launch_bounds__(1) void poly_sub_scalar_at_zero(
        scalar_t* d_buf,
        const scalar_t* d_scalar)
    {
        scalar_t a = d_buf[0];
        scalar_t b = *d_scalar;
        d_buf[0] = a - b;
    }

} // namespace polynomial
} // namespace zkpcuda
