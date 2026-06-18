#pragma once

#include "curve/jacobian_t.hpp"
#include "curve/xyzz_t.hpp"
#include "field/alt_bn128.hpp"

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

typedef jacobian_t<fp_t> point_t;
typedef xyzz_t<fp_t> bucket_t;
typedef bucket_t::affine_t affine_t;
typedef fr_t scalar_t;

namespace zkpcuda {
namespace pippenger {

    // ===============================================================================
    // ============================== PRE PROCESS ====================================
    // ===============================================================================

    __global__ void cukernel_from_monty(scalar_t* d_scalar, size_t n)
    {
        uint64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx >= n)
            return;

        scalar_t s = d_scalar[idx];
        s.from();
        d_scalar[idx] = s;
    }

#define DIGIT_BITS 64

    template <int ScalarBit, int ScalarLimbs>
    __global__ void cukernel_preprocess_scalars1(
        const uint64_t* d_scalars,
        uint64_t length,
        uint32_t win_bit,
        uint32_t win_num,
        uint32_t* d_count)
    {
        const uint32_t thread_idx = threadIdx.x;
        const uint32_t block_idx = blockIdx.x;
        const uint32_t block_num = gridDim.x;

        const uint64_t limbs_total = length * ScalarLimbs; // u64 total
        const uint64_t limbs_per_block = limbs_total / block_num; // u64 per block
        const uint64_t limbs_start = block_idx * limbs_per_block + thread_idx;
        const uint64_t limbs_end = block_idx != block_num - 1 // last blcock
            ? (block_idx + 1) * limbs_per_block
            : limbs_total;
        const uint32_t limbs_stride = blockDim.x; // threads per block

        const uint64_t bucket_mask = (1U << win_bit) - 1U;
        int32_t r, bottom_bits, pos_old;
        uint64_t s, s_plus;
        (void)pos_old; (void)s;
        for (uint64_t idx = limbs_start; idx < limbs_end; idx += limbs_stride) {
            if (idx >= limbs_end)
                return;

            uint64_t limb_data = d_scalars[idx];
            // [0,3], limbs idx in scalar
            uint32_t internal_limbs_idx = idx % ScalarLimbs;
            int32_t addr_start = internal_limbs_idx * DIGIT_BITS;
            int32_t addr_end = (internal_limbs_idx + 1) * DIGIT_BITS - 1;
            if (internal_limbs_idx == (ScalarBit - 1) / 64) // top limb: handles bits 704-753 or 192-254
                addr_end = win_bit * ((ScalarBit + win_bit - 1) / win_bit) - 1;

            int32_t i = addr_end - (addr_end % win_bit);
            int32_t W = win_num - 1 - (i / win_bit);
            while (i >= addr_start) {
                r = i % DIGIT_BITS;
                uint64_t win = (limb_data >> r) & bucket_mask;
                // Handle case where win_bit doesn't divide DIGIT_BITS
                bottom_bits = DIGIT_BITS - r;
                // detect when window overlaps digit boundary
                if (bottom_bits < win_bit && (internal_limbs_idx != (ScalarBit - 1) / 64)) {
                    s_plus = *(d_scalars + idx + 1); // need to read the next limb
                    win |= (s_plus << bottom_bits) & bucket_mask;
                }
                if (win != 0) {
                    atomicAdd(&d_count[W * (1 << win_bit) + win], 1);
                }
                W++;
                i -= win_bit;
            }
            __syncthreads();
        }
    }

    template <int ScalarBit, int ScalarLimbs>
    __global__ void cukernel_preprocess_scalars2(
        int win_bit,
        int win_num,
        int* d_wins_start,
        int* d_wins_end,
        int* count,
        int* pos,
        size_t N)
    {
        const int T = threadIdx.x, B = blockIdx.x, D = blockDim.x;
        (void)D;
        uint32_t bucket_num = (1 << win_bit);

        if (B < win_num && T == 0) {
            d_wins_start[B * bucket_num + 0] = 0;
            pos[B * bucket_num + 0] = 0;
            for (int i = 1; i < bucket_num; i++) {
                int temp = d_wins_start[B * bucket_num + i - 1] + count[B * bucket_num + i - 1];
                d_wins_start[B * bucket_num + i] = temp;
                pos[B * bucket_num + i] = temp;
                d_wins_end[B * bucket_num + i - 1] = temp;
            }
            d_wins_end[B * bucket_num + bucket_num - 1] = d_wins_start[B * bucket_num + bucket_num - 1] + count[B * bucket_num + bucket_num - 1];
        }
    }

    template <int ScalarBit, int ScalarLimbs>
    __global__ void cukernel_preprocess_scalars3(
        const int win_bit,
        const int win_num,
        const uint64_t* __restrict__ scalars,
        int* __restrict__ d_wins,
        int* __restrict__ pos,
        const size_t N)
    {
        const int T = threadIdx.x, B = blockIdx.x, D = blockDim.x, G = gridDim.x;
        const int batch_size = N / G; // number of uint64_t words each block processes
        const uint64_t bucket_mask = (1U << win_bit) - 1U;
        int n, b, end, addr_start, addr_end, q, i, r, bottom_bits, pos_old, W;
        (void)n; (void)b; (void)addr_start; (void)addr_end;
        uint64_t s, win, s_plus;
        bool if_plus = false;

        end = (B + 1) * batch_size;
        if (B == G - 1)
            end = N;

        i = win_bit * ((ScalarBit + win_bit - 1) / win_bit);
        for (int w = 0; w < win_num; ++w) {
            i -= win_bit;
            W = win_num - 1 - i / win_bit;
            q = i / DIGIT_BITS;
            r = i % DIGIT_BITS;
            bottom_bits = DIGIT_BITS - r;
            if_plus = (bottom_bits < win_bit) && ((q + 1) < ScalarLimbs);
            s_plus = 0UL;
            for (int j = B * batch_size + T; j < end; j += D) {
                s = scalars[j * ScalarLimbs + q];
                if (if_plus) {
                    s_plus = scalars[j * ScalarLimbs + q + 1];
                }
                win = (s >> r) & bucket_mask;
                win |= (s_plus << bottom_bits) & bucket_mask;

                if (win != 0) {
                    pos_old = atomicAdd(&pos[W * (1 << win_bit) + win], 1);
                    d_wins[W * N + pos_old] = j;
                    // __stcg(d_wins + W * N + pos_old, j);
                }
            }
            __syncthreads();
        }
    }

#undef DIGIT_BITS

    template <int ScalarBit, int ScalarLimbs>
    void preprocess_scalars(
        cudaStream_t& stream,
        uint64_t* d_scalars,
        size_t length,
        int win_bit,
        int win_num,
        int* d_pos,
        int* d_count,
        int* d_wins,
        int* d_wins_start,
        int* d_wins_end)
    {
        uint64_t bucket_num = (1 << win_bit) * win_num;
        cudaMemsetAsync(d_pos, 0, bucket_num * sizeof(uint32_t), stream);
        cudaMemsetAsync(d_count, 0, bucket_num * sizeof(uint32_t), stream);

        // convert scalars from montgomery form(aR) to integer form(a)
        cukernel_from_monty<<<(length + 127) / 128, 128, 0, stream>>>((scalar_t*)d_scalars, length);

        int process_parallel_level = 80;
        cukernel_preprocess_scalars1<ScalarBit, ScalarLimbs>
            <<<process_parallel_level, 256, 0, stream>>>((const uint64_t*)d_scalars, (uint64_t)length, win_bit, win_num, (uint32_t*)d_count);
        cukernel_preprocess_scalars2<ScalarBit, ScalarLimbs>
            <<<win_num, 1, 0, stream>>>(win_bit, win_num, d_wins_start, d_wins_end, d_count, d_pos, length);
        cukernel_preprocess_scalars3<ScalarBit, ScalarLimbs>
            <<<process_parallel_level, 256, 0, stream>>>(win_bit, win_num, d_scalars, d_wins, d_pos, length);
    }

    // ===============================================================================
    // ============================= MIXED ADD =======================================
    // ===============================================================================

    __global__ void cukernel_check_sparsity(
        int* d_wins_start,
        int* d_wins_end,
        float* d_sparsity,
        float sparsity_threshold,
        size_t scalars_num,
        int window_num,
        int bucket_num)
    {

        uint32_t global_idx = blockDim.x * blockIdx.x + threadIdx.x;
        uint32_t window_idx = global_idx / bucket_num;
        uint32_t bucket_idx = global_idx % bucket_num;
        if (window_idx >= window_num)
            return;
        if (bucket_idx >= bucket_num)
            return;

        uint32_t begin = d_wins_start[window_idx * bucket_num + bucket_idx];
        uint32_t end = d_wins_end[window_idx * bucket_num + bucket_idx];
        uint32_t num_element = end - begin;
        float sparsity = (float)num_element / (float)scalars_num;
        d_sparsity[window_idx * bucket_num + bucket_idx] = sparsity;
    }

    template <uint32_t TILE_PER_BLOCK>
    __global__ void cukernel_mixed_add_points_in_bucket(
        point_t* d_out, const affine_t* d_point, size_t point_size,
        const bool skip_dense, const float* d_sparsity, float sparsity_threshold,
        const int* d_wins, int* d_wins_start, int* d_wins_end,
        int window_size, int window_offset, int bucket_size)
    {

        uint32_t win_idx = window_offset + blockIdx.x / bucket_size;
        uint32_t bin_idx = blockIdx.x % bucket_size;
        if (win_idx >= window_size)
            return;
        if (skip_dense) {
            if (d_sparsity[win_idx * bucket_size + bin_idx] >= sparsity_threshold)
                return;
        }
        uint32_t begin = d_wins_start[win_idx * bucket_size + bin_idx];
        uint32_t end = d_wins_end[win_idx * bucket_size + bin_idx];
        uint32_t points_in_bucket = end - begin;
        d_wins = d_wins + win_idx * point_size + begin;

        // mixed add
        uint32_t point_idx = threadIdx.x;
        bucket_t sum;
        sum.inf();
        for (uint32_t i = point_idx; i < points_in_bucket; i += TILE_PER_BLOCK) {
            if (i < points_in_bucket) {
                sum.add(d_point[d_wins[i]]);
            }
        }
        __syncthreads();
        __shared__ bucket_t tile_bucket[TILE_PER_BLOCK];
        tile_bucket[point_idx] = sum;
        __syncthreads();

        // in block reduction
        for (int m = 1; m < TILE_PER_BLOCK; m *= 2) {
            int base = point_idx * (m * 2);
            if ((base + m) < TILE_PER_BLOCK) {
                bucket_t a1 = tile_bucket[base];
                bucket_t a2 = tile_bucket[base + m];
                a1.add(a2);
                tile_bucket[base] = a1;
            }
            __syncthreads();
        }

        uint64_t out_idx = win_idx * bucket_size + bin_idx;
        if (point_idx == 0) {
            bucket_t sum = tile_bucket[0];
            d_out[out_idx] = (point_t)sum;
        }
    }

    template <uint32_t TILE_PER_BLOCK>
    __global__ void cukernel_mixed_add_dense_bucket(
        bucket_t* d_out, const affine_t* d_point, size_t point_size,
        const int* d_wins, int* d_wins_start, int* d_wins_end,
        uint32_t window_size, uint32_t win_idx, uint32_t bucket_size, uint32_t bin_idx)
    {

        uint32_t begin = d_wins_start[win_idx * bucket_size + bin_idx];
        uint32_t end = d_wins_end[win_idx * bucket_size + bin_idx];
        uint32_t points_in_bucket = end - begin;
        uint32_t points_per_block = points_in_bucket / gridDim.x; // floor, not ceil
        begin = begin + blockIdx.x * points_per_block;
        d_wins = d_wins + win_idx * point_size + begin;
        uint32_t points_in_last_block = points_in_bucket - blockIdx.x * points_per_block;
        uint32_t length = blockIdx.x == (gridDim.x - 1) ? points_in_last_block : points_per_block;

        if (blockIdx.x * points_per_block > points_in_bucket) {
            bucket_t sum;
            sum.inf();
            if (threadIdx.x == 0) {
                d_out[blockIdx.x] = sum;
            }
            return;
        }

        // mixed add
        uint32_t point_idx = threadIdx.x;
        bucket_t sum;
        sum.inf();
        for (uint32_t i = point_idx; i < length; i += TILE_PER_BLOCK) {
            if (i < length) {
                sum.add(d_point[d_wins[i]]);
            }
        }
        __syncthreads();
        __shared__ bucket_t tile_bucket[TILE_PER_BLOCK];
        tile_bucket[point_idx] = sum;
        __syncthreads();

        // in block reduction
        for (int m = 1; m < TILE_PER_BLOCK; m *= 2) {
            int base = point_idx * (m * 2);
            if ((base + m) < TILE_PER_BLOCK) {
                bucket_t a1 = tile_bucket[base];
                bucket_t a2 = tile_bucket[base + m];
                a1.add(a2);
                tile_bucket[base] = a1;
            }
            __syncthreads();
        }

        if (point_idx == 0) {
            d_out[blockIdx.x] = tile_bucket[0];
        }
    }

    template <uint32_t TILE_PER_BLOCK>
    __global__ void cukernel_dense_bucket_acc_results(
        bucket_t* d_in, point_t* d_out,
        uint32_t in_num, uint32_t window_offset, uint32_t bucket_num, uint32_t bucket_offset)
    {
        // mixed add
        uint32_t point_idx = threadIdx.x;
        bucket_t sum;
        sum.inf();
        for (uint32_t i = point_idx; i < in_num; i += TILE_PER_BLOCK) {
            if (i < in_num) {
                sum.add(d_in[i]);
            }
        }
        __syncthreads();
        __shared__ bucket_t tile_bucket[TILE_PER_BLOCK];
        tile_bucket[point_idx] = sum;
        __syncthreads();

        // in block reduction
        for (int m = 1; m < TILE_PER_BLOCK; m *= 2) {
            int base = point_idx * (m * 2);
            if ((base + m) < TILE_PER_BLOCK) {
                bucket_t a1 = tile_bucket[base];
                bucket_t a2 = tile_bucket[base + m];
                a1.add(a2);
                tile_bucket[base] = a1;
            }
            __syncthreads();
        }

        uint64_t out_idx = window_offset * bucket_num + bucket_offset;
        if (point_idx == 0) {
            bucket_t sum = tile_bucket[0];
            d_out[out_idx] = (point_t)sum;
        }
    }

    template <uint32_t TILE_PER_BLOCK>
    void mixed_add_wins(cudaStream_t& _stream,
        point_t* d_out, const affine_t* d_point, size_t point_num,
        float SPARSITY_THRESHOLD, float* _sparsity, uint64_t* _dense_out,
        int* d_wins, int* d_wins_start, int* d_wins_end, uint32_t win_num, uint32_t bin_num)
    {

        // check sparsity
        float* d_sparsity = _sparsity;
        uint64_t total_bin_num = win_num * bin_num;
        uint64_t thread_per_block = 64;
        uint64_t block_num = (total_bin_num + thread_per_block - 1) / thread_per_block;
        cukernel_check_sparsity<<<block_num, thread_per_block, 0, _stream>>>(
            d_wins_start, d_wins_end,
            d_sparsity, SPARSITY_THRESHOLD,
            point_num, win_num, bin_num);

        // Load-bearing D->H sync: the host loop below branches on
        // `h_sparsity[i]` to dispatch dense-bucket kernels.
        float* h_sparsity = (float*)malloc(win_num * bin_num * sizeof(float));
        cudaMemcpyAsync(h_sparsity, d_sparsity, win_num * bin_num * sizeof(float), cudaMemcpyDeviceToHost, _stream);
        cudaStreamSynchronize(_stream);
        uint32_t dense_cnt = 0;
        for (uint32_t i = 0; i < win_num * bin_num; i++) {
            if (h_sparsity[i] >= SPARSITY_THRESHOLD) {
                dense_cnt++;
            }
        }

        uint32_t SPLIT_N_BLOCKS = 128;
        // 128 block per bucket, do not change!!!!
        // for d_dense_out is malloced at the beginning
        bucket_t* d_dense_out = (bucket_t*)_dense_out;
        if (dense_cnt > 0) {
            uint32_t offset_cnt = 0;
            for (uint32_t w = 0; w < win_num; w++) {
                for (uint32_t b = 0; b < bin_num; b++) {
                    if (h_sparsity[w * bin_num + b] >= SPARSITY_THRESHOLD) {
                        // printf("sparsity of [%d][%d] is [%f] \r\n", w, b, h_sparsity[w*bin_num + b]);
                        bucket_t* d_dense_offset = d_dense_out + (offset_cnt * 128);
                        if (SPLIT_N_BLOCKS * TILE_PER_BLOCK > point_num) {
                            SPLIT_N_BLOCKS = point_num / TILE_PER_BLOCK;
                        }
                        cukernel_mixed_add_dense_bucket<TILE_PER_BLOCK><<<SPLIT_N_BLOCKS, TILE_PER_BLOCK, 0, _stream>>>(
                            d_dense_offset, d_point, point_num,
                            d_wins, d_wins_start, d_wins_end,
                            win_num, w, bin_num, b);
                        cukernel_dense_bucket_acc_results<TILE_PER_BLOCK><<<1, TILE_PER_BLOCK, 0, _stream>>>(
                            d_dense_offset, d_out,
                            SPLIT_N_BLOCKS, w, bin_num, b);
                        offset_cnt++;
                    }
                }
            }
        }

        cukernel_mixed_add_points_in_bucket<TILE_PER_BLOCK><<<win_num * bin_num, TILE_PER_BLOCK, 0, _stream>>>(
            d_out, d_point, point_num,
            true /*skip dense bucket*/, d_sparsity, SPARSITY_THRESHOLD,
            d_wins, d_wins_start, d_wins_end,
            win_num, 0, bin_num);

        if (cudaPeekAtLastError() != cudaSuccess) {
            printf("zkpcuda::sppark_pippenger::mixed_add_wins Error : %s\n",
                cudaGetErrorString(cudaPeekAtLastError()));
        }

        if (h_sparsity != nullptr)
            free(h_sparsity);
        h_sparsity = nullptr;
    }

    // ===============================================================================
    // ============================= POST PROCESS ====================================
    // ===============================================================================

    template <uint32_t TILE_PER_BLOCK>
    __global__ void cukernel_reduce_all_wins_to_one(
        point_t* d_inout,
        uint32_t win_bit,
        uint32_t win_num,
        uint32_t elements_per_block)
    {
        // Block0: Bucket1
        // Block(2^win_bit-2): Bucket(2^win_bit-1)
        uint32_t tile_idx = threadIdx.x;
        uint32_t bucket_idx = blockIdx.x * elements_per_block + tile_idx;

        point_t acc;
        acc.identity();
        uint32_t bucket_num = 1 << win_bit;
        for (int i = 0; i < win_num; i++) {
            // multiply Q[i] by 2^win_bit
            // EC::mul_2exp(win_bit, acc, acc);
            acc.add(acc);
            for (int k = 1; k < win_bit; ++k) {
                acc.add(acc);
            }
            // add Q[i]
            if (bucket_num * i + bucket_idx < bucket_num * win_num)
                acc.add(d_inout[bucket_num * i + bucket_idx]);
        }
        // if TILE_PER_BLOCK does not divide bucket_num or bucket_num is smaller than 32, which happens when length is smaller 
        // than 1024, then OOB
        if (bucket_idx < bucket_num)
          d_inout[bucket_idx] = acc;
    }

    template <uint32_t TILE_PER_BLOCK>
    __global__ void cukernel_reduce_buckets_in_block_then_shift(
        const point_t* d_buckets,
        point_t* d_buckets_tmp,
        uint32_t win_bit,
        uint32_t win_bit_half)
    {
        __shared__ point_t tile_bucket[TILE_PER_BLOCK];
        const uint32_t tile_idx = threadIdx.x;
        const uint64_t block_offset = blockIdx.x * (1 << win_bit_half);

        point_t acc;
        acc.identity();
        for (uint32_t j = tile_idx; j < 1 << win_bit_half; j += TILE_PER_BLOCK) {
            if (j < 1 << win_bit_half) {
                acc.add(d_buckets[block_offset + j]);
            }
        }

        // in block reduction
        tile_bucket[tile_idx] = acc;
        __syncthreads();
        for (uint32_t stride = 1; stride < TILE_PER_BLOCK; stride *= 2) {
            if (tile_idx % (2 * stride) == 0) {
                tile_bucket[tile_idx].add(tile_bucket[tile_idx + stride]);
            }
            __syncthreads();
        }

        uint32_t block_bits = (1 << win_bit_half) * blockIdx.x;
        if (tile_idx == 0) {
            acc.identity();
            point_t res = tile_bucket[0];
            for (; block_bits > 0; block_bits >>= 1) {
                if (block_bits & 1) {
                    acc.add(res);
                }
                res.add(res); // res.dbl();
            }
            uint64_t out_off = blockIdx.x;
            d_buckets_tmp[out_off] = acc;
        }
    }

    template <uint32_t TILE_PER_BLOCK>
    __global__ void cukernel_reduce_buckets_across_block_then_shift(
        const point_t* d_buckets,
        point_t* d_buckets_tmp,
        uint32_t win_bit,
        uint32_t win_bit_half)
    {
        __shared__ point_t tile_bucket[TILE_PER_BLOCK];
        const uint32_t tile_idx = threadIdx.x;
        const uint64_t block_offset = blockIdx.x;

        point_t acc;
        acc.identity();
        for (uint32_t j = tile_idx; j < 1 << (win_bit - win_bit_half); j += TILE_PER_BLOCK) {
            if (j < 1 << win_bit_half) {
                const uint64_t tile_offset = j * (1 << win_bit_half);
                acc.add(d_buckets[block_offset + tile_offset]);
            }
        }

        // in block reduction
        tile_bucket[tile_idx] = acc;
        __syncthreads();
        for (uint32_t stride = 1; stride < TILE_PER_BLOCK; stride *= 2) {
            if (tile_idx % (2 * stride) == 0) {
                tile_bucket[tile_idx].add(tile_bucket[tile_idx + stride]);
            }
            __syncthreads();
        }

        uint32_t block_bits = blockIdx.x;
        if (tile_idx == 0) {
            acc.identity();
            point_t res = tile_bucket[0];
            for (; block_bits > 0; block_bits >>= 1) {
                if (block_bits & 1) {
                    acc.add(res);
                }
                res.add(res); // res.dbl();
            }
            uint64_t out_off = (1 << (win_bit - win_bit_half)) + blockIdx.x;
            d_buckets_tmp[out_off] = acc;
        }
    }

    template <uint32_t TILE_PER_BLOCK>
    __global__ void cukernel_ec_sum_intermidiates(point_t* X, point_t* Y, size_t n)
    {
        uint64_t idx = blockIdx.x * TILE_PER_BLOCK + threadIdx.x;
        if (idx >= n)
            return;

        point_t z, x, y;
        z.identity();
        z.add(X[idx]);
        z.add(Y[idx]);
        X[idx] = z;
    }

    template <uint32_t TILE_PER_BLOCK>
    __global__ void cukernel_to_affine(point_t* d_res, size_t n)
    {
        uint64_t idx = blockIdx.x * TILE_PER_BLOCK + threadIdx.x;
        if (idx >= n)
            return;

        point_t jacobian = d_res[idx];
        point_t affine = jacobian.to_affine();
        d_res[idx] = affine;
    }

    template <uint32_t TILE_PER_BLOCK>
    uint64_t* postprocess_buckets(
        cudaStream_t& stream,
        point_t* d_buckets,
        point_t* d_buckets_tmp,
        uint32_t win_bit,
        uint32_t win_bit_half,
        uint32_t win_num)
    {
        // reduce all d_wins to on one win with 2^win_bit buckets
        uint32_t bucket_num = (1 << win_bit);
        uint32_t bucket_per_block = TILE_PER_BLOCK;
        uint32_t block_num = (bucket_num + bucket_per_block - 1) / bucket_per_block;
        cukernel_reduce_all_wins_to_one<TILE_PER_BLOCK>
            <<<block_num, bucket_per_block, 0, stream>>>(
                d_buckets, win_bit, win_num, bucket_per_block);
        // reduce buckets to intermidiate results
        // 1. in block reduce
        // 2. across block reduce
        cukernel_reduce_buckets_in_block_then_shift<TILE_PER_BLOCK>
            <<<1 << (win_bit - win_bit_half), TILE_PER_BLOCK, 0, stream>>>(
                d_buckets, d_buckets_tmp, win_bit, win_bit_half);
        cukernel_reduce_buckets_across_block_then_shift<TILE_PER_BLOCK>
            <<<1 << win_bit_half, TILE_PER_BLOCK, 0, stream>>>(
                d_buckets, d_buckets_tmp, win_bit, win_bit_half);

        uint64_t n = (1 << (win_bit - win_bit_half)) + (1 << win_bit_half);
        uint64_t r = n & 1, m = n / 2;
        for (; m != 0; r = m & 1, m >>= 1) {
            // add in-block results (shifted)
            uint64_t block_num = (m + TILE_PER_BLOCK - 1) / TILE_PER_BLOCK;
            cukernel_ec_sum_intermidiates<TILE_PER_BLOCK>
                <<<block_num, TILE_PER_BLOCK, 0, stream>>>(d_buckets_tmp, d_buckets_tmp + m, m);
            if (r) // add across block results if current bit is 1
                cukernel_ec_sum_intermidiates<TILE_PER_BLOCK>
                    <<<1, TILE_PER_BLOCK, 0, stream>>>(d_buckets_tmp, d_buckets_tmp + 2 * m, 1);
        }
        // :::WARNING
        // As of https://github.com/privacy-scaling-explorations/halo2curves/pull/19, halo2curves has changed to `CurveExt` using homogeneous coordinates
        // Meanwhile CUDA still uses Jacobian coordinates
        // Therefore we convert to affine coordinates here to avoid ambiguity.
        // :::
        cukernel_to_affine<TILE_PER_BLOCK><<<1, 1, 0, stream>>>(d_buckets_tmp, 1);
        return (uint64_t*)d_buckets_tmp;
    }

} // namespace pippenger
} // namespace zkpcuda
