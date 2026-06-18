#pragma once

#include "common/scratch_span.h"
#include "field/alt_bn128.hpp"
#include <cuda.h>
#include <cuda_runtime.h>

typedef fr_t scalar_t;

/// do not change this variable,
/// it's also hardcoded in the Rust side.
/// halo2_proofs/src/arithmetic.rs
#define DENSE_POWER_DEGREE 10

namespace zkpcuda {
namespace omega {

    // GPU twiddle generation: build the omega power table on device.
    // use 1 thead to generate omega_lut
    // e.g. for degree= 9
    // omega_lut = [omega^(1<<9), omega^(1<<8), ..., omega^(1<<0)]
    __global__ __launch_bounds__(1) void cukernel_prepare_powers_init(
        scalar_t* d_omega_lut,
        scalar_t* d_omega,
        const uint32_t log_n)
    {
        // load omega/omega_inv from d_omega_lut[0]
        scalar_t omega;
        omega = d_omega_lut[0];

        // reverse order: 9, 8, 7, ..., 0
        // power: [512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
        // omega^(1<<log_n) = omega^(0) = 1
        for (int32_t idx = log_n; idx >= 0; --idx) {
            d_omega_lut[idx] = omega;
            omega *= omega;
        }

        // store omega^(1<<log_n) to d_omega
        // force write one
        d_omega[0] = scalar_t::one();
    }

    __global__ static void cukernel_prepare_powers(
        scalar_t* d_omega_lut,
        scalar_t* d_omega,
        const uint32_t n,
        const uint32_t noffset,
        const uint32_t nstep)
    {
        extern __shared__ scalar_t shared_powers[];
        const uint32_t tile_idx = threadIdx.x;
        const uint32_t block_idx = blockIdx.x;

        if (tile_idx == 0) {
            uint64_t omega_index = (block_idx << (n - noffset));
            shared_powers[0] = d_omega[omega_index]; // init with omega^n
        }
        __syncthreads();

        for (uint32_t level = 0; level < nstep; ++level) {
            if (tile_idx < (1 << level)) {
                const uint32_t id0 = tile_idx << (nstep - level);
                const uint32_t id1 = id0 | 1 << (nstep - level - 1);
                const uint64_t omega_index = (noffset + level + 2);
                scalar_t b_r = shared_powers[id0];
                scalar_t a_r = d_omega_lut[omega_index];
                shared_powers[id1] = b_r * a_r;
            }
            __syncthreads();
        }

        if (tile_idx < (1 << nstep)) {
            uint64_t omega_index = (block_idx << nstep | tile_idx) << (n - noffset - nstep);
            d_omega[omega_index] = shared_powers[tile_idx];
        }
    }

    void generate_omega(
        cudaStream_t& stream,
        scalar_t* d_omega,
        scalar_t* d_omega_lut,
        const uint32_t log_n)
    {
#define ELT_BYTES 32
#define PREPARE(offset, step) \
    cukernel_prepare_powers<<<1 << offset, 1 << step, ELT_BYTES << step, stream>>>(d_omega_lut, d_omega, log_n, offset, step)

        // 1<<6=64, 64*4=256 threads
        const uint32_t generator_block_n = 6;
        cukernel_prepare_powers_init<<<1, 1, 0, stream>>>(
            d_omega_lut,
            d_omega,
            log_n + 1 // must +1, tech debt
        );

        if ((log_n) % generator_block_n != 0) {
            PREPARE(0, (log_n) % generator_block_n);
        }
        for (uint32_t count = (log_n) % generator_block_n; count < (log_n); count += generator_block_n) {
            PREPARE(count, generator_block_n);
        }

#undef ELT_BYTES
#undef PREPARE
    }
    // generate omega_lut
    // e.g. for degree = 10, size_of_lut = 'one' + 10 = 11
    // omega_lut = [omega^(0), omega^(1<<0), omega^(1<<1), ..., omega^(1<<9)]
    __global__ static void generate_omega_log_lut(
        scalar_t* d_omega_lut,
        scalar_t* d_omega,
        const uint32_t log_n)
    {
        // init
        // load omega/omega_inv from d_omega[0]
        scalar_t omega = d_omega[0];
        d_omega_lut[0] = scalar_t::one();

        // order: [one, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        // power: [one, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        //        omega^(1<<log_n) = omega^(0) = 1
        for (uint32_t idx = 1; idx <= log_n; idx++) {
            d_omega_lut[idx] = omega;
            omega = omega * omega;
        }
    }

    __device__ __forceinline__ static uint32_t cuda_log_floor(uint64_t n)
    {
        uint64_t k = 0;
        while ((1 << k) < n) {
            k++;
        }
        if ((1 << k) > n) {
            k--;
        }
        return k;
    }

    __device__ __forceinline__ scalar_t func_compute_power_of_omega(
        const scalar_t* d_omega_lut,
        const uint32_t power)
    {
        if (power == 0) {
            return scalar_t::one();
        }

        uint32_t log = cuda_log_floor(power);
        uint32_t rest_pow = power - (1 << log);
        scalar_t res = d_omega_lut[log + 1];

        while (rest_pow != 0) {
            uint32_t log = cuda_log_floor(rest_pow);
            rest_pow = rest_pow - (1 << log);
            // res += (1<<log);
            // [0,1,2,3,4,...]
            // [x,0,1,2,3,4,...]
            // pow = 16, log = 4, offset = 4+1
            scalar_t omega = d_omega_lut[log + 1];
            res = res * omega;
        }

        return res;
    }

    __global__ __launch_bounds__(1) void compute_power_of_omega(
        scalar_t* d_res,
        scalar_t* d_omega_lut,
        uint32_t pow)
    {
        scalar_t power = func_compute_power_of_omega(d_omega_lut, pow);
        d_res[0] = power;
    }

    __global__ void mult_power_of_omega(
        scalar_t* d_data,
        scalar_t* d_omega_lut,
        uint32_t length)
    {
        const uint32_t tile_size = blockDim.x;
        const uint64_t stride = gridDim.x * tile_size;
        const uint64_t offset = blockIdx.x * blockDim.x + threadIdx.x;
        if (offset >= length)
            return;

        scalar_t data;
        scalar_t power = func_compute_power_of_omega(d_omega_lut, offset);
        scalar_t power_stride = func_compute_power_of_omega(d_omega_lut, stride);

        for (uint64_t idx = offset; idx < length; idx += stride) {
            if (idx >= length)
                break;
            d_data[idx] *= power;
            power *= power_stride;
        }
    }

    __global__ void cukernel_generate_omegadelta(
        scalar_t* d_omega_delta,
        uint32_t* d_mapping,
        scalar_t* d_omega_lut,
        scalar_t* d_delta_lut,
        uint64_t omega_start,
        uint64_t omega_end,
        uint64_t delta_colunm_offset)
    {
        uint64_t low_degree_lut_len = 1 << DENSE_POWER_DEGREE;
        const scalar_t* d_omega_lut_low = d_omega_lut;
        const scalar_t* d_omega_lut_high = d_omega_lut + low_degree_lut_len;

        // thread index
        const uint32_t tile_size = blockDim.x;
        const uint64_t stride = gridDim.x * tile_size;
        const uint64_t offset = blockIdx.x * blockDim.x + threadIdx.x;

        // the start and end position of omega, is not necessary to be aligned with the data index
        uint64_t data_idx = offset; // [0, omega_end-omega_start]
        for (uint64_t omega_idx = omega_start + offset; omega_idx <= omega_end; omega_idx += stride) { // [omega_start, omega_end]
            if (omega_idx > omega_end)
                break;

            uint64_t permutation_col = d_mapping[omega_idx * 2 + 0];
            uint64_t permutation_row = d_mapping[omega_idx * 2 + 1];
            scalar_t delta = d_delta_lut[permutation_col];
            scalar_t omega_low_degree = d_omega_lut_low[permutation_row % low_degree_lut_len];
            scalar_t omega_high_degree = d_omega_lut_high[permutation_row >> DENSE_POWER_DEGREE];
            scalar_t omega = omega_low_degree * omega_high_degree;
            d_omega_delta[data_idx] = omega * delta;

            data_idx += stride;
        }
    }

} // namespace omega
} // namespace zkpcuda

template <typename Scalar>
class DirectOmegaPowersGenerator {

public:
    DirectOmegaPowersGenerator() = default;

    static uint64_t get_required_memory_size(uint64_t log_n)
    {
        // from 0 to log_n: [omega^(1<<logn), ..., omega^(1<<1), omega^(1<<0)]
        uint64_t powers_lut_num = log_n + 1;
        uint64_t generate_num = 1 << log_n;
        return Scalar::ELT_BYTES * (generate_num + powers_lut_num);
    }

    static uint64_t get_powers_memory_size(uint64_t log_n)
    {
        return Scalar::ELT_BYTES << (log_n);
    }

    // Bytes that `run()` carves from the caller-provided ScratchSpan.
    // Pure host arithmetic for use in `_halo2_<kernel>_workspace_size`.
    static uint64_t get_run_scratch_size(uint64_t log_n)
    {
        return align_up(Scalar::ELT_BYTES * (log_n + 1), 32);
    }

    // `d_omega` is device-resident (one 32-byte scalar). `span` carries
    // the scratch budget reported by `get_run_scratch_size`. The caller
    // must keep `span`'s backing allocation live until they sync.
    static RustError run(
        cudaStream_t& stream,
        uint64_t* d_omega_powers,
        const uint64_t* d_omega,
        uint64_t log_n,
        ScratchSpan& span)
    {
        uint64_t* d_powers_lut = (uint64_t*)span.take(Scalar::ELT_BYTES * (log_n + 1));
        try {
            // Seed d_powers_lut[0] with caller's omega so generate_omega's
            // first kernel reads it as the recurrence base.
            CUDA_OK(cudaMemcpyAsync(d_powers_lut, d_omega, Scalar::ELT_BYTES, cudaMemcpyDeviceToDevice, stream));
            zkpcuda::omega::generate_omega(
                stream,
                (scalar_t*)d_omega_powers,
                (scalar_t*)d_powers_lut,
                log_n);
        } catch (const cuda_error& error) {
            return RustError(error.code(), error.what());
        };
        return cudaSuccess;
    }
};

// high and low degree lookup table
template <typename Scalar>
class LutOmegaPowersGenerator {
public:
    LutOmegaPowersGenerator(uint64_t log_n)
    {
        _low_degree_lut_len = 1 << _dense_degree;
        _high_degree_lut_len = 1 << (log_n - _dense_degree);
        _total_lookuptable_len = _low_degree_lut_len + _high_degree_lut_len;
    }

    uint64_t get_required_memory_size(uint64_t log_n)
    {
        uint64_t omega_num = 1;
        uint64_t omega_lut_num = log_n + 1;
        uint64_t res_lut_num = _total_lookuptable_len;
        return Scalar::ELT_BYTES * (omega_num + omega_lut_num + res_lut_num);
    }

    uint64_t get_lut_memory_size()
    {
        return Scalar::ELT_BYTES * _total_lookuptable_len;
    }

    // Bytes that `run()` carves from the caller-provided ScratchSpan.
    static uint64_t get_run_scratch_size(uint64_t log_n)
    {
        return align_up((uint64_t)Scalar::ELT_BYTES, 32)
            + align_up((uint64_t)Scalar::ELT_BYTES * (log_n + 1), 32);
    }

    // `d_omega_in` is device-resident (one 32-byte scalar): the root of unity
    // whose powers seed both the low- and high-degree lookup tables.
    RustError run(
        cudaStream_t& stream,
        uint64_t* d_powers_lut,
        const uint64_t* d_omega_in,
        uint64_t log_n,
        ScratchSpan& span)
    {
        uint64_t* d_omega = (uint64_t*)span.take(Scalar::ELT_BYTES);
        uint64_t* d_omega_lut = (uint64_t*)span.take(Scalar::ELT_BYTES * (log_n + 1));
        try {
            // Stage caller's omega into the two scratch slots that
            // generate_omega / compute_power_of_omega read from.
            CUDA_OK(cudaMemcpyAsync(d_omega_lut, d_omega_in, Scalar::ELT_BYTES, cudaMemcpyDeviceToDevice, stream));
            CUDA_OK(cudaMemcpyAsync(d_omega, d_omega_in, Scalar::ELT_BYTES, cudaMemcpyDeviceToDevice, stream));

            // generate low degree dense lut
            uint64_t* d_low_degree_omega = d_powers_lut;
            zkpcuda::omega::generate_omega(
                stream,
                (scalar_t*)d_low_degree_omega,
                (scalar_t*)d_omega_lut,
                _dense_degree);
            // generate high degree sparse lut (start from omega^dense_degree)
            uint64_t high_degree_log = log_n - _dense_degree;
            uint64_t* d_high_degree_lut = d_low_degree_omega + Scalar::ELT_LIMBS * _low_degree_lut_len;
            zkpcuda::omega::generate_omega_log_lut<<<1, 1, 0, stream>>>(
                (scalar_t*)d_omega_lut,
                (scalar_t*)d_omega,
                log_n);
            zkpcuda::omega::compute_power_of_omega<<<1, 1, 0, stream>>>(
                (scalar_t*)d_omega_lut,
                (scalar_t*)d_omega_lut,
                1 << _dense_degree);
            zkpcuda::omega::generate_omega(
                stream,
                (scalar_t*)d_high_degree_lut,
                (scalar_t*)d_omega_lut,
                high_degree_log);
        } catch (const cuda_error& error) {
            return RustError(error.code(), error.what());
        };
        return cudaSuccess;
    }

private:
    uint64_t _low_degree_lut_len { 0 };
    uint64_t _high_degree_lut_len { 0 };
    uint64_t _total_lookuptable_len { 0 };
    // Mirrors `halo2_proofs::arithmetic::DENSE_POWER_DEGREE`. Must stay in
    // sync with the Rust side — they size the same LUT.
    static const uint64_t _dense_degree { DENSE_POWER_DEGREE };
};

template <typename Scalar>
class OmegaDeltaGenerator {
public:
    OmegaDeltaGenerator(uint64_t log_n, uint64_t omega_start, uint64_t omega_end, uint64_t colunm_num, uint64_t colunm_offset)
    {
        // omega
        _omega_log = log_n;
        _omega_start = omega_start;
        _omega_end = omega_end;

        // delta
        _delta_colunm_num = colunm_num;
        _delta_colunm_offset = colunm_offset;
        for (uint64_t colunm = 1; colunm < colunm_num; colunm = colunm << 1) {
            _delta_colunm_log++;
        }
    }

    uint64_t get_omegadelta_memory_size()
    {
        // generate range: [_omega_start, _omega_start], so the total length need to +1
        return Scalar::ELT_BYTES * (_omega_end - _omega_start + 1);
    }

    uint64_t get_required_memory_size(uint64_t log_n)
    {
        LutOmegaPowersGenerator<Scalar> omega_generator(log_n);
        DirectOmegaPowersGenerator<Scalar> delta_generator;

        uint64_t mapping_size = (2 * sizeof(uint32_t)) << log_n;
        uint64_t omega_lut_required_size = omega_generator.get_required_memory_size(log_n);
        uint64_t detla_lut_required_size = delta_generator.get_required_memory_size(_delta_colunm_log);
        uint64_t omegadelta_generate_size = get_omegadelta_memory_size();

        uint64_t total_num = mapping_size + omega_lut_required_size + detla_lut_required_size + omegadelta_generate_size;
        return total_num;
    }

    // Bytes `run()` carves from the caller-provided ScratchSpan: two
    // outer LUTs (d_omega_lut, d_delta_lut) plus the bigger of the two
    // inner-generator scratch budgets (the inner runs are sequential and
    // both sync inside, so they share bytes via a dedicated sub-span).
    uint64_t get_run_scratch_size(uint64_t log_n)
    {
        LutOmegaPowersGenerator<Scalar> omega_generator(log_n);
        uint64_t omega_lut_size = omega_generator.get_lut_memory_size();
        uint64_t detla_lut_size = DirectOmegaPowersGenerator<Scalar>::get_powers_memory_size(_delta_colunm_log);
        uint64_t inner_run_size = omega_generator.get_run_scratch_size(log_n);
        uint64_t direct_run_size = DirectOmegaPowersGenerator<Scalar>::get_run_scratch_size(_delta_colunm_log);
        if (direct_run_size > inner_run_size) {
            inner_run_size = direct_run_size;
        }
        return align_up(omega_lut_size, 32)
            + align_up(detla_lut_size, 32)
            + align_up(inner_run_size, 32);
    }

    RustError run(
        cudaStream_t& stream,
        void* omegadelta_device,
        void* mapping_device,
        const void* d_omega,
        const void* d_delta,
        uint32_t log_n,
        ScratchSpan& span)
    {
        // generate omega&delta lookup table
        LutOmegaPowersGenerator<Scalar> omega_generator(log_n);
        DirectOmegaPowersGenerator<Scalar> delta_generator;
        uint64_t omega_lut_size = omega_generator.get_lut_memory_size();
        uint64_t detla_lut_size = delta_generator.get_powers_memory_size(_delta_colunm_log);
        uint64_t* d_omega_lut = (uint64_t*)span.take(omega_lut_size);
        uint64_t* d_delta_lut = (uint64_t*)span.take(detla_lut_size);

        // Carve a shared sub-span for the two inner generator runs.
        // omega_generator.run() syncs internally; delta_generator.run()
        // does NOT sync but the bytes it uses are not read afterwards
        // (cukernel_generate_omegadelta reads d_delta_lut, not the inner
        // d_powers_lut). So passing a fresh ScratchSpan covering the same
        // backing bytes to both runs is safe.
        uint64_t inner_run_size = omega_generator.get_run_scratch_size(log_n);
        uint64_t direct_run_size = DirectOmegaPowersGenerator<Scalar>::get_run_scratch_size(_delta_colunm_log);
        uint64_t inner_size = inner_run_size > direct_run_size ? inner_run_size : direct_run_size;
        uint8_t* inner_ptr = (uint8_t*)span.take(inner_size);

        try {
            ScratchSpan inner_span_omega { inner_ptr, (size_t)inner_size };
            omega_generator.run(stream, d_omega_lut, (const uint64_t*)d_omega, log_n, inner_span_omega);
            ScratchSpan inner_span_delta { inner_ptr, (size_t)inner_size };
            delta_generator.run(stream, d_delta_lut, (const uint64_t*)d_delta, _delta_colunm_log, inner_span_delta);
        } catch (const cuda_error& error) {
            return RustError(error.code(), error.what());
        };

        // generate omegadelta
        try {
            zkpcuda::omega::cukernel_generate_omegadelta<<<1024, 256, 0, stream>>>(
                (scalar_t*)omegadelta_device,
                (uint32_t*)mapping_device,
                (scalar_t*)d_omega_lut,
                (scalar_t*)d_delta_lut,
                _omega_start,
                _omega_end,
                _delta_colunm_offset);
        } catch (const cuda_error& error) {
            return RustError(error.code(), error.what());
        };
        return cudaSuccess;
    }

private:
    uint64_t _omega_log { 0 };
    uint64_t _omega_start { 0 };
    uint64_t _omega_end { 0 };
    uint64_t _delta_colunm_num { 0 };
    uint64_t _delta_colunm_log { 0 };
    uint64_t _delta_colunm_offset { 0 };
};
