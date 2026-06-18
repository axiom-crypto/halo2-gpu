#include <assert.h>
#include <chrono>
#include <map>
#include <string>
#include <vector>

#include "common/exception.h"
#include "common/halo2_ffi.h"
#include "common/scratch_span.h"

#include "kernel/inverse.h"
#include "kernel/permutation_commit.h"

using Scalar = utils::FFITraitObject;

// halo2_proofs/src/plonk/permutation/prover.rs
//     pub(in crate::plonk) fn commit
//         Z_i(X) denominator
//         denominator invert
//         Z_i(X) numerator
class PermutationCommitInfo {
public:
    static uint64_t get_required_memory_size(uint64_t poly_length)
    {
        const uint64_t field_size = Scalar::ELT_BYTES;
        uint64_t required_memory_size = 0;
        uint64_t poly_size = poly_length * field_size;
        // inout data
        required_memory_size += poly_size; // denominators
        required_memory_size += poly_size; // numerators
        required_memory_size += poly_size; // permutations
        required_memory_size += poly_size; // values
        // temp buffer
        required_memory_size += (tile_size + block_num) * field_size; // omeaga lut;
        return required_memory_size;
    }

    // 512*128 = 65536 elements per internal iter
    static const uint32_t block_num = 512;
    static const uint32_t tile_size = 128;
};

extern "C" uint64_t _halo2_permutation_product_max_len(
    uint64_t poly_length,
    uint64_t free_bytes)
{
    for (uint64_t _length = poly_length; _length > 0; _length = _length >> 1) {
        if (PermutationCommitInfo::get_required_memory_size(_length) < free_bytes) {
            return _length;
        }
    }
    return 0;
}

extern "C" uint64_t _halo2_permutation_product_workspace_size(uint64_t poly_length)
{
    return align_up(PermutationCommitInfo::get_required_memory_size(poly_length), 32);
}

extern "C" RustError _halo2_permutation_product(
    const Scalar* modified_values, // inout vec<Field>
    const Scalar* permutations, // input vec<FFITraitObject>
    const Scalar* values, // input vec<FFITraitObject>
    const scalar_t* d_beta, // input Field, device-resident (32 bytes)
    const scalar_t* d_gamma, // input Field, device-resident (32 bytes)
    const scalar_t* d_delta, // input Field, device-resident (32 bytes)
    scalar_t* d_omega, // input Field, device-resident (32 bytes)
    scalar_t* d_deltaomega, // inout Field, device-resident (32 bytes)
    uint64_t poly_length,
    uint64_t poly_offset,
    uint64_t batch_size,
    void* scratch,
    uint64_t scratch_bytes,
    cudaStream_t stream)
{
    ScratchSpan span { (uint8_t*)scratch, (size_t)scratch_bytes };
    uint64_t required_memory_size = PermutationCommitInfo::get_required_memory_size(poly_length);
    uint64_t* d_gpu_mem = (uint64_t*)span.take(required_memory_size);

    uint64_t field_size = Scalar::ELT_BYTES; // bytes
    uint64_t poly_size = poly_length * field_size; // bytes
    const uint32_t BLOCK_NUM = PermutationCommitInfo::block_num;
    const uint32_t TILE_SIZE = PermutationCommitInfo::tile_size;
    const uint32_t THREAD_NUM = TILE_SIZE;
    // host memory offset
    uint64_t* h_modified_values = (uint64_t*)((char*)(modified_values->ptr)); // + poly_offset*field_size);
    // gpu memory ptr
    uint64_t* d_denominators = d_gpu_mem;
    uint64_t* d_numerators = (uint64_t*)((char*)(d_denominators) + poly_size);
    uint64_t* d_permutations = (uint64_t*)((char*)(d_numerators) + poly_size);
    uint64_t* d_values = (uint64_t*)((char*)(d_permutations) + poly_size);
    uint64_t* d_omega_lut = (uint64_t*)((char*)(d_values) + poly_size);

    // run
    try {
        // init d_omega_lut (reads caller's device-resident omega)
        zkpcuda::permutation_commit::omega_lut_init<TILE_SIZE>
            <<<1, 1, 0, stream>>>(d_omega, (scalar_t*)d_omega_lut);
        zkpcuda::permutation_commit::omega_power_of_block<TILE_SIZE>
            <<<1, 1, 0, stream>>>((scalar_t*)d_omega_lut, BLOCK_NUM);
        // init d_numerators (set one)
        zkpcuda::permutation_commit::cuda_kernel_permutation_numerator_set_one<<<BLOCK_NUM, THREAD_NUM, 0, stream>>>((scalar_t*)d_numerators, poly_length);
        // input data
        CUDA_OK(cudaMemcpyAsync(d_denominators, h_modified_values, poly_size, cudaMemcpyHostToDevice, stream));

        for (uint32_t i = 0; i < batch_size; ++i) {
            uint64_t* h_permutations = (uint64_t*)((char*)(permutations[i].ptr) + poly_offset * field_size);
            uint64_t* h_values = (uint64_t*)((char*)(values[i].ptr) + poly_offset * field_size);
            CUDA_OK(cudaMemcpyAsync(d_permutations, h_permutations, poly_size, cudaMemcpyHostToDevice, stream));
            CUDA_OK(cudaMemcpyAsync(d_values, h_values, poly_size, cudaMemcpyHostToDevice, stream));
            // *denominators *= &(*beta * permuted_value + &*gamma + value);
            zkpcuda::permutation_commit::cuda_kernel_permutation_denominator<<<BLOCK_NUM, THREAD_NUM, 0, stream>>>(
                (scalar_t*)d_denominators,
                (scalar_t*)d_permutations,
                (scalar_t*)d_values,
                d_beta,
                d_gamma,
                poly_length);
            // *numerators *= &(deltaomega * &*beta + &*gamma + value);
            zkpcuda::permutation_commit::cuda_kernel_permutation_numerator<<<BLOCK_NUM, THREAD_NUM, 0, stream>>>(
                (scalar_t*)d_numerators,
                (scalar_t*)d_values,
                d_beta,
                d_gamma,
                (scalar_t*)d_omega_lut,
                d_deltaomega,
                poly_length);
            // deltaomega = deltaomega * delta (in-place update of caller's slot)
            zkpcuda::permutation_commit::cuda_kernel_permutation_multiply<<<1, 1, 0, stream>>>(
                d_deltaomega,
                d_delta,
                1);
        }

        zkpcuda::operation::batch_invert<0 /*fr*/>(
            stream,
            d_denominators,
            poly_length);
        zkpcuda::permutation_commit::cuda_kernel_permutation_multiply<<<BLOCK_NUM, THREAD_NUM, 0, stream>>>(
            (scalar_t*)d_denominators,
            (scalar_t*)d_numerators,
            poly_length);
    } catch (const cuda_error& error) {
        return RustError(error.code(), error.what());
    };

    // copy to host
    try {
        CUDA_OK(cudaMemcpyAsync(h_modified_values, d_denominators, poly_size, cudaMemcpyDeviceToHost, stream));
        CUDA_OK(cudaStreamSynchronize(stream));
    } catch (const cuda_error& error) {
        return RustError(error.code(), error.what());
    };
    return cudaSuccess;
}

// Device-input + device-output variant of `_halo2_permutation_product`.
// `modified_values_device->ptr` is a device pointer used in-place as the
// running denominator accumulator (no `d_denominators` scratch slot).
// Each `permutations_device[i].ptr` and `values_device[i].ptr` is a device
// pointer to a full-length column; the chunk slice is addressed via
// pointer arithmetic on `poly_offset * field_size` bytes.
class PermutationCommitInfoDeviceInputs {
public:
    static uint64_t get_required_memory_size(uint64_t poly_length)
    {
        const uint64_t field_size = Scalar::ELT_BYTES;
        uint64_t required_memory_size = 0;
        const uint64_t poly_size = poly_length * field_size;
        required_memory_size += poly_size; // numerators
        required_memory_size += (PermutationCommitInfo::tile_size + PermutationCommitInfo::block_num) * field_size; // omega lut
        return required_memory_size;
    }
};

extern "C" uint64_t _halo2_permutation_product_device_inputs_workspace_size(uint64_t poly_length)
{
    return align_up(PermutationCommitInfoDeviceInputs::get_required_memory_size(poly_length), 32);
}

extern "C" RustError _halo2_permutation_product_device_inputs(
    const Scalar* modified_values_device, // inout vec<Field>, device-resident accumulator
    const Scalar* permutations_device, // input vec<FFITraitObject>, .ptr is device
    const Scalar* values_device, // input vec<FFITraitObject>, .ptr is device
    const scalar_t* d_beta, // input Field, device-resident (32 bytes)
    const scalar_t* d_gamma, // input Field, device-resident (32 bytes)
    const scalar_t* d_delta, // input Field, device-resident (32 bytes)
    scalar_t* d_omega, // input Field, device-resident (32 bytes)
    scalar_t* d_deltaomega, // inout Field, device-resident (32 bytes)
    uint64_t poly_length,
    uint64_t poly_offset,
    uint64_t batch_size,
    void* scratch,
    uint64_t scratch_bytes,
    cudaStream_t stream)
{
    ScratchSpan span { (uint8_t*)scratch, (size_t)scratch_bytes };
    uint64_t required_memory_size = PermutationCommitInfoDeviceInputs::get_required_memory_size(poly_length);
    uint64_t* d_gpu_mem = (uint64_t*)span.take(required_memory_size);

    uint64_t field_size = Scalar::ELT_BYTES; // bytes
    uint64_t poly_size = poly_length * field_size; // bytes
    const uint32_t BLOCK_NUM = PermutationCommitInfo::block_num;
    const uint32_t TILE_SIZE = PermutationCommitInfo::tile_size;
    const uint32_t THREAD_NUM = TILE_SIZE;
    // Caller-owned device buffer doubles as the d_denominators accumulator.
    uint64_t* d_denominators = (uint64_t*)((char*)(modified_values_device->ptr));
    // gpu memory ptr
    uint64_t* d_numerators = d_gpu_mem;
    uint64_t* d_omega_lut = (uint64_t*)((char*)(d_numerators) + poly_size);

    // run
    try {
        zkpcuda::permutation_commit::omega_lut_init<TILE_SIZE>
            <<<1, 1, 0, stream>>>(d_omega, (scalar_t*)d_omega_lut);
        zkpcuda::permutation_commit::omega_power_of_block<TILE_SIZE>
            <<<1, 1, 0, stream>>>((scalar_t*)d_omega_lut, BLOCK_NUM);
        zkpcuda::permutation_commit::cuda_kernel_permutation_numerator_set_one<<<BLOCK_NUM, THREAD_NUM, 0, stream>>>((scalar_t*)d_numerators, poly_length);

        for (uint32_t i = 0; i < batch_size; ++i) {
            scalar_t* d_perm_i = (scalar_t*)((char*)(permutations_device[i].ptr) + poly_offset * field_size);
            scalar_t* d_val_i = (scalar_t*)((char*)(values_device[i].ptr) + poly_offset * field_size);
            zkpcuda::permutation_commit::cuda_kernel_permutation_denominator<<<BLOCK_NUM, THREAD_NUM, 0, stream>>>(
                (scalar_t*)d_denominators,
                d_perm_i,
                d_val_i,
                d_beta,
                d_gamma,
                poly_length);
            zkpcuda::permutation_commit::cuda_kernel_permutation_numerator<<<BLOCK_NUM, THREAD_NUM, 0, stream>>>(
                (scalar_t*)d_numerators,
                d_val_i,
                d_beta,
                d_gamma,
                (scalar_t*)d_omega_lut,
                d_deltaomega,
                poly_length);
            zkpcuda::permutation_commit::cuda_kernel_permutation_multiply<<<1, 1, 0, stream>>>(
                d_deltaomega,
                d_delta,
                1);
        }

        zkpcuda::operation::batch_invert<0 /*fr*/>(
            stream,
            d_denominators,
            poly_length);
        zkpcuda::permutation_commit::cuda_kernel_permutation_multiply<<<BLOCK_NUM, THREAD_NUM, 0, stream>>>(
            (scalar_t*)d_denominators,
            (scalar_t*)d_numerators,
            poly_length);
    } catch (const cuda_error& error) {
        return RustError(error.code(), error.what());
    };
    return cudaSuccess;
}
