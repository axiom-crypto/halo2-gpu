#include <assert.h>
#include <chrono>
#include <map>
#include <string>
#include <vector>

#include "common/exception.h"
#include "common/halo2_ffi.h"
#include "common/scratch_span.h"

#include "kernel/commit_product.h"
#include "kernel/inverse.h"

using Scalar = utils::FFITraitObject;

// halo2_proofs/src/plonk/lookup/prover.rs
//     pub(in crate::plonk) fn commit_product
//         lookup Z(X) denominator
//         lookup Z(X) denominator invert
//         lookup Z(X) numerator
class CommitProductInfo {
public:
    static uint64_t get_required_memory_size(uint64_t poly_length)
    {
        const uint64_t field_size = Scalar::ELT_BYTES;
        uint64_t required_memory_size = 0;
        uint64_t poly_size = poly_length * field_size;
        required_memory_size += poly_size; // lookup_denominator
        required_memory_size += poly_size; // permuted_input
        required_memory_size += poly_size; // permuted_table
        required_memory_size += poly_size; // compressed_input
        required_memory_size += poly_size; // compressed_table
        return required_memory_size;
    }
};

extern "C" uint64_t _halo2_commit_product_max_len(
    uint64_t poly_length,
    uint64_t free_bytes)
{
    for (uint64_t _length = poly_length; _length > 0; _length = _length >> 1) {
        if (CommitProductInfo::get_required_memory_size(_length) < free_bytes) {
            return _length;
        }
    }
    return 0;
}

extern "C" uint64_t _halo2_commit_product_workspace_size(uint64_t poly_length)
{
    return align_up(CommitProductInfo::get_required_memory_size(poly_length), 32);
}

extern "C" RustError _halo2_commit_product(
    const Scalar* lookup_product, // output vec<Field>
    const Scalar* permuted_input, // input vec<Field>
    const Scalar* permuted_table, // input vec<Field>
    const Scalar* compressed_input, // input vec<Field>
    const Scalar* compressed_table, // input vec<Field>
    const scalar_t* d_beta, // input Field, device-resident (32 bytes)
    const scalar_t* d_gamma, // input Field, device-resident (32 bytes)
    uint64_t poly_length,
    uint64_t poly_offset,
    void* scratch,
    uint64_t scratch_bytes,
    cudaStream_t stream)
{
    ScratchSpan span { (uint8_t*)scratch, (size_t)scratch_bytes };
    uint64_t required_memory_size = CommitProductInfo::get_required_memory_size(poly_length);
    uint64_t* d_gpu_mem = (uint64_t*)span.take(required_memory_size);

    uint64_t field_size = Scalar::ELT_BYTES; // bytes
    uint64_t poly_size = poly_length * field_size; // bytes
    // host memory offset
    uint64_t* h_lookup_product = (uint64_t*)((char*)(lookup_product->ptr) + poly_offset * field_size);
    uint64_t* h_permuted_input = (uint64_t*)((char*)(permuted_input->ptr) + poly_offset * field_size);
    uint64_t* h_permuted_table = (uint64_t*)((char*)(permuted_table->ptr) + poly_offset * field_size);
    uint64_t* h_compressed_input = (uint64_t*)((char*)(compressed_input->ptr) + poly_offset * field_size);
    uint64_t* h_compressed_table = (uint64_t*)((char*)(compressed_table->ptr) + poly_offset * field_size);
    // gpu memory ptr
    uint64_t* d_lookup_product = d_gpu_mem;
    uint64_t* d_permuted_input = (uint64_t*)((char*)(d_lookup_product) + poly_size);
    uint64_t* d_permuted_table = (uint64_t*)((char*)(d_permuted_input) + poly_size);
    uint64_t* d_compressed_input = (uint64_t*)((char*)(d_permuted_table) + poly_size);
    uint64_t* d_compressed_table = (uint64_t*)((char*)(d_compressed_input) + poly_size);

    // run
    try {
        // no need to set memory, cause it will be overwritten in the kernel
        CUDA_OK(cudaMemcpyAsync(d_permuted_input, h_permuted_input, poly_size, cudaMemcpyHostToDevice, stream));
        CUDA_OK(cudaMemcpyAsync(d_permuted_table, h_permuted_table, poly_size, cudaMemcpyHostToDevice, stream));
        CUDA_OK(cudaMemcpyAsync(d_compressed_input, h_compressed_input, poly_size, cudaMemcpyHostToDevice, stream));
        CUDA_OK(cudaMemcpyAsync(d_compressed_table, h_compressed_table, poly_size, cudaMemcpyHostToDevice, stream));
        // 2048*32 = 65536 elements per internal iter
        zkpcuda::commit_product::cuda_kernel_lookup_denominator<<<2048, 128, 0, stream>>>(
            (scalar_t*)d_lookup_product,
            (scalar_t*)d_permuted_input,
            (scalar_t*)d_permuted_table,
            d_beta,
            d_gamma,
            poly_length);
        zkpcuda::operation::batch_invert<0 /*fr*/>(
            stream,
            d_lookup_product,
            poly_length);
        zkpcuda::commit_product::cuda_kernel_lookup_numerator<<<2048, 128, 0, stream>>>(
            (scalar_t*)d_lookup_product,
            (scalar_t*)d_compressed_input,
            (scalar_t*)d_compressed_table,
            d_beta,
            d_gamma,
            poly_length);
    } catch (const cuda_error& error) {
        return RustError(error.code(), error.what());
    };

    // copy to host
    try {
        CUDA_OK(cudaMemcpyAsync(h_lookup_product, d_lookup_product, poly_size, cudaMemcpyDeviceToHost, stream));
        CUDA_OK(cudaStreamSynchronize(stream));
    } catch (const cuda_error& error) {
        return RustError(error.code(), error.what());
    };
    return cudaSuccess;
}

// Device-input + device-output variant of `_halo2_commit_product`.
//
// Every `.ptr` on the five `Scalar*` carriers is a **device pointer** to a
// full-length polynomial; the chunk slice is addressed via pointer
// arithmetic on `poly_offset * field_size` bytes. The same three compute
// kernels (`cuda_kernel_lookup_denominator`, `batch_invert`,
// `cuda_kernel_lookup_numerator`) run unchanged on the caller's device
// buffers — no `cudaMemcpyHostToDevice` for any input, no
// `cudaMemcpyDeviceToHost` for the output. No scratch buffer is required
// (the launcher allocates nothing; all five slots live in the caller's
// `DeviceBuffer<F>` allocations), so no `_workspace_size` sibling is
// emitted. This mirrors `_halo2_grand_product_device_inputs` (no scratch)
// and `_halo2_permutation_product_device_inputs` (device-pointer inputs
// addressed via `poly_offset * field_size`).
extern "C" RustError _halo2_commit_product_device_inputs(
    const Scalar* d_lookup_product, // out: .ptr is a device pointer to the full-length scalar buffer
    const Scalar* d_permuted_input, // in : .ptr is a device pointer to the full-length scalar buffer
    const Scalar* d_permuted_table, // in : .ptr is a device pointer to the full-length scalar buffer
    const Scalar* d_compressed_input, // in : .ptr is a device pointer to the full-length scalar buffer
    const Scalar* d_compressed_table, // in : .ptr is a device pointer to the full-length scalar buffer
    const scalar_t* d_beta, // input Field, device-resident (32 bytes)
    const scalar_t* d_gamma, // input Field, device-resident (32 bytes)
    uint64_t poly_length,
    uint64_t poly_offset,
    cudaStream_t stream)
{
    uint64_t field_size = Scalar::ELT_BYTES; // bytes
    // Device-side chunk pointers via pointer arithmetic (no H2D).
    scalar_t* d_lp = (scalar_t*)((char*)(d_lookup_product->ptr) + poly_offset * field_size);
    scalar_t* d_pi = (scalar_t*)((char*)(d_permuted_input->ptr) + poly_offset * field_size);
    scalar_t* d_pt = (scalar_t*)((char*)(d_permuted_table->ptr) + poly_offset * field_size);
    scalar_t* d_ci = (scalar_t*)((char*)(d_compressed_input->ptr) + poly_offset * field_size);
    scalar_t* d_ct = (scalar_t*)((char*)(d_compressed_table->ptr) + poly_offset * field_size);

    try {
        // <<<2048, 128>>> = 2048*128 = 262144 threads (same grid as the host variant).
        zkpcuda::commit_product::cuda_kernel_lookup_denominator<<<2048, 128, 0, stream>>>(
            d_lp,
            d_pi,
            d_pt,
            d_beta,
            d_gamma,
            poly_length);
        zkpcuda::operation::batch_invert<0 /*fr*/>(
            stream,
            (uint64_t*)d_lp,
            poly_length);
        zkpcuda::commit_product::cuda_kernel_lookup_numerator<<<2048, 128, 0, stream>>>(
            d_lp,
            d_ci,
            d_ct,
            d_beta,
            d_gamma,
            poly_length);
    } catch (const cuda_error& error) {
        return RustError(error.code(), error.what());
    };

    return cudaSuccess;
}
