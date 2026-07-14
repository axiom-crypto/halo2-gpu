#include <assert.h>
#include <chrono>
#include <map>
#include <string>
#include <vector>

#include "common/exception.h"
#include "common/halo2_ffi.h"
#include "common/scratch_span.h"

#include "kernel/polynomial.h"

using Scalar = utils::FFITraitObject;

// Evaluates a polynomial at a single point. The polynomial is read from a
// caller-owned device buffer; the point is staged into device memory; the
// result stays on device.
class PolyEvaluation {
public:
    static RustError run_device_input(
        cudaStream_t& stream,
        const uint64_t* d_poly_input,
        const Scalar* point,
        uint64_t length,
        uint64_t* d_gpu_mem,
        uint64_t** d_eval_result)
    {
        try {
            uint64_t* d_point = d_gpu_mem;
            uint64_t* d_pow_lut = d_point + Scalar::ELT_LIMBS;
            uint64_t* d_batch_res = d_pow_lut + (_tile_size + _block_num) * Scalar::ELT_LIMBS;
            uint64_t* d_res = d_batch_res + _block_num * Scalar::ELT_LIMBS;

            CUDA_OK(cudaMemcpyAsync(d_point, point->ptr, Scalar::ELT_BYTES, cudaMemcpyHostToDevice, stream));

            zkpcuda::polynomial::power_of_scalar_init<_tile_size><<<1, 1, 0, stream>>>(
                (const scalar_t*)d_point,
                (scalar_t*)d_pow_lut);
            zkpcuda::polynomial::power_of_scalar_block<_tile_size><<<1, 1, 0, stream>>>(
                (scalar_t*)d_pow_lut,
                _block_num);
            zkpcuda::polynomial::eval_polynomial_batch<_tile_size><<<_block_num, _tile_size, 0, stream>>>(
                (const scalar_t*)d_poly_input,
                (const scalar_t*)d_pow_lut,
                (scalar_t*)d_batch_res,
                length);
            zkpcuda::polynomial::eval_polynomial_epilogue<<<1, 1, 0, stream>>>(
                (scalar_t*)d_res,
                (const scalar_t*)d_batch_res,
                _block_num);

            *d_eval_result = d_res;
        } catch (const cuda_error& error) {
            return RustError(error.code(), error.what());
        };

        return cudaSuccess;
    }

    static uint64_t requiredMemorySize_device_input(uint64_t /*length*/)
    {
        // No `poly` slot; rest matches requiredMemorySize.
        uint64_t s = 0;
        s += Scalar::ELT_BYTES; // point
        s += (_tile_size + _block_num) * Scalar::ELT_BYTES; // power_lut
        s += _block_num * Scalar::ELT_BYTES; // batch_res
        s += Scalar::ELT_BYTES; // result
        return s;
    }

private:
    static const uint64_t _tile_size = 64;
    static const uint64_t _block_num = 256;
};

extern "C" uint64_t _halo2_eval_polynomial_workspace_size(uint64_t length)
{
    return align_up(PolyEvaluation::requiredMemorySize_device_input(length), 32);
}

extern "C" RustError _halo2_eval_polynomial(
    const void* d_poly,
    const Scalar* point,
    void** d_result_out,
    uint64_t length,
    void* scratch,
    uint64_t scratch_bytes,
    cudaStream_t stream)
{
    ScratchSpan span { (uint8_t*)scratch, (size_t)scratch_bytes };
    uint64_t required_memory_size = PolyEvaluation::requiredMemorySize_device_input(length);
    uint64_t* d_gpu_mem = (uint64_t*)span.take(required_memory_size);

    uint64_t* d_eval_result = nullptr;
    RustError status_run = PolyEvaluation::run_device_input(
        stream,
        (const uint64_t*)d_poly,
        point,
        length,
        d_gpu_mem,
        &d_eval_result);
    if (status_run.code != cudaSuccess) {
        return status_run;
    }
    *d_result_out = d_eval_result;
    return cudaSuccess;
}

// batch poly evaluation

class CudaEvalPolynomialBatchInfo {
public:
    CudaEvalPolynomialBatchInfo(uint64_t poly_length, uint64_t batch_size)
    {
        this->poly_length = poly_length;
        this->batch_size = batch_size;
        this->poly_size = poly_length * Scalar::ELT_BYTES;
    }

    uint64_t get_required_memory_size()
    {
        uint64_t required_memory_size = 0;
        // in/out
        required_memory_size += poly_size * 2; // poly_in[0] and poly_in[1] for double buffering
        required_memory_size += batch_size * Scalar::ELT_BYTES; // eval point
        required_memory_size += batch_size * Scalar::ELT_BYTES; // eval result
        // temp
        required_memory_size += (eval_tile_size + eval_block_num) * Scalar::ELT_BYTES; // power_lut
        required_memory_size += eval_block_num * Scalar::ELT_BYTES; // block_res
        return required_memory_size;
    }

    uint64_t poly_length = 0;
    uint64_t batch_size = 0;
    uint64_t poly_size = 0;
    static const uint32_t eval_tile_size = 64;
    static const uint32_t eval_block_num = 256;
};

extern "C" uint64_t _halo2_eval_poly_batch_max_len(
    uint64_t poly_length,
    uint64_t batch_size,
    uint64_t free_bytes)
{
    for (uint64_t _length = poly_length; _length > 0; _length = _length >> 1) {
        CudaEvalPolynomialBatchInfo batch_poly_eval_info(_length, batch_size);
        if (batch_poly_eval_info.get_required_memory_size() < free_bytes) {
            return _length;
        }
    }
    return 0;
}

extern "C" uint64_t _halo2_eval_polynomial_batch_workspace_size(
    uint64_t poly_length,
    uint64_t batch_size)
{
    CudaEvalPolynomialBatchInfo info(poly_length, batch_size);
    return align_up(info.get_required_memory_size(), 32);
}

extern "C" RustError _halo2_eval_polynomial_batch(
    const Scalar* poly_in_many, // input:  vec<poly>
    const Scalar* eval_point, // input:  vec<scalar>
    const Scalar* eval_result, // output: vec<scalar>
    uint64_t poly_offset,
    uint64_t poly_length,
    uint64_t batch_size,
    void* scratch,
    uint64_t scratch_bytes,
    cudaStream_t stream)
{
    ScratchSpan span { (uint8_t*)scratch, (size_t)scratch_bytes };
    // input/output
    uint64_t* d_poly_in[2] = { nullptr }; // vec<poly>, double buffer
    uint64_t* d_eval_point = nullptr; // vec<scalar>
    uint64_t* d_eval_result = nullptr; // vec<scalar>
    // temp buffer
    uint64_t* d_eval_pow_lut = nullptr;
    uint64_t* d_eval_block_res = nullptr;

    CudaEvalPolynomialBatchInfo batch_poly_eval_info(poly_length, batch_size);
    const uint32_t poly_size = batch_poly_eval_info.poly_size;
    const uint32_t eval_tile_size = CudaEvalPolynomialBatchInfo::eval_tile_size;
    const uint32_t eval_block_num = CudaEvalPolynomialBatchInfo::eval_block_num;
    uint64_t required_memory_size = batch_poly_eval_info.get_required_memory_size();
    uint64_t* d_gpu_mem = (uint64_t*)span.take(required_memory_size);

    // init
    try {
        // in/out
        d_poly_in[0] = d_gpu_mem;
        d_poly_in[1] = (uint64_t*)((char*)(d_poly_in[0]) + poly_size);
        d_eval_point = (uint64_t*)((char*)(d_poly_in[1]) + poly_size);
        d_eval_result = (uint64_t*)((char*)d_eval_point + batch_size * Scalar::ELT_BYTES);
        // temp
        d_eval_pow_lut = (uint64_t*)((char*)d_eval_result + batch_size * Scalar::ELT_BYTES);
        d_eval_block_res = (uint64_t*)((char*)d_eval_pow_lut + (eval_tile_size + eval_block_num) * Scalar::ELT_BYTES);
        CUDA_OK(cudaMemcpyAsync(d_eval_point, eval_point->ptr, Scalar::ELT_BYTES * batch_size, cudaMemcpyHostToDevice, stream));
        uint64_t* h_poly_in_ptr = poly_in_many[0].ptr + Scalar::ELT_LIMBS * poly_offset;
        CUDA_OK(cudaMemcpyAsync(d_poly_in[0], h_poly_in_ptr, poly_size, cudaMemcpyHostToDevice, stream));
    } catch (const cuda_error& error) {
        return RustError(error.code(), error.what());
    };

    for (uint32_t i = 0; i < batch_size; ++i) {
        try {
            uint64_t* d_eval_point_ = (uint64_t*)((char*)d_eval_point + i * Scalar::ELT_BYTES);
            uint64_t* d_eval_result_ = (uint64_t*)((char*)d_eval_result + i * Scalar::ELT_BYTES);
            zkpcuda::polynomial::power_of_scalar_init<eval_tile_size><<<1, 1, 0, stream>>>(
                (const scalar_t*)d_eval_point_,
                (scalar_t*)d_eval_pow_lut);
            zkpcuda::polynomial::power_of_scalar_block<eval_tile_size><<<1, 1, 0, stream>>>(
                (scalar_t*)d_eval_pow_lut,
                eval_block_num);
            zkpcuda::polynomial::eval_polynomial_batch<eval_tile_size><<<eval_block_num, eval_tile_size, 0, stream>>>(
                (const scalar_t*)d_poly_in[i & 1],
                (const scalar_t*)d_eval_pow_lut,
                (scalar_t*)d_eval_block_res,
                poly_length);
            zkpcuda::polynomial::eval_polynomial_epilogue<<<1, 1, 0, stream>>>(
                (scalar_t*)d_eval_result_,
                (const scalar_t*)d_eval_block_res,
                eval_block_num);
            if (i + 1 < batch_size) {
                uint64_t* h_poly_in_ptr = poly_in_many[i + 1].ptr + Scalar::ELT_LIMBS * poly_offset;
                CUDA_OK(cudaMemcpyAsync(d_poly_in[(i + 1) & 1], h_poly_in_ptr, poly_size, cudaMemcpyHostToDevice, stream));
            }
        } catch (const cuda_error& error) {
            return RustError(error.code(), error.what());
        };
    }

    // copy to host
    try {
        CUDA_OK(cudaMemcpyAsync(eval_result->ptr, d_eval_result, batch_size * Scalar::ELT_BYTES, cudaMemcpyDeviceToHost, stream));
        CUDA_OK(cudaStreamSynchronize(stream));
    } catch (const cuda_error& error) {
        return RustError(error.code(), error.what());
    };

    return cudaSuccess;
}

// Strided gather: `d_out[i * num_parts + p] = d_parts[p][i]`. All inputs
// are device-resident. No scratch.
extern "C" RustError _halo2_extended_from_lagrange_vec_device(
    void* d_out,                       // length n * num_parts
    const void* d_parts,               // [num_parts] device pointers (device array)
    uint32_t num_parts,
    uint64_t n,
    cudaStream_t stream)
{
    try {
        const uint32_t block_num = 512;
        const uint32_t tile_size = 256;
        zkpcuda::polynomial::extended_from_lagrange_vec_kernel
            <<<block_num, tile_size, 0, stream>>>(
                (scalar_t*)d_out,
                (const scalar_t* const*)d_parts,
                num_parts,
                n);
    } catch (const cuda_error& error) {
        return RustError(error.code(), error.what());
    };
    return cudaSuccess;
}

// `d_poly[i] *= d_t_evals[i % t_len]`. All inputs are device-resident.
extern "C" RustError _halo2_divide_by_vanishing_poly(
    void* d_poly,                  // in/out, length n
    const void* d_t_evals,         // length t_len
    uint32_t t_len,
    uint64_t n,
    cudaStream_t stream)
{
    try {
        const uint32_t block_num = 512;
        const uint32_t tile_size = 256;
        zkpcuda::polynomial::divide_by_vanishing_poly_kernel
            <<<block_num, tile_size, 0, stream>>>(
                (scalar_t*)d_poly,
                (const scalar_t*)d_t_evals,
                t_len,
                n);
    } catch (const cuda_error& error) {
        return RustError(error.code(), error.what());
    };
    return cudaSuccess;
}

// In-place `d_a[i] *= d_coset_powers[i % (coset_powers_len+1) - 1]` when
// `i % (coset_powers_len+1) != 0` (identity at `i_mod == 0`).
extern "C" RustError _halo2_distribute_powers_zeta(
    void* d_a,                          // in/out, length n
    const void* d_coset_powers,         // length coset_powers_len
    uint32_t coset_powers_len,
    uint64_t n,
    cudaStream_t stream)
{
    try {
        const uint32_t block_num = 512;
        const uint32_t tile_size = 256;
        zkpcuda::polynomial::distribute_powers_zeta_kernel
            <<<block_num, tile_size, 0, stream>>>(
                (scalar_t*)d_a,
                (const scalar_t*)d_coset_powers,
                coset_powers_len,
                n);
    } catch (const cuda_error& error) {
        return RustError(error.code(), error.what());
    };
    return cudaSuccess;
}

// `d_out[i] = d_a[i] * d_b[i]` over `length` elements. All buffers
// device-resident. No scratch. Used by the batch_invert_assigned device
// reconstruction kernel: the Rust caller stages numerators device-side
// and supplies pre-inverted denominators (with `None` denominators
// encoded as `F::ONE`), then this kernel writes the per-cell product.
extern "C" RustError _halo2_poly_elementwise_multiply(
    void* d_out,
    const void* d_a,
    const void* d_b,
    uint64_t length,
    cudaStream_t stream)
{
    try {
        const uint32_t block_num = 512;
        const uint32_t tile_size = 256;
        zkpcuda::polynomial::poly_elementwise_multiply<<<block_num, tile_size, 0, stream>>>(
            (scalar_t*)d_out,
            (const scalar_t*)d_a,
            (const scalar_t*)d_b,
            length);
    } catch (const cuda_error& error) {
        return RustError(error.code(), error.what());
    };
    return cudaSuccess;
}

// `d_out[i] = *d_scalar` for i in [0, length). Broadcast-fills a device buffer
// from a device-resident scalar so the caller avoids staging a full
// length-sized host buffer + H2D just to initialize to a constant.
extern "C" RustError _halo2_poly_fill_scalar(
    void* d_out,
    const void* d_scalar,
    uint64_t length,
    cudaStream_t stream)
{
    try {
        const uint32_t block_num = 512;
        const uint32_t tile_size = 256;
        zkpcuda::polynomial::poly_fill_scalar<<<block_num, tile_size, 0, stream>>>(
            (scalar_t*)d_out,
            (const scalar_t*)d_scalar,
            length);
    } catch (const cuda_error& error) {
        return RustError(error.code(), error.what());
    };
    return cudaSuccess;
}

// `d_acc[i] -= d_short[i]` for i in [0, short_len). Both buffers
// device-resident. No scratch. `d_acc` is updated in place; only its
// length-`short_len` prefix is touched.
extern "C" RustError _halo2_poly_sub_short_inplace(
    void* d_acc,
    const void* d_short,
    uint64_t short_len,
    cudaStream_t stream)
{
    if (short_len == 0) {
        return cudaSuccess;
    }
    try {
        const uint32_t tile_size = 256;
        const uint32_t block_num = (short_len + tile_size - 1) / tile_size;
        zkpcuda::polynomial::poly_sub_short_inplace<<<block_num, tile_size, 0, stream>>>(
            (scalar_t*)d_acc,
            (const scalar_t*)d_short,
            short_len);
    } catch (const cuda_error& error) {
        return RustError(error.code(), error.what());
    };
    return cudaSuccess;
}

// `d_out[i] = d_long[i] - d_short[i]` for i < short_len; `= d_long[i]`
// for short_len <= i < long_len. Both inputs and output device-resident.
// Out-of-place sibling of `_halo2_poly_sub_short_inplace`: lets the
// caller materialise a fresh `d_out` without a separate D2D clone of
// `d_long` (preserves `d_long` for a second consumer).
extern "C" RustError _halo2_poly_sub_short_out_of_place(
    void* d_out,
    const void* d_long,
    const void* d_short,
    uint64_t short_len,
    uint64_t long_len,
    cudaStream_t stream)
{
    if (long_len == 0) {
        return cudaSuccess;
    }
    try {
        const uint32_t tile_size = 256;
        const uint32_t block_num = (long_len + tile_size - 1) / tile_size;
        zkpcuda::polynomial::poly_sub_short_out_of_place
            <<<block_num, tile_size, 0, stream>>>(
                (scalar_t*)d_out,
                (const scalar_t*)d_long,
                (const scalar_t*)d_short,
                short_len,
                long_len);
    } catch (const cuda_error& error) {
        return RustError(error.code(), error.what());
    };
    return cudaSuccess;
}

// `d_buf[0] -= *d_scalar`. Both pointers device-resident. No scratch.
extern "C" RustError _halo2_poly_sub_scalar_at_zero(
    void* d_buf,
    const void* d_scalar,
    cudaStream_t stream)
{
    try {
        zkpcuda::polynomial::poly_sub_scalar_at_zero<<<1, 1, 0, stream>>>(
            (scalar_t*)d_buf,
            (const scalar_t*)d_scalar);
    } catch (const cuda_error& error) {
        return RustError(error.code(), error.what());
    };
    return cudaSuccess;
}

// `poly_acc[i] += scalar * poly_in[i]`. All buffers are device-resident;
// `d_poly_acc` is updated in place.
extern "C" RustError _halo2_poly_multiply_add(
    void* d_poly_acc,         // in/out
    const void* d_poly_in,
    const void* d_scalar,     // single field element
    uint64_t poly_length,
    cudaStream_t stream)
{
    try {
        const uint32_t block_num = 512;
        const uint32_t tile_size = 64; // thread_num = tile_size * 4 = 256
        zkpcuda::polynomial::poly_multiply_add<<<block_num, tile_size, 0, stream>>>(
            (scalar_t*)d_poly_acc,
            (const scalar_t*)d_poly_in,
            (const scalar_t*)d_scalar,
            poly_length);
    } catch (const cuda_error& error) {
        return RustError(error.code(), error.what());
    };

    return cudaSuccess;
}
