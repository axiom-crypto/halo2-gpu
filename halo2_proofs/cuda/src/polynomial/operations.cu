#include <assert.h>
#include <chrono>
#include <map>
#include <string>
#include <vector>

#include "common/exception.h"
#include "common/halo2_ffi.h"
#include "common/scratch_span.h"

#include "kernel/inverse.h"

using Scalar = utils::FFITraitObject;

// Device-input, in-place batch_invert.
extern "C" RustError _halo2_batch_invert(
    void* d_data,
    uint32_t field_type, // 0: Fr 1: Fp
    uint64_t length,
    cudaStream_t stream)
{
    if (field_type != 0 && field_type != 1) {
        return RustError(cudaErrorInvalidValue, "Invalid scalar type");
    }
    // batch_invert computes `block_num = ceil_div(length, inverse_per_block)`;
    // length == 0 yields a zero-block <<<0, ...>>> launch which is invalid.
    if (length == 0) {
        return RustError(cudaErrorInvalidValue, "_halo2_batch_invert: length must be >= 1\r\n");
    }

    try {
        if (field_type == 0)
            zkpcuda::operation::batch_invert<0 /*fr*/>(stream, (uint64_t*)d_data, length);
        else if (field_type == 1)
            zkpcuda::operation::batch_invert<1 /*fp*/>(stream, (uint64_t*)d_data, length);
        CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& error) {
        return RustError(error.code(), error.what());
    };

    return cudaSuccess;
}
