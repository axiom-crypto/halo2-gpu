#include <assert.h>
#include <cstdint>

#include "common/exception.h"
#include "common/halo2_ffi.h"
#include "common/scratch_span.h"

#include "kernel/omega.h"

using Scalar = utils::FFITraitObject;

// Pure host preflight: bytes the caller (Rust) must hand into
// `_halo2_power_of_omega` as the `scratch` buffer. Sums the three
// internal device buffers (d_omega, d_omega_lut, d_res).
extern "C" uint64_t _halo2_power_of_omega_workspace_size(uint32_t log_n)
{
    return align_up((uint64_t)Scalar::ELT_BYTES, 32)
        + align_up((uint64_t)(log_n + 1) * Scalar::ELT_BYTES, 32)
        + align_up((uint64_t)Scalar::ELT_BYTES, 32);
}

uint32_t bit_width(uint32_t x)
{
    if (x == 0)
        return 0;
    return 32 - __builtin_clz(x);
}

extern "C" RustError _halo2_power_of_omega(
    Scalar* res,
    Scalar* omega_lut,
    Scalar* omega,
    uint32_t log_n,
    uint32_t pow,
    void* scratch,
    uint64_t scratch_bytes,
    cudaStream_t stream)
{
    // `func_compute_power_of_omega` indexes `d_omega_lut[floor_log2(pow) + 1]`
    // with LUT length `log_n + 1`; `pow >= (1 << log_n)` reads past the LUT.
    if (log_n >= 32 || bit_width(pow) > log_n) {
        return RustError(cudaErrorInvalidValue, "_halo2_power_of_omega: pow must be < (1 << log_n) and log_n < 32\r\n");
    }
    ScratchSpan span { (uint8_t*)scratch, (size_t)scratch_bytes };
    uint64_t* d_omega = (uint64_t*)span.take(Scalar::ELT_BYTES);
    uint64_t* d_omega_lut = (uint64_t*)span.take((log_n + 1) * Scalar::ELT_BYTES);
    uint64_t* d_res = (uint64_t*)span.take(Scalar::ELT_BYTES);
    try {
        CUDA_OK(cudaMemcpyAsync(d_omega, omega->ptr, Scalar::ELT_BYTES, cudaMemcpyHostToDevice, stream));
        // phase 1: generate omega log lut
        zkpcuda::omega::generate_omega_log_lut<<<1, 1, 0, stream>>>(
            (scalar_t*)d_omega_lut,
            (scalar_t*)d_omega,
            log_n);
        CUDA_OK(cudaMemcpyAsync(omega_lut->ptr, d_omega_lut, (log_n + 1) * Scalar::ELT_BYTES, cudaMemcpyDeviceToHost, stream));
        // phase 2: compute power of omega
        zkpcuda::omega::compute_power_of_omega<<<1, 1, 0, stream>>>(
            (scalar_t*)d_res,
            (scalar_t*)d_omega_lut,
            pow);
        CUDA_OK(cudaMemcpyAsync(res->ptr, d_res, Scalar::ELT_BYTES, cudaMemcpyDeviceToHost, stream));
        CUDA_OK(cudaStreamSynchronize(stream));
    } catch (const cuda_error& error) {
        return RustError(error.code(), error.what());
    };
    return cudaSuccess;
}

// Pure host preflight: bytes for `_halo2_generate_omega_powers`. Funds only
// the inner generator's scratch; the result buffer and omega scalar are
// caller-allocated device pointers.
extern "C" uint64_t _halo2_generate_omega_powers_workspace_size(uint32_t log_n)
{
    return align_up(DirectOmegaPowersGenerator<Scalar>::get_run_scratch_size(log_n), 32);
}

extern "C" RustError _halo2_generate_omega_powers(
    void* d_omega_powers, // device pointer; caller-allocated result buffer
    const void* d_omega, // device pointer; caller-staged 32-byte omega scalar
    uint32_t log_n,
    void* scratch,
    uint64_t scratch_bytes,
    cudaStream_t stream)
{
    ScratchSpan span { (uint8_t*)scratch, (size_t)scratch_bytes };
    uint64_t* d_powers = (uint64_t*)d_omega_powers;
    const uint64_t* d_omega_staging = (const uint64_t*)d_omega;

    RustError status_run = DirectOmegaPowersGenerator<Scalar>::run(stream, d_powers, d_omega_staging, log_n, span);
    if (status_run.code != cudaSuccess) {
        return status_run;
    }
    return cudaSuccess;
}

// Pure host preflight: bytes for `_halo2_generate_omega_lut`. Caller's
// span funds the result LUT, a 32-byte omega staging slot, and the
// inner generator's scratch.
extern "C" uint64_t _halo2_generate_omega_lut_workspace_size(uint32_t log_n)
{
    LutOmegaPowersGenerator<Scalar> generator(log_n);
    return align_up(generator.get_lut_memory_size(), 32)
        + align_up((uint64_t)Scalar::ELT_BYTES, 32)
        + align_up(generator.get_run_scratch_size(log_n), 32);
}

extern "C" RustError _halo2_generate_omega_lut(
    Scalar* omega_lut,
    Scalar* omega,
    uint32_t log_n,
    void* scratch,
    uint64_t scratch_bytes,
    cudaStream_t stream)
{
    // LutOmegaPowersGenerator sizes the sparse LUT as `1 << (log_n - DENSE_POWER_DEGREE)`;
    // reject log_n outside [DENSE_POWER_DEGREE, DENSE_POWER_DEGREE + 32) so the
    // shift neither underflows nor exceeds the 32-bit int operand's width.
    if (log_n < DENSE_POWER_DEGREE || log_n >= DENSE_POWER_DEGREE + 32) {
        return RustError(cudaErrorInvalidValue, "_halo2_generate_omega_lut: log_n must be in [DENSE_POWER_DEGREE, DENSE_POWER_DEGREE + 32)\r\n");
    }
    ScratchSpan span { (uint8_t*)scratch, (size_t)scratch_bytes };
    LutOmegaPowersGenerator<Scalar> generator(log_n);
    uint64_t lut_memory_size = generator.get_lut_memory_size();
    uint64_t* d_omega_lut = (uint64_t*)span.take(lut_memory_size);
    uint64_t* d_omega_staging = (uint64_t*)span.take(Scalar::ELT_BYTES);

    try {
        CUDA_OK(cudaMemcpyAsync(d_omega_staging, omega->ptr, Scalar::ELT_BYTES, cudaMemcpyHostToDevice, stream));
    } catch (const cuda_error& error) {
        return RustError(error.code(), error.what());
    };

    RustError status_run = generator.run(stream, d_omega_lut, d_omega_staging, log_n, span);
    if (status_run.code != cudaSuccess) {
        return status_run;
    }

    try {
        CUDA_OK(cudaMemcpyAsync(omega_lut->ptr, d_omega_lut, lut_memory_size, cudaMemcpyDeviceToHost, stream));
        CUDA_OK(cudaStreamSynchronize(stream));
    } catch (const cuda_error& error) {
        return RustError(error.code(), error.what());
    };

    return cudaSuccess;
}

// `OmegaDeltaGenerator::run` carves its scratch (omega_lut, delta_lut,
// inner generator workspace) from the caller-provided span; this FFI
// surfaces a matching `_workspace_size` preflight, plus two 32-byte
// staging slots used here to host the omega/delta scalars in device
// memory before invoking the (device-omega-pointer) generator.
extern "C" uint64_t _halo2_generate_omegadelta_workspace_size(
    uint32_t log_n, uint32_t omega_start, uint32_t omega_end,
    uint32_t colunm_num, uint32_t colunm_offset)
{
    OmegaDeltaGenerator<Scalar> generator(log_n, omega_start, omega_end, colunm_num, colunm_offset);
    return align_up((uint64_t)Scalar::ELT_BYTES, 32)
        + align_up((uint64_t)Scalar::ELT_BYTES, 32)
        + generator.get_run_scratch_size(log_n);
}

extern "C" RustError _halo2_generate_omegadelta(
    void* omegadelta, // device out
    const void* mapping, // device in
    const void* omega_host, // host in
    const void* delta_host, // host in
    uint32_t log_n,
    uint32_t omega_start,
    uint32_t omega_end,
    uint32_t colunm_num,
    uint32_t colunm_offset,
    void* scratch,
    uint64_t scratch_bytes,
    cudaStream_t stream)
{
    // OmegaDeltaGenerator instantiates LutOmegaPowersGenerator(log_n); reject log_n
    // outside [DENSE_POWER_DEGREE, DENSE_POWER_DEGREE + 32) so the inner
    // `1 << (log_n - DENSE_POWER_DEGREE)` neither underflows nor exceeds the
    // 32-bit int operand's width.
    if (log_n < DENSE_POWER_DEGREE || log_n >= DENSE_POWER_DEGREE + 32) {
        return RustError(cudaErrorInvalidValue, "_halo2_generate_omegadelta: log_n must be in [DENSE_POWER_DEGREE, DENSE_POWER_DEGREE + 32)\r\n");
    }
    ScratchSpan span { (uint8_t*)scratch, (size_t)scratch_bytes };
    OmegaDeltaGenerator<Scalar> generator(log_n, omega_start, omega_end, colunm_num, colunm_offset);

    // Stage host omega/delta into 32-byte device slots.
    uint64_t* d_omega_staging = (uint64_t*)span.take(Scalar::ELT_BYTES);
    uint64_t* d_delta_staging = (uint64_t*)span.take(Scalar::ELT_BYTES);
    try {
        CUDA_OK(cudaMemcpyAsync(d_omega_staging, omega_host, Scalar::ELT_BYTES, cudaMemcpyHostToDevice, stream));
        CUDA_OK(cudaMemcpyAsync(d_delta_staging, delta_host, Scalar::ELT_BYTES, cudaMemcpyHostToDevice, stream));
    } catch (const cuda_error& error) {
        return RustError(error.code(), error.what());
    };

    RustError status_run = generator.run(
        stream,
        omegadelta,
        (void*)mapping,
        d_omega_staging,
        d_delta_staging,
        log_n,
        span);
    if (status_run.code != cudaSuccess) {
        return status_run;
    }

    return cudaSuccess;
}
