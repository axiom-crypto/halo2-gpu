#include <algorithm>
#include <assert.h>
#include <chrono>
#include <map>
#include <string>
#include <vector>

#include "common/exception.h"
#include "common/halo2_ffi.h"
#include "common/scratch_span.h"

#include "kernel/common.h"
#include "kernel/ntt_dif.h"
#include "kernel/ntt_dit.h"
#include "kernel/omega.h"
#include "kernel/polynomial.h"
#include "kernel/zeta.h"

using Scalar = utils::FFITraitObject;

enum NTT_TYPE {
    FFT = 1,
    iFFT = 2,
    cosetFFT = 3,
    icosetFFT = 4,
    iFFT_cosetFFT = 5,
    CosetFFT_Part = 6,
};

enum class TWIDDLE_TYPE {
    UNDEF,
    DENSE,
    SPARSE,
};

enum class BUFFFER_TYPE {
    UNDEF,
    INSUFFICIENT,
    NORMAL,
    DOUBLE,
};

/*=======================================*/
/*=============== twiddle ===============*/
/*=======================================*/
// Bytes the inner generator inside `twiddle_init` will carve from the
// caller's ScratchSpan. Used by the FFT workspace_size preflights.
static inline uint64_t twiddle_init_run_scratch_size(NTT_TYPE ntt_type, uint32_t log_n, uint32_t extend_log_n)
{
    if (ntt_type == cosetFFT || ntt_type == icosetFFT) {
        log_n = extend_log_n;
    }
    // We always pick TWIDDLE_TYPE::DENSE — the SPARSE branch in
    // twiddle_init is dead. Take the max of the two budgets
    // anyway, defensively, in case the policy is reverted via a
    // free_bytes_hint follow-up. Both helpers below are static and
    // depend only on the passed log_n, so we don't instantiate
    // `LutOmegaPowersGenerator(log_n - 1)` — that constructor does
    // `1 << (log_n - 1 - DENSE_POWER_DEGREE)` and underflows on
    // very small FFTs (log_n - 1 < DENSE_POWER_DEGREE).
    uint64_t direct_size = DirectOmegaPowersGenerator<Scalar>::get_run_scratch_size(log_n - 1);
    uint64_t sparse_size = LutOmegaPowersGenerator<Scalar>::get_run_scratch_size(log_n);
    return direct_size > sparse_size ? direct_size : sparse_size;
}

RustError twiddle_init(
    NTT_TYPE ntt_type, TWIDDLE_TYPE twiddle_type,
    uint32_t log_n, uint32_t extend_log_n,
    cudaStream_t& stream,
    const uint64_t* d_omega, uint64_t* d_twiddle,
    ScratchSpan& span)
{
    if (ntt_type == iFFT_cosetFFT) {
        return RustError(-1, "iFFT_cosetFFT not supported\r\n");
    }
    if (ntt_type == cosetFFT || ntt_type == icosetFFT) {
        log_n = extend_log_n;
    }

    if (twiddle_type == TWIDDLE_TYPE::DENSE) {
        RustError status_run = DirectOmegaPowersGenerator<Scalar>::run(stream, d_twiddle, d_omega, log_n - 1, span);
        if (status_run.code != cudaSuccess) {
            return status_run;
        }
    } else if (twiddle_type == TWIDDLE_TYPE::SPARSE) {
        LutOmegaPowersGenerator<Scalar> sparse_twiddle_generator(log_n - 1);
        RustError status_run = sparse_twiddle_generator.run(stream, d_twiddle, d_omega, log_n, span);
        if (status_run.code != cudaSuccess) {
            return status_run;
        }
    } else {
        char buf[128];
        sprintf(buf, "twiddle type %d not supported\r\n", static_cast<int>(twiddle_type));
        return RustError(-6, buf);
    }

    return cudaSuccess;
}

/*=======================================*/
/*============= FFT normal ==============*/
/*=======================================*/

class CudaFFTNormalInfo {
public:
    // Workspace sizing is deterministic: this constructor always picks
    // the dense twiddle generator and ignores any `free_bytes` hint. The
    // advisory chunker FFIs (`_halo2_*_max_len`,
    // `_halo2_get_fft_split_radix`, `_halo2_evaluate_h_max_rows`) accept a
    // `free_bytes` hint to size their output; the workspace_size preflight
    // and the exec FFIs unconditionally select the dense generator.
    CudaFFTNormalInfo(NTT_TYPE ntt_type, bool is_input_host, uint64_t log_n, uint64_t extend_log_n)
    {
        if (ntt_type == cosetFFT || ntt_type == icosetFFT) {
            log_n = extend_log_n;
        }
        this->ntt_type = ntt_type;
        data_memory_size = Scalar::ELT_BYTES << log_n;
        divisor_memory_size = Scalar::ELT_BYTES;
        // Inner CosetFFT_Part branches in `run_fft` need an extra omega LUT
        // of (log_n+1) field elements. We reserve the slot unconditionally
        // so the workspace size is the same for every NTT type and the
        // inner branch can carve from the same span without re-querying.
        omega_lut_memory_size = (log_n + 1) * Scalar::ELT_BYTES;

        twiddle_memory_size = DirectOmegaPowersGenerator<Scalar>::get_powers_memory_size(log_n - 1);
        twiddle_type = TWIDDLE_TYPE::DENSE;

        required_memory_size = 0;
        if (is_input_host) {
            required_memory_size += data_memory_size;
        }
        required_memory_size += divisor_memory_size;
        required_memory_size += twiddle_memory_size;
        required_memory_size += omega_lut_memory_size;
    }

    NTT_TYPE ntt_type = NTT_TYPE::FFT;
    TWIDDLE_TYPE twiddle_type = TWIDDLE_TYPE::DENSE;
    BUFFFER_TYPE bufffer_type = BUFFFER_TYPE::NORMAL;
    uint64_t required_memory_size = 0;
    uint64_t data_memory_size = 0;
    uint64_t twiddle_memory_size = 0;
    uint64_t divisor_memory_size = 0;
    uint64_t omega_lut_memory_size = 0;
};

// return true: enough memory, false: insufficient memory
extern "C" bool _halo2_fft_normal_check_memory(NTT_TYPE ntt_type, const void* input, uint32_t log_n, uint32_t extend_log_n)
{
    // All Rust call sites pass a host-borrowed pointer (often a dummy),
    // so the "is input on host?" flag is always true.
    (void)input;
    (void)ntt_type;
    (void)input;
    (void)log_n;
    (void)extend_log_n;
    // Deterministic dense + double-buffer sizing: always returns true.
    // Kernel scratch is Rust-owned and carved from the caller's
    // `ScratchSpan`; a true OOM surfaces from
    // `DeviceBuffer<u8>::with_capacity_on` on the Rust side. Retained as
    // an ABI shim.
    return true;
}

RustError cuda_ntt_init_normal(
    CudaFFTNormalInfo& fft_info,
    uint32_t log_n,
    cudaStream_t stream_main,
    ScratchSpan& span,
    void** d_data,
    void** d_twiddle,
    void** d_divisor)
{
    (void)log_n;
    (void)stream_main;
    try {
        *d_data = span.take(fft_info.data_memory_size);
        *d_twiddle = span.take(fft_info.twiddle_memory_size);
        *d_divisor = span.take(fft_info.divisor_memory_size);
    } catch (const cuda_error& error) {
        return RustError(error.code(), error.what());
    };

    return cudaSuccess;
}

// `inner_span` is passed by VALUE: each call gets a fresh cursor over
// its own LUT slot, so reinvoking run_fft / run_fft_many in a loop never
// drains a shared cursor across iterations.
//
// `input_on_device` selects the memcpy kind for the data-plane copy
// from `input` into `d_data`. When `false`, `input` is a host pointer
// and the copy is HostToDevice; when `true`, `input` is a device pointer
// and the copy is DeviceToDevice. The divisor copy is always
// HostToDevice — divisor is always a host scalar.
//
// `input == d_data` is permitted: the per-NTT_TYPE data-plane copy is
// elided in that case. CUDA documents `cudaMemcpyAsync` as requiring
// non-overlapping src/dst, so a `src == dst` self-copy is undefined
// behavior even if many drivers no-op it.
RustError run_fft(
    NTT_TYPE ntt_type, TWIDDLE_TYPE twiddle_type,
    uint64_t log_n, uint64_t extend_log_n,
    cudaStream_t stream,
    void* input, void* divisor,
    void* d_data_v, void* d_twiddle_v, void* d_divisor_v,
    ScratchSpan inner_span,
    bool input_on_device)
{
    uint64_t* d_data = (uint64_t*)d_data_v;
    uint64_t* d_twiddle = (uint64_t*)d_twiddle_v;
    uint64_t* d_divisor = (uint64_t*)d_divisor_v;
    bool is_twiddle_dense = twiddle_type == TWIDDLE_TYPE::DENSE;
    cudaMemcpyKind input_kind = input_on_device ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
    bool needs_input_copy = (input != (void*)d_data);
    try {
        CUDA_OK(cudaMemcpyAsync(d_divisor, divisor, Scalar::ELT_BYTES, cudaMemcpyHostToDevice, stream)); // common
        if (ntt_type == NTT_TYPE::FFT) {
            if (needs_input_copy)
                CUDA_OK(cudaMemcpyAsync(d_data, input, Scalar::ELT_BYTES << log_n, input_kind, stream));
            zkpcuda::common::revbin(stream, (scalar_t*)d_data, log_n);
#include "ntt_combine.h"
            zkpcuda::ntt::dit_module(log_n, batch_size, log_n % batch_size, combine_size_1, combine_size_2, is_twiddle_dense, (const scalar_t*)d_twiddle, (scalar_t*)d_data, stream);
        } else if (ntt_type == NTT_TYPE::iFFT) {
            if (needs_input_copy)
                CUDA_OK(cudaMemcpyAsync(d_data, input, Scalar::ELT_BYTES << log_n, input_kind, stream));
#include "ntt_combine.h"
            zkpcuda::ntt::dif_module(log_n, batch_size, log_n % batch_size, combine_size_1, combine_size_2, is_twiddle_dense, (const scalar_t*)d_twiddle, (scalar_t*)d_data, stream);
            zkpcuda::common::revbin(stream, (scalar_t*)d_data, log_n);
            zkpcuda::common::scale_by_constant(stream, (scalar_t*)d_data, (scalar_t*)d_divisor, log_n);
        } else if (ntt_type == NTT_TYPE::cosetFFT) {
            uint64_t log_n_in = log_n;
            log_n = extend_log_n;
            CUDA_OK(cudaMemsetAsync(d_data, 0, Scalar::ELT_BYTES << log_n, stream));
            if (needs_input_copy)
                CUDA_OK(cudaMemcpyAsync(d_data, input, Scalar::ELT_BYTES << log_n_in, input_kind, stream));
            zkpcuda::zeta::mul_by_zeta(log_n, false, false, (scalar_t*)d_data, stream);
            zkpcuda::common::revbin(stream, (scalar_t*)d_data, log_n);
#include "ntt_combine.h"
            zkpcuda::ntt::dit_module(log_n, batch_size, log_n % batch_size, combine_size_1, combine_size_2, is_twiddle_dense, (const scalar_t*)d_twiddle, (scalar_t*)d_data, stream);
        } else if (ntt_type == NTT_TYPE::CosetFFT_Part) {
            // Inner LUT carved from caller-provided span (sized by
            // `_halo2_fft_normal_workspace_size` to always include this
            // slot regardless of ntt_type).
            uint64_t* d_omega_lut = (uint64_t*)inner_span.take((log_n + 1) * Scalar::ELT_BYTES);
            if (needs_input_copy)
                CUDA_OK(cudaMemcpyAsync(d_data, input, Scalar::ELT_BYTES << log_n, input_kind, stream));
            zkpcuda::omega::generate_omega_log_lut<<<1, 1, 0, stream>>>((scalar_t*)d_omega_lut, (scalar_t*)d_divisor, log_n);
            auto length = 1 << log_n;
            auto threads = std::min(length, 1024);
            auto blocks = (length + threads - 1) / threads;

            zkpcuda::omega::mult_power_of_omega<<<blocks, threads, 0, stream>>>((scalar_t*)d_data, (scalar_t*)d_omega_lut, 1 << log_n);
            zkpcuda::common::revbin(stream, (scalar_t*)d_data, log_n);
#include "ntt_combine.h"
            zkpcuda::ntt::dit_module(log_n, batch_size, log_n % batch_size, combine_size_1, combine_size_2, is_twiddle_dense, (const scalar_t*)d_twiddle, (scalar_t*)d_data, stream);
        } else if (ntt_type == NTT_TYPE::icosetFFT) {
            uint64_t log_n_in = log_n;
            log_n = extend_log_n;
            CUDA_OK(cudaMemsetAsync(d_data, 0, Scalar::ELT_BYTES << log_n, stream));
            if (needs_input_copy)
                CUDA_OK(cudaMemcpyAsync(d_data, input, Scalar::ELT_BYTES << log_n_in, input_kind, stream));
#include "ntt_combine.h"
            zkpcuda::ntt::dif_module(log_n, batch_size, log_n % batch_size, combine_size_1, combine_size_2, is_twiddle_dense, (const scalar_t*)d_twiddle, (scalar_t*)d_data, stream);
            zkpcuda::common::revbin(stream, (scalar_t*)d_data, log_n);
            zkpcuda::common::scale_by_constant(stream, (scalar_t*)d_data, (scalar_t*)d_divisor, log_n);
            zkpcuda::zeta::mul_by_zeta(log_n, true, false, (scalar_t*)d_data, stream);
        }
    } catch (const cuda_error& error) {
        printf("%s\r\n", error.what()); // error hadling is not fully implemented in *_many(), better printf here
        return RustError(error.code(), error.what());
    };

    return cudaSuccess;
}

// TODO: refactor the in/out of fft_many to be raw pointer-based as well
//       then reuse run_fft() and delete run_fft_many()
// `inner_span` is passed by VALUE — see `run_fft` rationale.
RustError run_fft_many(
    NTT_TYPE ntt_type, TWIDDLE_TYPE twiddle_type,
    uint64_t log_n, uint64_t extend_log_n,
    cudaStream_t stream,
    uint64_t* input, uint64_t* h_divisor,
    uint64_t* d_data, uint64_t* d_twiddle, uint64_t* d_divisor,
    ScratchSpan inner_span,
    bool input_on_device)
{
    bool is_twiddle_dense = twiddle_type == TWIDDLE_TYPE::DENSE;
    cudaMemcpyKind input_kind = input_on_device ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
    try {
        if (ntt_type == NTT_TYPE::FFT) {
            CUDA_OK(cudaMemcpyAsync(d_data, input, Scalar::ELT_BYTES << log_n, input_kind, stream));
            zkpcuda::common::revbin(stream, (scalar_t*)d_data, log_n);
#include "ntt_combine.h"
            zkpcuda::ntt::dit_module(log_n, batch_size, log_n % batch_size, combine_size_1, combine_size_2, is_twiddle_dense, (const scalar_t*)d_twiddle, (scalar_t*)d_data, stream);
        } else if (ntt_type == NTT_TYPE::iFFT) {
            CUDA_OK(cudaMemcpyAsync(d_data, input, Scalar::ELT_BYTES << log_n, input_kind, stream));
            CUDA_OK(cudaMemcpyAsync(d_divisor, h_divisor, Scalar::ELT_BYTES, cudaMemcpyHostToDevice, stream));
#include "ntt_combine.h"
            zkpcuda::ntt::dif_module(log_n, batch_size, log_n % batch_size, combine_size_1, combine_size_2, is_twiddle_dense, (const scalar_t*)d_twiddle, (scalar_t*)d_data, stream);
            zkpcuda::common::revbin(stream, (scalar_t*)d_data, log_n);
            zkpcuda::common::scale_by_constant(stream, (scalar_t*)d_data, (scalar_t*)d_divisor, log_n);
        } else if (ntt_type == NTT_TYPE::cosetFFT) {
            uint64_t log_n_in = log_n;
            log_n = extend_log_n;
            CUDA_OK(cudaMemsetAsync(d_data, 0, Scalar::ELT_BYTES << log_n, stream));
            CUDA_OK(cudaMemcpyAsync(d_data, input, Scalar::ELT_BYTES << log_n_in, input_kind, stream));
            zkpcuda::zeta::mul_by_zeta(log_n, false, false, (scalar_t*)d_data, stream);
            zkpcuda::common::revbin(stream, (scalar_t*)d_data, log_n);
#include "ntt_combine.h"
            zkpcuda::ntt::dit_module(log_n, batch_size, log_n % batch_size, combine_size_1, combine_size_2, is_twiddle_dense, (const scalar_t*)d_twiddle, (scalar_t*)d_data, stream);
        } else if (ntt_type == NTT_TYPE::CosetFFT_Part) {
            // Inner LUT carved from caller-provided span (sized by
            // `_halo2_fft_many_workspace_size` to always include this slot).
            uint64_t* d_omega_lut = (uint64_t*)inner_span.take((log_n + 1) * Scalar::ELT_BYTES);
            CUDA_OK(cudaMemcpyAsync(d_divisor, h_divisor, Scalar::ELT_BYTES, cudaMemcpyHostToDevice, stream));
            CUDA_OK(cudaMemcpyAsync(d_data, input, Scalar::ELT_BYTES << log_n, input_kind, stream));
            zkpcuda::omega::generate_omega_log_lut<<<1, 1, 0, stream>>>((scalar_t*)d_omega_lut, (scalar_t*)d_divisor, log_n);

            auto length = 1 << log_n;
            auto threads = std::min(length, 1024);
            auto blocks = (length + threads - 1) / threads;
            zkpcuda::omega::mult_power_of_omega<<<blocks, threads, 0, stream>>>((scalar_t*)d_data, (scalar_t*)d_omega_lut, 1 << log_n);
            zkpcuda::common::revbin(stream, (scalar_t*)d_data, log_n);
#include "ntt_combine.h"
            zkpcuda::ntt::dit_module(log_n, batch_size, log_n % batch_size, combine_size_1, combine_size_2, is_twiddle_dense, (const scalar_t*)d_twiddle, (scalar_t*)d_data, stream);
        } else if (ntt_type == NTT_TYPE::icosetFFT) {
            uint64_t log_n_in = log_n;
            log_n = extend_log_n;
            CUDA_OK(cudaMemsetAsync(d_data, 0, Scalar::ELT_BYTES << log_n, stream));
            CUDA_OK(cudaMemcpyAsync(d_data, input, Scalar::ELT_BYTES << log_n_in, input_kind, stream));
            CUDA_OK(cudaMemcpyAsync(d_divisor, h_divisor, Scalar::ELT_BYTES, cudaMemcpyHostToDevice, stream));
#include "ntt_combine.h"
            zkpcuda::ntt::dif_module(log_n, batch_size, log_n % batch_size, combine_size_1, combine_size_2, is_twiddle_dense, (const scalar_t*)d_twiddle, (scalar_t*)d_data, stream);
            zkpcuda::common::revbin(stream, (scalar_t*)d_data, log_n);
            zkpcuda::common::scale_by_constant(stream, (scalar_t*)d_data, (scalar_t*)d_divisor, log_n);
            zkpcuda::zeta::mul_by_zeta(log_n, true, false, (scalar_t*)d_data, stream);
        }
    } catch (const cuda_error& error) {
        printf("%s\r\n", error.what()); // error hadling is not fully implemented in *_many(), better printf here
        return RustError(error.code(), error.what());
    };

    return cudaSuccess;
}

// fft without extend log_n
// Pure host preflight: returns the byte size Rust must hand into
// `_halo2_fft_normal` / `_halo2_cosetfft` as the `scratch` buffer. Sizes
// for the dense twiddle path unconditionally and always reserves an inner
// CosetFFT_Part LUT slot so the same workspace size works for every
// ntt_type. Also includes the `twiddle_init` inner-generator scratch
// since `DirectOmegaPowersGenerator::run` carves from the caller's span,
// and a 32-byte staging slot used by the launcher to host the omega
// scalar in device memory before handing it to `twiddle_init`.
extern "C" uint64_t _halo2_fft_normal_workspace_size(uint32_t ntt_type, uint32_t log_n, uint32_t extend_log_n)
{
    CudaFFTNormalInfo info((NTT_TYPE)ntt_type, /*is_input_host=*/true, log_n, extend_log_n);
    return align_up(info.data_memory_size, 32)
        + align_up(info.twiddle_memory_size, 32)
        + align_up(info.divisor_memory_size, 32)
        + align_up((uint64_t)Scalar::ELT_BYTES, 32)
        + align_up(twiddle_init_run_scratch_size((NTT_TYPE)ntt_type, log_n, extend_log_n), 32)
        + align_up(info.omega_lut_memory_size, 32);
}

extern "C" RustError _halo2_fft_normal(
    NTT_TYPE ntt_type, uint32_t log_n,
    const void* input, void* output,
    const void* omega, const void* divisor,
    void* scratch, uint64_t scratch_bytes,
    cudaStream_t stream)
{
    if (ntt_type == iFFT_cosetFFT) {
        return RustError(-1, "iFFT_cosetFFT not supported\r\n");
    }
    // Dense twiddle sizing uses `log_n - 1`; reject log_n == 0 before it underflows.
    if (log_n == 0) {
        return RustError(cudaErrorInvalidValue, "_halo2_fft_normal: log_n must be >= 1\r\n");
    }

    // All Rust call sites pass host-borrowed pointers for all four params.
    CudaFFTNormalInfo fft_info(
        ntt_type,
        /*is_input_host=*/true,
        log_n,
        log_n /*extend_log_n*/);

    ScratchSpan span { (uint8_t*)scratch, (size_t)scratch_bytes };

    void* d_data = nullptr;
    void* d_twiddle = nullptr;
    void* d_divisor = nullptr;
    RustError status_init = cuda_ntt_init_normal(
        fft_info, log_n,
        stream, span,
        &d_data, &d_twiddle, &d_divisor);
    if (status_init.code != cudaSuccess)
        return status_init;

    // Stage host omega into a 32-byte device slot before twiddle_init.
    uint64_t* d_omega_staging = (uint64_t*)span.take(Scalar::ELT_BYTES);
    try {
        CUDA_OK(cudaMemcpyAsync(d_omega_staging, omega, Scalar::ELT_BYTES, cudaMemcpyHostToDevice, stream));
    } catch (const cuda_error& error) {
        return RustError(error.code(), error.what());
    }

    // Carve twiddle_init's inner-generator scratch from the post-init span.
    // This must happen before `run_fft` (which may take the omega_lut slot)
    // so the layout matches `_halo2_fft_normal_workspace_size`.
    uint64_t ti_size = align_up(twiddle_init_run_scratch_size(fft_info.ntt_type, log_n, log_n), 32);
    ScratchSpan ti_span { (uint8_t*)span.take(ti_size), (size_t)ti_size };
    RustError status_twiddle = twiddle_init(
        fft_info.ntt_type, fft_info.twiddle_type,
        log_n, log_n /*extended*/,
        stream, d_omega_staging, (uint64_t*)d_twiddle,
        ti_span);
    if (status_twiddle.code != cudaSuccess) {
        return status_twiddle;
    }

    RustError status_fft = run_fft(
        fft_info.ntt_type,
        fft_info.twiddle_type,
        log_n, log_n,
        stream,
        (void*)input, (void*)divisor,
        d_data, d_twiddle, d_divisor,
        span,
        /*input_on_device=*/false);
    if (status_fft.code != cudaSuccess) {
        return status_fft;
    }

    // common: copy results back. Scratch lifetime ends when Rust drops the
    // backing DeviceBuffer<u8> at scope-end on its own stream.
    try {
        CUDA_OK(cudaMemcpyAsync(output, d_data, fft_info.data_memory_size, cudaMemcpyDeviceToHost, stream));
        CUDA_OK(cudaStreamSynchronize(stream));
    } catch (const cuda_error& error) {
        return RustError(error.code(), error.what());
    }
    return cudaSuccess;
}

// Workspace size for `_halo2_fft_normal_to_device`. Mirrors
// `_halo2_fft_normal_workspace_size` but excludes the `d_data` slot —
// the caller supplies `d_output`, which doubles as `d_data` for the
// in-place FFT body.
extern "C" uint64_t _halo2_fft_normal_to_device_workspace_size(uint32_t ntt_type, uint32_t log_n, uint32_t extend_log_n)
{
    CudaFFTNormalInfo info((NTT_TYPE)ntt_type, /*is_input_host=*/false, log_n, extend_log_n);
    return align_up(info.twiddle_memory_size, 32)
        + align_up(info.divisor_memory_size, 32)
        + align_up(twiddle_init_run_scratch_size((NTT_TYPE)ntt_type, log_n, extend_log_n), 32)
        + align_up(info.omega_lut_memory_size, 32);
}

// Device-input / device-output variant of `_halo2_fft_normal`.
//
// Accepts device-resident input AND writes device-resident output: the
// FFT runs in place on `d_output` after a DeviceToDevice pre-copy from
// `d_input`. `d_input == d_output` is permitted and elides the
// pre-copy entirely (CUDA `cudaMemcpyAsync` documents non-overlapping
// src/dst, so `src == dst` would be undefined behavior). For
// non-aliased calls, `d_input` and `d_output` must not partially
// overlap. `d_omega` is a device-resident 32-byte scalar; `h_divisor`
// is a host scalar.
//
// Supported `ntt_type`: `FFT`, `iFFT`, `CosetFFT_Part`. The `cosetFFT`
// and `icosetFFT` branches in `run_fft` issue a `cudaMemsetAsync` on
// `d_data` before the input copy, which would clobber an aliased input
// — so they are out of contract here; route those transforms through
// the dedicated `_halo2_cosetfft` entry point instead.
//
// Like `_halo2_fft_many_to_device`, this entry point does not
// `cudaStreamSynchronize` the main stream on return: the next kernel
// can chain on the same stream without an intervening fence.
extern "C" RustError _halo2_fft_normal_to_device(
    NTT_TYPE ntt_type, uint32_t log_n,
    const void* d_input, void* d_output,
    const void* d_omega, const void* h_divisor,
    void* scratch, uint64_t scratch_bytes,
    cudaStream_t stream)
{
    if (ntt_type != NTT_TYPE::FFT
        && ntt_type != NTT_TYPE::iFFT
        && ntt_type != NTT_TYPE::CosetFFT_Part) {
        return RustError(cudaErrorInvalidValue, "_halo2_fft_normal_to_device supports only FFT, iFFT, CosetFFT_Part\r\n");
    }
    // Dense twiddle sizing uses `log_n - 1`; reject log_n == 0 before it underflows.
    if (log_n == 0) {
        return RustError(cudaErrorInvalidValue, "_halo2_fft_normal_to_device: log_n must be >= 1\r\n");
    }

    CudaFFTNormalInfo fft_info(
        ntt_type,
        /*is_input_host=*/false,
        log_n,
        log_n /*extend_log_n*/);

    ScratchSpan span { (uint8_t*)scratch, (size_t)scratch_bytes };

    void* d_twiddle = nullptr;
    void* d_divisor = nullptr;
    try {
        d_twiddle = span.take(fft_info.twiddle_memory_size);
        d_divisor = span.take(fft_info.divisor_memory_size);
    } catch (const cuda_error& error) {
        return RustError(error.code(), error.what());
    }

    // Carve twiddle_init's inner-generator scratch before `run_fft`
    // (which may take the omega_lut slot), matching the layout assumed
    // by `_halo2_fft_normal_to_device_workspace_size`.
    uint64_t ti_size = align_up(twiddle_init_run_scratch_size(fft_info.ntt_type, log_n, log_n), 32);
    ScratchSpan ti_span { (uint8_t*)span.take(ti_size), (size_t)ti_size };
    RustError status_twiddle = twiddle_init(
        fft_info.ntt_type, fft_info.twiddle_type,
        log_n, log_n,
        stream, (const uint64_t*)d_omega, (uint64_t*)d_twiddle,
        ti_span);
    if (status_twiddle.code != cudaSuccess) {
        return status_twiddle;
    }

    RustError status_fft = run_fft(
        fft_info.ntt_type,
        fft_info.twiddle_type,
        log_n, log_n,
        stream,
        (void*)d_input, (void*)h_divisor,
        d_output, d_twiddle, d_divisor,
        span,
        /*input_on_device=*/true);
    if (status_fft.code != cudaSuccess) {
        return status_fft;
    }

    return cudaSuccess;
}

// out-of-place version
// input >>> output
extern "C" RustError _halo2_cosetfft(
    NTT_TYPE ntt_type,
    uint32_t log_n, uint32_t extend_log_n,
    const void* input, void* output,
    const void* omega, const void* divisor,
    void* scratch, uint64_t scratch_bytes,
    cudaStream_t stream)
{
    // Dense twiddle sizing on the active log uses `log - 1`; reject the
    // size-one case (cosetFFT/icosetFFT pick `extend_log_n`; others pick
    // `log_n`) before it underflows.
    if (log_n == 0 || extend_log_n == 0) {
        return RustError(cudaErrorInvalidValue, "_halo2_cosetfft: log_n and extend_log_n must be >= 1\r\n");
    }
    // All Rust call sites pass host-borrowed pointers.
    (void)divisor; // divisor unused in this entry point's body

    CudaFFTNormalInfo fft_info(
        ntt_type,
        /*is_input_host=*/true,
        log_n,
        extend_log_n);

    ScratchSpan span { (uint8_t*)scratch, (size_t)scratch_bytes };

    void* d_data = nullptr;
    void* d_twiddle = nullptr;
    void* d_divisor = nullptr;
    RustError status_init = cuda_ntt_init_normal(
        fft_info, extend_log_n,
        stream, span,
        &d_data, &d_twiddle, &d_divisor);
    if (status_init.code != cudaSuccess)
        return status_init;

    // Stage host omega into a 32-byte device slot before twiddle_init.
    uint64_t* d_omega_staging = (uint64_t*)span.take(Scalar::ELT_BYTES);
    try {
        CUDA_OK(cudaMemcpyAsync(d_omega_staging, omega, Scalar::ELT_BYTES, cudaMemcpyHostToDevice, stream));
    } catch (const cuda_error& error) {
        return RustError(error.code(), error.what());
    }

    uint64_t ti_size = align_up(twiddle_init_run_scratch_size(fft_info.ntt_type, log_n, extend_log_n), 32);
    ScratchSpan ti_span { (uint8_t*)span.take(ti_size), (size_t)ti_size };
    RustError status_twiddle = twiddle_init(
        fft_info.ntt_type, fft_info.twiddle_type, log_n, extend_log_n,
        stream, d_omega_staging, (uint64_t*)d_twiddle,
        ti_span);
    if (status_twiddle.code != cudaSuccess) {
        return status_twiddle;
    }

    bool is_twiddle_dense = fft_info.twiddle_type == TWIDDLE_TYPE::DENSE;
    try {
        size_t extended_data_size = Scalar::ELT_BYTES << (uint64_t)extend_log_n;
        CUDA_OK(cudaMemsetAsync(d_data, 0, extended_data_size, stream));
        CUDA_OK(cudaMemcpyAsync(d_data, input, Scalar::ELT_BYTES << (uint64_t)log_n, cudaMemcpyHostToDevice, stream));
        log_n = extend_log_n; // !!!
        zkpcuda::zeta::mul_by_zeta(log_n, false, false, (scalar_t*)d_data, stream);
        zkpcuda::common::revbin(stream, (scalar_t*)d_data, log_n);
#include "ntt_combine.h"
        zkpcuda::ntt::dit_module(log_n, batch_size, log_n % batch_size, combine_size_1, combine_size_2, is_twiddle_dense, (const scalar_t*)d_twiddle, (scalar_t*)d_data, stream);

        CUDA_OK(cudaMemcpyAsync(output, d_data, Scalar::ELT_BYTES << extend_log_n, cudaMemcpyDeviceToHost, stream));
        CUDA_OK(cudaStreamSynchronize(stream));
    } catch (const cuda_error& error) {
        return RustError(error.code(), error.what());
    };

    return cudaSuccess;
}

/*=======================================*/
/*=========== FFT split-radix ===========*/
/*=======================================*/

class CudaFFTSplitInfo {
public:
    static uint64_t get_required_memory_size(NTT_TYPE ntt_type, uint32_t log_n, uint32_t extend_log_n)
    {
        if (ntt_type == cosetFFT || ntt_type == icosetFFT) {
            log_n = extend_log_n;
        }

        LutOmegaPowersGenerator<Scalar> sparse_twiddle_generator(log_n - 1);
        uint64_t twiddle_memory_size = sparse_twiddle_generator.get_lut_memory_size();
        uint64_t data_memory_size = Scalar::ELT_BYTES << log_n;
        uint64_t divisor_memory_size = Scalar::ELT_BYTES;
        uint64_t required_memory_size = 0;
        required_memory_size += twiddle_memory_size;
        required_memory_size += data_memory_size;
        required_memory_size += divisor_memory_size;

        return required_memory_size;
    }
};

extern "C" int32_t _halo2_get_fft_split_radix(NTT_TYPE ntt_type, uint32_t log_n, uint32_t extend_log_n, uint64_t free_bytes)
{
    uint32_t log_split_radix = 0;
    uint64_t required_memory_size = 0;
    while (log_split_radix < log_n) {
        required_memory_size = CudaFFTSplitInfo::get_required_memory_size(
            ntt_type,
            log_n - log_split_radix,
            extend_log_n - log_split_radix);
        if (required_memory_size < free_bytes) {
            return log_split_radix;
        }
        log_split_radix++;
    }
    return -1; // insufficient memory
}

/*=======================================*/
/*============== FFT many ===============*/
/*=======================================*/

class CudaFFTManyInfo {
public:
    // Same deterministic-sizing posture as `CudaFFTNormalInfo`: always
    // dense twiddle, always double-buffered. For very large NTTs
    // (log_n >= 24) a future change can plumb a `free_bytes_hint` from Rust
    // and re-introduce the sparse / normal-buffer fallbacks here.
    CudaFFTManyInfo(NTT_TYPE ntt_type, uint32_t log_n, uint32_t extend_log_n)
    {
        if (ntt_type == cosetFFT || ntt_type == icosetFFT) {
            log_n = extend_log_n;
        }
        this->ntt_type = ntt_type;
        divisor_memory_size = Scalar::ELT_BYTES;
        data_memory_size = Scalar::ELT_BYTES << log_n;
        // Per-slot LUT size for the inner CosetFFT_Part branch in
        // `run_fft_many`. Two disjoint slots back the double-buffered
        // ping-pong (see `_halo2_fft_many` body). Reserved unconditionally
        // so the same workspace size works for every ntt_type.
        omega_lut_per_slot_size = (log_n + 1) * Scalar::ELT_BYTES;
        omega_lut_memory_size = 2 * omega_lut_per_slot_size;

        twiddle_memory_size = DirectOmegaPowersGenerator<Scalar>::get_powers_memory_size(log_n - 1);
        twiddle_type = TWIDDLE_TYPE::DENSE;
        bufffer_type = BUFFFER_TYPE::DOUBLE;

        uint64_t data_double_memory_size = 2 * data_memory_size;
        twiddle_memory_offset = data_double_memory_size;

        required_memory_size = divisor_memory_size
            + data_double_memory_size
            + twiddle_memory_size
            + omega_lut_memory_size;
    }

    NTT_TYPE ntt_type = NTT_TYPE::FFT;
    TWIDDLE_TYPE twiddle_type = TWIDDLE_TYPE::DENSE;
    BUFFFER_TYPE bufffer_type = BUFFFER_TYPE::DOUBLE;
    uint64_t required_memory_size = 0;
    uint64_t data_memory_size = 0;
    uint64_t twiddle_memory_size = 0;
    uint64_t twiddle_memory_offset = 0;
    uint64_t divisor_memory_size = 0;
    uint64_t omega_lut_per_slot_size = 0;
    uint64_t omega_lut_memory_size = 0; // = 2 * omega_lut_per_slot_size
};

// Pure host preflight: returns the byte size Rust must hand into
// `_halo2_fft_many` as the `scratch` buffer.
extern "C" uint64_t _halo2_fft_many_workspace_size(uint32_t ntt_type, uint32_t log_n, uint32_t extend_log_n)
{
    CudaFFTManyInfo info((NTT_TYPE)ntt_type, log_n, extend_log_n);
    // Layout matches the take order in `_halo2_fft_many`: one big slab
    // (data×2 + twiddle + divisor, offset-divided in `cuda_ntt_init_many`),
    // then `twiddle_init`'s inner-generator scratch, then two disjoint
    // per-slot LUT buffers.
    uint64_t slab_bytes = info.required_memory_size - info.omega_lut_memory_size;
    return align_up(slab_bytes, 32)
        + align_up(twiddle_init_run_scratch_size((NTT_TYPE)ntt_type, log_n, extend_log_n), 32)
        + align_up(info.omega_lut_per_slot_size, 32)
        + align_up(info.omega_lut_per_slot_size, 32);
}

// Device-output many-FFT runs sequentially on the caller's explicit stream
// and writes directly into the final output buffers, so it needs only one
// data slot and one inner CosetFFT_Part LUT slot.
extern "C" uint64_t _halo2_fft_many_to_device_workspace_size(
    uint32_t ntt_type,
    uint32_t log_n,
    uint32_t extend_log_n)
{
    CudaFFTManyInfo info((NTT_TYPE)ntt_type, log_n, extend_log_n);
    return align_up(info.twiddle_memory_size, 32)
        + align_up(info.divisor_memory_size, 32)
        + align_up(twiddle_init_run_scratch_size((NTT_TYPE)ntt_type, log_n, extend_log_n), 32)
        + align_up(info.omega_lut_per_slot_size, 32);
}

// Always returns true; FFT memory size is deterministic per
// `_halo2_fft_normal_check_memory`'s sizing model. Retained for ABI
// stability with existing callers.
extern "C" bool _halo2_fft_many_check_memory(NTT_TYPE ntt_type, uint32_t log_n, uint32_t extend_log_n)
{
    (void)ntt_type;
    (void)log_n;
    (void)extend_log_n;
    return true;
}

RustError cuda_ntt_init_many(
    CudaFFTManyInfo& fft_info,
    uint32_t log_n,
    uint32_t extend_log_n,
    cudaStream_t stream_main,
    ScratchSpan& span,
    uint64_t** fft_gpu_mem,
    uint64_t** fft_data,
    uint64_t** fft_twiddle,
    uint64_t** fft_divisor)
{
    (void)log_n;
    (void)extend_log_n;
    (void)stream_main;
    try {
        // Carve the data+twiddle+divisor slab out of the caller's span.
        uint64_t slab_bytes = fft_info.required_memory_size - fft_info.omega_lut_memory_size;
        *fft_gpu_mem = (uint64_t*)span.take(slab_bytes);
    } catch (const cuda_error& error) {
        return RustError(error.code(), error.what());
    };

    *fft_data = (uint64_t*)((char*)(*fft_gpu_mem));
    *fft_twiddle = (uint64_t*)((char*)(*fft_data) + fft_info.twiddle_memory_offset);
    *fft_divisor = (uint64_t*)((char*)(*fft_twiddle) + fft_info.twiddle_memory_size);
    return cudaSuccess;
}

// out-of-place version
// input >>> output
extern "C" RustError _halo2_fft_many(
    NTT_TYPE ntt_type,
    uint32_t num_many, uint32_t log_n, uint32_t extend_log_n,
    Scalar* input, Scalar* output,
    const void* d_omega, Scalar* divisor,
    void* scratch, uint64_t scratch_bytes,
    cudaStream_t stream)
{

    if (ntt_type == icosetFFT) {
        return RustError(2, "icosetFFT not supported now");
    }
    // Dense twiddle sizing on the active log uses `log - 1`; reject the
    // size-one case before it underflows.
    if (log_n == 0 || extend_log_n == 0) {
        return RustError(cudaErrorInvalidValue, "_halo2_fft_many: log_n and extend_log_n must be >= 1\r\n");
    }

    uint64_t* fft_gpu_mem = nullptr;
    uint64_t* fft_data = nullptr;
    uint64_t* fft_twiddle = nullptr;
    uint64_t* fft_divisor = nullptr;
    CudaFFTManyInfo fft_info(ntt_type, log_n, extend_log_n);
    ScratchSpan span { (uint8_t*)scratch, (size_t)scratch_bytes };
    RustError status_init = cuda_ntt_init_many(
        fft_info, log_n, extend_log_n,
        stream, span,
        &fft_gpu_mem, &fft_data, &fft_twiddle, &fft_divisor);
    if (status_init.code != cudaSuccess)
        return status_init;
    uint64_t* d_databuf[2];
    d_databuf[0] = fft_data;
    d_databuf[1] = (uint64_t*)((char*)d_databuf[0] + fft_info.data_memory_size);

    uint64_t ti_size = align_up(twiddle_init_run_scratch_size(fft_info.ntt_type, log_n, extend_log_n), 32);
    ScratchSpan ti_span { (uint8_t*)span.take(ti_size), (size_t)ti_size };
    RustError status_twiddle = twiddle_init(
        fft_info.ntt_type, fft_info.twiddle_type, log_n, extend_log_n,
        stream, (const uint64_t*)d_omega, fft_twiddle,
        ti_span);
    if (status_twiddle.code != cudaSuccess) {
        return status_twiddle;
    }

    // Two disjoint LUT slots, one per double-buffer half.
    uint8_t* lut_a_ptr = (uint8_t*)span.take(fft_info.omega_lut_per_slot_size);
    uint8_t* lut_b_ptr = (uint8_t*)span.take(fft_info.omega_lut_per_slot_size);
    ScratchSpan inner_a { lut_a_ptr, (size_t)fft_info.omega_lut_per_slot_size };
    ScratchSpan inner_b { lut_b_ptr, (size_t)fft_info.omega_lut_per_slot_size };

    try {
        // Always double-buffered; there is no single-buffer NORMAL path.
        for (uint32_t i = 0; i < num_many; i += 2) {
            run_fft_many(fft_info.ntt_type, fft_info.twiddle_type, log_n, extend_log_n, stream, input[i].ptr, divisor->ptr, d_databuf[0], fft_twiddle, fft_divisor, inner_a, /*input_on_device=*/false);
            if (i + 1 < num_many) {
                run_fft_many(fft_info.ntt_type, fft_info.twiddle_type, log_n, extend_log_n, stream, input[i + 1].ptr, divisor->ptr, d_databuf[1], fft_twiddle, fft_divisor, inner_b, /*input_on_device=*/false);
            }
            CUDA_OK(cudaMemcpyAsync(output[i].ptr, d_databuf[0], Scalar::ELT_BYTES << extend_log_n, cudaMemcpyDeviceToHost, stream));
            if (i + 1 < num_many) {
                CUDA_OK(cudaMemcpyAsync(output[i + 1].ptr, d_databuf[1], Scalar::ELT_BYTES << extend_log_n, cudaMemcpyDeviceToHost, stream));
            }
        }
    } catch (const cuda_error& error) {
        return RustError(error.code(), error.what());
    };

    try {
        CUDA_OK(cudaStreamSynchronize(stream));
    } catch (const cuda_error& error) {
        return RustError(error.code(), error.what());
    };

    return cudaSuccess;
}

// Device-output variant of `_halo2_fft_many`.
//
// Same algebra as `_halo2_fft_many`, but each `output[i].ptr` is a device
// pointer and each FFT writes directly into that buffer on the caller's
// explicit stream, so the result can feed straight into a subsequent GPU
// kernel. `input_on_device` selects the data-plane copy kind: `false` reads
// each `input[i].ptr` as a host pointer (HostToDevice); `true` reads it as a
// device pointer (DeviceToDevice). `divisor` is host-borrowed (each
// divisor-bearing NTT_TYPE issues its own 32-byte H2D inside `run_fft_many`);
// `d_omega` is a device-resident scalar uploaded by the Rust caller.
extern "C" RustError _halo2_fft_many_to_device(
    NTT_TYPE ntt_type,
    uint32_t num_many, uint32_t log_n, uint32_t extend_log_n,
    Scalar* input, Scalar* output,
    const void* d_omega, Scalar* divisor,
    bool input_on_device,
    void* scratch, uint64_t scratch_bytes,
    cudaStream_t stream)
{

    if (ntt_type == icosetFFT) {
        return RustError(2, "icosetFFT not supported now");
    }
    // Dense twiddle sizing on the active log uses `log - 1`; reject the
    // size-one case before it underflows.
    if (log_n == 0 || extend_log_n == 0) {
        return RustError(cudaErrorInvalidValue, "_halo2_fft_many_to_device: log_n and extend_log_n must be >= 1\r\n");
    }

    CudaFFTManyInfo fft_info(ntt_type, log_n, extend_log_n);
    ScratchSpan span { (uint8_t*)scratch, (size_t)scratch_bytes };
    uint64_t* fft_twiddle = nullptr;
    uint64_t* fft_divisor = nullptr;

    try {
        fft_twiddle = (uint64_t*)span.take(fft_info.twiddle_memory_size);
        fft_divisor = (uint64_t*)span.take(fft_info.divisor_memory_size);
    } catch (const cuda_error& error) {
        return RustError(error.code(), error.what());
    }

    uint64_t ti_size = align_up(twiddle_init_run_scratch_size(fft_info.ntt_type, log_n, extend_log_n), 32);
    ScratchSpan ti_span { (uint8_t*)span.take(ti_size), (size_t)ti_size };
    RustError status_twiddle = twiddle_init(
        fft_info.ntt_type, fft_info.twiddle_type, log_n, extend_log_n,
        stream, (const uint64_t*)d_omega, fft_twiddle,
        ti_span);
    if (status_twiddle.code != cudaSuccess)
        return status_twiddle;

    uint8_t* lut_ptr = (uint8_t*)span.take(fft_info.omega_lut_per_slot_size);
    ScratchSpan inner { lut_ptr, (size_t)fft_info.omega_lut_per_slot_size };

    try {
        for (uint32_t i = 0; i < num_many; ++i) {
            run_fft_many(
                fft_info.ntt_type,
                fft_info.twiddle_type,
                log_n,
                extend_log_n,
                stream,
                input[i].ptr,
                divisor->ptr,
                (uint64_t*)output[i].ptr,
                fft_twiddle,
                fft_divisor,
                inner,
                input_on_device);
        }
    } catch (const cuda_error& error) {
        return RustError(error.code(), error.what());
    }

    return cudaSuccess;
}
