#pragma once

// Halo2-gpu C++ <-> Rust FFI glue.
//
//   1. `RustError` — the ABI-pinned return struct for kernel FFI.
//   2. `utils::FFITraitObject` — opaque handle for polynomial/scalar arrays
//      crossing the FFI. Matches the Rust `FFITraitObject { ptr: usize }`.
//
// Memory-ownership contract: Rust owns ALL device memory seen by kernels
// AND queries free GPU memory on their behalf. Each kernel FFI that needs
// internal scratch takes `(void* scratch, uint64_t scratch_bytes)` plus a
// pure-host `_halo2_<kernel>_workspace_size(...)` preflight, and
// sub-allocates via `ScratchSpan::take(...)` (see `common/scratch_span.h`).
// Each `_halo2_*_max_len` / `_halo2_msm_max_length` /
// `_halo2_evaluate_h_max_rows` / `_halo2_get_fft_split_radix` advisory FFI
// takes a `uint64_t free_bytes` parameter — Rust queries `cudaMemGetInfo`
// once at the call site (see
// `halo2_proofs::cuda::utils::query_device_free_bytes_for_chunking`).
// C++ kernels do not call `cudaMalloc*` and do not query `cudaMemGetInfo`.

#include <cstdint>
#include <cstring>
#include <string>

#include <cuda_runtime.h>

// ABI-stable error struct returned by kernel FFIs. Matches Rust's
// `halo2_proofs::cuda::error::CudaError { code: i32, message: *mut i8 }`.
// Exclusively returned by value. No destructor: Rust frees `message`.
struct RustError {
    int code;
    char* message;

    RustError(int e = 0)
        : code(e)
        , message(nullptr)
    {
    }
    RustError(int e, const std::string& str)
        : code(e)
    {
        message = str.empty() ? nullptr : strdup(str.c_str());
    }
    RustError(int e, const char* str)
        : code(e)
    {
        message = str == nullptr ? nullptr : strdup(str);
    }
};

namespace utils {

// Opaque handle passed across the FFI for scalar / point / polynomial
// arrays. Matches the Rust-side `FFITraitObject { ptr: usize }`; the
// `ELT_*` constants encode BN254 `Fr` element size (256 bits = 32 bytes).
struct FFITraitObject {
    uint64_t* ptr { nullptr };
    static constexpr uint64_t ELT_BYTES = 32;
    static constexpr uint64_t ELT_LIMBS = ELT_BYTES / sizeof(uint64_t);
};

} // namespace utils
