#include <assert.h>
#include <chrono>
#include <cmath>
#include <map>
#include <string>
#include <vector>

#include "common/exception.h"
#include "common/halo2_ffi.h"
#include "common/scratch_span.h"

#include "kernel/pippenger.h"

#define MAX_MAX_LEN_LOG 28

static constexpr uint64_t SCALAR_BIT = 254;
static constexpr uint64_t AFFINE_POINT_SIZE = 64;

using Scalar = utils::FFITraitObject;
using Point = utils::FFITraitObject;
using Output = utils::FFITraitObject;

// Computes the scratch layout and per-slot offsets for the Pippenger MSM
// kernel chain.
class CudaMsmInfo {
public:
    CudaMsmInfo(uint64_t length)
    {
        set_params(length);
    }

    void set_params(uint64_t length)
    {
        // basic
        length_ = length;
        win_bit_ = (uint64_t)log2(length_) / 2;
        win_bit_half_ = (win_bit_ + 1) / 2;
        win_num_ = (SCALAR_BIT + win_bit_ - 1) / win_bit_;
        bin_num_ = 1 << win_bit_;

        // sparsity
        // Inclusive upper bound on dense buckets PER WINDOW. Per-window
        // sparsities sum to <= 1 and a bucket is dense iff its float sparsity
        // >= threshold, so at most ceil(1/threshold) buckets qualify (for 0.10:
        // exactly 10, since 10 buckets of 26/260 = 0.10f each fit). The old
        // `(uint64_t)(1.0 / SPARSITY_THRESHOLD)` truncated 0.10f's reciprocal
        // (9.9999998) to 9 and undersized BOTH the worklist and the
        // `d_dense_out` arena by one slice/window; ceil in double fixes it.
        MAX_DENSE_BUCKET_NUM = (uint64_t)ceil(1.0 / (double)SPARSITY_THRESHOLD);
        DENSE_SPLIT_N_BLOCKS = 128; // 128 for sppark:  sizeof(bucket_t)
        size_dense_out_ = win_num_ * (MAX_DENSE_BUCKET_NUM * DENSE_SPLIT_N_BLOCKS) * 128;
        size_sparsity_ = win_num_ * bin_num_ * sizeof(float); // wins*bins
        // e.g. 16win*(128*10) = 20480 * sizeof(bucket_t) = 20480 * 128bytes

        // S1b: device dense-bucket worklist. `size_dense_worklist_` matches the
        // `d_dense_out` arena capacity (win_num * MAX_DENSE_BUCKET_NUM slices),
        // which is the maximum number of dense buckets per MSM; `d_dense_cnt`
        // is the single atomic compaction counter.
        size_dense_worklist_ = win_num_ * MAX_DENSE_BUCKET_NUM * sizeof(int);
        size_dense_cnt_ = sizeof(int);

        // memory
        size_scalars_ = (length_ + 1) * Scalar::ELT_BYTES;
        size_points_ = AFFINE_POINT_SIZE * length_;
        size_out_ = 3 * Scalar::ELT_BYTES * win_num_ * bin_num_; // point
        size_out2_ = 3 * Scalar::ELT_BYTES * ((1 << (win_bit_ - win_bit_half_)) + (1 << win_bit_half_)); // point
        size_wins_ = length_ * win_num_ * 2 * sizeof(int); // [win_idx][bin_idx]
        size_wins_start_ = bin_num_ * win_num_ * sizeof(int);
        size_wins_end_ = bin_num_ * win_num_ * sizeof(int);
        size_pos_ = bin_num_ * win_num_ * sizeof(int);
        size_count_ = bin_num_ * win_num_ * sizeof(int);
    }

    uint64_t get_required_memory_size()
    {
        uint64_t required_memory_size = 0;
        required_memory_size += size_dense_out_;
        required_memory_size += size_sparsity_;
        required_memory_size += size_scalars_;
        required_memory_size += size_points_;
        required_memory_size += size_out_;
        required_memory_size += size_out2_;
        required_memory_size += size_wins_;
        required_memory_size += size_wins_start_;
        required_memory_size += size_wins_end_;
        required_memory_size += size_pos_;
        required_memory_size += size_count_;
        required_memory_size += size_dense_worklist_;
        required_memory_size += size_dense_cnt_;
        return required_memory_size;
    }

    uint64_t get_max_supported_length_on_device(uint64_t free_bytes)
    {
        for (uint32_t log_n = MAX_MAX_LEN_LOG; log_n > 0; --log_n) {
            set_params(1 << log_n);
            if (get_required_memory_size() < free_bytes) {
                return (1 << log_n);
            }
        }
        return 0;
    }

    // win and bin param
    uint64_t length_ = 0;
    uint64_t win_bit_, win_bit_half_, win_num_, bin_num_ = 0;
    // sparsity
    const float SPARSITY_THRESHOLD = 0.10;
    uint32_t MAX_DENSE_BUCKET_NUM = 0;
    uint32_t DENSE_SPLIT_N_BLOCKS = 0;
    uint64_t size_dense_out_ = 0;
    // size
    uint64_t size_scalars_ = 0;
    uint64_t size_points_ = 0;
    uint64_t size_sparsity_ = 0;
    uint64_t size_out_ = 0;
    uint64_t size_out2_ = 0;
    uint64_t size_wins_ = 0;
    uint64_t size_wins_start_ = 0;
    uint64_t size_wins_end_ = 0;
    uint64_t size_pos_ = 0;
    uint64_t size_count_ = 0;
    uint64_t size_dense_worklist_ = 0;
    uint64_t size_dense_cnt_ = 0;
};

extern "C" uint64_t _halo2_msm_max_length(uint64_t free_bytes)
{
    uint64_t default_length = 1 << MAX_MAX_LEN_LOG;
    CudaMsmInfo msm_info(default_length);
    return msm_info.get_max_supported_length_on_device(free_bytes);
}

// Pure host preflight: bytes for `_halo2_multiexp` ScratchSpan.
extern "C" uint64_t _halo2_multiexp_workspace_size(uint64_t length)
{
    CudaMsmInfo msm_info(length);
    return align_up(msm_info.get_required_memory_size(), 32);
}

// `include_point_slot` / `include_scalar_slot` control whether the
// per-MSM scratch carves device buffers for points / scalars. Set
// `false` on paths whose bases or scalars are caller-owned device
// pointers; the corresponding `*d_point` / `*d_scalar` is left null.
RustError cuda_msm_init(
    /*params*/
    CudaMsmInfo& msm_info,
    bool include_point_slot,
    bool include_scalar_slot,
    /*gpu*/
    cudaStream_t stream_main,
    ScratchSpan& span,
    uint64_t** d_gpu_mem,
    /*pointers*/
    uint64_t** d_scalar,
    uint64_t** d_point,
    uint64_t** d_out,
    uint64_t** d_out2,
    uint64_t** d_dense_out,
    float** d_sparsity,
    int** d_wins,
    int** d_wins_start,
    int** d_wins_end,
    int** d_pos,
    int** d_count,
    int** d_dense_worklist,
    int** d_dense_cnt)
{
    (void)stream_main;
    uint64_t required_memory_size = msm_info.get_required_memory_size();
    if (!include_point_slot) {
        required_memory_size -= msm_info.size_points_;
    }
    if (!include_scalar_slot) {
        required_memory_size -= msm_info.size_scalars_;
    }
    *d_gpu_mem = (uint64_t*)span.take(required_memory_size);

    char* base = (char*)(*d_gpu_mem);
    uint64_t off = 0;
    if (include_scalar_slot) {
        *d_scalar = (uint64_t*)(base + off);
        off += msm_info.size_scalars_;
    } else {
        *d_scalar = nullptr;
    }
    if (include_point_slot) {
        *d_point = (uint64_t*)(base + off);
        off += msm_info.size_points_;
    } else {
        *d_point = nullptr;
    }
    *d_out = (uint64_t*)(base + off);
    off += msm_info.size_out_;
    *d_out2 = (uint64_t*)(base + off);
    off += msm_info.size_out2_;
    *d_dense_out = (uint64_t*)(base + off);
    off += msm_info.size_dense_out_;
    *d_sparsity = (float*)(base + off);
    off += msm_info.size_sparsity_;
    *d_wins = (int*)(base + off);
    off += msm_info.size_wins_;
    *d_wins_start = (int*)(base + off);
    off += msm_info.size_wins_start_;
    *d_wins_end = (int*)(base + off);
    off += msm_info.size_wins_end_;
    *d_pos = (int*)(base + off);
    off += msm_info.size_pos_;
    *d_count = (int*)(base + off);
    off += msm_info.size_count_;
    *d_dense_worklist = (int*)(base + off);
    off += msm_info.size_dense_worklist_;
    *d_dense_cnt = (int*)(base + off);
    // Catch slot-offset drift vs the Rust-side workspace-size calculation
    // before the kernel reads past its scratch (debug builds only).
    assert(off + msm_info.size_dense_cnt_ <= required_memory_size);

    return cudaSuccess;
}

extern "C" RustError _halo2_multiexp(
    const Scalar* scalar,
    const Point* point,
    const Output* out,
    uint64_t length,
    void* scratch,
    uint64_t scratch_bytes,
    cudaStream_t stream)
{
    ScratchSpan span { (uint8_t*)scratch, (size_t)scratch_bytes };

    // malloc device memory
    uint64_t* d_gpu_mem = nullptr;
    // set device ptr
    uint64_t* d_scalar = nullptr;
    uint64_t* d_point = nullptr;
    uint64_t* d_out = nullptr;
    uint64_t* d_out2 = nullptr;
    uint64_t* d_dense_out = nullptr;
    float* d_sparsity = nullptr;
    int* d_wins = nullptr;
    int* d_wins_start = nullptr;
    int* d_wins_end = nullptr;
    int* d_pos = nullptr;
    int* d_count = nullptr;
    int* d_dense_worklist = nullptr;
    int* d_dense_cnt = nullptr;

    CudaMsmInfo msm_info(length);
    RustError state_init = cuda_msm_init(
        msm_info, /*include_point_slot=*/true, /*include_scalar_slot=*/true,
        stream, span, &d_gpu_mem,
        &d_scalar, &d_point, &d_out, &d_out2,
        &d_dense_out, &d_sparsity,
        &d_wins, &d_wins_start, &d_wins_end,
        &d_pos, &d_count,
        &d_dense_worklist, &d_dense_cnt);
    if (state_init.code != cudaSuccess)
        return state_init;

    // preprocess scalar & copy points to gpu
    try {
        // scalar
        CUDA_OK(cudaMemcpyAsync(d_scalar, scalar->ptr, length * Scalar::ELT_BYTES, cudaMemcpyHostToDevice, stream));
        zkpcuda::pippenger::preprocess_scalars<SCALAR_BIT, 4>(
            stream, d_scalar, length,
            msm_info.win_bit_, msm_info.win_num_,
            d_pos, d_count,
            d_wins, d_wins_start, d_wins_end);
        // points
        CUDA_OK(cudaMemcpyAsync(d_point, point->ptr, AFFINE_POINT_SIZE * length, cudaMemcpyHostToDevice, stream));
    } catch (const cuda_error& error) {
        return RustError(error.code(), error.what());
    };

    // ec::mixed_add
    zkpcuda::pippenger::mixed_add_wins<32>(
        stream, (point_t*)d_out, (affine_t*)d_point, length,
        msm_info.SPARSITY_THRESHOLD, d_sparsity, d_dense_out, d_wins, d_wins_start, d_wins_end,
        msm_info.win_num_, msm_info.bin_num_,
        d_dense_worklist, d_dense_cnt, msm_info.MAX_DENSE_BUCKET_NUM);
    auto d_res = zkpcuda::pippenger::postprocess_buckets<32>(
        stream, (point_t*)d_out, (point_t*)d_out2,
        msm_info.win_bit_, msm_info.win_bit_half_, msm_info.win_num_);

    // get result
    try {
        uint64_t size_jac_point = 3 * Scalar::ELT_BYTES;
        CUDA_OK(cudaMemcpyAsync(out->ptr, d_res, size_jac_point, cudaMemcpyDeviceToHost, stream));
        CUDA_OK(cudaStreamSynchronize(stream));
    } catch (const cuda_error& error) {
        return RustError(error.code(), error.what());
    };

    return cudaSuccess;
}

// Workspace size for `_halo2_multiexp_device_bases`. Mirrors
// `_halo2_multiexp_workspace_size` but excludes the point slot: the
// caller supplies device-resident bases via `d_bases`, so no points
// need to be staged inside the per-call scratch.
extern "C" uint64_t _halo2_multiexp_device_bases_workspace_size(uint64_t length)
{
    CudaMsmInfo msm_info(length);
    return align_up(msm_info.get_required_memory_size() - msm_info.size_points_, 32);
}

// Device-bases variant of `_halo2_multiexp`.
//
// `d_bases` is a caller-owned device pointer to a contiguous run of
// affine points of length at least `length` (the caller typically
// caches an SRS device mirror across many calls). The scalars are
// host-resident and the output is a host Jacobian point.
// `cuda_msm_init(include_point_slot=false)` omits the base-point scratch
// slot, since the bases live in the caller's device buffer.
extern "C" RustError _halo2_multiexp_device_bases(
    const Scalar* h_scalar,
    const void* d_bases,
    const Output* h_out,
    uint64_t length,
    void* scratch,
    uint64_t scratch_bytes,
    cudaStream_t stream)
{
    ScratchSpan span { (uint8_t*)scratch, (size_t)scratch_bytes };

    uint64_t* d_gpu_mem = nullptr;
    uint64_t* d_scalar = nullptr;
    uint64_t* d_point_slot = nullptr; // unused — set to nullptr by cuda_msm_init
    uint64_t* d_out = nullptr;
    uint64_t* d_out2 = nullptr;
    uint64_t* d_dense_out = nullptr;
    float* d_sparsity = nullptr;
    int* d_wins = nullptr;
    int* d_wins_start = nullptr;
    int* d_wins_end = nullptr;
    int* d_pos = nullptr;
    int* d_count = nullptr;
    int* d_dense_worklist = nullptr;
    int* d_dense_cnt = nullptr;

    CudaMsmInfo msm_info(length);
    RustError state_init = cuda_msm_init(
        msm_info, /*include_point_slot=*/false, /*include_scalar_slot=*/true,
        stream, span, &d_gpu_mem,
        &d_scalar, &d_point_slot, &d_out, &d_out2,
        &d_dense_out, &d_sparsity,
        &d_wins, &d_wins_start, &d_wins_end,
        &d_pos, &d_count,
        &d_dense_worklist, &d_dense_cnt);
    if (state_init.code != cudaSuccess)
        return state_init;

    // Kernel reads bases directly from the caller's device pointer.
    uint64_t* d_point = (uint64_t*)d_bases;

    try {
        CUDA_OK(cudaMemcpyAsync(d_scalar, h_scalar->ptr, length * Scalar::ELT_BYTES, cudaMemcpyHostToDevice, stream));
        zkpcuda::pippenger::preprocess_scalars<SCALAR_BIT, 4>(
            stream, d_scalar, length,
            msm_info.win_bit_, msm_info.win_num_,
            d_pos, d_count,
            d_wins, d_wins_start, d_wins_end);
    } catch (const cuda_error& error) {
        return RustError(error.code(), error.what());
    };

    zkpcuda::pippenger::mixed_add_wins<32>(
        stream, (point_t*)d_out, (affine_t*)d_point, length,
        msm_info.SPARSITY_THRESHOLD, d_sparsity, d_dense_out, d_wins, d_wins_start, d_wins_end,
        msm_info.win_num_, msm_info.bin_num_,
        d_dense_worklist, d_dense_cnt, msm_info.MAX_DENSE_BUCKET_NUM);
    auto d_res = zkpcuda::pippenger::postprocess_buckets<32>(
        stream, (point_t*)d_out, (point_t*)d_out2,
        msm_info.win_bit_, msm_info.win_bit_half_, msm_info.win_num_);

    try {
        uint64_t size_jac_point = 3 * Scalar::ELT_BYTES;
        CUDA_OK(cudaMemcpyAsync(h_out->ptr, d_res, size_jac_point, cudaMemcpyDeviceToHost, stream));
        CUDA_OK(cudaStreamSynchronize(stream));
    } catch (const cuda_error& error) {
        return RustError(error.code(), error.what());
    };

    return cudaSuccess;
}

// Workspace size for `_halo2_multiexp_device_scalars_device_bases`.
// Bases are caller-owned device pointers, but the scalar scratch slot
// is preserved: `preprocess_scalars` calls `from_monty` which mutates
// its input in place, so the caller's scalars are D→D copied into the
// scratch slot before running the kernel chain.
extern "C" uint64_t _halo2_multiexp_device_scalars_device_bases_workspace_size(uint64_t length)
{
    CudaMsmInfo msm_info(length);
    return align_up(msm_info.get_required_memory_size() - msm_info.size_points_, 32);
}

// Device-scalars + device-bases variant of `_halo2_multiexp`.
//
// `d_scalar_caller` and `d_bases` are caller-owned device pointers.
// The point slot is dropped via `cuda_msm_init(include_point_slot=false)`
// (bases live in the caller's device buffer). The scalar scratch slot
// is retained: `preprocess_scalars` invokes `cukernel_from_monty` which
// MUTATES the input scalars in place (Montgomery → canonical), so the
// caller's buffer is D→D copied into the scratch slot before the kernel
// chain runs, leaving the caller's buffer intact.
extern "C" RustError _halo2_multiexp_device_scalars_device_bases(
    const void* d_scalar_caller,
    const void* d_bases,
    const Output* h_out,
    uint64_t length,
    void* scratch,
    uint64_t scratch_bytes,
    cudaStream_t stream)
{
    ScratchSpan span { (uint8_t*)scratch, (size_t)scratch_bytes };

    uint64_t* d_gpu_mem = nullptr;
    uint64_t* d_scalar = nullptr;
    uint64_t* d_point_slot = nullptr; // unused — set to nullptr by cuda_msm_init
    uint64_t* d_out = nullptr;
    uint64_t* d_out2 = nullptr;
    uint64_t* d_dense_out = nullptr;
    float* d_sparsity = nullptr;
    int* d_wins = nullptr;
    int* d_wins_start = nullptr;
    int* d_wins_end = nullptr;
    int* d_pos = nullptr;
    int* d_count = nullptr;
    int* d_dense_worklist = nullptr;
    int* d_dense_cnt = nullptr;

    CudaMsmInfo msm_info(length);
    RustError state_init = cuda_msm_init(
        msm_info, /*include_point_slot=*/false, /*include_scalar_slot=*/true,
        stream, span, &d_gpu_mem,
        &d_scalar, &d_point_slot, &d_out, &d_out2,
        &d_dense_out, &d_sparsity,
        &d_wins, &d_wins_start, &d_wins_end,
        &d_pos, &d_count,
        &d_dense_worklist, &d_dense_cnt);
    if (state_init.code != cudaSuccess)
        return state_init;

    // Kernel reads bases directly from the caller's device pointer.
    uint64_t* d_point = (uint64_t*)d_bases;

    try {
        // D→D copy caller's scalars into the scratch slot — the
        // `preprocess_scalars` chain mutates this buffer in place
        // (`from_monty`), so the caller's buffer must NOT be touched.
        CUDA_OK(cudaMemcpyAsync(d_scalar, d_scalar_caller, length * Scalar::ELT_BYTES, cudaMemcpyDeviceToDevice, stream));
        zkpcuda::pippenger::preprocess_scalars<SCALAR_BIT, 4>(
            stream, d_scalar, length,
            msm_info.win_bit_, msm_info.win_num_,
            d_pos, d_count,
            d_wins, d_wins_start, d_wins_end);
        // Surface kernel-launch errors before the host-read sync below.
        CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& error) {
        return RustError(error.code(), error.what());
    };

    zkpcuda::pippenger::mixed_add_wins<32>(
        stream, (point_t*)d_out, (affine_t*)d_point, length,
        msm_info.SPARSITY_THRESHOLD, d_sparsity, d_dense_out, d_wins, d_wins_start, d_wins_end,
        msm_info.win_num_, msm_info.bin_num_,
        d_dense_worklist, d_dense_cnt, msm_info.MAX_DENSE_BUCKET_NUM);
    auto d_res = zkpcuda::pippenger::postprocess_buckets<32>(
        stream, (point_t*)d_out, (point_t*)d_out2,
        msm_info.win_bit_, msm_info.win_bit_half_, msm_info.win_num_);

    try {
        // Surface launch errors before the host-bound copy + sync that lets
        // the Rust caller read `h_out->ptr` after return.
        CUDA_OK(cudaGetLastError());
        uint64_t size_jac_point = 3 * Scalar::ELT_BYTES;
        CUDA_OK(cudaMemcpyAsync(h_out->ptr, d_res, size_jac_point, cudaMemcpyDeviceToHost, stream));
        CUDA_OK(cudaStreamSynchronize(stream));
    } catch (const cuda_error& error) {
        return RustError(error.code(), error.what());
    };

    return cudaSuccess;
}
