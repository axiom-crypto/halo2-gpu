#include <cstdint>

#include <cub/device/device_merge_sort.cuh>
#include <cub/device/device_select.cuh>

#include "common/exception.h"
#include "common/halo2_ffi.h"
#include "common/scratch_span.h"

#include "kernel/common.h"

using Scalar = utils::FFITraitObject;

namespace {

// 256-bit BN254 Fr canonical-form key. Limbs are stored in little-endian
// word order: value = limbs[0] | limbs[1] << 64 | limbs[2] << 128 |
// limbs[3] << 192. The lookup permutation sort order must agree with the
// host `Vec<Fr>::par_sort`, whose `Ord` impl compares the canonical
// 32-byte little-endian representation iterating bytes in reverse —
// equivalent to a big-endian compare on the 256-bit big integer, i.e.
// `limbs[3]` first, then `limbs[2]`, `limbs[1]`, `limbs[0]`.
//
// Why a dedicated POD rather than reusing `scalar_t` (= `fr_t`):
//   1. CUB::DeviceMergeSort requires a trivially-copyable POD key type;
//      `scalar_t` is a templated `blst_256_t` class with a private `val`
//      and Montgomery arithmetic methods — heavier than needed and not
//      guaranteed trivially-copyable across all CUB versions.
//   2. Sort keys live in canonical (not Montgomery) form. `scalar_t`
//      advertises Montgomery semantics; storing canonical bytes in a
//      `scalar_t` would create a type-system trap where any subsequent
//      `*` would silently produce nonsense.
//   3. Decouples the sort key's 32-byte canonical layout from the
//      host-blst field representation (see
//      `cuda/include/field/alt_bn128.hpp`), keeping the CUB sort
//      comparator independent of that field path.
struct Key256 {
    uint64_t limbs[4];
};

struct Key256LessBE {
    __host__ __device__ __forceinline__ bool operator()(
        const Key256& a, const Key256& b) const
    {
        if (a.limbs[3] != b.limbs[3]) return a.limbs[3] < b.limbs[3];
        if (a.limbs[2] != b.limbs[2]) return a.limbs[2] < b.limbs[2];
        if (a.limbs[1] != b.limbs[1]) return a.limbs[1] < b.limbs[1];
        return a.limbs[0] < b.limbs[0];
    }
};

__device__ __forceinline__ bool key_eq(const Key256& a, const Key256& b)
{
    return a.limbs[0] == b.limbs[0]
        && a.limbs[1] == b.limbs[1]
        && a.limbs[2] == b.limbs[2]
        && a.limbs[3] == b.limbs[3];
}

__device__ __forceinline__ int key_cmp_be(const Key256& a, const Key256& b)
{
    if (a.limbs[3] != b.limbs[3]) return a.limbs[3] < b.limbs[3] ? -1 : 1;
    if (a.limbs[2] != b.limbs[2]) return a.limbs[2] < b.limbs[2] ? -1 : 1;
    if (a.limbs[1] != b.limbs[1]) return a.limbs[1] < b.limbs[1] ? -1 : 1;
    if (a.limbs[0] != b.limbs[0]) return a.limbs[0] < b.limbs[0] ? -1 : 1;
    return 0;
}

__device__ bool key_in_sorted_array(
    const Key256* sorted, uint64_t n, const Key256& target)
{
    int64_t lo = 0;
    int64_t hi = (int64_t)n - 1;
    while (lo <= hi) {
        int64_t mid = (lo + hi) >> 1;
        int c = key_cmp_be(sorted[mid], target);
        if (c == 0) return true;
        if (c < 0) lo = mid + 1;
        else hi = mid - 1;
    }
    return false;
}

__global__ void build_canonical_keys_kernel(
    Key256* d_keys,
    uint32_t* d_indices,
    const scalar_t* d_values,
    uint64_t length)
{
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= length) return;
    scalar_t v = d_values[tid];
    v.from(); // Montgomery -> canonical
    const uint64_t* src = reinterpret_cast<const uint64_t*>(&v);
    Key256 k;
    k.limbs[0] = src[0];
    k.limbs[1] = src[1];
    k.limbs[2] = src[2];
    k.limbs[3] = src[3];
    d_keys[tid] = k;
    d_indices[tid] = (uint32_t)tid;
}

__global__ void gather_mont_kernel(
    scalar_t* d_out,
    const scalar_t* d_src,
    const uint32_t* d_indices,
    uint64_t length)
{
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= length) return;
    d_out[tid] = d_src[d_indices[tid]];
}

__global__ void head_flags_kernel(
    uint32_t* d_is_head,
    uint32_t* d_not_head,
    const Key256* d_sorted_keys,
    uint64_t length)
{
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= length) return;
    bool head = (tid == 0) || !key_eq(d_sorted_keys[tid], d_sorted_keys[tid - 1]);
    d_is_head[tid] = head ? 1u : 0u;
    d_not_head[tid] = head ? 0u : 1u;
}

__global__ void leftover_flag_kernel(
    uint32_t* d_leftover_flag,
    const Key256* d_sorted_table_keys,
    const Key256* d_sorted_input_keys,
    uint64_t input_length,
    uint64_t table_length)
{
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= table_length) return;
    Key256 v = d_sorted_table_keys[tid];
    bool is_first_in_table
        = (tid == 0) || !key_eq(v, d_sorted_table_keys[tid - 1]);
    bool is_in_input
        = is_first_in_table
        && key_in_sorted_array(d_sorted_input_keys, input_length, v);
    d_leftover_flag[tid] = (is_first_in_table && is_in_input) ? 0u : 1u;
}

__global__ void iota_u32_kernel(uint32_t* d_iota, uint64_t length)
{
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= length) return;
    d_iota[tid] = (uint32_t)tid;
}

__global__ void reverse_scatter_kernel(
    scalar_t* d_permuted_table,
    const uint32_t* d_repeated_rows,
    const scalar_t* d_leftover_asc,
    const uint32_t* d_m)
{
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t m = *d_m;
    if (tid >= (uint64_t)m) return;
    uint32_t row = d_repeated_rows[m - 1 - (uint32_t)tid];
    d_permuted_table[row] = d_leftover_asc[tid];
}

__host__ __device__ inline uint64_t kernel_blocks(uint64_t n, uint64_t tpb)
{
    return (n + tpb - 1) / tpb;
}

// Query CUB scratch requirements for the three primitive calls used by
// `_halo2_permute_expression_pair`. Returns the max across primitives so
// a single arena suffices for all stages.
inline uint64_t cub_max_temp_bytes(uint64_t U)
{
    size_t sort_bytes = 0;
    cub::DeviceMergeSort::SortPairs(
        nullptr, sort_bytes,
        (Key256*)nullptr, (uint32_t*)nullptr,
        (int64_t)U, Key256LessBE{});

    size_t select_u32_bytes = 0;
    cub::DeviceSelect::Flagged(
        nullptr, select_u32_bytes,
        (uint32_t*)nullptr, (uint32_t*)nullptr, (uint32_t*)nullptr,
        (uint32_t*)nullptr, (int)U);

    size_t select_fr_bytes = 0;
    cub::DeviceSelect::Flagged(
        nullptr, select_fr_bytes,
        (Key256*)nullptr, (uint32_t*)nullptr, (Key256*)nullptr,
        (uint32_t*)nullptr, (int)U);

    uint64_t m = (uint64_t)sort_bytes;
    if ((uint64_t)select_u32_bytes > m) m = (uint64_t)select_u32_bytes;
    if ((uint64_t)select_fr_bytes > m) m = (uint64_t)select_fr_bytes;
    return m;
}

} // namespace

extern "C" uint64_t _halo2_permute_expression_pair_workspace_size(
    uint64_t n,
    uint64_t usable_rows)
{
    (void)n;
    uint64_t U = usable_rows;
    if (U == 0) return 0;

    uint64_t total = 0;
    total += align_up(U * (uint64_t)sizeof(Key256), 32);     // d_in_keys
    total += align_up(U * (uint64_t)sizeof(uint32_t), 32);   // d_in_indices
    total += align_up(U * (uint64_t)sizeof(Key256), 32);     // d_tab_keys
    total += align_up(U * (uint64_t)sizeof(uint32_t), 32);   // d_tab_indices
    total += align_up(U * (uint64_t)Scalar::ELT_BYTES, 32);  // d_sorted_in_mont
    total += align_up(U * (uint64_t)Scalar::ELT_BYTES, 32);  // d_sorted_tab_mont
    total += align_up(U * (uint64_t)sizeof(uint32_t), 32);   // d_is_head
    total += align_up(U * (uint64_t)sizeof(uint32_t), 32);   // d_not_head
    total += align_up(U * (uint64_t)sizeof(uint32_t), 32);   // d_leftover_flag
    total += align_up(U * (uint64_t)Scalar::ELT_BYTES, 32);  // d_leftover_asc
    total += align_up(U * (uint64_t)sizeof(uint32_t), 32);   // d_iota
    total += align_up(U * (uint64_t)sizeof(uint32_t), 32);   // d_repeated_rows
    total += align_up((uint64_t)sizeof(uint32_t), 32);       // d_leftover_count
    total += align_up((uint64_t)sizeof(uint32_t), 32);       // d_repeated_count
    total += align_up(cub_max_temp_bytes(U), 32);            // d_cub_temp
    return total;
}

extern "C" RustError _halo2_permute_expression_pair(
    const void* d_compressed_input,
    const void* d_compressed_table,
    void* d_permuted_input,
    void* d_permuted_table,
    uint64_t n,
    uint64_t usable_rows,
    void* scratch,
    uint64_t scratch_bytes,
    cudaStream_t stream)
{
    if (n == 0) return cudaSuccess;
    uint64_t U = usable_rows;

    try {
        CUDA_OK(cudaMemsetAsync(
            d_permuted_input, 0,
            n * (uint64_t)Scalar::ELT_BYTES, stream));
        CUDA_OK(cudaMemsetAsync(
            d_permuted_table, 0,
            n * (uint64_t)Scalar::ELT_BYTES, stream));
        if (U == 0) return cudaSuccess;

        ScratchSpan span { (uint8_t*)scratch, (size_t)scratch_bytes };
        Key256* d_in_keys
            = (Key256*)span.take(U * sizeof(Key256));
        uint32_t* d_in_indices
            = (uint32_t*)span.take(U * sizeof(uint32_t));
        Key256* d_tab_keys
            = (Key256*)span.take(U * sizeof(Key256));
        uint32_t* d_tab_indices
            = (uint32_t*)span.take(U * sizeof(uint32_t));
        scalar_t* d_sorted_in
            = (scalar_t*)span.take(U * (uint64_t)Scalar::ELT_BYTES);
        scalar_t* d_sorted_tab
            = (scalar_t*)span.take(U * (uint64_t)Scalar::ELT_BYTES);
        uint32_t* d_is_head
            = (uint32_t*)span.take(U * sizeof(uint32_t));
        uint32_t* d_not_head
            = (uint32_t*)span.take(U * sizeof(uint32_t));
        uint32_t* d_leftover_flag
            = (uint32_t*)span.take(U * sizeof(uint32_t));
        scalar_t* d_leftover_asc
            = (scalar_t*)span.take(U * (uint64_t)Scalar::ELT_BYTES);
        uint32_t* d_iota
            = (uint32_t*)span.take(U * sizeof(uint32_t));
        uint32_t* d_repeated_rows
            = (uint32_t*)span.take(U * sizeof(uint32_t));
        uint32_t* d_leftover_count
            = (uint32_t*)span.take(sizeof(uint32_t));
        uint32_t* d_repeated_count
            = (uint32_t*)span.take(sizeof(uint32_t));

        uint64_t cub_max = cub_max_temp_bytes(U);
        void* d_cub_temp = span.take((size_t)cub_max);

        const uint32_t TPB = 128;
        uint64_t blocks_U = kernel_blocks(U, TPB);

        build_canonical_keys_kernel<<<blocks_U, TPB, 0, stream>>>(
            d_in_keys, d_in_indices,
            (const scalar_t*)d_compressed_input, U);
        build_canonical_keys_kernel<<<blocks_U, TPB, 0, stream>>>(
            d_tab_keys, d_tab_indices,
            (const scalar_t*)d_compressed_table, U);

        size_t tmp = (size_t)cub_max;
        CUDA_OK(cub::DeviceMergeSort::SortPairs(
            d_cub_temp, tmp,
            d_in_keys, d_in_indices,
            (int64_t)U, Key256LessBE{}, stream));
        tmp = (size_t)cub_max;
        CUDA_OK(cub::DeviceMergeSort::SortPairs(
            d_cub_temp, tmp,
            d_tab_keys, d_tab_indices,
            (int64_t)U, Key256LessBE{}, stream));

        gather_mont_kernel<<<blocks_U, TPB, 0, stream>>>(
            d_sorted_in, (const scalar_t*)d_compressed_input,
            d_in_indices, U);
        gather_mont_kernel<<<blocks_U, TPB, 0, stream>>>(
            d_sorted_tab, (const scalar_t*)d_compressed_table,
            d_tab_indices, U);

        head_flags_kernel<<<blocks_U, TPB, 0, stream>>>(
            d_is_head, d_not_head, d_in_keys, U);

        leftover_flag_kernel<<<blocks_U, TPB, 0, stream>>>(
            d_leftover_flag, d_tab_keys, d_in_keys, U, U);

        tmp = (size_t)cub_max;
        CUDA_OK(cub::DeviceSelect::Flagged(
            d_cub_temp, tmp,
            (Key256*)d_sorted_tab, d_leftover_flag,
            (Key256*)d_leftover_asc, d_leftover_count,
            (int)U, stream));

        iota_u32_kernel<<<blocks_U, TPB, 0, stream>>>(d_iota, U);
        tmp = (size_t)cub_max;
        CUDA_OK(cub::DeviceSelect::Flagged(
            d_cub_temp, tmp,
            d_iota, d_not_head, d_repeated_rows, d_repeated_count,
            (int)U, stream));

        CUDA_OK(cudaMemcpyAsync(
            d_permuted_input, d_sorted_in,
            U * (uint64_t)Scalar::ELT_BYTES,
            cudaMemcpyDeviceToDevice, stream));
        CUDA_OK(cudaMemcpyAsync(
            d_permuted_table, d_sorted_in,
            U * (uint64_t)Scalar::ELT_BYTES,
            cudaMemcpyDeviceToDevice, stream));

        reverse_scatter_kernel<<<blocks_U, TPB, 0, stream>>>(
            (scalar_t*)d_permuted_table,
            d_repeated_rows, d_leftover_asc, d_repeated_count);
    } catch (const cuda_error& error) {
        return RustError(error.code(), error.what());
    }

    return cudaSuccess;
}
