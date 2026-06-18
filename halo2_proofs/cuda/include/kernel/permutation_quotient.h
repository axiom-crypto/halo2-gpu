#pragma once

#include "field/alt_bn128.hpp"
#include "kernel/quotient.h"
#include <cuda.h>
#include <cuda_runtime.h>

// Permutation-argument quotient kernel for halo2. Mirrors the CPU loop in
// `halo2_proofs/src/cpu/evaluator.rs::permutation_quotient_cpu_chunk`
// — see that function for the algebraic derivation. The kernel must stay
// byte-for-byte algebraically equivalent: `gas_cost` is the correctness
// gate.
//
// Layout:
//   - `d_perm_prod_cosets_arr`, `d_perm_cosets_arr`, and
//     `d_column_values_arr` are *device-resident arrays of device
//     pointers* (one entry per polynomial). Pointers reference per-part
//     device buffers (extended-coset evaluations of advice / fixed /
//     instance / permutation cosets) produced upstream by
//     `coeff_to_extended_part_many_device`. This matches the
//     stark-backend `MainMatrixPtrs` pattern.
//
// The kernel is `__global__ static` to keep it translation-unit-local;
// `blockDim.x` is read at runtime instead of templating on a tile size.

namespace zkpcuda {
namespace quotient {

    // Bit-by-bit fast exponentiation. `idx` is at most the row count
    // (≤ 2^32 in practice), so the loop runs at most ~32 iterations.
    __device__ inline scalar_t omega_pow_idx(scalar_t omega, uint64_t idx)
    {
        scalar_t result = scalar_t::one();
        scalar_t b = omega;
        while (idx > 0) {
            if (idx & 1ull) {
                result = result * b;
            }
            b = b * b;
            idx >>= 1;
        }
        return result;
    }

    // Per-row constraints (matches the CPU reference order):
    //   1. value = value*y + (1 - z_0[idx]) * l0[idx]                      (first set)
    //   2. value = value*y + (z_l[idx]^2 - z_l[idx]) * l_last[idx]         (last set)
    //   3. for set_idx in 1..n_sets:
    //        value = value*y + (z_set[idx] - z_{set-1}[r_last]) * l0[idx]
    //   4. for set_idx in 0..n_sets:
    //        left  = z_set[r_next] * Π_j (col[set,j][idx] + beta * perm[set,j][idx] + gamma)
    //        right = z_set[idx]    * Π_j (col[set,j][idx] + current_delta_j      + gamma)
    //        value = value*y + (left - right) * l_active[idx]
    //        (current_delta_j advances by *= DELTA in inner loop, seeded
    //         once per row by delta_start * (current_extended_omega * omega^idx))
    __global__ static void cuda_kernel_permutation_quotient(
        scalar_t* __restrict__ d_values,                           // [length] in/out
        const scalar_t* const* __restrict__ d_perm_prod_cosets_arr, // [n_sets] device pointers
        const scalar_t* const* __restrict__ d_perm_cosets_arr,      // [n_perm_cols] device pointers
        const scalar_t* const* __restrict__ d_column_values_arr,    // [n_perm_cols] device pointers
        const scalar_t* __restrict__ d_l0,                          // [length]
        const scalar_t* __restrict__ d_l_last,                      // [length]
        const scalar_t* __restrict__ d_l_active_row,                // [length]
        scalar_t beta,
        scalar_t gamma,
        scalar_t y,
        scalar_t delta,
        scalar_t delta_start,
        scalar_t current_extended_omega,
        scalar_t omega,
        uint64_t n_sets,
        uint64_t chunk_len,
        uint64_t n_perm_cols,
        int32_t last_rotation,
        int32_t rot_scale,
        int32_t isize_,
        uint64_t length)
    {
        const uint64_t offset = (uint64_t)blockDim.x * blockIdx.x + threadIdx.x;
        const uint64_t stride = (uint64_t)blockDim.x * gridDim.x;
        const scalar_t one = scalar_t::one();
        const scalar_t omega_step = omega_pow_idx(omega, stride);
        scalar_t beta_term = current_extended_omega * omega_pow_idx(omega, offset);

        for (uint64_t row = offset; row < length; row += stride) {
            const int32_t r_next = lookups_rotation((int32_t)row, 1, rot_scale, isize_);
            const int32_t r_last = lookups_rotation((int32_t)row, last_rotation, rot_scale, isize_);

            scalar_t value = d_values[row];
            const scalar_t l0 = d_l0[row];
            const scalar_t l_last = d_l_last[row];
            const scalar_t l_active = d_l_active_row[row];

            // 1. First set: value = value*y + (1 - z_0[idx]) * l0
            const scalar_t first_z = d_perm_prod_cosets_arr[0][row];
            value = value * y + ((one - first_z) * l0);

            // 2. Last set: value = value*y + (z_l^2 - z_l) * l_last
            const scalar_t last_z = d_perm_prod_cosets_arr[n_sets - 1][row];
            value = value * y + ((last_z * last_z - last_z) * l_last);

            // 3. Inter-set: l_0 * (z_i - z_{i-1}[r_last])
            for (uint64_t set_idx = 1; set_idx < n_sets; ++set_idx) {
                const scalar_t z_curr = d_perm_prod_cosets_arr[set_idx][row];
                const scalar_t z_prev_rlast = d_perm_prod_cosets_arr[set_idx - 1][(uint64_t)r_last];
                value = value * y + ((z_curr - z_prev_rlast) * l0);
            }

            // 4. Per-set grand product over the chunk_len columns assigned to this set.
            //    `current_delta` advances per inner column iteration (NOT reset per set
            //    — matches the CPU reference exactly).
            scalar_t current_delta = delta_start * beta_term;

            for (uint64_t set_idx = 0; set_idx < n_sets; ++set_idx) {
                const scalar_t* z_ptr = d_perm_prod_cosets_arr[set_idx];
                const scalar_t z = z_ptr[row];
                const scalar_t z_rnext = z_ptr[(uint64_t)r_next];

                scalar_t left = z_rnext;
                scalar_t right = z;

                // The last set may have fewer than `chunk_len` columns
                // (matches CPU `column_values.chunks(chunk_len)` semantics
                // when `n_perm_cols % chunk_len != 0`).
                const uint64_t cols_start = set_idx * chunk_len;
                const uint64_t cols_in_this_set =
                    (cols_start + chunk_len <= n_perm_cols) ? chunk_len : (n_perm_cols - cols_start);

                // Single pass over the columns in this set: reuse `col_val`
                // for both the left and right factors and advance
                // `current_delta` in lockstep.
                for (uint64_t j = 0; j < cols_in_this_set; ++j) {
                    const uint64_t col = cols_start + j;
                    const scalar_t col_val = d_column_values_arr[col][row];
                    const scalar_t shifted_col = col_val + gamma;
                    const scalar_t perm_val = d_perm_cosets_arr[col][row];
                    left = left * (shifted_col + beta * perm_val);
                    right = right * (shifted_col + current_delta);
                    current_delta = current_delta * delta;
                }

                value = value * y + ((left - right) * l_active);
            }

            d_values[row] = value;
            beta_term = beta_term * omega_step;
        }
    }

} // namespace quotient
} // namespace zkpcuda
