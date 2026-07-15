//! Byte-equivalence coverage for the device dense-bucket accumulation path.
//!
//! Every case compares the device
//! `multiexp_gpu_device_scalars_device_bases_chunked` path against the
//! `best_multiexp_cpu` oracle and asserts **equal final affine points**
//! (`gpu.to_affine() == cpu.to_affine()`), for both the unchunked
//! (`chunk_size == n`) and chunked (`chunk_size == n/4 + 1`, short tail)
//! dispatch. Dense-bucket accumulation only batches independent bucket sums,
//! so the MSM result must stay byte-identical to a plain summation.
//!
//! Run single-threaded (shared GPU ctx + global mem-manager mutex):
//!   CUDA_VISIBLE_DEVICES=7 cargo test \
//!     --test multiexp_dense_bucket_edge_equivalence -- --test-threads=1
//!
//! Coverage rationale for the byte-identity-risk edges:
//! - interspersed **zero scalars** exercise the `if (win != 0)` scatter skip;
//! - **identity/point-at-infinity bases** exercise the `sum.inf()` bucket seed
//!   and `affine_t::is_inf()` in `xyzz_t::add`;
//! - **all-zero scalars** drive the whole MSM to identity, exercising
//!   `to_affine` of an identity accumulator;
//! - a **single nonzero coeff** amid zeros produces **single-element buckets**
//!   (one per nonzero window digit) and leaves nearly every other bucket
//!   **empty** (`begin == end`);
//! - a large equal-valued coeff **prefix** forces **dense buckets**
//!   (see `dense_constant` doc), exercising the dense-accumulation kernels.

use ff::{Field, PrimeField};
use group::{Curve, Group, GroupEncoding};
use halo2_axiom_gpu::cpu::arithmetic::best_multiexp_cpu;
use halo2_axiom_gpu::cuda::funcs::multiexp_gpu_device_scalars_device_bases_chunked;
use halo2_axiom_gpu::cuda::utils::HALO2_GPU_CTX;
use halo2curves::bn256::{Fr, G1Affine, G1};
use openvm_cuda_common::copy::MemCopyH2D;
use rand_core::OsRng;

/// A fixed nonzero field element with nonzero limbs across the whole 254-bit
/// width, so its `win_bit`-wide Pippenger windows are nonzero in (essentially)
/// every window.
///
/// Dense-bucket guarantee: if the first `m` coeffs all equal `K` (nonzero),
/// then for every window `W` whose digit `d = digit_W(K)` is nonzero, all `m`
/// of them land in the SAME bucket `(W, d)`, so
/// `sparsity(W,d) = num_element / N >= m / N`. Choosing `m >= 0.25 * N` makes
/// `sparsity >= 0.25 > SPARSITY_THRESHOLD (0.10)`, so at least one bucket is
/// dense and the dense-accumulation kernels are traversed. Because `K != 0`,
/// at least one window has a nonzero digit, so at least one dense bucket
/// always exists regardless of the exact digit decomposition.
fn dense_constant() -> Fr {
    Fr::from_raw([
        0x9e37_79b9_7f4a_7c15,
        0xbf58_476d_1ce4_e5b9,
        0x94d0_49bb_1331_11eb,
        0x022b_5f6a_3c1d_0e8f,
    ])
}

fn rand_bases(n: usize) -> Vec<G1Affine> {
    (0..n)
        .map(|_| (G1::generator() * Fr::random(OsRng)).to_affine())
        .collect()
}

/// Byte-identity assertion: compare the SERIALIZED affine encoding
/// (`GroupEncoding::to_bytes`, the canonical compressed form used by
/// proof/transcript point emission), which is strictly stronger than
/// `G1Affine == G1Affine` for the byte-identity gate.
fn assert_affine_bytes_eq(gpu: &G1Affine, cpu: &G1Affine, label: &str) {
    assert_eq!(
        gpu.to_bytes().as_ref(),
        cpu.to_bytes().as_ref(),
        "serialized affine bytes disagree with best_multiexp_cpu [{label}]"
    );
}

/// Compare the device MSM against `best_multiexp_cpu` for BOTH the unchunked
/// and chunked dispatch. CPU oracle is computed once.
fn assert_chunked_and_unchunked(coeffs: &[Fr], bases: &[G1Affine], label: &str) {
    assert_eq!(coeffs.len(), bases.len());
    let n = coeffs.len();

    let cpu = best_multiexp_cpu::<G1Affine>(coeffs, bases).to_affine();

    let d_coeffs = coeffs.to_device_on(&HALO2_GPU_CTX).expect("H2D coeffs");
    let d_bases = bases.to_device_on(&HALO2_GPU_CTX).expect("H2D bases");

    // Unchunked: one Pippenger chain over the whole input.
    let unchunked =
        multiexp_gpu_device_scalars_device_bases_chunked::<G1Affine>(&d_coeffs, &d_bases, n)
            .expect("unchunked device MSM")
            .to_affine();
    assert_affine_bytes_eq(&unchunked, &cpu, &format!("{label} / unchunked"));

    // Chunked: >= 2 chunks with a short tail, exercising the host fold.
    let chunk_size = n / 4 + 1;
    let chunked = multiexp_gpu_device_scalars_device_bases_chunked::<G1Affine>(
        &d_coeffs, &d_bases, chunk_size,
    )
    .expect("chunked device MSM")
    .to_affine();
    assert_affine_bytes_eq(&chunked, &cpu, &format!("{label} / chunked"));
}

#[test]
fn random_equiv_log_n_14() {
    let n = 1usize << 14;
    let coeffs: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();
    let bases = rand_bases(n);
    assert_chunked_and_unchunked(&coeffs, &bases, "random log_n=14");
}

#[test]
fn random_equiv_log_n_20() {
    let n = 1usize << 20;
    let coeffs: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();
    let bases = rand_bases(n);
    assert_chunked_and_unchunked(&coeffs, &bases, "random log_n=20");
}

/// Dense-bucket coverage at production-like size. First quarter of the coeffs
/// share the fixed constant `K` (see `dense_constant`), guaranteeing dense
/// buckets (sparsity >= 0.25). In the chunked dispatch the first chunk
/// (`n/4 + 1` elements, all but one equal to `K`) is itself dense, so the
/// dense path is traversed in both dispatches.
#[test]
fn dense_bucket_equiv_log_n_20() {
    let n = 1usize << 20;
    let k = dense_constant();
    let m = n / 4; // 25% share of the scalars => sparsity >= 0.25 > 0.10
    let mut coeffs: Vec<Fr> = Vec::with_capacity(n);
    coeffs.extend(std::iter::repeat_n(k, m));
    coeffs.extend((m..n).map(|_| Fr::random(OsRng)));
    let bases = rand_bases(n);
    assert_chunked_and_unchunked(&coeffs, &bases, "dense log_n=20");
}

/// Dense-bucket coverage at the GPU threshold size (cheap).
#[test]
fn dense_bucket_equiv_log_n_14() {
    let n = 1usize << 14;
    let k = dense_constant();
    let m = n / 4;
    let mut coeffs: Vec<Fr> = Vec::with_capacity(n);
    coeffs.extend(std::iter::repeat_n(k, m));
    coeffs.extend((m..n).map(|_| Fr::random(OsRng)));
    let bases = rand_bases(n);
    assert_chunked_and_unchunked(&coeffs, &bases, "dense log_n=14");
}

/// Interspersed zero scalars (every 3rd) + identity/point-at-infinity bases
/// (every 5th). Exercises the `win != 0` scatter skip and the `is_inf` affine
/// add path together, mixed with dense-forcing constants so the dense kernels
/// also see the infinity handling.
#[test]
fn zero_scalars_and_identity_bases_log_n_14() {
    let n = 1usize << 14;
    let k = dense_constant();
    let id = G1::identity().to_affine();

    let coeffs: Vec<Fr> = (0..n)
        .map(|i| {
            if i % 3 == 0 {
                Fr::ZERO // interspersed zero scalar
            } else if i % 7 == 0 {
                k // repeated constant -> dense bucket
            } else {
                Fr::random(OsRng)
            }
        })
        .collect();
    let bases: Vec<G1Affine> = (0..n)
        .map(|i| {
            if i % 5 == 0 {
                id // point at infinity base
            } else {
                (G1::generator() * Fr::random(OsRng)).to_affine()
            }
        })
        .collect();

    assert_chunked_and_unchunked(&coeffs, &bases, "zeros + identity bases log_n=14");
}

/// All-zero scalars: the whole MSM collapses to the identity, exercising
/// `to_affine` of an identity accumulator and all-empty buckets.
#[test]
fn all_zero_scalars_identity_result_log_n_14() {
    let n = 1usize << 14;
    let coeffs = vec![Fr::ZERO; n];
    let bases = rand_bases(n);

    let cpu = best_multiexp_cpu::<G1Affine>(&coeffs, &bases).to_affine();
    assert_eq!(cpu, G1::identity().to_affine(), "oracle must be identity");

    let d_coeffs = coeffs.to_device_on(&HALO2_GPU_CTX).expect("H2D coeffs");
    let d_bases = bases.to_device_on(&HALO2_GPU_CTX).expect("H2D bases");
    let gpu = multiexp_gpu_device_scalars_device_bases_chunked::<G1Affine>(&d_coeffs, &d_bases, n)
        .expect("device MSM")
        .to_affine();
    assert_affine_bytes_eq(&gpu, &cpu, "all-zero scalars (identity result)");
}

/// A single nonzero coeff amid `n-1` zeros: each window in which the lone
/// scalar has a nonzero digit yields a **single-element bucket**, and every
/// other bucket is **empty**. Exercises the 1-member reduction (1 real point +
/// 31 infinity lanes) and the empty-bucket (`begin == end`) seed.
#[test]
fn single_and_empty_buckets_log_n_14() {
    let n = 1usize << 14;
    let mut coeffs = vec![Fr::ZERO; n];
    coeffs[0] = dense_constant();
    let bases = rand_bases(n);
    assert_chunked_and_unchunked(&coeffs, &bases, "single/empty buckets log_n=14");
}

/// Scalar for group `g` (0..10) of the max-dense-per-window adversarial case.
///
/// At `n ≈ 2^16` the Pippenger `win_bit` is 8, so each window is exactly one
/// scalar byte. Byte `j` (for `j` in 0..31) is set to a value in `1..=250`
/// that is DISTINCT across `g` (`((g + j) % 250) + 1`), and byte 31 is left 0
/// (keeps the value `< 2^248 < modulus`, so `from_repr` succeeds). Two
/// different groups therefore never share a digit in any window, so all 10
/// groups occupy 10 distinct dense buckets in every one of the 31 low windows.
fn ten_group_scalar(g: usize) -> Fr {
    let mut bytes = [0u8; 32];
    for (j, b) in bytes.iter_mut().enumerate().take(31) {
        *b = (((g + j) % 250) + 1) as u8;
    }
    // byte 31 stays 0 (top window digit 0, and value < 2^248 < modulus)
    Option::<Fr>::from(Fr::from_repr(bytes)).expect("in-field scalar")
}

/// Arena/worklist sizing at the maximum dense-buckets-per-window.
///
/// With `win_bit == 8` each window is one scalar byte; `ten_group_scalar`
/// makes 10 digit-distinct groups, each repeated `PER_GROUP` times with
/// sparsity `PER_GROUP/n ≈ 0.10 ≥ SPARSITY_THRESHOLD`. That yields 10 dense
/// buckets in each of the 31 low windows == **310 dense buckets**, the worst
/// case. `MAX_DENSE_BUCKET_NUM` must be the inclusive ceiling `10`, which
/// sizes both the `d_dense_out` arena and the worklist for `win_num*10 = 320
/// ≥ 310` slices; a truncating floor of `9` would size them for only
/// `win_num*9 = 288` slices, so ~22 dense buckets would be dropped/OOB and the
/// result would diverge from the oracle. `n` is deliberately non-power-of-two.
#[test]
fn ten_dense_buckets_per_window_overflow() {
    const GROUPS: usize = 10;
    const PER_GROUP: usize = 6554; // 6554/65540 = 0.10001 >= SPARSITY_THRESHOLD
    let n = GROUPS * PER_GROUP; // 65540, non-power-of-two
    let mut coeffs: Vec<Fr> = Vec::with_capacity(n);
    for g in 0..GROUPS {
        let k = ten_group_scalar(g);
        coeffs.extend(std::iter::repeat_n(k, PER_GROUP));
    }
    let bases = rand_bases(n);
    assert_chunked_and_unchunked(&coeffs, &bases, "10 dense buckets/window overflow");
}

/// SPLIT_N_BLOCKS floor for sub-`TILE` dense chunks.
///
/// A valid dense chunk with `4 <= point_num < TILE_PER_BLOCK (=32)`: all
/// scalars equal, so every window has a single dense bucket (sparsity 1.0).
/// Here `point_num / 32` truncates to `0`; without a floor of one block, no
/// split blocks would launch and the dense bucket would be written as infinity
/// with its summands lost. The `>= 1` clamp makes a single block sum the whole
/// bucket. Verified against the oracle.
#[test]
fn small_dense_chunk_sub_tile() {
    let n = 20usize; // 4 <= 20 < 32
    let coeffs = vec![dense_constant(); n];
    let bases = rand_bases(n);

    let cpu = best_multiexp_cpu::<G1Affine>(&coeffs, &bases).to_affine();
    let d_coeffs = coeffs.to_device_on(&HALO2_GPU_CTX).expect("H2D coeffs");
    let d_bases = bases.to_device_on(&HALO2_GPU_CTX).expect("H2D bases");
    // Unchunked (single chunk == whole input): a chunked dispatch here would
    // hit a sub-`MSM_MIN_KERNEL_LEN` tail chunk (separate concern), so the
    // sub-TILE dense path is exercised with one chunk of `point_num = n`.
    let gpu = multiexp_gpu_device_scalars_device_bases_chunked::<G1Affine>(&d_coeffs, &d_bases, n)
        .expect("device MSM")
        .to_affine();
    assert_affine_bytes_eq(&gpu, &cpu, "sub-TILE dense chunk (n=20)");
}
