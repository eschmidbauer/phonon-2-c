/*  blas.c – Hand-rolled fp32 sgemm plus the quantised-B GEMMs Phonon-2 needs. No BLAS library anywhere.
 *
 *  fp32 (p2_sgemm / p2_sgemm_st): register-blocked NEON / AVX2 micro-kernels with cache-blocked B packing, from
 *  parakeet-unified-en-0.6b-c.
 *
 *  Quantised B operands, exact mode: expanded to fp32 inside the pack step, so the inner kernel is the fp32 one and the
 *  result equals an fp32 GEMM on the expanded weights:
 *    p2_sgemm_q8b  int8 [K, N] with one fp32 scale per column
 *    p2_sgemm_fvb  five-value [K, N]: w = sign · (flag ? hi[n] : lo[n]); 2-bit signs + 1-bit flags (blas.h)
 *  These partition N (not M) across threads so each weight is expanded once per GEMM, not once per thread.
 *  Int8 modes: see the dot-product section further down.
 *
 *  Compile with `-O3 -march=native -ffast-math`.
 */
#include "phonon2.h"
#include "blas.h"
#include "threadpool.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>

/* -DP2_SCALAR builds the scalar fallbacks alone (no NEON, AVX2 or dot-product code), to test them on a machine that
 * has SIMD; the same switch is honoured in tensor_ops.c and encoder.c. */
#if (defined(__aarch64__) || defined(__ARM_NEON)) && !defined(P2_SCALAR)
  #include <arm_neon.h>
  #define P2_NEON 1
#endif

#if defined(__AVX2__) && defined(__FMA__) && !defined(P2_SCALAR)
  #include <immintrin.h>
  #define P2_AVX2 1
#endif

/* WebAssembly with -msimd128 (web/wasm/build.sh): the fp32 micro-kernels in wasm_simd128.h terms. The int8 paths
 * need SDOT or AVX2, so this build runs the exact GEMM mode; the kernel below was 94 % of its time as scalar code. */
#if defined(__wasm_simd128__) && !defined(P2_SCALAR)
  #include <wasm_simd128.h>
  #define P2_WASM 1
#endif

/* How many K-iterations ahead to prefetch B rows for. */
#define P2_PF_DIST 8

#if defined(__GNUC__) || defined(__clang__)
  #define P2_PREFETCH(addr) __builtin_prefetch((addr), 0, 1)
#else
  #define P2_PREFETCH(addr) ((void)0)
#endif

/* Columns per B panel: the micro-kernel's width, and the unit quantised weights are expanded in. */
#define NR 16
/* Columns per packed B block. */
#define NC 256

/* ── SGEMM helpers ────────────────────────────────────────────────────────── */

static void scale_c(float *C, int M, int N, int ldc, float beta)
{
    if (beta == 1.0f) return;
    if (beta == 0.0f) {
        for (int i = 0; i < M; i++) memset(C + (size_t)i * ldc, 0, N * sizeof(float));
        return;
    }
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            C[(size_t)i * ldc + j] *= beta;
}

/* ── Micro-kernels and their packing buffer ──────────────────────────────── */

/* ── NEON kernels: MR×16 NN (MR = 1, 2, 3, 4) ─────────────────────────────── */

#if P2_NEON

#define NEON_KERNEL_NN(MR_VAL, SUFFIX)                                     \
static inline void kernel_nn_##SUFFIX##x16_neon(int K, float alpha,        \
                                          const float *A, int lda,         \
                                          const float *B, int ldb,         \
                                          float *C, int ldc)               \
{                                                                          \
    float32x4_t c[MR_VAL][4];                                              \
    for (int i = 0; i < MR_VAL; i++) {                                     \
        c[i][0] = vld1q_f32(C + i * ldc + 0);                              \
        c[i][1] = vld1q_f32(C + i * ldc + 4);                              \
        c[i][2] = vld1q_f32(C + i * ldc + 8);                              \
        c[i][3] = vld1q_f32(C + i * ldc + 12);                             \
    }                                                                      \
    for (int k = 0; k < K; k++) {                                          \
        if (k + P2_PF_DIST < K)                                            \
            P2_PREFETCH(B + (k + P2_PF_DIST) * ldb);                       \
        float32x4_t b0 = vld1q_f32(B + k * ldb + 0);                       \
        float32x4_t b1 = vld1q_f32(B + k * ldb + 4);                       \
        float32x4_t b2 = vld1q_f32(B + k * ldb + 8);                       \
        float32x4_t b3 = vld1q_f32(B + k * ldb + 12);                      \
        for (int i = 0; i < MR_VAL; i++) {                                 \
            float32x4_t a = vdupq_n_f32(alpha * A[i * lda + k]);           \
            c[i][0] = vfmaq_f32(c[i][0], a, b0);                           \
            c[i][1] = vfmaq_f32(c[i][1], a, b1);                           \
            c[i][2] = vfmaq_f32(c[i][2], a, b2);                           \
            c[i][3] = vfmaq_f32(c[i][3], a, b3);                           \
        }                                                                  \
    }                                                                      \
    for (int i = 0; i < MR_VAL; i++) {                                     \
        vst1q_f32(C + i * ldc + 0,  c[i][0]);                              \
        vst1q_f32(C + i * ldc + 4,  c[i][1]);                              \
        vst1q_f32(C + i * ldc + 8,  c[i][2]);                              \
        vst1q_f32(C + i * ldc + 12, c[i][3]);                              \
    }                                                                      \
}

NEON_KERNEL_NN(1, 1)
NEON_KERNEL_NN(2, 2)
NEON_KERNEL_NN(3, 3)
NEON_KERNEL_NN(4, 4)

#define NEON_KERNEL_NT(MR_VAL, SUFFIX)                                     \
static inline void kernel_nt_##SUFFIX##x16_neon(int K, float alpha,        \
                                          const float *A, int lda,         \
                                          const float *B, int ldb,         \
                                          float *C, int ldc)               \
{                                                                          \
    for (int i = 0; i < MR_VAL; i++) {                                     \
        const float *a = A + i * lda;                                      \
        float *c = C + i * ldc;                                            \
        for (int j = 0; j < 16; j++) {                                     \
            const float *b = B + j * ldb;                                  \
            float32x4_t acc = vdupq_n_f32(0.0f);                           \
            int k = 0;                                                     \
            for (; k + 4 <= K; k += 4) {                                   \
                float32x4_t av = vld1q_f32(a + k);                         \
                float32x4_t bv = vld1q_f32(b + k);                         \
                acc = vfmaq_f32(acc, av, bv);                              \
            }                                                              \
            float s = vaddvq_f32(acc);                                     \
            for (; k < K; k++) s += a[k] * b[k];                           \
            c[j] += alpha * s;                                             \
        }                                                                  \
    }                                                                      \
}

NEON_KERNEL_NT(1, 1)
NEON_KERNEL_NT(2, 2)
NEON_KERNEL_NT(3, 3)
NEON_KERNEL_NT(4, 4)

#endif /* P2_NEON */

/* ── AVX2 kernels: MR×16 NN (MR = 1, 2, 3, 4) ─────────────────────────────── */

#if P2_AVX2

#define AVX2_KERNEL_NN(MR_VAL, SUFFIX)                                     \
static inline void kernel_nn_##SUFFIX##x16_avx2(int K, float alpha,        \
                                          const float *A, int lda,         \
                                          const float *B, int ldb,         \
                                          float *C, int ldc)               \
{                                                                          \
    __m256 c0[MR_VAL], c1[MR_VAL];                                         \
    for (int i = 0; i < MR_VAL; i++) {                                     \
        c0[i] = _mm256_loadu_ps(C + i * ldc + 0);                          \
        c1[i] = _mm256_loadu_ps(C + i * ldc + 8);                          \
    }                                                                      \
    for (int k = 0; k < K; k++) {                                          \
        if (k + P2_PF_DIST < K)                                            \
            P2_PREFETCH(B + (k + P2_PF_DIST) * ldb);                       \
        __m256 b0 = _mm256_loadu_ps(B + k * ldb + 0);                      \
        __m256 b1 = _mm256_loadu_ps(B + k * ldb + 8);                      \
        for (int i = 0; i < MR_VAL; i++) {                                 \
            __m256 a = _mm256_set1_ps(alpha * A[i * lda + k]);             \
            c0[i] = _mm256_fmadd_ps(a, b0, c0[i]);                         \
            c1[i] = _mm256_fmadd_ps(a, b1, c1[i]);                         \
        }                                                                  \
    }                                                                      \
    for (int i = 0; i < MR_VAL; i++) {                                     \
        _mm256_storeu_ps(C + i * ldc + 0, c0[i]);                          \
        _mm256_storeu_ps(C + i * ldc + 8, c1[i]);                          \
    }                                                                      \
}

AVX2_KERNEL_NN(1, 1)
AVX2_KERNEL_NN(2, 2)
AVX2_KERNEL_NN(3, 3)
AVX2_KERNEL_NN(4, 4)

#define AVX2_KERNEL_NT(MR_VAL, SUFFIX)                                     \
static inline void kernel_nt_##SUFFIX##x16_avx2(int K, float alpha,        \
                                          const float *A, int lda,         \
                                          const float *B, int ldb,         \
                                          float *C, int ldc)               \
{                                                                          \
    for (int i = 0; i < MR_VAL; i++) {                                     \
        const float *a = A + i * lda;                                      \
        float *c = C + i * ldc;                                            \
        for (int j = 0; j < 16; j++) {                                     \
            const float *b = B + j * ldb;                                  \
            __m256 acc = _mm256_setzero_ps();                              \
            int k = 0;                                                     \
            for (; k + 8 <= K; k += 8) {                                   \
                __m256 av = _mm256_loadu_ps(a + k);                        \
                __m256 bv = _mm256_loadu_ps(b + k);                        \
                acc = _mm256_fmadd_ps(av, bv, acc);                        \
            }                                                              \
            __m128 lo = _mm256_castps256_ps128(acc);                       \
            __m128 hi = _mm256_extractf128_ps(acc, 1);                     \
            __m128 s4 = _mm_add_ps(lo, hi);                                \
            __m128 s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));             \
            __m128 s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));         \
            float s = _mm_cvtss_f32(s1);                                   \
            for (; k < K; k++) s += a[k] * b[k];                           \
            c[j] += alpha * s;                                             \
        }                                                                  \
    }                                                                      \
}

AVX2_KERNEL_NT(1, 1)
AVX2_KERNEL_NT(2, 2)
AVX2_KERNEL_NT(3, 3)
AVX2_KERNEL_NT(4, 4)

#endif /* P2_AVX2 */

/* ── wasm SIMD128 kernels: MR×16 NN and NT (MR = 1, 2, 3, 4), the NEON ones ── */

#if P2_WASM

/* c += a · b: fused where the build has relaxed SIMD (-mrelaxed-simd, f32x4.relaxed_madd), else a multiply and an
 * add – the wasm core profile has no FMA. The accumulators stay in registers across the whole K loop, which is what
 * the scalar fallback could not do (the compiler cannot prove C and B apart, so it reloaded and stored the tile on
 * every k) and why it ran 4× slower than this. */
#ifdef __wasm_relaxed_simd__
  #define P2_WASM_FMA(c, a, b) wasm_f32x4_relaxed_madd((a), (b), (c))
#else
  #define P2_WASM_FMA(c, a, b) wasm_f32x4_add((c), wasm_f32x4_mul((a), (b)))
#endif

#define WASM_KERNEL_NN(MR_VAL, SUFFIX)                                     \
static inline void kernel_nn_##SUFFIX##x16_wasm(int K, float alpha,        \
                                          const float *A, int lda,         \
                                          const float *B, int ldb,         \
                                          float *C, int ldc)               \
{                                                                          \
    v128_t c[MR_VAL][4];                                                   \
    for (int i = 0; i < MR_VAL; i++) {                                     \
        c[i][0] = wasm_v128_load(C + i * ldc + 0);                         \
        c[i][1] = wasm_v128_load(C + i * ldc + 4);                         \
        c[i][2] = wasm_v128_load(C + i * ldc + 8);                         \
        c[i][3] = wasm_v128_load(C + i * ldc + 12);                        \
    }                                                                      \
    for (int k = 0; k < K; k++) {                                          \
        v128_t b0 = wasm_v128_load(B + k * ldb + 0);                       \
        v128_t b1 = wasm_v128_load(B + k * ldb + 4);                       \
        v128_t b2 = wasm_v128_load(B + k * ldb + 8);                       \
        v128_t b3 = wasm_v128_load(B + k * ldb + 12);                      \
        for (int i = 0; i < MR_VAL; i++) {                                 \
            v128_t a = wasm_f32x4_splat(alpha * A[i * lda + k]);           \
            c[i][0] = P2_WASM_FMA(c[i][0], a, b0);                         \
            c[i][1] = P2_WASM_FMA(c[i][1], a, b1);                         \
            c[i][2] = P2_WASM_FMA(c[i][2], a, b2);                         \
            c[i][3] = P2_WASM_FMA(c[i][3], a, b3);                         \
        }                                                                  \
    }                                                                      \
    for (int i = 0; i < MR_VAL; i++) {                                     \
        wasm_v128_store(C + i * ldc + 0,  c[i][0]);                        \
        wasm_v128_store(C + i * ldc + 4,  c[i][1]);                        \
        wasm_v128_store(C + i * ldc + 8,  c[i][2]);                        \
        wasm_v128_store(C + i * ldc + 12, c[i][3]);                        \
    }                                                                      \
}

WASM_KERNEL_NN(1, 1)
WASM_KERNEL_NN(2, 2)
WASM_KERNEL_NN(3, 3)
WASM_KERNEL_NN(4, 4)

#define WASM_KERNEL_NT(MR_VAL, SUFFIX)                                     \
static inline void kernel_nt_##SUFFIX##x16_wasm(int K, float alpha,        \
                                          const float *A, int lda,         \
                                          const float *B, int ldb,         \
                                          float *C, int ldc)               \
{                                                                          \
    for (int i = 0; i < MR_VAL; i++) {                                     \
        const float *a = A + i * lda;                                      \
        float *c = C + i * ldc;                                            \
        for (int j = 0; j < 16; j++) {                                     \
            const float *b = B + j * ldb;                                  \
            v128_t acc = wasm_f32x4_splat(0.0f);                           \
            int k = 0;                                                     \
            for (; k + 4 <= K; k += 4)                                     \
                acc = P2_WASM_FMA(acc, wasm_v128_load(a + k),              \
                                       wasm_v128_load(b + k));             \
            float s = wasm_f32x4_extract_lane(acc, 0)                      \
                    + wasm_f32x4_extract_lane(acc, 1)                      \
                    + wasm_f32x4_extract_lane(acc, 2)                      \
                    + wasm_f32x4_extract_lane(acc, 3);                     \
            for (; k < K; k++) s += a[k] * b[k];                           \
            c[j] += alpha * s;                                             \
        }                                                                  \
    }                                                                      \
}

WASM_KERNEL_NT(1, 1)
WASM_KERNEL_NT(2, 2)
WASM_KERNEL_NT(3, 3)
WASM_KERNEL_NT(4, 4)

#endif /* P2_WASM */

/* ── Scalar fallback kernels (only compiled when no SIMD is available) ────── */

#if !P2_NEON && !P2_AVX2 && !P2_WASM
static inline void kernel_nn_generic(int MR_, int K, float alpha,
                                     const float *A, int lda,
                                     const float *B, int ldb,
                                     float *C, int ldc)
{
    for (int i = 0; i < MR_; i++) {
        float *c = C + i * ldc;
        for (int k = 0; k < K; k++) {
            float a = alpha * A[i * lda + k];
            const float *b = B + k * ldb;
            for (int j = 0; j < 16; j++) c[j] += a * b[j];
        }
    }
}

static inline void kernel_nt_generic(int MR_, int K, float alpha,
                                     const float *A, int lda,
                                     const float *B, int ldb,
                                     float *C, int ldc)
{
    for (int i = 0; i < MR_; i++) {
        const float *a = A + i * lda;
        float *c = C + i * ldc;
        for (int j = 0; j < 16; j++) {
            const float *b = B + j * ldb;
            float acc = 0.0f;
            for (int k = 0; k < K; k++) acc += a[k] * b[k];
            c[j] += alpha * acc;
        }
    }
}
#endif

/* ── Kernel dispatch ──────────────────────────────────────────────────────── */

static void dispatch_nn(int MR_, int K, float alpha,
                        const float *A, int lda,
                        const float *B, int ldb,
                        float *C, int ldc)
{
#if P2_NEON
    switch (MR_) {
        case 1: kernel_nn_1x16_neon(K, alpha, A, lda, B, ldb, C, ldc); return;
        case 2: kernel_nn_2x16_neon(K, alpha, A, lda, B, ldb, C, ldc); return;
        case 3: kernel_nn_3x16_neon(K, alpha, A, lda, B, ldb, C, ldc); return;
        case 4: kernel_nn_4x16_neon(K, alpha, A, lda, B, ldb, C, ldc); return;
    }
#elif P2_AVX2
    switch (MR_) {
        case 1: kernel_nn_1x16_avx2(K, alpha, A, lda, B, ldb, C, ldc); return;
        case 2: kernel_nn_2x16_avx2(K, alpha, A, lda, B, ldb, C, ldc); return;
        case 3: kernel_nn_3x16_avx2(K, alpha, A, lda, B, ldb, C, ldc); return;
        case 4: kernel_nn_4x16_avx2(K, alpha, A, lda, B, ldb, C, ldc); return;
    }
#elif P2_WASM
    switch (MR_) {
        case 1: kernel_nn_1x16_wasm(K, alpha, A, lda, B, ldb, C, ldc); return;
        case 2: kernel_nn_2x16_wasm(K, alpha, A, lda, B, ldb, C, ldc); return;
        case 3: kernel_nn_3x16_wasm(K, alpha, A, lda, B, ldb, C, ldc); return;
        case 4: kernel_nn_4x16_wasm(K, alpha, A, lda, B, ldb, C, ldc); return;
    }
#else
    kernel_nn_generic(MR_, K, alpha, A, lda, B, ldb, C, ldc);
#endif
}

static void dispatch_nt(int MR_, int K, float alpha,
                        const float *A, int lda,
                        const float *B, int ldb,
                        float *C, int ldc)
{
#if P2_NEON
    switch (MR_) {
        case 1: kernel_nt_1x16_neon(K, alpha, A, lda, B, ldb, C, ldc); return;
        case 2: kernel_nt_2x16_neon(K, alpha, A, lda, B, ldb, C, ldc); return;
        case 3: kernel_nt_3x16_neon(K, alpha, A, lda, B, ldb, C, ldc); return;
        case 4: kernel_nt_4x16_neon(K, alpha, A, lda, B, ldb, C, ldc); return;
    }
#elif P2_AVX2
    switch (MR_) {
        case 1: kernel_nt_1x16_avx2(K, alpha, A, lda, B, ldb, C, ldc); return;
        case 2: kernel_nt_2x16_avx2(K, alpha, A, lda, B, ldb, C, ldc); return;
        case 3: kernel_nt_3x16_avx2(K, alpha, A, lda, B, ldb, C, ldc); return;
        case 4: kernel_nt_4x16_avx2(K, alpha, A, lda, B, ldb, C, ldc); return;
    }
#elif P2_WASM
    switch (MR_) {
        case 1: kernel_nt_1x16_wasm(K, alpha, A, lda, B, ldb, C, ldc); return;
        case 2: kernel_nt_2x16_wasm(K, alpha, A, lda, B, ldb, C, ldc); return;
        case 3: kernel_nt_3x16_wasm(K, alpha, A, lda, B, ldb, C, ldc); return;
        case 4: kernel_nt_4x16_wasm(K, alpha, A, lda, B, ldb, C, ldc); return;
    }
#else
    kernel_nt_generic(MR_, K, alpha, A, lda, B, ldb, C, ldc);
#endif
}

/* ── Column-edge (N < 16) kernels ─────────────────────────────────────────── */

static void update_edge_nn(int M, int N, int K, float alpha,
                           const float *A, int lda,
                           const float *B, int ldb,
                           float *C, int ldc)
{
#if P2_NEON
    int Nv4 = (N / 4) * 4;
    for (int i = 0; i < M; i++) {
        float *c = C + i * ldc;
        for (int k = 0; k < K; k++) {
            if (k + P2_PF_DIST < K)
                P2_PREFETCH(B + (k + P2_PF_DIST) * ldb);
            float a_s = alpha * A[i * lda + k];
            float32x4_t a = vdupq_n_f32(a_s);
            const float *b = B + k * ldb;
            int j = 0;
            for (; j + 4 <= Nv4; j += 4) {
                float32x4_t bv = vld1q_f32(b + j);
                float32x4_t cv = vld1q_f32(c + j);
                vst1q_f32(c + j, vfmaq_f32(cv, a, bv));
            }
            for (; j < N; j++) c[j] += a_s * b[j];
        }
    }
#elif P2_AVX2
    int Nv8 = (N / 8) * 8;
    for (int i = 0; i < M; i++) {
        float *c = C + i * ldc;
        for (int k = 0; k < K; k++) {
            if (k + P2_PF_DIST < K)
                P2_PREFETCH(B + (k + P2_PF_DIST) * ldb);
            float a_s = alpha * A[i * lda + k];
            __m256 a = _mm256_set1_ps(a_s);
            const float *b = B + k * ldb;
            int j = 0;
            for (; j + 8 <= Nv8; j += 8) {
                __m256 bv = _mm256_loadu_ps(b + j);
                __m256 cv = _mm256_loadu_ps(c + j);
                _mm256_storeu_ps(c + j, _mm256_fmadd_ps(a, bv, cv));
            }
            for (; j < N; j++) c[j] += a_s * b[j];
        }
    }
#else
    for (int i = 0; i < M; i++) {
        float *c = C + i * ldc;
        for (int k = 0; k < K; k++) {
            float a = alpha * A[i * lda + k];
            const float *b = B + k * ldb;
            for (int j = 0; j < N; j++) c[j] += a * b[j];
        }
    }
#endif
}

static void update_edge_nt(int M, int N, int K, float alpha,
                           const float *A, int lda,
                           const float *B, int ldb,
                           float *C, int ldc)
{
#if P2_NEON
    for (int i = 0; i < M; i++) {
        const float *a = A + i * lda;
        float *c = C + i * ldc;
        for (int j = 0; j < N; j++) {
            const float *b = B + j * ldb;
            float32x4_t acc = vdupq_n_f32(0.0f);
            int k = 0;
            for (; k + 4 <= K; k += 4) {
                float32x4_t av = vld1q_f32(a + k);
                float32x4_t bv = vld1q_f32(b + k);
                acc = vfmaq_f32(acc, av, bv);
            }
            float s = vaddvq_f32(acc);
            for (; k < K; k++) s += a[k] * b[k];
            c[j] += alpha * s;
        }
    }
#elif P2_AVX2
    for (int i = 0; i < M; i++) {
        const float *a = A + i * lda;
        float *c = C + i * ldc;
        for (int j = 0; j < N; j++) {
            const float *b = B + j * ldb;
            __m256 acc = _mm256_setzero_ps();
            int k = 0;
            for (; k + 8 <= K; k += 8) {
                __m256 av = _mm256_loadu_ps(a + k);
                __m256 bv = _mm256_loadu_ps(b + k);
                acc = _mm256_fmadd_ps(av, bv, acc);
            }
            __m128 lo = _mm256_castps256_ps128(acc);
            __m128 hi = _mm256_extractf128_ps(acc, 1);
            __m128 s4 = _mm_add_ps(lo, hi);
            __m128 s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
            __m128 s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));
            float s = _mm_cvtss_f32(s1);
            for (; k < K; k++) s += a[k] * b[k];
            c[j] += alpha * s;
        }
    }
#else
    for (int i = 0; i < M; i++) {
        const float *a = A + i * lda;
        float *c = C + i * ldc;
        for (int j = 0; j < N; j++) {
            const float *b = B + j * ldb;
            float acc = 0.0f;
            for (int k = 0; k < K; k++) acc += a[k] * b[k];
            c[j] += alpha * acc;
        }
    }
#endif
}

/* ── Blocking and the per-thread packing buffer ───────────────────────────── */

#define MR 4

/* B_pack (KC × NC fp32) ≈ 256 KB: fits L2 on Apple M-series and typical x86 parts. */
#define KC 256
#define B_PACK_BYTES ((size_t)KC * NC * sizeof(float))

/* Thread-local packing buffer, allocated on first use, reclaimed when the thread exits. */
static pthread_key_t  tls_B_pack_key;
static pthread_once_t tls_B_pack_once = PTHREAD_ONCE_INIT;

static void tls_B_pack_destroy(void *p) { free(p); }

static void tls_B_pack_init(void)
{
    pthread_key_create(&tls_B_pack_key, tls_B_pack_destroy);
}

static float *get_B_pack(void)
{
    pthread_once(&tls_B_pack_once, tls_B_pack_init);
    float *p = (float *)pthread_getspecific(tls_B_pack_key);
    if (p == NULL) {
        p = (float *)aligned_alloc(64, B_PACK_BYTES);
        if (p != NULL)
            pthread_setspecific(tls_B_pack_key, p);
    }
    return p;
}

/* ── fp32 SGEMM (hand-rolled) ─────────────────────────────────────────────── */

/* Pack one [kb × NR] slice of B so pack[k * NR + j] == B[k * ldb + j]; the micro-kernels then read it with ldb = NR. */
static inline void pack_B_panel(const float *B, int ldb, float *pack, int kb)
{
    for (int k = 0; k < kb; k++)
        memcpy(pack + k * NR, B + (size_t)k * ldb, NR * sizeof(float));
}

static void pack_B(const float *B, int ldb, float *pack, int kb, int nb_full)
{
    int panels = nb_full / NR;
    for (int p = 0; p < panels; p++)
        pack_B_panel(B + p * NR, ldb, pack + p * (size_t)kb * NR, kb);
}

struct sgemm_nn_args {
    int M, N, K;
    float alpha;
    const float *A; int lda;
    const float *B; int ldb;
    float *C; int ldc;
};

/* Each thread takes an MR-aligned band of rows across all (kc, jc) blocks, with its own B_pack. */
static void sgemm_nn_worker(int tid, int nthreads, void *aux)
{
    struct sgemm_nn_args *a = (struct sgemm_nn_args *)aux;
    int M = a->M, N = a->N, K = a->K, lda = a->lda, ldb = a->ldb, ldc = a->ldc;
    float alpha = a->alpha;
    const float *A = a->A, *B = a->B;
    float *C = a->C;

    int Nf = (N / NR) * NR;

    int tiles = (M + MR - 1) / MR;
    int first_tile = (tiles * tid) / nthreads;
    int last_tile  = (tiles * (tid + 1)) / nthreads;
    int m_start = first_tile * MR;
    int m_end   = last_tile  * MR;
    if (m_end > M) m_end = M;
    if (m_start >= m_end) return;

    float *B_pack = get_B_pack();
    if (!B_pack) {
        int i = m_start;
        while (i < m_end) {
            int mr = m_end - i; if (mr > MR) mr = MR;
            for (int j = 0; j < Nf; j += NR)
                dispatch_nn(mr, K, alpha, A + (size_t)i * lda, lda,
                            B + j, ldb, C + (size_t)i * ldc + j, ldc);
            if (Nf < N)
                update_edge_nn(mr, N - Nf, K, alpha, A + (size_t)i * lda, lda,
                               B + Nf, ldb, C + (size_t)i * ldc + Nf, ldc);
            i += mr;
        }
        return;
    }

    for (int kc = 0; kc < K; kc += KC) {
        int kb = (kc + KC <= K) ? KC : (K - kc);

        for (int jc = 0; jc < Nf; jc += NC) {
            int nb      = (jc + NC <= Nf) ? NC : (Nf - jc);
            int nb_full = (nb / NR) * NR;

            pack_B(B + (size_t)kc * ldb + jc, ldb, B_pack, kb, nb_full);

            int i = m_start;
            while (i < m_end) {
                int mr = m_end - i; if (mr > MR) mr = MR;
                int panels = nb_full / NR;
                for (int p = 0; p < panels; p++) {
                    dispatch_nn(mr, kb, alpha,
                                A + (size_t)i * lda + kc,          lda,
                                B_pack + (size_t)p * kb * NR,      NR,
                                C + (size_t)i * ldc + jc + p * NR, ldc);
                }
                i += mr;
            }
        }

        if (Nf < N) {
            int i = m_start;
            while (i < m_end) {
                int mr = m_end - i; if (mr > MR) mr = MR;
                update_edge_nn(mr, N - Nf, kb, alpha,
                               A + (size_t)i * lda + kc,   lda,
                               B + (size_t)kc * ldb + Nf,  ldb,
                               C + (size_t)i * ldc + Nf,   ldc);
                i += mr;
            }
        }
    }
}

static void sgemm_nn(int M, int N, int K, float alpha,
                     const float *A, int lda,
                     const float *B, int ldb,
                     float *C, int ldc)
{
    struct sgemm_nn_args args = { M, N, K, alpha, A, lda, B, ldb, C, ldc };
    p2_parallel(p2_pool(), sgemm_nn_worker, &args);
}

struct sgemm_nt_args {
    int M, N, K;
    float alpha;
    const float *A; int lda;
    const float *B; int ldb;
    float *C; int ldc;
};

/* NT needs no packing: each output is a dot product along contiguous K. */
static void sgemm_nt_worker(int tid, int nthreads, void *aux)
{
    struct sgemm_nt_args *a = (struct sgemm_nt_args *)aux;
    int M = a->M, N = a->N, K = a->K, lda = a->lda, ldb = a->ldb, ldc = a->ldc;
    float alpha = a->alpha;
    const float *A = a->A, *B = a->B;
    float *C = a->C;

    int Nf = (N / NR) * NR;

    int tiles = (M + MR - 1) / MR;
    int first_tile = (tiles * tid) / nthreads;
    int last_tile  = (tiles * (tid + 1)) / nthreads;
    int m_start = first_tile * MR;
    int m_end   = last_tile  * MR;
    if (m_end > M) m_end = M;
    if (m_start >= m_end) return;

    int i = m_start;
    while (i < m_end) {
        int mr = m_end - i; if (mr > MR) mr = MR;
        for (int j = 0; j < Nf; j += NR) {
            dispatch_nt(mr, K, alpha,
                        A + (size_t)i * lda,     lda,
                        B + (size_t)j * ldb,     ldb,
                        C + (size_t)i * ldc + j, ldc);
        }
        if (Nf < N) {
            update_edge_nt(mr, N - Nf, K, alpha,
                           A + (size_t)i * lda,       lda,
                           B + (size_t)Nf * ldb,      ldb,
                           C + (size_t)i * ldc + Nf,  ldc);
        }
        i += mr;
    }
}

static void sgemm_nt(int M, int N, int K, float alpha,
                     const float *A, int lda,
                     const float *B, int ldb,
                     float *C, int ldc)
{
    struct sgemm_nt_args args = { M, N, K, alpha, A, lda, B, ldb, C, ldc };
    p2_parallel(p2_pool(), sgemm_nt_worker, &args);
}

/* Rarely used TN / TT paths. */
static void sgemm_tn(int M, int N, int K, float alpha,
                     const float *A, int lda,
                     const float *B, int ldb,
                     float *C, int ldc)
{
    for (int i = 0; i < M; i++) {
        float *c = C + (size_t)i * ldc;
        for (int k = 0; k < K; k++) {
            float a = alpha * A[(size_t)k * lda + i];
            const float *b = B + (size_t)k * ldb;
            for (int j = 0; j < N; j++) c[j] += a * b[j];
        }
    }
}

static void sgemm_tt(int M, int N, int K, float alpha,
                     const float *A, int lda,
                     const float *B, int ldb,
                     float *C, int ldc)
{
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) acc += A[(size_t)k * lda + i] * B[(size_t)j * ldb + k];
            C[(size_t)i * ldc + j] += alpha * acc;
        }
}

void p2_sgemm(P2Trans transA, P2Trans transB,
              int M, int N, int K,
              float alpha, const float *A, int lda,
              const float *B, int ldb,
              float beta, float *C, int ldc)
{
    scale_c(C, M, N, ldc, beta);
    if (alpha == 0.0f) return;

    int tA = (transA == P2_TRANS);
    int tB = (transB == P2_TRANS);
    if      (!tA && !tB) sgemm_nn(M, N, K, alpha, A, lda, B, ldb, C, ldc);
    else if (!tA &&  tB) sgemm_nt(M, N, K, alpha, A, lda, B, ldb, C, ldc);
    else if ( tA && !tB) sgemm_tn(M, N, K, alpha, A, lda, B, ldb, C, ldc);
    else                 sgemm_tt(M, N, K, alpha, A, lda, B, ldb, C, ldc);
}

void p2_sgemm_st(P2Trans transA, P2Trans transB,
                 int M, int N, int K,
                 float alpha, const float *A, int lda,
                 const float *B, int ldb,
                 float beta, float *C, int ldc)
{
    scale_c(C, M, N, ldc, beta);
    if (alpha == 0.0f) return;

    int tA = (transA == P2_TRANS);
    int tB = (transB == P2_TRANS);
    if (!tA && !tB) {
        struct sgemm_nn_args args = { M, N, K, alpha, A, lda, B, ldb, C, ldc };
        sgemm_nn_worker(0, 1, &args);
    } else if (!tA && tB) {
        struct sgemm_nt_args args = { M, N, K, alpha, A, lda, B, ldb, C, ldc };
        sgemm_nt_worker(0, 1, &args);
    } else if (tA && !tB) {
        sgemm_tn(M, N, K, alpha, A, lda, B, ldb, C, ldc);
    } else {
        sgemm_tt(M, N, K, alpha, A, lda, B, ldb, C, ldc);
    }
}

/* ── Quantised-B GEMMs ─────────────────────────────────────────────────────── */

/* Byte → four 2-bit sign codes (as -1/0/+1) and byte → eight flag bits, for the AVX2 and scalar unpackers. */
static signed char   SIGN_LUT[256][4];
static unsigned char FLAG_LUT[256][8];
static pthread_once_t lut_once = PTHREAD_ONCE_INIT;

static void lut_init(void)
{
    for (int b = 0; b < 256; b++) {
        for (int j = 0; j < 4; j++) SIGN_LUT[b][j] = (signed char)(((b >> (2 * j)) & 3) - 1);
        for (int j = 0; j < 8; j++) FLAG_LUT[b][j] = (unsigned char)((b >> j) & 1);
    }
}

/* Expand a [kb × NR] slice of int8 B, columns col..col+15, into fp32 rows of stride ldp, multiplying each column by
 * its own scale. */
static inline void pack_B_panel_q8c(const signed char *B, int ldb, const float *sc16,
                                    float *pack, int ldp, int kb)
{
#if P2_NEON
    float32x4_t vs0 = vld1q_f32(sc16 + 0), vs1 = vld1q_f32(sc16 + 4);
    float32x4_t vs2 = vld1q_f32(sc16 + 8), vs3 = vld1q_f32(sc16 + 12);
#elif P2_AVX2
    __m256 vs_lo = _mm256_loadu_ps(sc16), vs_hi = _mm256_loadu_ps(sc16 + 8);
#endif
    for (int k = 0; k < kb; k++) {
        const signed char *src = B + (size_t)k * ldb;
        float *dst = pack + (size_t)k * ldp;
#if P2_NEON
        int8x16_t i8 = vld1q_s8(src);
        int16x8_t lo = vmovl_s8(vget_low_s8(i8));
        int16x8_t hi = vmovl_s8(vget_high_s8(i8));
        vst1q_f32(dst + 0,  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))),  vs0));
        vst1q_f32(dst + 4,  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))), vs1));
        vst1q_f32(dst + 8,  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))),  vs2));
        vst1q_f32(dst + 12, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))), vs3));
#elif P2_AVX2
        __m128i i8 = _mm_loadu_si128((const __m128i *)src);
        __m256 f_lo = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(i8));
        __m256 f_hi = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(i8, 8)));
        _mm256_storeu_ps(dst + 0, _mm256_mul_ps(f_lo, vs_lo));
        _mm256_storeu_ps(dst + 8, _mm256_mul_ps(f_hi, vs_hi));
#else
        for (int j = 0; j < NR; j++) dst[j] = (float)src[j] * sc16[j];
#endif
    }
}

/* Expand a [kb × NR] slice of five-value B, rows k0.., columns col..col+15, into fp32 rows of stride ldp.
 * mag = lo + flag · (hi − lo) is exact: lo, hi are fp16 values held in fp32, their difference is exact in fp32, and
 * lo + (hi − lo) rounds back to hi. So the expansion reproduces {0, ±lo, ±hi} bit for bit. */
static inline void pack_B_panel_fv(const FiveValueB *b, int N, int k0, int col,
                                   float *pack, int ldp, int kb)
{
    const int sign_stride = N / 4, flag_stride = N / 8;
#if P2_NEON
    const uint8x16_t rep4 = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
    const int8x16_t  sh2  = {0, -2, -4, -6, 0, -2, -4, -6, 0, -2, -4, -6, 0, -2, -4, -6};
    const uint8x16_t rep8 = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1};
    const int8x16_t  sh1  = {0, -1, -2, -3, -4, -5, -6, -7, 0, -1, -2, -3, -4, -5, -6, -7};
    const uint8x16_t m3 = vdupq_n_u8(3), m1 = vdupq_n_u8(1);
    const int8x16_t  one = vdupq_n_s8(1);
    float32x4_t lo[4], d[4];
    for (int q = 0; q < 4; q++) {
        lo[q] = vld1q_f32(b->lo + col + 4 * q);
        d[q]  = vsubq_f32(vld1q_f32(b->hi + col + 4 * q), lo[q]);
    }
    for (int k = 0; k < kb; k++) {
        uint32_t s32; uint16_t f16;
        memcpy(&s32, b->signs + (size_t)(k0 + k) * sign_stride + (col >> 2), 4);
        memcpy(&f16, b->flags + (size_t)(k0 + k) * flag_stride + (col >> 3), 2);
        uint8x16_t sb = vqtbl1q_u8(vreinterpretq_u8_u32(vdupq_n_u32(s32)), rep4);
        int8x16_t  sgn = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshlq_u8(sb, sh2), m3)), one);
        uint8x16_t fb = vqtbl1q_u8(vreinterpretq_u8_u16(vdupq_n_u16(f16)), rep8);
        int8x16_t  flg = vreinterpretq_s8_u8(vandq_u8(vshlq_u8(fb, sh1), m1));
        int16x8_t s_lo = vmovl_s8(vget_low_s8(sgn)), s_hi = vmovl_s8(vget_high_s8(sgn));
        int16x8_t f_lo = vmovl_s8(vget_low_s8(flg)), f_hi = vmovl_s8(vget_high_s8(flg));
        float32x4_t s[4] = {
            vcvtq_f32_s32(vmovl_s16(vget_low_s16(s_lo))), vcvtq_f32_s32(vmovl_s16(vget_high_s16(s_lo))),
            vcvtq_f32_s32(vmovl_s16(vget_low_s16(s_hi))), vcvtq_f32_s32(vmovl_s16(vget_high_s16(s_hi))) };
        float32x4_t f[4] = {
            vcvtq_f32_s32(vmovl_s16(vget_low_s16(f_lo))), vcvtq_f32_s32(vmovl_s16(vget_high_s16(f_lo))),
            vcvtq_f32_s32(vmovl_s16(vget_low_s16(f_hi))), vcvtq_f32_s32(vmovl_s16(vget_high_s16(f_hi))) };
        float *dst = pack + (size_t)k * ldp;
        for (int q = 0; q < 4; q++)
            vst1q_f32(dst + 4 * q, vmulq_f32(s[q], vfmaq_f32(lo[q], f[q], d[q])));
    }
#elif P2_AVX2
    __m256 lo_lo = _mm256_loadu_ps(b->lo + col), lo_hi = _mm256_loadu_ps(b->lo + col + 8);
    __m256 d_lo = _mm256_sub_ps(_mm256_loadu_ps(b->hi + col), lo_lo);
    __m256 d_hi = _mm256_sub_ps(_mm256_loadu_ps(b->hi + col + 8), lo_hi);
    for (int k = 0; k < kb; k++) {
        const unsigned char *s = b->signs + (size_t)(k0 + k) * sign_stride + (col >> 2);
        const unsigned char *f = b->flags + (size_t)(k0 + k) * flag_stride + (col >> 3);
        signed char sg[16]; unsigned char fl[16];
        for (int q = 0; q < 4; q++) memcpy(sg + 4 * q, SIGN_LUT[s[q]], 4);
        memcpy(fl, FLAG_LUT[f[0]], 8);
        memcpy(fl + 8, FLAG_LUT[f[1]], 8);
        __m128i sg8 = _mm_loadu_si128((const __m128i *)sg);
        __m128i fl8 = _mm_loadu_si128((const __m128i *)fl);
        __m256 s_lo = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(sg8));
        __m256 s_hi = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(sg8, 8)));
        __m256 f_lo = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(fl8));
        __m256 f_hi = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(fl8, 8)));
        float *dst = pack + (size_t)k * ldp;
        _mm256_storeu_ps(dst + 0, _mm256_mul_ps(s_lo, _mm256_fmadd_ps(f_lo, d_lo, lo_lo)));
        _mm256_storeu_ps(dst + 8, _mm256_mul_ps(s_hi, _mm256_fmadd_ps(f_hi, d_hi, lo_hi)));
    }
#else
    for (int k = 0; k < kb; k++) {
        const unsigned char *s = b->signs + (size_t)(k0 + k) * sign_stride + (col >> 2);
        const unsigned char *f = b->flags + (size_t)(k0 + k) * flag_stride + (col >> 3);
        float *dst = pack + (size_t)k * ldp;
        for (int j = 0; j < NR; j++) {
            int sg = SIGN_LUT[s[j >> 2]][j & 3];
            int fl = FLAG_LUT[f[j >> 3]][j & 7];
            dst[j] = (float)sg * (fl ? b->hi[col + j] : b->lo[col + j]);
        }
    }
#endif
}

struct sgemm_packed_args {
    int M, N, K;
    float alpha;
    const float *A; int lda;
    float *C; int ldc;
    const signed char *q8; const float *scales; int ldb;   /* int8 path */
    const FiveValueB *fv;                                   /* five-value path */
};

/* One five-value weight, expanded – for the no-buffer fallback. */
static inline float fv_weight(const FiveValueB *b, int N, int k, int n)
{
    int code = (b->signs[(size_t)k * (N / 4) + (n >> 2)] >> (2 * (n & 3))) & 3;
    int flag = (b->flags[(size_t)k * (N / 8) + (n >> 3)] >> (n & 7)) & 1;
    return (float)(code - 1) * (flag ? b->hi[n] : b->lo[n]);
}

/* Each thread takes a range of NR-wide column panels and runs every row of A over them, so the weight is expanded
 * exactly once per GEMM. Within the range, (kc, jc) blocks keep B_pack at KC × NC. */
static void sgemm_packed_worker(int tid, int nthreads, void *aux)
{
    struct sgemm_packed_args *a = (struct sgemm_packed_args *)aux;
    int M = a->M, N = a->N, K = a->K, lda = a->lda, ldc = a->ldc;
    float alpha = a->alpha;
    const float *A = a->A;
    float *C = a->C;

    int panels_total = N / NR;
    int p0 = (panels_total * tid) / nthreads;
    int p1 = (panels_total * (tid + 1)) / nthreads;
    if (p0 >= p1) return;
    int n0 = p0 * NR, n1 = p1 * NR;

    float *B_pack = get_B_pack();
    if (!B_pack) {
        /* Correct-but-slow fallback: expand one weight at a time. */
        for (int i = 0; i < M; i++)
            for (int k = 0; k < K; k++) {
                float ax = alpha * A[(size_t)i * lda + k];
                for (int n = n0; n < n1; n++) {
                    float w = a->fv ? fv_weight(a->fv, N, k, n)
                                    : (float)a->q8[(size_t)k * a->ldb + n] * a->scales[n];
                    C[(size_t)i * ldc + n] += ax * w;
                }
            }
        return;
    }

    for (int kc = 0; kc < K; kc += KC) {
        int kb = (kc + KC <= K) ? KC : (K - kc);

        for (int jc = n0; jc < n1; jc += NC) {
            int nb = (jc + NC <= n1) ? NC : (n1 - jc);
            int panels = nb / NR;

            for (int p = 0; p < panels; p++) {
                float *pack = B_pack + (size_t)p * kb * NR;
                if (a->fv)
                    pack_B_panel_fv(a->fv, N, kc, jc + p * NR, pack, NR, kb);
                else
                    pack_B_panel_q8c(a->q8 + (size_t)kc * a->ldb + jc + p * NR, a->ldb,
                                     a->scales + jc + p * NR, pack, NR, kb);
            }

            for (int i = 0; i < M; i += MR) {
                int mr = M - i; if (mr > MR) mr = MR;
                for (int p = 0; p < panels; p++) {
                    dispatch_nn(mr, kb, alpha,
                                A + (size_t)i * lda + kc,          lda,
                                B_pack + (size_t)p * kb * NR,      NR,
                                C + (size_t)i * ldc + jc + p * NR, ldc);
                }
            }
        }
    }
}

static void require_panel_aligned(int N, const char *who)
{
    if (N % NR != 0) {
        fprintf(stderr, "%s: N = %d is not a multiple of %d\n", who, N, NR);
        abort();
    }
}

static void sgemm_packed(struct sgemm_packed_args *args)
{
    p2_parallel(p2_pool(), sgemm_packed_worker, args);
}

/* ── int8 dot-product GEMM for quantised B ───────────────────────────────── */
/*
 *  The fast, inexact path. A is quantised per row to int8 (symmetric, scale = max|row| / 127); B stays exactly what
 *  the container holds. A five-value weight is sign · (flag ? hi : lo) = sign · lo + (sign · flag) · (hi − lo), so two
 *  int8 planes in {−1, 0, +1}, S = sign and T = sign · flag, give
 *
 *      C[m][n] = a_scale[m] · ( lo[n] · Σ_k q[m][k] S[k][n]  +  (hi[n] − lo[n]) · Σ_k q[m][k] T[k][n] )
 *
 *  with both sums exact in int32. An int8 weight table is one plane with its own column scale. The sums run on
 *  NEON SDOT (vdotq_laneq_s32): 16 multiply-adds per instruction against 4 for an fp32 FMA.
 *
 *  Layouts. A: [Mp/4 tiles][K/4 quads][4 rows][4 k] int8, Mp = M rounded up to 4, zero rows beyond M.
 *  B: per 16-column panel, [K/4 quads][16 columns][4 k] int8 — what vst4q_s8 produces from four consecutive rows.
 *  Each thread owns a range of panels, packs all K rows of its block once (int8, so a 4096 × 256 block is 1 MB per
 *  plane) and runs every A tile over it, so the weight is expanded once per GEMM.
 *
 *  Only on AArch64 with the dot-product extension; elsewhere, and under P2_EXACT=1, the fp32 expansion path runs.
 */

#if defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD) && !defined(P2_SCALAR)
#define P2_DOTPROD 1
#endif

#define KC_I8 1024   /* K rows per packed-B block: 16 KB per plane per panel, so a block stays in L1 across every A tile */

enum { GEMM_INT8 = 0, GEMM_EXACT = 1, GEMM_INT8W = 2 };

static int gemm_mode = -1;
static pthread_once_t gemm_mode_once = PTHREAD_ONCE_INIT;

/* P2_GEMM = int8 (A quantised, weights exact, two planes) | int8w (A and weights int8, one plane) | exact (fp32);
 * P2_EXACT=1 is the same as P2_GEMM=exact. Without the dot-product extension everything is exact. */
static void gemm_mode_init(void)
{
    const char *e = getenv("P2_EXACT");
    const char *g = getenv("P2_GEMM");
    int mode = GEMM_INT8W;
    if (g && strcmp(g, "exact") == 0) mode = GEMM_EXACT;
    else if (g && strcmp(g, "int8") == 0) mode = GEMM_INT8;
    else if (g && strcmp(g, "int8w") == 0) mode = GEMM_INT8W;
    if (e && e[0] && e[0] != '0') mode = GEMM_EXACT;
#if !P2_DOTPROD
    mode = GEMM_EXACT;
#endif
    gemm_mode = mode;
}

static int current_mode(void)
{
    pthread_once(&gemm_mode_once, gemm_mode_init);
    return gemm_mode;
}

static int use_exact(void) { return current_mode() == GEMM_EXACT; }

int p2_gemm_exact(void) { return use_exact(); }

const char *p2_gemm_mode(void)
{
    switch (current_mode()) {
    case GEMM_INT8:  return "int8 dot products, exact weights";
    case GEMM_INT8W: return "int8 dot products, int8 weights";
    default:         return "fp32 expansion (exact)";
    }
}

#if P2_DOTPROD

/* Per-thread buffers: the quantised A (grows with M × K) and the packed B block (fixed, two planes). */
typedef struct {
    signed char *a;      float *a_scales; size_t a_cap; int a_rows;
    signed char *b;      size_t b_cap;
} I8Bufs;

static pthread_key_t  tls_i8_key;
static pthread_once_t tls_i8_once = PTHREAD_ONCE_INIT;

static void tls_i8_destroy(void *p)
{
    I8Bufs *b = (I8Bufs *)p;
    if (b) { free(b->a); free(b->a_scales); free(b->b); free(b); }
}

static void tls_i8_init(void) { pthread_key_create(&tls_i8_key, tls_i8_destroy); }

static I8Bufs *get_i8_bufs(size_t a_bytes, int a_rows, size_t b_bytes)
{
    pthread_once(&tls_i8_once, tls_i8_init);
    I8Bufs *b = (I8Bufs *)pthread_getspecific(tls_i8_key);
    if (!b) {
        b = (I8Bufs *)calloc(1, sizeof(*b));
        if (!b) return NULL;
        pthread_setspecific(tls_i8_key, b);
    }
    if (a_bytes > b->a_cap || a_rows > b->a_rows) {
        free(b->a); free(b->a_scales);
        b->a = (signed char *)aligned_alloc(64, (a_bytes + 63) / 64 * 64);
        b->a_scales = (float *)malloc((size_t)a_rows * sizeof(float));
        b->a_cap = a_bytes; b->a_rows = a_rows;
        if (!b->a || !b->a_scales) { b->a_cap = 0; b->a_rows = 0; return NULL; }
    }
    if (b_bytes > b->b_cap) {
        free(b->b);
        b->b = (signed char *)aligned_alloc(64, (b_bytes + 63) / 64 * 64);
        b->b_cap = b_bytes;
        if (!b->b) { b->b_cap = 0; return NULL; }
    }
    return b;
}

/* Quantise one row of K floats symmetrically to int8 and scatter it into the tile layout: dst0 is the row's first
 * quad (tile base + 4·r), successive quads are 16 bytes apart. */
static inline void quant_pack_row(const float *row, int K, signed char *dst0, float *scale_out)
{
    float32x4_t vmax = vdupq_n_f32(0.0f);
    int k = 0;
    for (; k + 4 <= K; k += 4) vmax = vmaxq_f32(vmax, vabsq_f32(vld1q_f32(row + k)));
    float amax = vmaxvq_f32(vmax);
    for (; k < K; k++) { float v = fabsf(row[k]); if (v > amax) amax = v; }
    float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
    *scale_out = scale;
    float32x4_t vinv = vdupq_n_f32(1.0f / scale);
    k = 0;
    for (; k + 16 <= K; k += 16) {
        int32x4_t i0 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(row + k),      vinv));
        int32x4_t i1 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(row + k + 4),  vinv));
        int32x4_t i2 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(row + k + 8),  vinv));
        int32x4_t i3 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(row + k + 12), vinv));
        int8x16_t q = vcombine_s8(vqmovn_s16(vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1))),
                                  vqmovn_s16(vcombine_s16(vqmovn_s32(i2), vqmovn_s32(i3))));
        int32x4_t quads = vreinterpretq_s32_s8(q);
        signed char *dst = dst0 + (size_t)(k / 4) * 16;
        vst1q_lane_s32((int32_t *)(dst),      quads, 0);
        vst1q_lane_s32((int32_t *)(dst + 16), quads, 1);
        vst1q_lane_s32((int32_t *)(dst + 32), quads, 2);
        vst1q_lane_s32((int32_t *)(dst + 48), quads, 3);
    }
    for (; k < K; k++) {
        int v = (int)lrintf(row[k] / scale);
        if (v > 127) v = 127; else if (v < -127) v = -127;
        dst0[(size_t)(k / 4) * 16 + (k & 3)] = (signed char)v;
    }
}

static inline void zero_pad_row(int K, signed char *dst0)
{
    for (int kq = 0; kq < K / 4; kq++) memset(dst0 + (size_t)kq * 16, 0, 4);
}

static inline signed char *row_quad0(signed char *Ap, int K, int m)
{
    return Ap + ((size_t)(m / 4) * (K / 4)) * 16 + (m % 4) * 4;
}

/* Quantise rows of A into the quad layout, one 4-row tile at a time (parallel over tiles). */
struct quant_args { const float *A; int M, K, lda; signed char *Ap; float *scales; };

static void quant_rows_worker(int tid, int nt, void *aux)
{
    struct quant_args *a = (struct quant_args *)aux;
    int tiles = (a->M + 3) / 4;
    int t0 = (tiles * tid) / nt, t1 = (tiles * (tid + 1)) / nt;
    for (int t = t0; t < t1; t++)
        for (int r = 0; r < 4; r++) {
            int m = 4 * t + r;
            signed char *dst0 = row_quad0(a->Ap, a->K, m);
            if (m >= a->M) zero_pad_row(a->K, dst0);
            else quant_pack_row(a->A + (size_t)m * a->lda, a->K, dst0, a->scales + m);
        }
}

int p2_quant_prepare(QuantA *qa, int M, int K)
{
    if (use_exact() || K % 4 != 0) return -1;
    int Mp = (M + 3) / 4 * 4;
    I8Bufs *b = get_i8_bufs((size_t)Mp * K, Mp, 0);
    if (!b) return -1;
    qa->q = b->a; qa->scales = b->a_scales; qa->M = M; qa->K = K;
    for (int m = M; m < Mp; m++) zero_pad_row(K, row_quad0(b->a, K, m));
    return 0;
}

void p2_quant_row(const QuantA *qa, int m, const float *row)
{
    quant_pack_row(row, qa->K, row_quad0(qa->q, qa->K, m), qa->scales + m);
}

/* Sixteen columns of one five-value row → S (sign) and T (sign · flag) as int8 in {−1, 0, +1}. */
static inline void fv_row16_i8(const FiveValueB *b, int N, int k, int col, int8x16_t *S, int8x16_t *T)
{
    const uint8x16_t rep4 = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
    const int8x16_t  sh2  = {0, -2, -4, -6, 0, -2, -4, -6, 0, -2, -4, -6, 0, -2, -4, -6};
    const uint8x16_t rep8 = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1};
    const int8x16_t  sh1  = {0, -1, -2, -3, -4, -5, -6, -7, 0, -1, -2, -3, -4, -5, -6, -7};
    uint32_t s32; uint16_t f16;
    memcpy(&s32, b->signs + (size_t)k * (N / 4) + (col >> 2), 4);
    memcpy(&f16, b->flags + (size_t)k * (N / 8) + (col >> 3), 2);
    uint8x16_t sb = vqtbl1q_u8(vreinterpretq_u8_u32(vdupq_n_u32(s32)), rep4);
    int8x16_t sgn = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshlq_u8(sb, sh2), vdupq_n_u8(3))), vdupq_n_s8(1));
    uint8x16_t fb = vqtbl1q_u8(vreinterpretq_u8_u16(vdupq_n_u16(f16)), rep8);
    int8x16_t flg = vreinterpretq_s8_u8(vandq_u8(vshlq_u8(fb, sh1), vdupq_n_u8(1)));
    *S = sgn;
    *T = vandq_s8(sgn, vnegq_s8(flg));   /* flag 1 → mask 0xFF keeps the sign, flag 0 → 0 */
}

/* Rows [k0, k0 + kb) of one 16-column five-value panel → S and T planes, [kb/4 quads][16 cols][4 k]. */
static void pack_fv_panel(const FiveValueB *b, int N, int k0, int kb, int col, signed char *S, signed char *T)
{
    for (int kq = 0; kq < kb / 4; kq++) {
        int k = k0 + 4 * kq;
        int8x16x4_t s4, t4;
        fv_row16_i8(b, N, k,     col, &s4.val[0], &t4.val[0]);
        fv_row16_i8(b, N, k + 1, col, &s4.val[1], &t4.val[1]);
        fv_row16_i8(b, N, k + 2, col, &s4.val[2], &t4.val[2]);
        fv_row16_i8(b, N, k + 3, col, &s4.val[3], &t4.val[3]);
        vst4q_s8(S + (size_t)kq * 64, s4);
        vst4q_s8(T + (size_t)kq * 64, t4);
    }
}

/* The same rows as one plane of int8 weight codes: sign · (flag ? 127 : clo[n]), clo[n] = round(127 · lo[n] / hi[n]),
 * to be scaled by hi[n] / 127. lo lands on a 127-step grid of hi (≤ hi/254 off); the signs and hi stay exact. */
static void pack_fv_codes_panel(const FiveValueB *b, int N, int k0, int kb, int col, int8x16_t clo, signed char *P)
{
    const int8x16_t v127 = vdupq_n_s8(127);
    for (int kq = 0; kq < kb / 4; kq++) {
        int k = k0 + 4 * kq;
        int8x16x4_t c4;
        for (int i = 0; i < 4; i++) {
            int8x16_t s, t;
            fv_row16_i8(b, N, k + i, col, &s, &t);
            /* t = s where the flag is set: a nonzero t marks a hi weight */
            uint8x16_t is_hi = vtstq_s8(t, t);
            int8x16_t mag = vbslq_s8(is_hi, v127, clo);
            c4.val[i] = vmulq_s8(s, mag);
        }
        vst4q_s8(P + (size_t)kq * 64, c4);
    }
}

/* Rows [k0, k0 + kb) of one 16-column int8 panel (row stride ldb). */
static void pack_q8_panel(const signed char *B, int ldb, int k0, int kb, int col, signed char *P)
{
    const signed char *src = B + (size_t)k0 * ldb + col;
    for (int kq = 0; kq < kb / 4; kq++) {
        int8x16x4_t v;
        v.val[0] = vld1q_s8(src + (size_t)(4 * kq)     * ldb);
        v.val[1] = vld1q_s8(src + (size_t)(4 * kq + 1) * ldb);
        v.val[2] = vld1q_s8(src + (size_t)(4 * kq + 2) * ldb);
        v.val[3] = vld1q_s8(src + (size_t)(4 * kq + 3) * ldb);
        vst4q_s8(P + (size_t)kq * 64, v);
    }
}

/* One 4 × 16 tile over the whole K: acc[r][j] holds columns 4j..4j+3 of row r. */
static inline void kernel_i8_4x16(const signed char *Ap, const signed char *Bp, int Kq, int32x4_t acc[4][4])
{
    for (int r = 0; r < 4; r++)
        for (int j = 0; j < 4; j++) acc[r][j] = vdupq_n_s32(0);
    for (int kq = 0; kq < Kq; kq++) {
        int8x16_t a  = vld1q_s8(Ap + (size_t)kq * 16);
        const signed char *bp = Bp + (size_t)kq * 64;
        int8x16_t b0 = vld1q_s8(bp), b1 = vld1q_s8(bp + 16), b2 = vld1q_s8(bp + 32), b3 = vld1q_s8(bp + 48);
        acc[0][0] = vdotq_laneq_s32(acc[0][0], b0, a, 0);
        acc[0][1] = vdotq_laneq_s32(acc[0][1], b1, a, 0);
        acc[0][2] = vdotq_laneq_s32(acc[0][2], b2, a, 0);
        acc[0][3] = vdotq_laneq_s32(acc[0][3], b3, a, 0);
        acc[1][0] = vdotq_laneq_s32(acc[1][0], b0, a, 1);
        acc[1][1] = vdotq_laneq_s32(acc[1][1], b1, a, 1);
        acc[1][2] = vdotq_laneq_s32(acc[1][2], b2, a, 1);
        acc[1][3] = vdotq_laneq_s32(acc[1][3], b3, a, 1);
        acc[2][0] = vdotq_laneq_s32(acc[2][0], b0, a, 2);
        acc[2][1] = vdotq_laneq_s32(acc[2][1], b1, a, 2);
        acc[2][2] = vdotq_laneq_s32(acc[2][2], b2, a, 2);
        acc[2][3] = vdotq_laneq_s32(acc[2][3], b3, a, 2);
        acc[3][0] = vdotq_laneq_s32(acc[3][0], b0, a, 3);
        acc[3][1] = vdotq_laneq_s32(acc[3][1], b1, a, 3);
        acc[3][2] = vdotq_laneq_s32(acc[3][2], b2, a, 3);
        acc[3][3] = vdotq_laneq_s32(acc[3][3], b3, a, 3);
    }
}

struct sgemm_i8_args {
    int M, N, K;
    float alpha;
    const signed char *Ap; const float *a_scales;
    float *C; int ldc;
    const FiveValueB *fv;                                   /* five-value */
    int codes;                                              /* five-value as one plane of int8 codes */
    const signed char *q8; const float *scales; int ldb;    /* int8 table: one plane, its column scale */
    int store_first;                                        /* beta == 0: the first K block stores into C, no memset */
    atomic_int next_panel;                                  /* dynamic scheduling: the next 16-column panel to take */
};

/* Threads take panels from a shared counter rather than a static range: on a chip with performance and efficiency
 * cores an equal split waits for the slowest core. Within a panel, K runs in KC_I8 blocks whose packed rows stay
 * in L1 while every A tile passes over them; each block's int32 sums are scaled and added into C. */
static void sgemm_i8_worker(int tid, int nthreads, void *aux)
{
    (void)tid; (void)nthreads;
    struct sgemm_i8_args *a = (struct sgemm_i8_args *)aux;
    int M = a->M, N = a->N, K = a->K, ldc = a->ldc;
    int tiles = (M + 3) / 4, Kq_total = K / 4;
    int panels_total = N / NR;

    size_t plane_bytes = (size_t)(KC_I8 / 4) * 64;
    I8Bufs *bufs = get_i8_bufs(0, 0, 2 * plane_bytes);
    if (!bufs) { fprintf(stderr, "sgemm_i8: out of memory\n"); abort(); }
    signed char *S = bufs->b, *T = bufs->b + plane_bytes;

    int p;
    while ((p = atomic_fetch_add(&a->next_panel, 1)) < panels_total) {
        int col = p * NR;

        /* Per-column epilogue factors: lo and hi − lo (two planes), hi / 127 (codes), or the int8 table's scale. */
        float32x4_t c1[4], c2[4];
        int8x16_t clo = vdupq_n_s8(0);
        if (a->fv && a->codes) {
            signed char clo_b[16];
            for (int j = 0; j < NR; j++) {
                float lo = a->fv->lo[col + j], hi = a->fv->hi[col + j];
                int c = hi > 0.0f ? (int)lrintf(127.0f * lo / hi) : 0;
                clo_b[j] = (signed char)(c < 0 ? 0 : c > 127 ? 127 : c);
            }
            clo = vld1q_s8(clo_b);
            for (int j = 0; j < 4; j++)
                c1[j] = vmulq_n_f32(vld1q_f32(a->fv->hi + col + 4 * j), 1.0f / 127.0f);
        } else if (a->fv) {
            for (int j = 0; j < 4; j++) {
                c1[j] = vld1q_f32(a->fv->lo + col + 4 * j);
                c2[j] = vsubq_f32(vld1q_f32(a->fv->hi + col + 4 * j), c1[j]);
            }
        } else {
            for (int j = 0; j < 4; j++) c1[j] = vld1q_f32(a->scales + col + 4 * j);
        }
        int two_planes = a->fv && !a->codes;

        for (int kc = 0; kc < K; kc += KC_I8) {
            int kb = (kc + KC_I8 <= K) ? KC_I8 : (K - kc);
            int Kq = kb / 4;
            const signed char *Bp = S;
            if (a->fv && a->codes && a->fv->codes)
                Bp = a->fv->codes + ((size_t)p * K + kc) * NR;   /* pre-packed: nothing to expand */
            else if (a->fv && a->codes) pack_fv_codes_panel(a->fv, N, kc, kb, col, clo, S);
            else if (a->fv)             pack_fv_panel(a->fv, N, kc, kb, col, S, T);
            else                        pack_q8_panel(a->q8, a->ldb, kc, kb, col, S);

            for (int t = 0; t < tiles; t++) {
                const signed char *Ap = a->Ap + ((size_t)t * Kq_total + kc / 4) * 16;
                int32x4_t accS[4][4], accT[4][4];
                kernel_i8_4x16(Ap, Bp, Kq, accS);
                if (two_planes) kernel_i8_4x16(Ap, T, Kq, accT);
                int store = a->store_first && kc == 0;
                for (int r = 0; r < 4; r++) {
                    int m = 4 * t + r;
                    if (m >= M) break;
                    float32x4_t as = vdupq_n_f32(a->a_scales[m] * a->alpha);
                    float *crow = a->C + (size_t)m * ldc + col;
                    for (int j = 0; j < 4; j++) {
                        float32x4_t f = vmulq_f32(vcvtq_f32_s32(accS[r][j]), c1[j]);
                        if (two_planes) f = vfmaq_f32(f, vcvtq_f32_s32(accT[r][j]), c2[j]);
                        if (store) vst1q_f32(crow + 4 * j, vmulq_f32(f, as));
                        else       vst1q_f32(crow + 4 * j, vfmaq_f32(vld1q_f32(crow + 4 * j), f, as));
                    }
                }
            }
        }
    }
}

struct pack_codes_args { const FiveValueB *b; int K, N; signed char *dst; atomic_int next; };

static void pack_codes_worker(int tid, int nthreads, void *aux)
{
    (void)tid; (void)nthreads;
    struct pack_codes_args *a = (struct pack_codes_args *)aux;
    int panels = a->N / NR, p;
    while ((p = atomic_fetch_add(&a->next, 1)) < panels) {
        int col = p * NR;
        signed char clo_b[16];
        for (int j = 0; j < NR; j++) {
            float lo = a->b->lo[col + j], hi = a->b->hi[col + j];
            int c = hi > 0.0f ? (int)lrintf(127.0f * lo / hi) : 0;
            clo_b[j] = (signed char)(c < 0 ? 0 : c > 127 ? 127 : c);
        }
        pack_fv_codes_panel(a->b, a->N, 0, a->K, col, vld1q_s8(clo_b), a->dst + (size_t)p * a->K * NR);
    }
}

signed char *p2_fv_pack_codes(const FiveValueB *b, int K, int N)
{
    if (K % 4 != 0 || N % NR != 0) return NULL;
    signed char *codes = (signed char *)malloc((size_t)K * N);
    if (!codes) return NULL;
    struct pack_codes_args a = { b, K, N, codes, 0 };
    atomic_init(&a.next, 0);
    p2_parallel(p2_pool(), pack_codes_worker, &a);
    return codes;
}

/* The int8 GEMM on already quantised A: C = alpha · A · B + beta · C with beta ∈ {0, 1} (0 stores, 1 accumulates;
 * other values need scale_c first). */
static void sgemm_i8_run(int M, int N, int K, float alpha, const signed char *Ap, const float *a_scales,
                         const FiveValueB *fv, const signed char *q8, const float *scales, int ldb,
                         int beta_zero, float *C, int ldc)
{
    struct sgemm_i8_args args = { M, N, K, alpha, Ap, a_scales, C, ldc,
                                  fv, current_mode() == GEMM_INT8W, q8, scales, ldb, beta_zero, 0 };
    atomic_init(&args.next_panel, 0);
    p2_parallel(p2_pool(), sgemm_i8_worker, &args);
}

/* Quantise A on the pool, then run the int8 GEMM. Returns 0, or -1 when the shape is not supported. */
static int sgemm_i8(int M, int N, int K, float alpha, const float *A, int lda,
                    const FiveValueB *fv, const signed char *q8, const float *scales, int ldb,
                    int beta_zero, float *C, int ldc)
{
    if (K % 4 != 0) return -1;
    int Mp = (M + 3) / 4 * 4;
    I8Bufs *bufs = get_i8_bufs((size_t)Mp * K, Mp, 0);
    if (!bufs) return -1;

    struct quant_args qa = { A, M, K, lda, bufs->a, bufs->a_scales };
    p2_parallel(p2_pool(), quant_rows_worker, &qa);
    sgemm_i8_run(M, N, K, alpha, bufs->a, bufs->a_scales, fv, q8, scales, ldb, beta_zero, C, ldc);
    return 0;
}

void p2_sgemm_fvb_q8a(int M, int N, int K, float alpha, const QuantA *A, const FiveValueB *B,
                         float beta, float *C, int ldc)
{
    require_panel_aligned(N, "p2_sgemm_fvb_q8a");
    if (alpha == 0.0f) { scale_c(C, M, N, ldc, beta); return; }
    if (beta != 0.0f && beta != 1.0f) scale_c(C, M, N, ldc, beta);
    sgemm_i8_run(M, N, K, alpha, A->q, A->scales, B, NULL, NULL, 0, beta == 0.0f, C, ldc);
}

void p2_sgemm_q8b_q8a(int M, int N, int K, float alpha, const QuantA *A, const signed char *B, const float *scales,
                         int ldb, float beta, float *C, int ldc)
{
    require_panel_aligned(N, "p2_sgemm_q8b_q8a");
    if (alpha == 0.0f) { scale_c(C, M, N, ldc, beta); return; }
    if (beta != 0.0f && beta != 1.0f) scale_c(C, M, N, ldc, beta);
    sgemm_i8_run(M, N, K, alpha, A->q, A->scales, NULL, B, scales, ldb, beta == 0.0f, C, ldc);
}

#else /* !P2_DOTPROD: no int8 path; prepare says so and nothing else is ever called */

signed char *p2_fv_pack_codes(const FiveValueB *b, int K, int N) { (void)b; (void)K; (void)N; return NULL; }
int p2_quant_prepare(QuantA *qa, int M, int K) { (void)qa; (void)M; (void)K; return -1; }
void p2_quant_row(const QuantA *qa, int m, const float *row) { (void)qa; (void)m; (void)row; abort(); }
void p2_sgemm_fvb_q8a(int M, int N, int K, float alpha, const QuantA *A, const FiveValueB *B,
                         float beta, float *C, int ldc)
{ (void)M; (void)N; (void)K; (void)alpha; (void)A; (void)B; (void)beta; (void)C; (void)ldc; abort(); }
void p2_sgemm_q8b_q8a(int M, int N, int K, float alpha, const QuantA *A, const signed char *B, const float *scales,
                         int ldb, float beta, float *C, int ldc)
{ (void)M; (void)N; (void)K; (void)alpha; (void)A; (void)B; (void)scales; (void)ldb; (void)beta; (void)C; (void)ldc; abort(); }

#endif /* P2_DOTPROD */

void p2_sgemm_q8b(int M, int N, int K,
                     float alpha, const float *A, int lda,
                     const signed char *B, const float *scales, int ldb,
                     float beta, float *C, int ldc)
{
    require_panel_aligned(N, "p2_sgemm_q8b");
    if (alpha == 0.0f) { scale_c(C, M, N, ldc, beta); return; }

#if P2_DOTPROD
    if (!use_exact() && K % 4 == 0) {
        if (beta != 0.0f && beta != 1.0f) scale_c(C, M, N, ldc, beta);
        if (sgemm_i8(M, N, K, alpha, A, lda, NULL, B, scales, ldb, beta == 0.0f, C, ldc) == 0) return;
    }
#endif
    scale_c(C, M, N, ldc, beta);
    struct sgemm_packed_args args = { M, N, K, alpha, A, lda, C, ldc, B, scales, ldb, NULL };
    sgemm_packed(&args);
}

void p2_sgemm_fvb(int M, int N, int K,
                     float alpha, const float *A, int lda,
                     const FiveValueB *B,
                     float beta, float *C, int ldc)
{
    require_panel_aligned(N, "p2_sgemm_fvb");
    pthread_once(&lut_once, lut_init);
    if (alpha == 0.0f) { scale_c(C, M, N, ldc, beta); return; }

#if P2_DOTPROD
    if (!use_exact() && K % 4 == 0) {
        if (beta != 0.0f && beta != 1.0f) scale_c(C, M, N, ldc, beta);
        if (sgemm_i8(M, N, K, alpha, A, lda, B, NULL, NULL, 0, beta == 0.0f, C, ldc) == 0) return;
    }
#endif
    scale_c(C, M, N, ldc, beta);
    struct sgemm_packed_args args = { M, N, K, alpha, A, lda, C, ldc, NULL, NULL, 0, B };
    sgemm_packed(&args);
}
