/*  tensor_ops.c – Elementwise and reduction kernels, plus the weight-kind GEMM dispatch.
 *
 *  Hot ops (LayerNorm, softmax, depthwise conv, Swish) get explicit NEON / AVX2 inner loops and a batched "_rows"
 *  variant that parallelises across rows through the thread pool. Scalar fallbacks are always provided.
 *
 *  The SIMD expf uses a 5-term polynomial around 0 with a 2^k range reduction; ~1e-6 relative, well inside float32
 *  for softmax and sigmoid.
 */

#include "phonon2.h"
#include "blas.h"
#include "threadpool.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#if (defined(__aarch64__) || defined(__ARM_NEON)) && !defined(P2_SCALAR)
  #include <arm_neon.h>
  #define P2_NEON 1
#endif

#if defined(__AVX2__) && defined(__FMA__) && !defined(P2_SCALAR)
  #include <immintrin.h>
  #define P2_AVX2 1
#endif

/* ── Weight GEMM dispatch ────────────────────────────────────────────────── */

void p2_wmatmul(const float *A, const Weight *W, float *C, int M, float alpha, float beta)
{
    int K = W->K, N = W->N;
    switch (W->kind) {
    case P2_W_FV: {
        FiveValueB b = { W->signs, W->flags, W->lo, W->hi, W->codes };
        p2_sgemm_fvb(M, N, K, alpha, A, K, &b, beta, C, N);
        break;
    }
    case P2_W_I8:
        p2_sgemm_q8b(M, N, K, alpha, A, K, W->q8, W->scales, N, beta, C, N);
        break;
    default:
        p2_sgemm(P2_NO_TRANS, P2_NO_TRANS, M, N, K, alpha, A, K, W->f32, N, beta, C, N);
        break;
    }
}

void p2_wmatmul_q8a(const QuantA *A, const Weight *W, float *C, float alpha, float beta)
{
    switch (W->kind) {
    case P2_W_FV: {
        FiveValueB b = { W->signs, W->flags, W->lo, W->hi, W->codes };
        p2_sgemm_fvb_q8a(A->M, W->N, W->K, alpha, A, &b, beta, C, W->N);
        break;
    }
    case P2_W_I8:
        p2_sgemm_q8b_q8a(A->M, W->N, W->K, alpha, A, W->q8, W->scales, W->N, beta, C, W->N);
        break;
    default:
        fprintf(stderr, "p2_wmatmul_q8a: fp32 weight has no int8 path\n");
        abort();
    }
}

/* ── Row GEMV: y = W x ──────────────────────────────────────────────────── */

static inline float dot_f32(const float *a, const float *b, int n)
{
    int i = 0;
#if P2_NEON
    float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f), acc3 = vdupq_n_f32(0.0f);
    for (; i + 16 <= n; i += 16) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(a + i),      vld1q_f32(b + i));
        acc1 = vfmaq_f32(acc1, vld1q_f32(a + i + 4),  vld1q_f32(b + i + 4));
        acc2 = vfmaq_f32(acc2, vld1q_f32(a + i + 8),  vld1q_f32(b + i + 8));
        acc3 = vfmaq_f32(acc3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }
    float s = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
#elif P2_AVX2
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    for (; i + 16 <= n; i += 16) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),     _mm256_loadu_ps(b + i),     acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), acc1);
    }
    __m256 acc = _mm256_add_ps(acc0, acc1);
    __m128 lo = _mm256_castps256_ps128(acc), hi = _mm256_extractf128_ps(acc, 1);
    __m128 s4 = _mm_add_ps(lo, hi);
    __m128 s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    float s = _mm_cvtss_f32(_mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1)));
#else
    float s = 0.0f;
#endif
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}

float p2_dot_f32(const float *a, const float *b, int n)
{
    return dot_f32(a, b, n);
}

struct gemv_args { const float *W; const float *x; float *y; int M; int K; };

static void gemv_worker(int tid, int nt, void *aux)
{
    struct gemv_args *a = (struct gemv_args *)aux;
    int first = (int)(((long long)a->M * tid) / nt);
    int last  = (int)(((long long)a->M * (tid + 1)) / nt);
    for (int r = first; r < last; r++)
        a->y[r] = dot_f32(a->W + (size_t)r * a->K, a->x, a->K);
}

/* Below this many multiply-adds the fork-join costs more than it saves. */
#define P2_GEMV_PARALLEL_MIN (2u << 20)

void p2_gemv_rows(const float *W, const float *x, float *y, int M, int K)
{
    struct gemv_args args = { W, x, y, M, K };
    if ((size_t)M * (size_t)K >= P2_GEMV_PARALLEL_MIN)
        p2_parallel(p2_pool(), gemv_worker, &args);
    else
        gemv_worker(0, 1, &args);
}


/* ── SIMD expf approximation ─────────────────────────────────────────────── */

#if P2_NEON
static inline float32x4_t p2_exp_neon(float32x4_t x)
{
    x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(-87.0f)), vdupq_n_f32(87.0f));

    const float32x4_t log2e = vdupq_n_f32(1.44269504088896341f);
    const float32x4_t ln2   = vdupq_n_f32(0.69314718055994529f);

    float32x4_t kf = vrndnq_f32(vmulq_f32(x, log2e));
    int32x4_t   ki = vcvtq_s32_f32(kf);
    float32x4_t r  = vsubq_f32(x, vmulq_f32(kf, ln2));

    float32x4_t y = vdupq_n_f32(1.0f / 120.0f);
    y = vfmaq_f32(vdupq_n_f32(1.0f / 24.0f),  y, r);
    y = vfmaq_f32(vdupq_n_f32(1.0f / 6.0f),   y, r);
    y = vfmaq_f32(vdupq_n_f32(0.5f),          y, r);
    y = vfmaq_f32(vdupq_n_f32(1.0f),          y, r);
    y = vfmaq_f32(vdupq_n_f32(1.0f),          y, r);

    int32x4_t pow2k = vshlq_n_s32(vaddq_s32(ki, vdupq_n_s32(127)), 23);
    return vmulq_f32(y, vreinterpretq_f32_s32(pow2k));
}
#endif

#if P2_AVX2
static inline __m256 p2_exp_avx2(__m256 x)
{
    x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-87.0f)), _mm256_set1_ps(87.0f));

    const __m256 log2e = _mm256_set1_ps(1.44269504088896341f);
    const __m256 ln2   = _mm256_set1_ps(0.69314718055994529f);

    __m256  kf = _mm256_round_ps(_mm256_mul_ps(x, log2e), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256i ki = _mm256_cvtps_epi32(kf);
    __m256  r  = _mm256_sub_ps(x, _mm256_mul_ps(kf, ln2));

    __m256 y = _mm256_set1_ps(1.0f / 120.0f);
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(1.0f / 24.0f));
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(1.0f / 6.0f));
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(0.5f));
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(1.0f));
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(1.0f));

    __m256i pow2k = _mm256_slli_epi32(_mm256_add_epi32(ki, _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(y, _mm256_castsi256_ps(pow2k));
}
#endif

#if P2_NEON
static inline float p2_hsum_neon(float32x4_t v) { return vaddvq_f32(v); }
static inline float p2_hmax_neon(float32x4_t v) { return vmaxvq_f32(v); }
#endif
#if P2_AVX2
static inline float p2_hsum_avx2(__m256 v)
{
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s4 = _mm_add_ps(lo, hi);
    __m128 s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    __m128 s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));
    return _mm_cvtss_f32(s1);
}
static inline float p2_hmax_avx2(__m256 v)
{
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 m4 = _mm_max_ps(lo, hi);
    __m128 m2 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
    __m128 m1 = _mm_max_ss(m2, _mm_shuffle_ps(m2, m2, 1));
    return _mm_cvtss_f32(m1);
}
#endif

/* ── Layer normalization (eps 1e-5, as nn.LayerNorm) ─────────────────────── */

void p2_layer_norm(const float *x, const float *gamma, const float *beta_,
                   float *out, int len)
{
#if P2_NEON
    float32x4_t vsum = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 4 <= len; i += 4) vsum = vaddq_f32(vsum, vld1q_f32(x + i));
    float sum = p2_hsum_neon(vsum);
    for (; i < len; i++) sum += x[i];
    float mean = sum / (float)len;

    float32x4_t vmean = vdupq_n_f32(mean);
    float32x4_t vvar  = vdupq_n_f32(0.0f);
    i = 0;
    for (; i + 4 <= len; i += 4) {
        float32x4_t d = vsubq_f32(vld1q_f32(x + i), vmean);
        vvar = vfmaq_f32(vvar, d, d);
    }
    float var_sum = p2_hsum_neon(vvar);
    for (; i < len; i++) { float d = x[i] - mean; var_sum += d * d; }
    float var = var_sum / (float)len;
    float inv_std = 1.0f / sqrtf(var + 1e-5f);

    float32x4_t vinv = vdupq_n_f32(inv_std);
    i = 0;
    for (; i + 4 <= len; i += 4) {
        float32x4_t xv = vsubq_f32(vld1q_f32(x + i), vmean);
        float32x4_t g  = vld1q_f32(gamma + i);
        float32x4_t b  = vld1q_f32(beta_ + i);
        vst1q_f32(out + i, vfmaq_f32(b, vmulq_f32(xv, vinv), g));
    }
    for (; i < len; i++)
        out[i] = (x[i] - mean) * inv_std * gamma[i] + beta_[i];

#elif P2_AVX2
    __m256 vsum = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= len; i += 8) vsum = _mm256_add_ps(vsum, _mm256_loadu_ps(x + i));
    float sum = p2_hsum_avx2(vsum);
    for (; i < len; i++) sum += x[i];
    float mean = sum / (float)len;

    __m256 vmean = _mm256_set1_ps(mean);
    __m256 vvar  = _mm256_setzero_ps();
    i = 0;
    for (; i + 8 <= len; i += 8) {
        __m256 d = _mm256_sub_ps(_mm256_loadu_ps(x + i), vmean);
        vvar = _mm256_fmadd_ps(d, d, vvar);
    }
    float var_sum = p2_hsum_avx2(vvar);
    for (; i < len; i++) { float d = x[i] - mean; var_sum += d * d; }
    float var = var_sum / (float)len;
    float inv_std = 1.0f / sqrtf(var + 1e-5f);

    __m256 vinv = _mm256_set1_ps(inv_std);
    i = 0;
    for (; i + 8 <= len; i += 8) {
        __m256 xv = _mm256_sub_ps(_mm256_loadu_ps(x + i), vmean);
        __m256 g  = _mm256_loadu_ps(gamma + i);
        __m256 b  = _mm256_loadu_ps(beta_ + i);
        _mm256_storeu_ps(out + i, _mm256_fmadd_ps(_mm256_mul_ps(xv, vinv), g, b));
    }
    for (; i < len; i++)
        out[i] = (x[i] - mean) * inv_std * gamma[i] + beta_[i];

#else
    double mean_d = 0.0;
    for (int i = 0; i < len; i++) mean_d += x[i];
    float mean = (float)(mean_d / len);

    double var_d = 0.0;
    for (int i = 0; i < len; i++) { double d = x[i] - mean; var_d += d * d; }
    float inv_std = 1.0f / sqrtf((float)(var_d / len) + 1e-5f);

    for (int i = 0; i < len; i++)
        out[i] = (x[i] - mean) * inv_std * gamma[i] + beta_[i];
#endif
}

struct ln_args {
    const float *x; const float *gamma; const float *beta_;
    float *out; int n_rows; int D;
};

static void ln_worker(int tid, int nt, void *aux)
{
    struct ln_args *a = (struct ln_args *)aux;
    int first = (int)(((long long)a->n_rows * tid) / nt);
    int last  = (int)(((long long)a->n_rows * (tid + 1)) / nt);
    for (int r = first; r < last; r++)
        p2_layer_norm(a->x + (size_t)r * a->D, a->gamma, a->beta_,
                      a->out + (size_t)r * a->D, a->D);
}

void p2_layer_norm_rows(const float *x, const float *gamma, const float *beta_,
                        float *out, int n_rows, int D)
{
    struct ln_args args = {x, gamma, beta_, out, n_rows, D};
    p2_parallel(p2_pool(), ln_worker, &args);
}

struct ln_ip_args {
    float *x; const float *gamma; const float *beta_; int n_rows; int D;
};

static void ln_ip_worker(int tid, int nt, void *aux)
{
    struct ln_ip_args *a = (struct ln_ip_args *)aux;
    int first = (int)(((long long)a->n_rows * tid) / nt);
    int last  = (int)(((long long)a->n_rows * (tid + 1)) / nt);
    float row_buf[P2_D_MODEL];
    for (int r = first; r < last; r++) {
        float *row = a->x + (size_t)r * a->D;
        p2_layer_norm(row, a->gamma, a->beta_, row_buf, a->D);
        memcpy(row, row_buf, a->D * sizeof(float));
    }
}

void p2_layer_norm_rows_inplace(float *x, const float *gamma, const float *beta_,
                                int n_rows, int D)
{
    struct ln_ip_args args = {x, gamma, beta_, n_rows, D};
    p2_parallel(p2_pool(), ln_ip_worker, &args);
}

/* LayerNorm straight into a QuantA: the normalised row only ever exists in a stack buffer. */
struct lnq_args {
    const float *x; const float *gamma; const float *beta_; const QuantA *qa; int n_rows; int D;
};

static void lnq_worker(int tid, int nt, void *aux)
{
    struct lnq_args *a = (struct lnq_args *)aux;
    int first = (int)(((long long)a->n_rows * tid) / nt);
    int last  = (int)(((long long)a->n_rows * (tid + 1)) / nt);
    float row_buf[P2_D_MODEL];
    for (int r = first; r < last; r++) {
        p2_layer_norm(a->x + (size_t)r * a->D, a->gamma, a->beta_, row_buf, a->D);
        p2_quant_row(a->qa, r, row_buf);
    }
}

void p2_layer_norm_quant_rows(const float *x, const float *gamma, const float *beta_, const QuantA *qa, int n_rows, int D)
{
    struct lnq_args args = {x, gamma, beta_, qa, n_rows, D};
    p2_parallel(p2_pool(), lnq_worker, &args);
}

/* ── Softmax ─────────────────────────────────────────────────────────────── */

void p2_softmax(float *x, int len)
{
#if P2_NEON
    float32x4_t vmx = vdupq_n_f32(x[0]);
    int i = 0;
    for (; i + 4 <= len; i += 4) vmx = vmaxq_f32(vmx, vld1q_f32(x + i));
    float mx = p2_hmax_neon(vmx);
    for (; i < len; i++) if (x[i] > mx) mx = x[i];

    float32x4_t vsum = vdupq_n_f32(0.0f);
    float32x4_t vmx4 = vdupq_n_f32(mx);
    i = 0;
    for (; i + 4 <= len; i += 4) {
        float32x4_t e = p2_exp_neon(vsubq_f32(vld1q_f32(x + i), vmx4));
        vst1q_f32(x + i, e);
        vsum = vaddq_f32(vsum, e);
    }
    float sum = p2_hsum_neon(vsum);
    for (; i < len; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }

    float32x4_t vinv = vdupq_n_f32(1.0f / sum);
    i = 0;
    for (; i + 4 <= len; i += 4) vst1q_f32(x + i, vmulq_f32(vld1q_f32(x + i), vinv));
    for (; i < len; i++) x[i] *= (1.0f / sum);

#elif P2_AVX2
    __m256 vmx = _mm256_set1_ps(x[0]);
    int i = 0;
    for (; i + 8 <= len; i += 8) vmx = _mm256_max_ps(vmx, _mm256_loadu_ps(x + i));
    float mx = p2_hmax_avx2(vmx);
    for (; i < len; i++) if (x[i] > mx) mx = x[i];

    __m256 vsum = _mm256_setzero_ps();
    __m256 vmx8 = _mm256_set1_ps(mx);
    i = 0;
    for (; i + 8 <= len; i += 8) {
        __m256 e = p2_exp_avx2(_mm256_sub_ps(_mm256_loadu_ps(x + i), vmx8));
        _mm256_storeu_ps(x + i, e);
        vsum = _mm256_add_ps(vsum, e);
    }
    float sum = p2_hsum_avx2(vsum);
    for (; i < len; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }

    __m256 vinv = _mm256_set1_ps(1.0f / sum);
    i = 0;
    for (; i + 8 <= len; i += 8)
        _mm256_storeu_ps(x + i, _mm256_mul_ps(_mm256_loadu_ps(x + i), vinv));
    for (; i < len; i++) x[i] *= (1.0f / sum);

#else
    float mx = x[0];
    for (int i = 1; i < len; i++) if (x[i] > mx) mx = x[i];
    float sum = 0.0f;
    for (int i = 0; i < len; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
    float inv = 1.0f / sum;
    for (int i = 0; i < len; i++) x[i] *= inv;
#endif
}

struct softmax_rows_args { float *x; int n_rows; int len; };

static void softmax_rows_worker(int tid, int nt, void *aux)
{
    struct softmax_rows_args *a = (struct softmax_rows_args *)aux;
    int first = (int)(((long long)a->n_rows * tid) / nt);
    int last  = (int)(((long long)a->n_rows * (tid + 1)) / nt);
    for (int r = first; r < last; r++)
        p2_softmax(a->x + (size_t)r * a->len, a->len);
}

void p2_softmax_rows(float *x, int n_rows, int len)
{
    struct softmax_rows_args args = {x, n_rows, len};
    p2_parallel(p2_pool(), softmax_rows_worker, &args);
}

/* ── Swish / SiLU: x · sigmoid(x) ────────────────────────────────────────── */

/* The scalar tails use p2_sigmoid (phonon2.h): clamped like p2_exp_*. The unclamped form here is what the x86 SSE
 * build turned into NaN under -ffast-math (verification.md, *Portability*); the NEON and AVX2 loops never saw it
 * because their lengths are multiples of the vector width, so these tails never ran. */
#define sigmoid_f p2_sigmoid

static void swish_range(float *x, int len)
{
#if P2_NEON
    const float32x4_t one = vdupq_n_f32(1.0f);
    int i = 0;
    for (; i + 4 <= len; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        float32x4_t e = p2_exp_neon(vnegq_f32(v));
        float32x4_t s = vdivq_f32(one, vaddq_f32(one, e));
        vst1q_f32(x + i, vmulq_f32(v, s));
    }
    for (; i < len; i++) x[i] *= sigmoid_f(x[i]);

#elif P2_AVX2
    const __m256 one = _mm256_set1_ps(1.0f);
    int i = 0;
    for (; i + 8 <= len; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        __m256 e = p2_exp_avx2(_mm256_sub_ps(_mm256_setzero_ps(), v));
        __m256 s = _mm256_div_ps(one, _mm256_add_ps(one, e));
        _mm256_storeu_ps(x + i, _mm256_mul_ps(v, s));
    }
    for (; i < len; i++) x[i] *= sigmoid_f(x[i]);

#else
    for (int i = 0; i < len; i++) x[i] *= sigmoid_f(x[i]);
#endif
}

struct swish_args { float *x; int len; };

static void swish_worker(int tid, int nt, void *aux)
{
    struct swish_args *a = (struct swish_args *)aux;
    /* 16-element-aligned ranges keep every thread on the SIMD path. */
    int chunks = (a->len + 15) / 16;
    int c0 = (int)(((long long)chunks * tid) / nt), c1 = (int)(((long long)chunks * (tid + 1)) / nt);
    int i0 = c0 * 16, i1 = c1 * 16 < a->len ? c1 * 16 : a->len;
    if (i0 < i1) swish_range(a->x + i0, i1 - i0);
}

/* The feed-forward hidden state is T × 4096: worth spreading over the pool. */
#define P2_SWISH_PARALLEL_MIN (1 << 15)

void p2_swish(float *x, int len)
{
    if (len >= P2_SWISH_PARALLEL_MIN) {
        struct swish_args a = { x, len };
        p2_parallel(p2_pool(), swish_worker, &a);
    } else {
        swish_range(x, len);
    }
}

/* SiLU of each row straight into a QuantA; the activated row only ever exists in a stack buffer. */
struct swq_args { const float *h; const QuantA *qa; int n_rows; int len; };

static void swq_worker(int tid, int nt, void *aux)
{
    struct swq_args *a = (struct swq_args *)aux;
    int first = (int)(((long long)a->n_rows * tid) / nt);
    int last  = (int)(((long long)a->n_rows * (tid + 1)) / nt);
    float row_buf[P2_FF_DIM];
    for (int r = first; r < last; r++) {
        memcpy(row_buf, a->h + (size_t)r * a->len, a->len * sizeof(float));
        swish_range(row_buf, a->len);
        p2_quant_row(a->qa, r, row_buf);
    }
}

void p2_swish_quant_rows(const float *h, const QuantA *qa, int n_rows, int len)
{
    struct swq_args args = { h, qa, n_rows, len };
    p2_parallel(p2_pool(), swq_worker, &args);
}

/* ── ReLU ────────────────────────────────────────────────────────────────── */

void p2_relu(float *x, int len)
{
#if P2_NEON
    const float32x4_t z = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 4 <= len; i += 4) vst1q_f32(x + i, vmaxq_f32(vld1q_f32(x + i), z));
    for (; i < len; i++) if (x[i] < 0.0f) x[i] = 0.0f;
#elif P2_AVX2
    const __m256 z = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= len; i += 8) _mm256_storeu_ps(x + i, _mm256_max_ps(_mm256_loadu_ps(x + i), z));
    for (; i < len; i++) if (x[i] < 0.0f) x[i] = 0.0f;
#else
    for (int i = 0; i < len; i++) if (x[i] < 0.0f) x[i] = 0.0f;
#endif
}

/* ── Bias add ────────────────────────────────────────────────────────────── */

void p2_bias_add(float *x, const float *b, int len)
{
#if P2_NEON
    int i = 0;
    for (; i + 4 <= len; i += 4)
        vst1q_f32(x + i, vaddq_f32(vld1q_f32(x + i), vld1q_f32(b + i)));
    for (; i < len; i++) x[i] += b[i];
#elif P2_AVX2
    int i = 0;
    for (; i + 8 <= len; i += 8)
        _mm256_storeu_ps(x + i, _mm256_add_ps(_mm256_loadu_ps(x + i), _mm256_loadu_ps(b + i)));
    for (; i < len; i++) x[i] += b[i];
#else
    for (int i = 0; i < len; i++) x[i] += b[i];
#endif
}

void p2_bias_add_rows(float *x, const float *b, int n_rows, int len)
{
    for (int r = 0; r < n_rows; r++)
        p2_bias_add(x + (size_t)r * len, b, len);
}

/* ── 1-D depthwise convolution (symmetric padding, parallel over channels) ── */

static void depthwise_conv1d_channel(const float *x, const float *w, float bias,
                                     float *out, int T, int K)
{
    int pad = (K - 1) / 2;

    for (int t = 0; t < pad && t < T; t++) {
        float val = bias;
        for (int k = 0; k < K; k++) {
            int src_t = t - pad + k;
            if (src_t >= 0 && src_t < T) val += x[src_t] * w[k];
        }
        out[t] = val;
    }

    int t_lo = pad;
    int t_hi = T - pad;
    if (t_hi < t_lo) t_hi = t_lo;

#if P2_NEON
    int t = t_lo;
    float32x4_t vbias = vdupq_n_f32(bias);
    for (; t + 4 <= t_hi; t += 4) {
        float32x4_t acc = vbias;
        for (int k = 0; k < K; k++)
            acc = vfmaq_n_f32(acc, vld1q_f32(x + t - pad + k), w[k]);
        vst1q_f32(out + t, acc);
    }
    for (; t < t_hi; t++) {
        float val = bias;
        for (int k = 0; k < K; k++) val += x[t - pad + k] * w[k];
        out[t] = val;
    }
#elif P2_AVX2
    int t = t_lo;
    __m256 vbias = _mm256_set1_ps(bias);
    for (; t + 8 <= t_hi; t += 8) {
        __m256 acc = vbias;
        for (int k = 0; k < K; k++)
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(x + t - pad + k), _mm256_set1_ps(w[k]), acc);
        _mm256_storeu_ps(out + t, acc);
    }
    for (; t < t_hi; t++) {
        float val = bias;
        for (int k = 0; k < K; k++) val += x[t - pad + k] * w[k];
        out[t] = val;
    }
#else
    for (int t = t_lo; t < t_hi; t++) {
        float val = bias;
        for (int k = 0; k < K; k++) val += x[t - pad + k] * w[k];
        out[t] = val;
    }
#endif

    for (int t = t_hi; t < T; t++) {
        if (t < pad) continue;
        float val = bias;
        for (int k = 0; k < K; k++) {
            int src_t = t - pad + k;
            if (src_t >= 0 && src_t < T) val += x[src_t] * w[k];
        }
        out[t] = val;
    }
}

struct dw_args {
    const float *x; const float *w; const float *b;
    float *out; int C; int T; int K;
};

static void dw_worker(int tid, int nt, void *aux)
{
    struct dw_args *a = (struct dw_args *)aux;
    int first = (int)(((long long)a->C * tid) / nt);
    int last  = (int)(((long long)a->C * (tid + 1)) / nt);
    for (int c = first; c < last; c++) {
        float bias = a->b ? a->b[c] : 0.0f;
        depthwise_conv1d_channel(a->x + (size_t)c * a->T,
                                 a->w + (size_t)c * a->K,
                                 bias,
                                 a->out + (size_t)c * a->T,
                                 a->T, a->K);
    }
}

void p2_depthwise_conv1d(const float *x, const float *w, const float *b,
                         float *out, int C, int T, int K)
{
    struct dw_args args = {x, w, b, out, C, T, K};
    p2_parallel(p2_pool(), dw_worker, &args);
}
