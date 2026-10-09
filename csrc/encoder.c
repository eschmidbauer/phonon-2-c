/*  encoder.c – FastConformer encoder forward pass.
 *
 *  mel [T × 128] → convolutional subsampling (8×) → 24 conformer blocks → encoder_projector → frames [T' × 640].
 *  Mirrors transformers' ParakeetEncoder for parakeet-tdt-0.6b-v3: no biases on the linears, no input scaling,
 *  relative-position attention with per-layer bias_u / bias_v, BatchNorm after the depthwise conv (folded into its
 *  weights by the converter).
 */

#include "phonon2.h"
#include "blas.h"
#include "threadpool.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdatomic.h>

#if (defined(__aarch64__) || defined(__ARM_NEON)) && !defined(P2_SCALAR)
  #include <arm_neon.h>
  #define P2_ENC_NEON 1
#endif
#if defined(__AVX2__) && defined(__FMA__) && !defined(P2_SCALAR)
  #include <immintrin.h>
  #define P2_ENC_AVX2 1
#endif

int p2_verbose = 1;

/* ── Helpers ─────────────────────────────────────────────────────────────── */

static float *alloc_f(size_t n)
{
    return (float *)calloc(n, sizeof(float));
}

/* Bump-allocator workspace for the conformer blocks: carved per sub-function, reset between them. */
typedef struct { float *buf; size_t cap; } CfmWS;
typedef struct { float *buf; size_t off; } WSA;

static float *ws_get(WSA *a, size_t n)
{
    float *p = a->buf + a->off;
    a->off += n;
    return p;
}

/* Peak workspace (floats) over the three sub-functions; self-attention dominates. */
static size_t conformer_ws_peak(int T)
{
    size_t D = P2_D_MODEL, H = P2_N_HEADS, Dh = P2_HEAD_DIM, FF = P2_FF_DIM;
    size_t t = (size_t)T, L = 2 * t - 1;

    size_t sa = 0;
    sa += t * D;             /* ln */
    sa += t * 3 * D;         /* QKV */
    sa += H * t * Dh * 2;    /* Q_u, Q_v */
    sa += L * D;             /* P */
    sa += H * t * Dh * 2;    /* K_t, V_h */
    sa += H * L * Dh;        /* P_t */
    sa += H * t * t;         /* ac (becomes the scores in place) */
    sa += H * t * L;         /* bd */
    sa += H * t * Dh;        /* attn_h */
    sa += t * D;             /* attn */
    sa += t * D;             /* out */

    size_t ff = t * D + t * FF + t * D;
    size_t cm = t * D + t * 2 * D + D * t + D * t + t * D + t * D;

    size_t peak = sa;
    if (ff > peak) peak = ff;
    if (cm > peak) peak = cm;
    return peak;
}

/* ── Relative positional encoding ────────────────────────────────────────── */

/* [2T-1 × D]: row j holds position (T-1-j); even columns sin, odd columns cos, with the frequency product taken in
 * float32 exactly as transformers' ParakeetEncoderRelPositionalEncoding does it. */
static float *rel_pos_encoding(int T)
{
    int D = P2_D_MODEL, L = 2 * T - 1;
    float *pe = (float *)malloc((size_t)L * D * sizeof(float));
    float inv[P2_D_MODEL / 2];
    for (int i = 0; i < D / 2; i++)
        inv[i] = 1.0f / powf(10000.0f, (float)(2 * i) / (float)D);
    for (int j = 0; j < L; j++) {
        float pos = (float)(T - 1 - j);
        float *row = pe + (size_t)j * D;
        for (int i = 0; i < D / 2; i++) {
            float f = inv[i] * pos;
            row[2 * i]     = sinf(f);
            row[2 * i + 1] = cosf(f);
        }
    }
    return pe;
}

/* ── Convolutional subsampling ───────────────────────────────────────────── */

/* Conv2d(1 → C, 3×3, stride 2, pad 1) on one [H × W] plane; parallel over output channels. */
struct conv_full_args { const float *x; int H, W, Ho, Wo; const float *w, *b; float *out; int C; };

static void conv3x3s2_full_worker(int tid, int nt, void *aux)
{
    struct conv_full_args *a = (struct conv_full_args *)aux;
    int c0 = (a->C * tid) / nt, c1 = (a->C * (tid + 1)) / nt;
    for (int c = c0; c < c1; c++) {
        const float *w = a->w + c * 9;
        float *out = a->out + (size_t)c * a->Ho * a->Wo;
        for (int oh = 0; oh < a->Ho; oh++) {
            for (int ow = 0; ow < a->Wo; ow++) {
                float acc = a->b[c];
                for (int kh = 0; kh < 3; kh++) {
                    int ih = 2 * oh - 1 + kh;
                    if (ih < 0 || ih >= a->H) continue;
                    const float *xr = a->x + (size_t)ih * a->W;
                    for (int kw = 0; kw < 3; kw++) {
                        int iw = 2 * ow - 1 + kw;
                        if (iw >= 0 && iw < a->W) acc += xr[iw] * w[kh * 3 + kw];
                    }
                }
                out[(size_t)oh * a->Wo + ow] = acc;
            }
        }
    }
}

/* Depthwise Conv2d(C, 3×3, stride 2, pad 1, groups = C) on [C × H × W]; parallel over channels. */
static void conv3x3s2_dw_worker(int tid, int nt, void *aux)
{
    struct conv_full_args *a = (struct conv_full_args *)aux;
    int c0 = (a->C * tid) / nt, c1 = (a->C * (tid + 1)) / nt;
    for (int c = c0; c < c1; c++) {
        const float *w = a->w + c * 9;
        const float *x = a->x + (size_t)c * a->H * a->W;
        float *out = a->out + (size_t)c * a->Ho * a->Wo;
        for (int oh = 0; oh < a->Ho; oh++) {
            for (int ow = 0; ow < a->Wo; ow++) {
                float acc = a->b[c];
                for (int kh = 0; kh < 3; kh++) {
                    int ih = 2 * oh - 1 + kh;
                    if (ih < 0 || ih >= a->H) continue;
                    const float *xr = x + (size_t)ih * a->W;
                    for (int kw = 0; kw < 3; kw++) {
                        int iw = 2 * ow - 1 + kw;
                        if (iw >= 0 && iw < a->W) acc += xr[iw] * w[kh * 3 + kw];
                    }
                }
                out[(size_t)oh * a->Wo + ow] = acc;
            }
        }
    }
}

static inline int conv_out(int n) { return (n + 2 - 3) / 2 + 1; }

static void conv3x3s2(const float *x, int C_in, int H, int W, const float *w, const float *b,
                      float *out, int C, int depthwise)
{
    struct conv_full_args args = { x, H, W, conv_out(H), conv_out(W), w, b, out, C };
    (void)C_in;
    p2_parallel(p2_pool(), depthwise ? conv3x3s2_dw_worker : conv3x3s2_full_worker, &args);
}

/* Pointwise Conv2d(C → C) is a GEMM over the flattened plane: out[C, HW] = W[C, C] @ x[C, HW] + b. */
static void conv1x1(const float *x, const float *w, const float *b, float *out, int C, int HW)
{
    p2_sgemm(P2_NO_TRANS, P2_NO_TRANS, C, HW, C, 1.0f, w, C, x, HW, 0.0f, out, HW);
    for (int c = 0; c < C; c++) {
        float *row = out + (size_t)c * HW;
        float bc = b[c];
        for (int i = 0; i < HW; i++) row[i] += bc;
    }
}

/*  Input:  mel [T × 128] time-major (one [H=T × W=128] plane)
 *  Output: x [T' × 1024], T' = ((T-1)/2+1 → three times)
 *
 *   Conv2d 1→256 k3 s2 p1 + ReLU → [256, T/2, 64]
 *   dw Conv2d k3 s2 + pw Conv2d + ReLU → [256, T/4, 32]
 *   dw Conv2d k3 s2 + pw Conv2d + ReLU → [256, T/8, 16]
 *   reshape [T', 256·16 = 4096] (channel-major within a frame) → Linear 4096→1024 + bias
 */
static int pre_encode(const SubsamplingWeights *w, const float *mel, int T, float **out_ptr)
{
    int C = P2_SUB_CHANNELS;
    int H1 = conv_out(T), W1 = conv_out(P2_N_MELS);
    float *s1 = alloc_f((size_t)C * H1 * W1);
    conv3x3s2(mel, 1, T, P2_N_MELS, w->conv0_w, w->conv0_b, s1, C, 0);
    p2_relu(s1, C * H1 * W1);

    int H2 = conv_out(H1), W2 = conv_out(W1);
    float *s2_dw = alloc_f((size_t)C * H2 * W2);
    conv3x3s2(s1, C, H1, W1, w->conv2_w, w->conv2_b, s2_dw, C, 1);
    free(s1);
    float *s2 = alloc_f((size_t)C * H2 * W2);
    conv1x1(s2_dw, w->conv3_w, w->conv3_b, s2, C, H2 * W2);
    free(s2_dw);
    p2_relu(s2, C * H2 * W2);

    int H3 = conv_out(H2), W3 = conv_out(W2);
    float *s3_dw = alloc_f((size_t)C * H3 * W3);
    conv3x3s2(s2, C, H2, W2, w->conv5_w, w->conv5_b, s3_dw, C, 1);
    free(s2);
    float *s3 = alloc_f((size_t)C * H3 * W3);
    conv1x1(s3_dw, w->conv6_w, w->conv6_b, s3, C, H3 * W3);
    free(s3_dw);
    p2_relu(s3, C * H3 * W3);

    int Tp = H3;
    int feat_dim = C * W3;
    float *flat = alloc_f((size_t)Tp * feat_dim);
    for (int c = 0; c < C; c++)
        for (int h = 0; h < H3; h++)
            memcpy(flat + (size_t)h * feat_dim + c * W3, s3 + ((size_t)c * H3 + h) * W3, W3 * sizeof(float));
    free(s3);

    float *x = alloc_f((size_t)Tp * P2_D_MODEL);
    p2_wmatmul(flat, &w->linear_w, x, Tp, 1.0f, 0.0f);
    free(flat);
    p2_bias_add_rows(x, w->linear_b, Tp, P2_D_MODEL);

    *out_ptr = x;
    return Tp;
}

/* ── Parallel-for over row ranges ────────────────────────────────────────── */

/* The layout shuffles between the GEMMs (head gathers, transposes, GLU, residual adds) are memory-bound and, done on
 * the calling thread, left the rest of the pool idle for a large share of a long-form pass. Each is a function over a
 * row range dispatched through here. */
typedef void (*RowFn)(int r0, int r1, void *arg);
struct rows_args { int n; RowFn fn; void *arg; };

static void rows_worker(int tid, int nt, void *aux)
{
    struct rows_args *a = (struct rows_args *)aux;
    int r0 = (int)(((long long)a->n * tid) / nt);
    int r1 = (int)(((long long)a->n * (tid + 1)) / nt);
    if (r0 < r1) a->fn(r0, r1, a->arg);
}

static void parallel_rows(int n, RowFn fn, void *arg)
{
    struct rows_args a = { n, fn, arg };
    p2_parallel(p2_pool(), rows_worker, &a);
}

/* x += alpha · y over a range of 1024-wide rows. */
struct axpy_args { float *x; const float *y; float alpha; };

static void axpy_rows(int r0, int r1, void *arg)
{
    struct axpy_args *a = (struct axpy_args *)arg;
    size_t i = (size_t)r0 * P2_D_MODEL, n = (size_t)r1 * P2_D_MODEL;
    float *x = a->x;
    const float *y = a->y;
#if P2_ENC_NEON
    float32x4_t va = vdupq_n_f32(a->alpha);
    for (; i + 4 <= n; i += 4)
        vst1q_f32(x + i, vfmaq_f32(vld1q_f32(x + i), vld1q_f32(y + i), va));
#elif P2_ENC_AVX2
    __m256 va = _mm256_set1_ps(a->alpha);
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(x + i, _mm256_fmadd_ps(_mm256_loadu_ps(y + i), va, _mm256_loadu_ps(x + i)));
#endif
    for (; i < n; i++)
        x[i] += a->alpha * y[i];
}

static void add_residual(float *x, const float *y, int T, float alpha)
{
    struct axpy_args a = { x, y, alpha };
    parallel_rows(T, axpy_rows, &a);
}

/* ── All heads' GEMMs in one dispatch ────────────────────────────────────── */

/* C[h] = A[h] @ B[h] (fp32, NN) for every head at once: work items are (head, row block) pairs taken from a
 * counter, each run through the single-threaded packed sgemm. One dispatch per head was 24 dispatches per block;
 * one head per thread left ten of eighteen cores idle. */
#define HEAD_ROWS 48

struct heads_args {
    int H, M, N, K;
    const float *A; size_t sA; int lda;
    const float *B; size_t sB; int ldb;
    float       *C; size_t sC; int ldc;
    int blocks;
    atomic_int next;
};

static void heads_worker(int tid, int nt, void *aux)
{
    (void)tid; (void)nt;
    struct heads_args *a = (struct heads_args *)aux;
    int item;
    while ((item = atomic_fetch_add(&a->next, 1)) < a->H * a->blocks) {
        int h = item / a->blocks, r0 = (item % a->blocks) * HEAD_ROWS;
        int rows = a->M - r0 < HEAD_ROWS ? a->M - r0 : HEAD_ROWS;
        p2_sgemm_st(P2_NO_TRANS, P2_NO_TRANS, rows, a->N, a->K, 1.0f,
                    a->A + h * a->sA + (size_t)r0 * a->lda, a->lda,
                    a->B + h * a->sB, a->ldb,
                    0.0f, a->C + h * a->sC + (size_t)r0 * a->ldc, a->ldc);
    }
}

static void heads_gemm(int H, int M, int N, int K,
                       const float *A, size_t sA, int lda,
                       const float *B, size_t sB, int ldb,
                       float *C, size_t sC, int ldc)
{
    struct heads_args a = { H, M, N, K, A, sA, lda, B, sB, ldb, C, sC, ldc, (M + HEAD_ROWS - 1) / HEAD_ROWS, 0 };
    atomic_init(&a.next, 0);
    p2_parallel(p2_pool(), heads_worker, &a);
}

/* The second QuantA of an int8 block: the first prepare having succeeded, the int8 path is committed, and a failure
 * here is an allocation failure for a larger buffer – there is no fp32 fallback half-way through a block. */
static void quant_prepare_or_die(QuantA *qa, int M, int K)
{
    if (p2_quant_prepare(qa, M, K) != 0) {
        fprintf(stderr, "encoder: out of memory for the int8 activation buffer (%d × %d)\n", M, K);
        abort();
    }
}

/* ── Feed-forward module ─────────────────────────────────────────────────── */

/* In the int8 modes the normalised and the activated rows are quantised as they are produced and never written as
 * fp32, and the residual is added in the GEMM epilogue. Exact mode keeps the reference arithmetic – separate buffers,
 * separate add – so its token-for-token parity with torch is untouched. */
static void feed_forward(const float *norm_w, const float *norm_b, const FFWeights *ff,
                         float *x, int T, CfmWS *ws)
{
    int D = P2_D_MODEL, FF = P2_FF_DIM;
    WSA a = {ws->buf, 0};
    float *h = ws_get(&a, (size_t)T * FF);

    QuantA qa;
    if (p2_quant_prepare(&qa, T, D) == 0) {
        p2_layer_norm_quant_rows(x, norm_w, norm_b, &qa, T, D);
        p2_wmatmul_q8a(&qa, &ff->linear1_w, h, 1.0f, 0.0f);
        QuantA qh;
        quant_prepare_or_die(&qh, T, FF);
        p2_swish_quant_rows(h, &qh, T, FF);
        p2_wmatmul_q8a(&qh, &ff->linear2_w, x, 0.5f, 1.0f);   /* x += ½ · ff(x) */
        return;
    }

    float *ln  = ws_get(&a, (size_t)T * D);
    float *out = ws_get(&a, (size_t)T * D);
    p2_layer_norm_rows(x, norm_w, norm_b, ln, T, D);
    p2_wmatmul(ln, &ff->linear1_w, h, T, 1.0f, 0.0f);
    p2_swish(h, T * FF);
    p2_wmatmul(h, &ff->linear2_w, out, T, 1.0f, 0.0f);
    add_residual(x, out, T, 0.5f);   /* the conformer's half-step feed-forward residual */
}

/* ── Attention pieces ────────────────────────────────────────────────────── */

/* Head gather for frames [t0, t1): Q_u = Q + bias_u and Q_v = Q + bias_v as [H, T, Dh]; K transposed into [H, Dh, T];
 * V as [H, T, Dh]. */
struct gather_args { const float *QKV; const float *bias_u, *bias_v; float *Q_u, *Q_v, *K_t, *V_h; int T; };

static void gather_heads(int t0, int t1, void *arg)
{
    struct gather_args *g = (struct gather_args *)arg;
    int D = P2_D_MODEL, H = P2_N_HEADS, Dh = P2_HEAD_DIM, T = g->T;
    for (int t = t0; t < t1; t++) {
        const float *row = g->QKV + (size_t)t * 3 * D;
        for (int h = 0; h < H; h++) {
            const float *q_src = row + h * Dh, *k_src = row + D + h * Dh, *v_src = row + 2 * D + h * Dh;
            const float *pu = g->bias_u + h * Dh, *pv = g->bias_v + h * Dh;
            float *qu = g->Q_u + ((size_t)h * T + t) * Dh;
            float *qv = g->Q_v + ((size_t)h * T + t) * Dh;
            float *kt = g->K_t + (size_t)h * Dh * T + t;
#if P2_ENC_NEON
            for (int d = 0; d < Dh; d += 4) {
                float32x4_t q = vld1q_f32(q_src + d);
                vst1q_f32(qu + d, vaddq_f32(q, vld1q_f32(pu + d)));
                vst1q_f32(qv + d, vaddq_f32(q, vld1q_f32(pv + d)));
            }
#elif P2_ENC_AVX2
            for (int d = 0; d < Dh; d += 8) {
                __m256 q = _mm256_loadu_ps(q_src + d);
                _mm256_storeu_ps(qu + d, _mm256_add_ps(q, _mm256_loadu_ps(pu + d)));
                _mm256_storeu_ps(qv + d, _mm256_add_ps(q, _mm256_loadu_ps(pv + d)));
            }
#else
            for (int d = 0; d < Dh; d++) {
                qu[d] = q_src[d] + pu[d];
                qv[d] = q_src[d] + pv[d];
            }
#endif
            for (int d = 0; d < Dh; d++)
                kt[(size_t)d * T] = k_src[d];
            memcpy(g->V_h + ((size_t)h * T + t) * Dh, v_src, Dh * sizeof(float));
        }
    }
}

/* P [2T-1, D] → P_t [H, Dh, 2T-1] for positions [p0, p1). */
struct ptrans_args { const float *P; float *P_t; int L; };

static void transpose_pos(int p0, int p1, void *arg)
{
    struct ptrans_args *a = (struct ptrans_args *)arg;
    int D = P2_D_MODEL, H = P2_N_HEADS, Dh = P2_HEAD_DIM, L = a->L;
    for (int p = p0; p < p1; p++)
        for (int h = 0; h < H; h++) {
            const float *src = a->P + (size_t)p * D + h * Dh;
            float *dst = a->P_t + (size_t)h * Dh * L + p;
            for (int d = 0; d < Dh; d++)
                dst[(size_t)d * L] = src[d];
        }
}

/* scores = softmax((ac + shift(bd)) / sqrt(Dh)) for rows [r0, r1) of the [H·T, T] score matrix, in place in ac.
 * Transformer-XL's relative shift written out: query i and key j need relative position i − j, which sits at index
 * T−1−i+j of bd's row (its 2T−1 positions run from T−1 down to −(T−1)), so every result row is a contiguous window
 * of a bd row. No padded copy, no shuffle, and the softmax follows in the same pass over the row. */
struct scores_args { float *ac; const float *bd; int T; float scale; };

static void combine_softmax(int r0, int r1, void *arg)
{
    struct scores_args *s = (struct scores_args *)arg;
    int T = s->T, L = 2 * T - 1;
    for (int r = r0; r < r1; r++) {
        int i = r % T;
        float *a = s->ac + (size_t)r * T;
        const float *b = s->bd + (size_t)r * L + (T - 1 - i);
        int j = 0;
#if P2_ENC_NEON
        float32x4_t vs = vdupq_n_f32(s->scale);
        for (; j + 4 <= T; j += 4)
            vst1q_f32(a + j, vmulq_f32(vaddq_f32(vld1q_f32(a + j), vld1q_f32(b + j)), vs));
#elif P2_ENC_AVX2
        __m256 vs = _mm256_set1_ps(s->scale);
        for (; j + 8 <= T; j += 8)
            _mm256_storeu_ps(a + j, _mm256_mul_ps(_mm256_add_ps(_mm256_loadu_ps(a + j), _mm256_loadu_ps(b + j)), vs));
#endif
        for (; j < T; j++)
            a[j] = (a[j] + b[j]) * s->scale;
        p2_softmax(a, T);
    }
}

/* attn_h [H, T, Dh] → attn [T, D] for frames [t0, t1); or, with a QuantA, straight into its int8 rows. */
struct concat_args { const float *attn_h; float *attn; const QuantA *qa; int T; };

static void concat_heads(int t0, int t1, void *arg)
{
    struct concat_args *c = (struct concat_args *)arg;
    int D = P2_D_MODEL, H = P2_N_HEADS, Dh = P2_HEAD_DIM, T = c->T;
    float row[P2_D_MODEL];
    for (int t = t0; t < t1; t++) {
        float *dst = c->qa ? row : c->attn + (size_t)t * D;
        for (int h = 0; h < H; h++)
            memcpy(dst + h * Dh, c->attn_h + ((size_t)h * T + t) * Dh, Dh * sizeof(float));
        if (c->qa) p2_quant_row(c->qa, t, row);
    }
}

/* ── Multi-head self-attention with relative positional encoding ─────────── */

/* Keys and position keys are gathered already transposed ([H, Dh, T] and [H, Dh, 2T-1]) so that every score product
 * is a plain NN GEMM on the register-blocked kernel, and each head's GEMM is spread over the whole thread pool. The
 * reference ran one head per thread through a dot-product NT kernel: with 8 heads on 18 cores that left most of the
 * machine idle, and that kernel was the single largest item in the profile. */
static void self_attention(const float *norm_w, const float *norm_b, const MHAWeights *mha,
                           const float *pe_slice, float *x, int T, CfmWS *ws)
{
    int D = P2_D_MODEL, H = P2_N_HEADS, Dh = P2_HEAD_DIM;
    int L = 2 * T - 1;
    WSA a = {ws->buf, 0};

    /* Fused QKV projection: [T, D] @ [D, 3D]; row layout Q | K | V. */
    float *QKV = ws_get(&a, (size_t)T * 3 * D);
    QuantA qa;
    int int8 = p2_quant_prepare(&qa, T, D) == 0;
    if (int8) {
        p2_layer_norm_quant_rows(x, norm_w, norm_b, &qa, T, D);
        p2_wmatmul_q8a(&qa, &mha->qkv_w, QKV, 1.0f, 0.0f);
    } else {
        float *ln = ws_get(&a, (size_t)T * D);
        p2_layer_norm_rows(x, norm_w, norm_b, ln, T, D);
        p2_wmatmul(ln, &mha->qkv_w, QKV, T, 1.0f, 0.0f);
    }

    float *Q_u = ws_get(&a, (size_t)H * T * Dh);
    float *Q_v = ws_get(&a, (size_t)H * T * Dh);
    float *K_t = ws_get(&a, (size_t)H * Dh * T);
    float *V_h = ws_get(&a, (size_t)H * T * Dh);
    struct gather_args g = { QKV, mha->bias_u, mha->bias_v, Q_u, Q_v, K_t, V_h, T };
    parallel_rows(T, gather_heads, &g);

    /* Positional projection: P = pe @ W_pos, [2T-1, D], then per head transposed to [H, Dh, 2T-1]. */
    float *P = ws_get(&a, (size_t)L * D);
    p2_wmatmul(pe_slice, &mha->pos_w, P, L, 1.0f, 0.0f);
    float *P_t = ws_get(&a, (size_t)H * Dh * L);
    struct ptrans_args pt = { P, P_t, L };
    parallel_rows(L, transpose_pos, &pt);

    /* AC = Q_u @ K^T: [H, T, T] */
    float *ac = ws_get(&a, (size_t)H * T * T);
    heads_gemm(H, T, T, Dh, Q_u, (size_t)T * Dh, Dh, K_t, (size_t)Dh * T, T, ac, (size_t)T * T, T);

    /* BD = Q_v @ P^T: [H, T, 2T-1] */
    float *bd = ws_get(&a, (size_t)H * T * L);
    heads_gemm(H, T, L, Dh, Q_v, (size_t)T * Dh, Dh, P_t, (size_t)Dh * L, L, bd, (size_t)T * L, L);

    /* scores = softmax((ac + shift(bd)) / sqrt(Dh)) per row */
    struct scores_args sc = { ac, bd, T, 1.0f / sqrtf((float)Dh) };
    parallel_rows(H * T, combine_softmax, &sc);

    /* attn_h = scores @ V: [H, T, Dh] */
    float *attn_h = ws_get(&a, (size_t)H * T * Dh);
    heads_gemm(H, T, Dh, T, ac, (size_t)T * T, T, V_h, (size_t)T * Dh, Dh, attn_h, (size_t)T * Dh, Dh);

    /* Concatenate heads → [T, D], project, add. */
    if (int8) {
        QuantA qo;
        quant_prepare_or_die(&qo, T, D);
        struct concat_args cc = { attn_h, NULL, &qo, T };
        parallel_rows(T, concat_heads, &cc);
        p2_wmatmul_q8a(&qo, &mha->out_w, x, 1.0f, 1.0f);   /* x += attn @ W_o */
    } else {
        float *attn = ws_get(&a, (size_t)T * D);
        struct concat_args cc = { attn_h, attn, NULL, T };
        parallel_rows(T, concat_heads, &cc);
        float *out = ws_get(&a, (size_t)T * D);
        p2_wmatmul(attn, &mha->out_w, out, T, 1.0f, 0.0f);
        add_residual(x, out, T, 1.0f);
    }
}

/* ── Convolution module ──────────────────────────────────────────────────── */

/* GLU over the channel axis into channel-major [D, T], for channels [d0, d1): gated = first half · sigmoid(second half). */
struct glu_args { const float *pw1; float *gated; int T; };

static void glu_to_channel_major(int d0, int d1, void *arg)
{
    struct glu_args *g = (struct glu_args *)arg;
    int D = P2_D_MODEL, T = g->T;
    for (int t = 0; t < T; t++) {
        const float *row = g->pw1 + (size_t)t * 2 * D;
        for (int d = d0; d < d1; d++)
            g->gated[(size_t)d * T + t] = row[d] / (1.0f + p2_expneg(row[D + d]));   /* the clamp alone: same rounding as before */
    }
}

/* dw [D, T] → act [T, D] for frames [t0, t1); or, with a QuantA, straight into its int8 rows. */
struct ct_args { const float *dw; float *act; const QuantA *qa; int T; };

static void channel_major_to_rows(int t0, int t1, void *arg)
{
    struct ct_args *c = (struct ct_args *)arg;
    int D = P2_D_MODEL, T = c->T;
    float row[P2_D_MODEL];
    for (int t = t0; t < t1; t++) {
        float *dst = c->qa ? row : c->act + (size_t)t * D;
        for (int d = 0; d < D; d++)
            dst[d] = c->dw[(size_t)d * T + t];
        if (c->qa) p2_quant_row(c->qa, t, row);
    }
}

/* LN → pointwise_conv1 (as a [T, D] @ [D, 2D] GEMM) → GLU → depthwise conv over time (+ folded BatchNorm) → SiLU
 * → pointwise_conv2 ([T, D] @ [D, D]) → residual. The depthwise conv runs channel-major, so the activation is
 * transposed into and out of that layout around it. */
static void conv_module(const float *norm_w, const float *norm_b, const ConvModWeights *cw,
                        float *x, int T, CfmWS *ws)
{
    int D = P2_D_MODEL;
    WSA a = {ws->buf, 0};

    float *pw1 = ws_get(&a, (size_t)T * 2 * D);
    QuantA qa;
    int int8 = p2_quant_prepare(&qa, T, D) == 0;
    if (int8) {
        p2_layer_norm_quant_rows(x, norm_w, norm_b, &qa, T, D);
        p2_wmatmul_q8a(&qa, &cw->pw1_w, pw1, 1.0f, 0.0f);
    } else {
        float *ln = ws_get(&a, (size_t)T * D);
        p2_layer_norm_rows(x, norm_w, norm_b, ln, T, D);
        p2_wmatmul(ln, &cw->pw1_w, pw1, T, 1.0f, 0.0f);
    }

    float *gated = ws_get(&a, (size_t)D * T);
    struct glu_args g = { pw1, gated, T };
    parallel_rows(D, glu_to_channel_major, &g);

    float *dw = ws_get(&a, (size_t)D * T);
    p2_depthwise_conv1d(gated, cw->dw_w, cw->dw_b, dw, D, T, P2_CONV_KERNEL);
    p2_swish(dw, D * T);

    if (int8) {
        QuantA qc;
        quant_prepare_or_die(&qc, T, D);
        struct ct_args ct = { dw, NULL, &qc, T };
        parallel_rows(T, channel_major_to_rows, &ct);
        p2_wmatmul_q8a(&qc, &cw->pw2_w, x, 1.0f, 1.0f);   /* x += conv(x) */
    } else {
        float *act = ws_get(&a, (size_t)T * D);
        struct ct_args ct = { dw, act, NULL, T };
        parallel_rows(T, channel_major_to_rows, &ct);
        float *pw2 = ws_get(&a, (size_t)T * D);
        p2_wmatmul(act, &cw->pw2_w, pw2, T, 1.0f, 0.0f);
        add_residual(x, pw2, T, 1.0f);
    }
}

/* ── Conformer block ─────────────────────────────────────────────────────── */

static void conformer_block(const ConformerBlock *blk, const float *pe_slice, float *x, int T, CfmWS *ws)
{
    feed_forward(blk->norm_ff1_w, blk->norm_ff1_b, &blk->ff1, x, T, ws);
    self_attention(blk->norm_sa_w, blk->norm_sa_b, &blk->mha, pe_slice, x, T, ws);
    conv_module(blk->norm_conv_w, blk->norm_conv_b, &blk->conv, x, T, ws);
    feed_forward(blk->norm_ff2_w, blk->norm_ff2_b, &blk->ff2, x, T, ws);
    p2_layer_norm_rows_inplace(x, blk->norm_out_w, blk->norm_out_b, T, P2_D_MODEL);
}

/* ── The block stack: full-length or chunked ─────────────────────────────── */

static void encode_blocks(const P2Model *m, float *x, int T)
{
    float *pe = rel_pos_encoding(T);
    CfmWS ws;
    ws.cap = conformer_ws_peak(T);
    ws.buf = (float *)malloc(ws.cap * sizeof(float));

    for (int i = 0; i < P2_N_BLOCKS; i++) {
        if (p2_verbose)
            fprintf(stderr, "  encoder block %d/%d\r", i + 1, P2_N_BLOCKS);
        conformer_block(&m->enc.blocks[i], pe, x, T, &ws);
    }

    free(ws.buf);
    free(pe);
    if (p2_verbose) fprintf(stderr, "\n");
}

/* Integer from the environment, or the default when unset or unparsable. */
static int env_int(const char *name, int fallback)
{
    const char *v = getenv(name);
    if (!v || !*v) return fallback;
    char *end;
    long n = strtol(v, &end, 10);
    return (*end == '\0' && n > 0) ? (int)n : fallback;
}

static void encode_blocks_chunked(const P2Model *m, const float *x, int T, float *out)
{
    int C  = env_int("P2_CHUNK_SIZE", P2_CHUNK_SIZE);
    int ov = env_int("P2_CHUNK_OVERLAP", P2_CHUNK_OVERLAP);
    int n_chunks = (T + C - 1) / C;
    int D = P2_D_MODEL;

    int common_in_len = C + 2 * ov;
    if (common_in_len > T) common_in_len = T;
    float *pe_common = rel_pos_encoding(common_in_len);

    CfmWS ws;
    ws.cap = conformer_ws_peak(common_in_len);
    ws.buf = (float *)malloc(ws.cap * sizeof(float));

    for (int ci = 0; ci < n_chunks; ci++) {
        int out_start = ci * C;
        int out_end   = out_start + C;
        if (out_end > T) out_end = T;
        int out_len = out_end - out_start;

        int in_start = out_start - ov;
        if (in_start < 0) in_start = 0;
        int in_end = out_end + ov;
        if (in_end > T) in_end = T;
        int in_len = in_end - in_start;
        int left_ov = out_start - in_start;

        if (p2_verbose)
            fprintf(stderr, "  chunk %d/%d: frames [%d, %d) input [%d, %d)\r",
                    ci + 1, n_chunks, out_start, out_end, in_start, in_end);

        float *chunk = alloc_f((size_t)in_len * D);
        memcpy(chunk, x + (size_t)in_start * D, (size_t)in_len * D * sizeof(float));

        float *pe = pe_common;
        int own_pe = 0;
        if (in_len != common_in_len) {
            pe = rel_pos_encoding(in_len);
            own_pe = 1;
        }

        for (int i = 0; i < P2_N_BLOCKS; i++)
            conformer_block(&m->enc.blocks[i], pe, chunk, in_len, &ws);

        if (own_pe) free(pe);

        memcpy(out + (size_t)out_start * D, chunk + (size_t)left_ov * D, (size_t)out_len * D * sizeof(float));
        free(chunk);
    }

    free(ws.buf);
    free(pe_common);
    if (p2_verbose) fprintf(stderr, "\n");
}

/* ── Public encoder API ──────────────────────────────────────────────────── */

/* P2_CHUNK in the environment: unset / "auto" → chunk beyond P2_CHUNK_THRESHOLD; "0" / "off" → never; "1" / "on" → always. */
static int should_chunk(int Tp)
{
    const char *env = getenv("P2_CHUNK");
    if (env) {
        if (env[0] == '0' || strcmp(env, "off") == 0) return 0;
        if (env[0] == '1' || strcmp(env, "on")  == 0) return 1;
    }
    return Tp > P2_CHUNK_THRESHOLD;
}

static int encode(const P2Model *m, const float *mel, int n_frames, int chunked, float **out, int *out_len)
{
    float *x;
    int Tp = pre_encode(&m->enc.subsampling, mel, n_frames, &x);

    if (chunked) {
        if (p2_verbose)
            fprintf(stderr, "  chunked: %d frames, chunk=%d, overlap=%d\n", Tp,
                    env_int("P2_CHUNK_SIZE", P2_CHUNK_SIZE), env_int("P2_CHUNK_OVERLAP", P2_CHUNK_OVERLAP));
        float *encoded = alloc_f((size_t)Tp * P2_D_MODEL);
        encode_blocks_chunked(m, x, Tp, encoded);
        free(x);
        x = encoded;
    } else {
        encode_blocks(m, x, Tp);
    }

    /* encoder_projector: the 640-dim frames the joint network consumes. */
    float *proj = alloc_f((size_t)Tp * P2_JOINT_DIM);
    p2_wmatmul(x, &m->enc.projector_w, proj, Tp, 1.0f, 0.0f);
    p2_bias_add_rows(proj, m->enc.projector_b, Tp, P2_JOINT_DIM);
    free(x);

    *out = proj;
    *out_len = Tp;
    return 0;
}

int p2_encoder(const P2Model *m, const float *mel, int n_frames, float **out, int *out_len)
{
    int Tp = conv_out(conv_out(conv_out(n_frames)));
    return encode(m, mel, n_frames, should_chunk(Tp), out, out_len);
}

int p2_encoder_chunked(const P2Model *m, const float *mel, int n_frames, float **out, int *out_len)
{
    return encode(m, mel, n_frames, 1, out, out_len);
}
