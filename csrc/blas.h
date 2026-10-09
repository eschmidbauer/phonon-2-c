/*  blas.h – The GEMM layer, all hand-rolled: nothing here comes from a BLAS library.
 *
 *  fp32: a row-major sgemm (register-blocked NEON / AVX2 micro-kernels, cache-blocked B packing) in blas.c, plus a
 *  single-threaded variant for use inside pool workers.
 *
 *  Quantised B operands: in exact mode the weight is expanded to fp32 inside the pack step, so the inner kernel stays
 *  fp32 and the result equals an fp32 GEMM on the expanded weights; in the int8 modes A is quantised per row and the
 *  product runs on integer dot products (NEON SDOT).
 *
 *  Until 2026-10-06 the fp32 sgemm could optionally come from Accelerate / OpenBLAS (-DUSE_CBLAS). It matched the
 *  hand-rolled build in the default int8 mode and was 1.4× faster only in exact mode on Apple silicon (its AMX units);
 *  the option was removed so that the library has one code path and no dependency beyond libc, libm and pthreads.
 */
#ifndef P2_BLAS_H
#define P2_BLAS_H

#include "phonon2.h"

typedef enum {
    P2_NO_TRANS = 111,
    P2_TRANS    = 112
} P2Trans;

/* C = alpha · op(A) · op(B) + beta · C.  op(A) is M×K, op(B) is K×N, C is M×N, all row-major: A[i, j] = A[i * lda + j]. */
void p2_sgemm(P2Trans transA, P2Trans transB,
              int M, int N, int K,
              float alpha, const float *A, int lda,
              const float *B, int ldb,
              float beta, float *C, int ldc);

/* Same, never dispatching to the thread pool – for use inside p2_parallel workers. */
void p2_sgemm_st(P2Trans transA, P2Trans transB,
                 int M, int N, int K,
                 float alpha, const float *A, int lda,
                 const float *B, int ldb,
                 float beta, float *C, int ldc);

/* Five-value B operand [K, N]: w[k][n] = sign · (flag ? hi[n] : lo[n]). */
typedef struct {
    const unsigned char *signs;  /* [K][N/4]: 2-bit codes, 0 → -1, 1 → 0, 2 → +1; weight n at bits 2*(n&3) */
    const unsigned char *flags;  /* [K][N/8]: bit (n&7) set → |w| = hi[n], clear → |w| = lo[n] */
    const float         *lo;     /* [N] */
    const float         *hi;     /* [N] */
    const signed char   *codes;  /* optional: the int8w codes pre-packed for the kernel (p2_fv_pack_codes), or NULL */
} FiveValueB;

/* The int8w codes of a whole five-value weight – sign · (flag ? 127 : round(127 · lo/hi)) – packed as the kernel
 * reads them: per 16-column panel, [K/4 quads][16 cols][4 k], panels consecutive; K × N bytes. With it in
 * FiveValueB.codes the int8w GEMM skips the per-GEMM expansion of the planes, which is most of a short utterance's
 * encoder time. Parallel over the pool. NULL without the int8 path or on allocation failure; caller frees. */
signed char *p2_fv_pack_codes(const FiveValueB *b, int K, int N);

/* C = alpha · A · B + beta · C with B int8 [K, N] (ldb bytes per row) and one fp32 scale per column. N % 16 == 0. */
void p2_sgemm_q8b(int M, int N, int K,
                  float alpha, const float *A, int lda,
                  const signed char *B, const float *scales, int ldb,
                  float beta, float *C, int ldc);

/* C = alpha · A · B + beta · C with B five-value [K, N]. N % 16 == 0. */
void p2_sgemm_fvb(int M, int N, int K,
                  float alpha, const float *A, int lda,
                  const FiveValueB *B,
                  float beta, float *C, int ldc);

/* How the quantised GEMMs run: "int8 dot products" (A quantised per row to int8, NEON SDOT; the default on AArch64
 * with the dot-product extension) or "fp32 expansion (exact)" (P2_EXACT=1 in the environment, or any other CPU). */
const char *p2_gemm_mode(void);
int p2_gemm_exact(void);

/* Pre-quantised A (see QuantA in phonon2.h). prepare returns -1 when the int8 path is not in use, and the caller
 * takes the fp32 route; otherwise the buffer is the calling thread's, with the padding rows beyond M zeroed. */
int  p2_quant_prepare(QuantA *qa, int M, int K);
void p2_quant_row(const QuantA *qa, int m, const float *row);   /* quantise and pack one row; any thread may call it */

/* The int8 GEMMs on pre-quantised A. beta is 0 (store) or 1 (accumulate). */
void p2_sgemm_fvb_q8a(int M, int N, int K, float alpha, const QuantA *A, const FiveValueB *B,
                      float beta, float *C, int ldc);
void p2_sgemm_q8b_q8a(int M, int N, int K, float alpha, const QuantA *A, const signed char *B, const float *scales,
                      int ldb, float beta, float *C, int ldc);

#endif /* P2_BLAS_H */
