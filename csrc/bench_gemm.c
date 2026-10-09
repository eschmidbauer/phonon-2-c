/*  bench_gemm.c – Throughput of the five-value GEMM on the encoder's shapes, M = one 32 s chunk (416 frames).
 *
 *  make -C csrc bench_gemm && P2_GEMM=int8w ./csrc/bench_gemm      (P2_GEMM = exact / int8 / int8w, P2_THREADS=N)
 *  Random planes, so only speed is measured. */
#include "phonon2.h"
#include "blas.h"
#include "threadpool.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define now p2_now_sec   /* the engine's clock */

static FiveValueB make_fv(int K, int N)
{
    FiveValueB b;
    unsigned char *s = malloc((size_t)K * N / 4), *f = malloc((size_t)K * N / 8);
    float *lo = malloc(N * sizeof(float)), *hi = malloc(N * sizeof(float));
    for (size_t i = 0; i < (size_t)K * N / 4; i++) {   /* codes 0,1,2 only (3 is unused) */
        unsigned v = 0;
        for (int j = 0; j < 4; j++) v |= (unsigned)(rand() % 3) << (2 * j);
        s[i] = (unsigned char)v;
    }
    for (size_t i = 0; i < (size_t)K * N / 8; i++) f[i] = (unsigned char)(rand() & 0xFF);
    for (int n = 0; n < N; n++) { lo[n] = 0.01f + 0.01f * (rand() % 100) / 100.0f; hi[n] = lo[n] * (1.5f + (rand() % 100) / 100.0f); }
    b.signs = s; b.flags = f; b.lo = lo; b.hi = hi;
    return b;
}

int main(void)
{
    int M = 416;
    int shapes[][2] = { {1024, 4096}, {4096, 1024}, {1024, 3072}, {1024, 1024}, {1024, 2048} };
    const char *names[] = { "ff linear1", "ff linear2", "qkv", "o_proj / pos / pw2", "pw1" };
    p2_pool();
    printf("mode: %s, %d threads, M = %d\n", p2_gemm_mode(), p2_pool_nthreads(p2_pool()), M);
    double total_s = 0, total_mac = 0;
    for (int i = 0; i < 5; i++) {
        int K = shapes[i][0], N = shapes[i][1];
        FiveValueB b = make_fv(K, N);
        float *A = malloc((size_t)M * K * sizeof(float)), *C = calloc((size_t)M * N, sizeof(float));
        for (size_t j = 0; j < (size_t)M * K; j++) A[j] = (rand() / (float)RAND_MAX - 0.5f) * 4.0f;
        p2_sgemm_fvb(M, N, K, 1.0f, A, K, &b, 0.0f, C, N);   /* warm-up */
        int reps = 10;
        double t0 = now();
        for (int r = 0; r < reps; r++) p2_sgemm_fvb(M, N, K, 1.0f, A, K, &b, 0.0f, C, N);
        double dt = (now() - t0) / reps;
        double mac = (double)M * K * N;
        printf("  %-20s K=%4d N=%4d  %7.2f ms  %6.0f GMAC/s\n", names[i], K, N, dt * 1e3, mac / dt / 1e9);
        total_s += dt; total_mac += mac;
        free(A); free(C);
    }
    printf("  per conformer block (ff ×4, qkv, pos, out, pw1, pw2): %.1f ms  → ×24 = %.2f s per chunk-pass\n",
           total_s * 1e3 + 0 /* approx */, total_s * 24);
    (void)total_mac;
    return 0;
}
