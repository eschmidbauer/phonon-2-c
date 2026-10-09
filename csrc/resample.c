/*  resample.c – Sample-rate conversion to the model's 16 kHz.
 *
 *  A windowed-sinc polyphase filter for the reduced rational ratio L/M = sr_out/sr_in: the prototype low-pass is
 *  designed at the common rate sr_in·L (Kaiser window, β = 9, about 80 dB of stop band), cut off at 0.46 of the lower
 *  of the two rates – anti-aliasing when going down, anti-imaging when going up – and split into L phases. An output
 *  sample is one dot product of its phase's taps with a window of the input. The window spans RS_TAPS input samples,
 *  times ceil(M/L) when decimating, so the filter is always about RS_TAPS output samples long and its transition band
 *  scales with the output rate: a fixed 48 input taps gave 48 kHz → 16 kHz a 1 ms filter, a transition band inside
 *  the passband and a 0.2 % gain droop at 1 kHz. 44.1 kHz → 16 kHz is L = 160, M = 441; 48 kHz → 16 kHz is
 *  L = 1, M = 3.
 */

#include "phonon2.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define RS_TAPS   48      /* input samples per output sample, before scaling by the decimation factor */
#define RS_BETA   9.0     /* Kaiser β */
#define RS_CUTOFF 0.46    /* of the lower sample rate */
#define RS_MAX_L  65536   /* phases; every pair of common rates reduces to far fewer */
#define RS_MAX_RATIO 256  /* at most this much decimation or interpolation (4 MHz … 62.5 Hz for 16 kHz out) */
#define RS_MAX_TABLE (1LL << 24)   /* floats in the tap table (64 MB); 44.1 kHz → 16 kHz needs 23 k */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Modified Bessel function of the first kind, order 0, by its power series. */
static double bessel_i0(double x)
{
    double sum = 1.0, term = 1.0, y = x * x / 4.0;
    for (int k = 1; k < 64; k++) {
        term *= y / ((double)k * k);
        sum += term;
        if (term < 1e-12 * sum) break;
    }
    return sum;
}

static unsigned gcd_u(unsigned a, unsigned b)
{
    while (b) { unsigned t = a % b; a = b; b = t; }
    return a;
}

/* The stateful converter. Output sample i sits at position t = i·M + center of the common rate and is the dot product
 * of its phase's taps with the taps_n input samples ending at q0 = t / L, so it can be produced as soon as input q0
 * has arrived; `hist` keeps the last taps_n inputs across feeds (zeros before the first, as the batch version's left
 * padding), and finish() supplies the right padding so that the whole-file output count ⌈n·L/M⌉ is reached. The
 * batch p2_resample is open → feed → finish, so the two agree bit for bit. */
struct P2Resampler {
    int        sr_in, sr_out, L, M, taps_n, center;
    float     *taps;                 /* [L][taps_n], reversed per phase */
    float     *hist;                 /* the last taps_n inputs (global indices n_in − taps_n … n_in − 1) */
    float     *work;                 /* hist followed by a feed's input */
    int        work_cap;
    long long  n_in;                 /* inputs received */
    long long  next;                 /* the next output index */
};

P2Resampler *p2_resampler_open(int sr_in, int sr_out)
{
    if (sr_in <= 0 || sr_out <= 0) return NULL;
    P2Resampler *r = (P2Resampler *)calloc(1, sizeof(*r));
    if (!r) return NULL;
    r->sr_in = sr_in; r->sr_out = sr_out;
    if (sr_in == sr_out) return r;   /* a copy; no filter */

    unsigned g = gcd_u((unsigned)sr_in, (unsigned)sr_out);
    int L = (int)(sr_out / g), M = (int)(sr_in / g);
    if (L > RS_MAX_L) { free(r); return NULL; }

    /* Prototype filter at rate sr_in·L, length N = taps·L, stored per phase with the taps reversed so that an output
     * sample is a forward dot product over a contiguous input window. The ratio and the table are bounded: a WAV
     * header claiming a 4 GHz rate would otherwise ask for millions of taps per phase and a tap table of tens of
     * gigabytes, which a lazy malloc grants and the design loop then never finishes (found by tests/fuzz). */
    if ((long long)M > (long long)RS_MAX_RATIO * L || (long long)L > (long long)RS_MAX_RATIO * M) {   /* nor a 16,000× interpolation of a "1 Hz" file */
        free(r);
        return NULL;
    }
    int taps_n = RS_TAPS * (M > L ? (M + L - 1) / L : 1);
    if ((long long)taps_n * L > RS_MAX_TABLE) { free(r); return NULL; }
    int N = taps_n * L;
    int center = N / 2;
    double nu = RS_CUTOFF * (sr_in < sr_out ? sr_in : sr_out) / ((double)sr_in * L);   /* cycles per sample at sr_in·L */
    double i0_beta = bessel_i0(RS_BETA);
    float *taps = (float *)malloc((size_t)L * taps_n * sizeof(float));
    float *hist = (float *)calloc((size_t)taps_n, sizeof(float));
    if (!taps || !hist) { free(taps); free(hist); free(r); return NULL; }
    for (int phase = 0; phase < L; phase++) {
        float *h = taps + (size_t)phase * taps_n;
        double sum = 0.0;
        for (int j = 0; j < taps_n; j++) {
            int n = phase + j * L;
            double m = n - center;
            double sinc = m == 0.0 ? 1.0 : sin(2.0 * M_PI * nu * m) / (2.0 * M_PI * nu * m);
            double rr = 2.0 * n / (N - 1) - 1.0;
            double w = bessel_i0(RS_BETA * sqrt(rr * rr < 1.0 ? 1.0 - rr * rr : 0.0)) / i0_beta;
            h[taps_n - 1 - j] = (float)(sinc * w);
            sum += sinc * w;
        }
        /* Each branch sums to one: exact DC gain and no imaging of a constant, whatever the truncation left of the
         * ideal L·2ν·sinc sum (0.05 % short at 48 input taps). */
        for (int j = 0; j < taps_n; j++) h[j] = (float)(h[j] / sum);
    }
    r->L = L; r->M = M; r->taps_n = taps_n; r->center = center;
    r->taps = taps; r->hist = hist;
    return r;
}

void p2_resampler_close(P2Resampler *r)
{
    if (!r) return;
    free(r->taps);
    free(r->hist);
    free(r->work);
    free(r);
}

/* Emit every output whose input window ends within the inputs available: the history plus `n` new samples in
 * work[taps_n ..], of which `n_real` are real (the rest right padding at finish). */
static int resampler_run(P2Resampler *r, int n, long long count_limit, float **out, int *n_out)
{
    long long avail = r->n_in + n;                   /* inputs now known, as a global count */
    long long first = r->next, last = first;         /* outputs [first, last) */
    while (last < count_limit && (last * r->M + r->center) / r->L < avail) last++;
    if (last - first > 0x7FFFFFFF) return -1;        /* more outputs than an int can count (a 256× feed of 10 M samples) */
    float *y = (float *)malloc((size_t)((last - first) > 0 ? (last - first) : 1) * sizeof(float));
    if (!y) return -1;
    for (long long i = first; i < last; i++) {
        long long t = i * (long long)r->M + r->center;   /* position at the common rate, filter centred */
        int phase = (int)(t % r->L);
        long long q0 = t / r->L;                         /* newest input sample the window ends on */
        long long start = q0 - (r->taps_n - 1);          /* oldest; global index, may be negative at the start */
        /* work[w] holds global input index (n_in − taps_n) + w */
        long long w = start - (r->n_in - r->taps_n);
        y[i - first] = p2_dot_f32(r->taps + (size_t)phase * r->taps_n, r->work + w, r->taps_n);
    }
    r->next = last;
    *out = y;
    *n_out = (int)(last - first);
    return 0;
}

/* Lay out hist followed by `n` samples (the input, or zeros) in work, and keep the last taps_n as the new history. */
static int resampler_stage(P2Resampler *r, const float *in, int n)
{
    if (n > 0x7FFFFFFF - r->taps_n) return -1;
    int need = r->taps_n + n;
    if (need > r->work_cap) {
        float *w = (float *)realloc(r->work, (size_t)need * sizeof(float));
        if (!w) return -1;
        r->work = w; r->work_cap = need;
    }
    memcpy(r->work, r->hist, (size_t)r->taps_n * sizeof(float));
    if (in) memcpy(r->work + r->taps_n, in, (size_t)n * sizeof(float));
    else    memset(r->work + r->taps_n, 0, (size_t)n * sizeof(float));
    /* the new history: the last taps_n of hist ++ input */
    if (n >= r->taps_n) memcpy(r->hist, r->work + n, (size_t)r->taps_n * sizeof(float));
    else {
        memmove(r->hist, r->hist + n, (size_t)(r->taps_n - n) * sizeof(float));
        memcpy(r->hist + r->taps_n - n, r->work + r->taps_n, (size_t)n * sizeof(float));
    }
    return 0;
}

int p2_resampler_feed(P2Resampler *r, const float *in, int n_in, float **out, int *n_out)
{
    if (n_in < 0 || (n_in > 0 && !in)) return -1;
    if (!r->taps) {   /* same rate: a copy */
        float *y = (float *)malloc((size_t)(n_in > 0 ? n_in : 1) * sizeof(float));
        if (!y) return -1;
        memcpy(y, in, (size_t)n_in * sizeof(float));
        r->n_in += n_in;
        *out = y; *n_out = n_in;
        return 0;
    }
    if (resampler_stage(r, in, n_in) != 0) return -1;
    int rc = resampler_run(r, n_in, (long long)1 << 62, out, n_out);
    r->n_in += n_in;
    return rc;
}

int p2_resampler_finish(P2Resampler *r, float **out, int *n_out)
{
    if (!r->taps) {
        *out = (float *)malloc(sizeof(float));
        *n_out = 0;
        return *out ? 0 : -1;
    }
    /* Right padding: zeros enough for every remaining output up to the whole-signal count ⌈n·L/M⌉. */
    long long count = (r->n_in * r->L + r->M - 1) / r->M;
    int pad = r->taps_n;
    if (resampler_stage(r, NULL, pad) != 0) return -1;
    int rc = resampler_run(r, pad, count, out, n_out);
    r->n_in += pad;   /* further feeds would be wrong; finish ends the stream */
    return rc;
}

int p2_resample(const float *in, int n_in, int sr_in, int sr_out, float **out, int *n_out)
{
    if (n_in < 0) return -1;
    P2Resampler *r = p2_resampler_open(sr_in, sr_out);
    if (!r) return -1;
    float *a = NULL, *b = NULL;
    int na = 0, nb = 0, rc = -1;
    if (p2_resampler_feed(r, in, n_in, &a, &na) == 0 && p2_resampler_finish(r, &b, &nb) == 0 &&
        na <= 0x7FFFFFFF - nb) {
        float *y = (float *)malloc((size_t)((na + nb) > 0 ? (na + nb) : 1) * sizeof(float));
        if (y) {
            memcpy(y, a, (size_t)na * sizeof(float));
            memcpy(y + na, b, (size_t)nb * sizeof(float));
            *out = y; *n_out = na + nb;
            rc = 0;
        }
    }
    free(a); free(b);
    p2_resampler_close(r);
    return rc;
}
