/*  segment.c – Pause-aligned windows for long audio, and the energy measures the engine uses to notice a dropout.
 *
 *  Phonon-2 is a ≤ 35 s model. Given a window that begins mid-speech it can go silent for 10–30 s – the decoder
 *  emits blanks over clear speech and resumes later – and torch does exactly the same on the same samples (see
 *  .docs/verification.md, "Long-form"). The engine's inherited fixed 32 s chunking therefore scored 25–36 % WER on
 *  concatenated LibriSpeech chapters where a full-length pass scored 1.3 %. Audio longer than P2_WINDOW_MAX_S is
 *  instead cut into 25–35 s windows, each ending in the middle of the longest pause the band offers, and every
 *  window is encoded and decoded on its own. The pause gate is Fermion's live-session one (50 ms blocks, RMS at or
 *  below max(0.004, 0.18 × the window's peak so far)), and the cut falls at 30 s when the band holds no pause, so
 *  the windows are the ones fermion/_speech/segment.py `plan()` produces. P2_WINDOW_MAX in the environment (seconds)
 *  moves the whole band for experiments: band [max − 10, max], target max − 5.
 *
 *  Starting at a pause makes the dropout rare, not impossible; engine.c also checks every decoded window for a
 *  stretch of speech energy with no token (p2_block_rms / p2_speech_gate / p2_speech_share) and re-decodes such a
 *  window as two halves split at its longest pause (p2_split_point). The speech measure is the share of blocks above three times the window's
 *  10th-percentile RMS – its own noise floor. Calibrated on 21 dropouts and 15 real gaps of 2 s or more (read speech
 *  from two LibriSpeech splits and a phone call): every dropout scores 0.42 or more, every pause in read speech
 *  about 0, and only a phone call's wordless but loud stretches (hold tone, line noise) score high too – a re-decode
 *  wasted, not a wrong transcript. A share of the window's peak (the pause gate) was tried first: dropouts ran
 *  0.05–0.55 because soft speakers sit under 0.18 × peak, and real pauses 0.07–0.23, so the two overlapped.
 */

#include "phonon2.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define P2_WINDOW_MAX_S      35.0   /* up to this length: one window */
#define P2_WINDOW_BAND_S     10.0   /* a cut is searched for in the last this-many seconds before the maximum */
#define P2_WINDOW_BLOCK_S    0.05
#define P2_WINDOW_FLOOR_RMS  0.004f
#define P2_WINDOW_GATE_RATIO 0.18f   /* a pause: blocks at or below this share of the peak */
#define P2_SPEECH_FLOOR_RATIO 3.0f   /* speech: blocks above this multiple of the window's 10th-percentile RMS */

/* Zero-padded 50 ms block RMS over the whole signal. */
static float *block_rms(const float *audio, int n, int block, int *n_blocks)
{
    int nb = (n + block - 1) / block;
    float *rms = (float *)malloc((size_t)(nb > 0 ? nb : 1) * sizeof(float));
    if (!rms) return NULL;
    for (int b = 0; b < nb; b++) {
        double acc = 0.0;
        int lo = b * block, hi = lo + block < n ? lo + block : n;
        for (int i = lo; i < hi; i++) acc += (double)audio[i] * audio[i];
        rms[b] = (float)sqrt(acc / block);
    }
    *n_blocks = nb;
    return rms;
}

/* Fermion's gate for a stretch of blocks: 0.18 of its peak, never below the room-tone floor. */
static float gate_for(const float *rms, int b_from, int b_to)
{
    float peak = 0.0f;
    for (int b = b_from; b < b_to; b++) if (rms[b] > peak) peak = rms[b];
    float gate = P2_WINDOW_GATE_RATIO * peak;
    return gate < P2_WINDOW_FLOOR_RMS ? P2_WINDOW_FLOOR_RMS : gate;
}

/* The middle (in samples) of the longest run of quiet blocks inside blocks [b0, b1), ties going to the run nearest
 * `target`; -1 when no block is quiet. */
static int quiet_cut(const float *rms, int b0, int b1, int block, float gate, int target)
{
    int cut = -1, best_len = 0, best_dist = 0;
    for (int i = b0; i < b1; ) {
        if (rms[i] > gate) { i++; continue; }
        int j = i;
        while (j < b1 && rms[j] <= gate) j++;
        int mid = (int)(((long long)i + j) * block / 2);
        int len = j - i, dist = abs(mid - target);
        if (len > best_len || (len == best_len && dist < best_dist)) {
            best_len = len; best_dist = dist; cut = mid;
        }
        i = j;
    }
    return cut;
}

static double window_max_seconds(void)
{
    const char *env = getenv("P2_WINDOW_MAX");
    if (env && *env) {
        double v = atof(env);
        if (v >= P2_WINDOW_BAND_S + 5.0) return v;
    }
    return P2_WINDOW_MAX_S;
}

int p2_window_max_samples(int sample_rate)
{
    return (int)(window_max_seconds() * sample_rate);
}

int p2_first_cut(const float *audio, int n_samples, int sample_rate)
{
    double max_s = window_max_seconds();
    int single  = (int)(max_s * sample_rate);
    int lo = (int)((max_s - P2_WINDOW_BAND_S) * sample_rate);
    int target = (int)((max_s - P2_WINDOW_BAND_S / 2) * sample_rate);
    if (n_samples <= single) return n_samples;

    /* Blocks are aligned to this window's start, so a stream that plans one window at a time (engine.c) cuts
     * exactly where a whole file does. */
    int block = (int)(P2_WINDOW_BLOCK_S * sample_rate), nb;
    float *rms = block_rms(audio, single, block, &nb);
    if (!rms) return -1;   /* never a silent fixed cut: that is the failure mode the windows exist to avoid */
    int b0 = (lo + block - 1) / block;   /* first block fully inside the band */
    int b1 = single / block;             /* one past the last block inside it */
    int cut = quiet_cut(rms, b0, b1, block, gate_for(rms, 0, b1), target);
    free(rms);
    if (cut < 0) cut = target;
    if (cut < lo) cut = lo;
    if (cut > single) cut = single;
    return cut;
}

int p2_plan_windows(const float *audio, int n_samples, int sample_rate, P2Window **out)
{
    *out = NULL;
    if (n_samples <= 0 || sample_rate <= 0) return 0;
    int band_lo = (int)((window_max_seconds() - P2_WINDOW_BAND_S) * sample_rate);
    P2Window *w = (P2Window *)malloc((size_t)(n_samples / band_lo + 2) * sizeof(P2Window));
    if (!w) return 0;

    int count = 0, start = 0;
    for (;;) {
        int cut = p2_first_cut(audio + start, n_samples - start, sample_rate);
        if (cut < 0) { free(w); return 0; }
        cut += start;
        w[count].start = start; w[count].end = cut;
        count++;
        if (cut >= n_samples) break;
        start = cut;
    }
    *out = w;
    return count;
}

int p2_split_point(const float *audio, int n_samples, int sample_rate)
{
    int block = (int)(P2_WINDOW_BLOCK_S * sample_rate), nb;
    float *rms = block_rms(audio, n_samples, block, &nb);
    if (!rms) return n_samples / 2;
    int b0 = (n_samples / 3 + block - 1) / block, b1 = (2 * n_samples / 3) / block;
    int cut = quiet_cut(rms, b0, b1, block, gate_for(rms, 0, nb), n_samples / 2);
    free(rms);
    return cut < 0 ? n_samples / 2 : cut;
}

static int cmp_float(const void *a, const void *b)
{
    float x = *(const float *)a, y = *(const float *)b;
    return (x > y) - (x < y);
}

float *p2_block_rms(const float *audio, int n_samples, int sample_rate, int *n_blocks)
{
    if (n_samples <= 0) { *n_blocks = 0; return NULL; }
    return block_rms(audio, n_samples, (int)(P2_WINDOW_BLOCK_S * sample_rate), n_blocks);
}

float p2_speech_gate(const float *rms, int n_blocks)
{
    /* The window's own noise floor: the 10th percentile of its block RMS (pauses and room tone); speech sits well
     * above it whatever the speaker's level, where a share of the peak does not (soft speakers fell under 0.18 × peak). */
    if (n_blocks <= 0) return P2_WINDOW_FLOOR_RMS;
    float *sorted = (float *)malloc((size_t)n_blocks * sizeof(float));
    if (!sorted) return -1.0f;   /* the caller skips the check: a floor gate would call every loud gap a dropout */
    memcpy(sorted, rms, (size_t)n_blocks * sizeof(float));
    qsort(sorted, (size_t)n_blocks, sizeof(float), cmp_float);
    float gate = P2_SPEECH_FLOOR_RATIO * sorted[n_blocks / 10];
    free(sorted);
    return gate < P2_WINDOW_FLOOR_RMS ? P2_WINDOW_FLOOR_RMS : gate;
}

float p2_speech_share(const float *rms, int n_blocks, int sample_rate, int from, int to, float gate)
{
    if (to <= from || n_blocks <= 0) return 0.0f;
    int block = (int)(P2_WINDOW_BLOCK_S * sample_rate);
    int b0 = from / block, b1 = (to + block - 1) / block, loud = 0;
    if (b1 > n_blocks) b1 = n_blocks;
    for (int b = b0; b < b1; b++) if (rms[b] > gate) loud++;
    return b1 > b0 ? (float)loud / (b1 - b0) : 0.0f;
}
