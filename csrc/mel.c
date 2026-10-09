/*  mel.c – Log-mel features matching transformers' ParakeetFeatureExtractor.
 *
 *  pre-emphasis 0.97 → STFT with torch.stft's conventions (n_fft 512, hop 160, 256 zeros of padding each side, a
 *  symmetric Hann window of 400 zero-padded to 512 and centred) → power → slaney mel (128 bins, 0–8 kHz) →
 *  log(x + 2^-24) → per-feature normalisation over the valid frames: (x − mean) / (std + 1e-5), unbiased variance.
 *
 *  The extractor marks n_samples / 160 frames valid and masks the one STFT frame after them; only the valid frames are
 *  computed and returned here, which gives the encoder the same input it sees under transformers.
 */

#include "phonon2.h"
#include "threadpool.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define LOG_ZERO_GUARD 5.9604644775390625e-08f   /* 2^-24 */
#define NORMALIZE_EPS  1e-5f

/* ── Real FFT (radix-2 Cooley-Tukey) ─────────────────────────────────────── */

/* In-place FFT of N complex values (interleaved real/imag). N must be a power of 2. */
static void fft_inplace(float *data, int N)
{
    for (int i = 1, j = 0; i < N; i++) {
        int bit = N >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j) {
            float tr = data[2 * i], ti = data[2 * i + 1];
            data[2 * i] = data[2 * j];
            data[2 * i + 1] = data[2 * j + 1];
            data[2 * j] = tr;
            data[2 * j + 1] = ti;
        }
    }

    for (int len = 2; len <= N; len <<= 1) {
        double angle = -2.0 * M_PI / len;
        float wr = (float)cos(angle), wi = (float)sin(angle);
        for (int i = 0; i < N; i += len) {
            float cur_r = 1.0f, cur_i = 0.0f;
            for (int j = 0; j < len / 2; j++) {
                int u = i + j, v = i + j + len / 2;
                float tr = data[2 * v] * cur_r - data[2 * v + 1] * cur_i;
                float ti = data[2 * v] * cur_i + data[2 * v + 1] * cur_r;
                data[2 * v]     = data[2 * u]     - tr;
                data[2 * v + 1] = data[2 * u + 1] - ti;
                data[2 * u]     += tr;
                data[2 * u + 1] += ti;
                float new_r = cur_r * wr - cur_i * wi;
                float new_i = cur_r * wi + cur_i * wr;
                cur_r = new_r;
                cur_i = new_i;
            }
        }
    }
}

/* ── Slaney mel filterbank (librosa.filters.mel with norm="slaney") ──────── */

static double hz_to_mel(double hz)
{
    if (hz < 1000.0) return hz * 3.0 / 200.0;
    return 15.0 + 27.0 * log(hz / 1000.0) / log(6.4);
}

static double mel_to_hz(double mel)
{
    if (mel < 15.0) return mel * 200.0 / 3.0;
    return 1000.0 * exp((mel - 15.0) * log(6.4) / 27.0);
}

/* [n_mels × (n_fft/2 + 1)] */
static float *make_mel_filterbank(int n_mels, int n_fft, int sr)
{
    int n_freqs = n_fft / 2 + 1;
    float *fb = (float *)calloc((size_t)n_mels * n_freqs, sizeof(float));

    double mel_lo = hz_to_mel(0.0);
    double mel_hi = hz_to_mel(sr / 2.0);
    double *mels = (double *)malloc((n_mels + 2) * sizeof(double));
    for (int i = 0; i < n_mels + 2; i++)
        mels[i] = mel_to_hz(mel_lo + (mel_hi - mel_lo) * i / (n_mels + 1));

    for (int m = 0; m < n_mels; m++) {
        double lo = mels[m], center = mels[m + 1], hi = mels[m + 2];
        double norm = 2.0 / (hi - lo);
        for (int f = 0; f < n_freqs; f++) {
            double hz = (double)sr * f / n_fft;
            double val = 0.0;
            if (hz >= lo && hz <= center)
                val = (hz - lo) / (center - lo);
            else if (hz > center && hz <= hi)
                val = (hi - hz) / (hi - center);
            fb[m * n_freqs + f] = (float)(val * norm);
        }
    }

    free(mels);
    return fb;
}

/* ── STFT + mel, parallel across frames ──────────────────────────────────── */

struct mel_stft_args {
    const float *padded;
    const float *window;     /* [n_fft], the Hann window already centred and zero-padded */
    const float *fb;
    int n_fft, n_freqs, n_mels, n_frames, hop_len;
    float *mel;              /* [n_frames × n_mels] */
};

static void mel_stft_worker(int tid, int nt, void *aux)
{
    struct mel_stft_args *a = (struct mel_stft_args *)aux;
    int f0 = (a->n_frames * tid) / nt;
    int f1 = (a->n_frames * (tid + 1)) / nt;
    if (f0 >= f1) return;

    float *fft_buf = (float *)malloc(2 * a->n_fft * sizeof(float));
    float *power   = (float *)malloc(a->n_freqs * sizeof(float));

    for (int f = f0; f < f1; f++) {
        const float *frame = a->padded + (size_t)f * a->hop_len;
        for (int i = 0; i < a->n_fft; i++) {
            fft_buf[2 * i]     = frame[i] * a->window[i];
            fft_buf[2 * i + 1] = 0.0f;
        }
        fft_inplace(fft_buf, a->n_fft);

        for (int k = 0; k < a->n_freqs; k++) {
            float re = fft_buf[2 * k], im = fft_buf[2 * k + 1];
            power[k] = re * re + im * im;
        }

        float *out = a->mel + (size_t)f * a->n_mels;
        for (int m = 0; m < a->n_mels; m++) {
            const float *row = a->fb + (size_t)m * a->n_freqs;
            float val = 0.0f;
            for (int k = 0; k < a->n_freqs; k++) val += row[k] * power[k];
            out[m] = logf(val + LOG_ZERO_GUARD);
        }
    }

    free(power);
    free(fft_buf);
}

/* ── Public API ──────────────────────────────────────────────────────────── */

int p2_mel_spectrogram(const float *audio, int n_samples, float **out_mel, int *out_frames)
{
    int n_fft = P2_N_FFT, win_len = P2_WIN_LEN, hop_len = P2_HOP_LEN, n_mels = P2_N_MELS;
    int n_freqs = n_fft / 2 + 1;

    int n_frames = n_samples / hop_len;
    if (n_frames < 2) {
        fprintf(stderr, "audio too short: %d samples\n", n_samples);
        return -1;
    }

    /* Pre-emphasis, then n_fft/2 zeros on each side (torch.stft center=True, pad_mode="constant"). */
    int pad = n_fft / 2;
    float *padded = (float *)calloc((size_t)n_samples + 2 * pad, sizeof(float));
    padded[pad] = audio[0];
    for (int i = 1; i < n_samples; i++)
        padded[pad + i] = audio[i] - P2_PREEMPHASIS * audio[i - 1];

    /* Symmetric Hann of win_len, centred inside the n_fft frame. */
    float *window = (float *)calloc(n_fft, sizeof(float));
    int off = (n_fft - win_len) / 2;
    for (int i = 0; i < win_len; i++)
        window[off + i] = (float)(0.5 - 0.5 * cos(2.0 * M_PI * i / (win_len - 1)));

    float *fb = make_mel_filterbank(n_mels, n_fft, P2_SAMPLE_RATE);
    float *mel = (float *)malloc((size_t)n_frames * n_mels * sizeof(float));

    struct mel_stft_args args = { padded, window, fb, n_fft, n_freqs, n_mels, n_frames, hop_len, mel };
    p2_parallel(p2_pool(), mel_stft_worker, &args);

    free(fb);
    free(window);
    free(padded);

    /* Per-feature normalisation over the valid frames. */
    double mean[P2_N_MELS], var[P2_N_MELS];
    memset(mean, 0, sizeof(mean));
    memset(var, 0, sizeof(var));
    for (int f = 0; f < n_frames; f++)
        for (int m = 0; m < n_mels; m++) mean[m] += mel[(size_t)f * n_mels + m];
    for (int m = 0; m < n_mels; m++) mean[m] /= n_frames;
    for (int f = 0; f < n_frames; f++)
        for (int m = 0; m < n_mels; m++) {
            double d = mel[(size_t)f * n_mels + m] - mean[m];
            var[m] += d * d;
        }
    float inv[P2_N_MELS], mu[P2_N_MELS];
    for (int m = 0; m < n_mels; m++) {
        float std = (float)sqrt(var[m] / (n_frames - 1));
        inv[m] = 1.0f / (std + NORMALIZE_EPS);
        mu[m] = (float)mean[m];
    }
    for (int f = 0; f < n_frames; f++)
        for (int m = 0; m < n_mels; m++)
            mel[(size_t)f * n_mels + m] = (mel[(size_t)f * n_mels + m] - mu[m]) * inv[m];

    *out_mel = mel;
    *out_frames = n_frames;
    return 0;
}
