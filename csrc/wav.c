/*  wav.c – WAV reader: PCM 8/16/24/32-bit and 32-bit float, any channel count (mixed down to mono), plain or
 *  WAVE_FORMAT_EXTENSIBLE headers, unknown chunks skipped. Returns float32 samples at the file's own rate, which
 *  p2_transcribe takes as is (it resamples to the model's 16 kHz with p2_resample).
 */

#include "phonon2.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define WAVE_FORMAT_PCM        1
#define WAVE_FORMAT_IEEE_FLOAT 3
#define WAVE_FORMAT_EXTENSIBLE 0xFFFE

static uint16_t rd16(const unsigned char *p) { return (uint16_t)(p[0] | (p[1] << 8)); }
static uint32_t rd32(const unsigned char *p) { return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24); }

/* One interleaved frame of `channels` samples at byte `p` → the mean of the channels as a float in [-1, 1). */
static float frame_to_mono(const unsigned char *p, int channels, int format, int bytes_per_sample)
{
    float sum = 0.0f;
    for (int c = 0; c < channels; c++, p += bytes_per_sample) {
        float v;
        if (format == WAVE_FORMAT_IEEE_FLOAT) {
            float f; memcpy(&f, p, 4); v = f;
        } else {
            switch (bytes_per_sample) {
            case 1:  v = ((int)p[0] - 128) / 128.0f; break;
            case 2:  v = (int16_t)rd16(p) / 32768.0f; break;
            case 3:  v = (float)((int32_t)((uint32_t)p[0] << 8 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 24) >> 8) / 8388608.0f; break;
            default: v = (int32_t)rd32(p) / 2147483648.0f; break;
            }
        }
        sum += v;
    }
    return sum / channels;
}

void p2_pcm16le_to_float(const unsigned char *bytes, int n_frames, int channels, float *out)
{
    if (channels < 1) channels = 1;
    for (int i = 0; i < n_frames; i++)
        out[i] = frame_to_mono(bytes + (size_t)i * 2 * channels, channels, WAVE_FORMAT_PCM, 2);
}

int p2_wav_read(const char *path, float **out_samples, int *out_count, int *out_sr, int *out_channels)
{
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return -1; }

    unsigned char hdr[12];
    if (fread(hdr, 1, 12, f) != 12 || memcmp(hdr, "RIFF", 4) != 0 || memcmp(hdr + 8, "WAVE", 4) != 0) {
        fprintf(stderr, "%s: not a WAV file\n", path);
        fclose(f);
        return -1;
    }

    int format = 0, channels = 0, bits = 0;
    uint32_t sample_rate = 0;
    unsigned char *data = NULL;
    uint32_t data_size = 0;
    int found_fmt = 0;

    for (;;) {
        unsigned char ch[8];
        if (fread(ch, 1, 8, f) != 8) break;
        uint32_t size = rd32(ch + 4);
        if (memcmp(ch, "fmt ", 4) == 0) {
            unsigned char fmt[40];
            uint32_t n = size < sizeof(fmt) ? size : sizeof(fmt);
            if (fread(fmt, 1, n, f) != n) break;
            if (size > n) fseek(f, size - n, SEEK_CUR);
            if (size < 16) break;   /* a fmt chunk is at least 16 bytes; shorter is a broken file, not a format */
            format      = rd16(fmt);
            channels    = rd16(fmt + 2);
            sample_rate = rd32(fmt + 4);
            bits        = rd16(fmt + 14);
            if (format == WAVE_FORMAT_EXTENSIBLE && size >= 40)
                format = rd16(fmt + 24);   /* first two bytes of the sub-format GUID */
            found_fmt = 1;
        } else if (memcmp(ch, "data", 4) == 0) {
            long here = ftell(f);
            fseek(f, 0, SEEK_END);
            long remaining = ftell(f) - here;
            fseek(f, here, SEEK_SET);
            if (size > (uint32_t)remaining || size == 0xFFFFFFFFu) size = (uint32_t)remaining;   /* streamed files lie here */
            data = (unsigned char *)malloc(size);
            if (!data || fread(data, 1, size, f) != size) {
                fprintf(stderr, "%s: truncated data chunk\n", path);
                free(data);
                fclose(f);
                return -1;
            }
            data_size = size;
            break;
        } else {
            fseek(f, size, SEEK_CUR);
        }
        if (size & 1) fseek(f, 1, SEEK_CUR);   /* chunks are padded to even sizes */
    }
    fclose(f);

    if (!found_fmt || !data) {
        fprintf(stderr, "%s: missing fmt or data chunk\n", path);
        free(data);
        return -1;
    }
    int bps = bits / 8;
    int ok = (format == WAVE_FORMAT_PCM && (bits == 8 || bits == 16 || bits == 24 || bits == 32)) ||
             (format == WAVE_FORMAT_IEEE_FLOAT && bits == 32);
    /* The rate is reported as an int; a header claiming more than INT_MAX Hz is garbage, not a rate. */
    if (!ok || channels < 1 || sample_rate == 0 || sample_rate > 0x7FFFFFFFu) {
        fprintf(stderr, "%s: unsupported format %d / %d-bit / %d channel(s)\n", path, format, bits, channels);
        free(data);
        return -1;
    }

    int frame_bytes = bps * channels;
    int n = (int)(data_size / frame_bytes);
    float *samples = (float *)malloc((size_t)n * sizeof(float));
    if (!samples) { free(data); return -1; }
    for (int i = 0; i < n; i++)
        samples[i] = frame_to_mono(data + (size_t)i * frame_bytes, channels, format, bps);
    free(data);
    /* A float file can carry NaNs and infinities; one would silence a whole window through the mel normalisation
     * (tests/fuzz). They become zero here, at the file boundary, so every consumer – the CLI's own resample and
     * feature paths included – sees finite samples. */
    if (format == WAVE_FORMAT_IEEE_FLOAT)
        for (int i = 0; i < n; i++)
            if (!p2_is_finite(samples[i])) samples[i] = 0.0f;

    *out_samples = samples;
    *out_count = n;
    *out_sr = (int)sample_rate;
    *out_channels = channels;
    return 0;
}
