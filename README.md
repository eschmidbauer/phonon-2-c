# phonon-2-c

A pure-C inference engine for [Phonon-2](https://huggingface.co/FermionResearch/Phonon-2), Fermion Research's
five-value re-training of NVIDIA's Parakeet-TDT 0.6B v3 — by its model card the most accurate open English speech
recogniser under 900 MB (5.21 % average WER on the Open ASR Leaderboard's seven sets). Version 0.1.0.

The engine keeps the model's own weight format: every encoder linear weight is one of five values per row,
`{0, ±lo, ±hi}`, stored as a 2-bit sign plane and a 1-bit plane. The GEMMs run on NEON int8 dot products straight
from those planes, or — in exact mode — on fp32 after expanding a 16-column panel at a time. The whole model runs
from **255 MB of weights**; the exact mode reproduces the `transformers` reference **token for token**, the default
mode transcribes a 256 s call in **1.7 s** (150× realtime) and a 3 s utterance in **0.03 s** on an M5 Max.

```bash
$ ./csrc/phonon2 c_weights recording.wav
Okay. Um I'm looking to see if you guys have any Mercedes bins in the S series …

$ ./csrc/phonon2 --json c_weights recording.wav
{"text":"Okay. Um I'm …","words":[{"text":"Okay.","start":12.800,"end":13.200}, …
```

No dependencies beyond libc, libm and pthreads — every GEMM, the mel front end and the resampler are in the tree.

## Features

- **FastConformer encoder** (24 blocks, d=1024, 8-head relative-position attention), **2-layer LSTM prediction
  network**, **TDT joint network** with a 5-way duration head, **greedy token-and-duration decoding** — all in C
- **Five-value GEMM**: the 216 quantised encoder weights stay quantised on disk and in memory (229 MB). Default: the
  activations are quantised per row to int8 and the weights go through NEON `SDOT` as int8 codes — 2.3 TMAC/s on
  18 cores, 4× the fp32 path. `--exact` keeps everything fp32 on the original weights.
- **Verified against transformers' `ParakeetForTDT`**: in exact mode, identical token ids on a 10 s clip and on a
  256 s call given the same features; in the default mode, the same WER on LibriSpeech
- **Log-mel front end matching transformers' `ParakeetFeatureExtractor`** (pre-emphasis, zero-padded STFT, slaney mel,
  per-feature normalisation)
- **Any WAV in**: 8/16/24/32-bit PCM or 32-bit float, any sample rate, any channel count — mixed down and resampled
  to 16 kHz by the engine's own polyphase resampler, with no change in WER
- **A C library** behind the CLI: `p2_engine_open` / `p2_transcribe` / `p2_result_free` give text, word times and
  tokens from samples at any rate; a **stream** (`p2_stream_feed` / `p2_stream_next` / `p2_stream_partial`) for
  audio that arrives over time; every stage is exported too; CMake installs it with a `phonon2::phonon2` target
- **Python bindings** in one dependency-free file (`python/phonon2.py`, `ctypes` over the shared library)
- **Word timestamps** from the decoder's emission frames (`--json`)
- **Long audio cut at pauses** into 25–35 s windows (the model's regime; the same windows as Fermion's runtime),
  each decoded on its own, with a re-decode of any window the model went silent in — 1.79 % / 3.19 % WER on
  5–10 minute LibriSpeech recordings (full-length attention: 1.82 / 3.40; fixed 32 s chunking: 40 %)
- **Thread pool** over every GEMM, norm, softmax and layout shuffle; **NEON** and **AVX2** inner loops with scalar fallbacks
- **Concurrent transcription** of several files over one loaded model

## Quick start

```bash
# 1. build
make -C csrc                     # ./csrc/phonon2, ./csrc/libphonon2.a and the shared ./csrc/libphonon2.dylib|.so

# 2. weights: downloads the converted weights (187 MB) from huggingface.co/eschmidbauer/phonon-2-c, checks the
#    pinned sha256 and unpacks them into c_weights/ (255 MB); `make -C csrc weights` is the same thing
scripts/get_weights.sh

# 3. run (any WAV: PCM or float, any rate, mono or multi-channel)
./csrc/phonon2 c_weights recording.wav
```

CMake works too: `cmake -S . -B build && cmake --build build`; `cmake --install build` puts `libphonon2.a`, the
headers (`include/phonon2/`), the exported `phonon2::phonon2` target and the binary under the prefix.
`FetchContent` on this repository gives the same target.

The weights can also be built from Phonon-2's own release instead of downloaded: `pip install -r requirements.txt`
(numpy, zstandard, huggingface_hub) and `python convert_weights.py` fetch `phonon-2.bps.tar.zst` (164 MB) from
FermionResearch/Phonon-2 and write the same `c_weights/` byte for byte; see "How the weights are converted".

The int8 GEMMs need AArch64 with the dot-product extension (any Apple silicon, Neoverse, Cortex-A75+); every other
CPU — x86-64 with or without AVX2, older ARM — builds and runs the exact fp32 path, at roughly the `exact` speeds
below scaled by the machine. The x86-64 AVX2 and SSE2 builds, the generic armv8 build and the scalar-only build
(`-DP2_SCALAR`) all give the same tokens as the native exact mode on the test clips (x86 run under Rosetta 2).

### Options

```
phonon2 [options] <weights_dir> <audio.wav> [audio2.wav ...]
  --json          text plus one {text, start, end} per word
  --tokens        token ids instead of text
  --mel-in FILE   read log-mel features (raw float32 [frames × 128]) instead of computing them
  --mel-out FILE  write the computed features in that layout
  --threads N     thread-pool size (default: online CPUs; also P2_THREADS)
  --exact         fp32 arithmetic on the five-value weights instead of int8 dot products (also P2_EXACT=1)
  --cache-codes   pre-pack the int8 weight codes once (604 MB of RAM): short utterances ~30 % faster (also P2_CACHE_CODES=1)
  --repeat N      run the pipeline N times on one file and report the fastest
  --quiet         no encoder progress on stderr
  --stream        no audio files: 16-bit mono PCM on stdin, one line per finished 25–35 s window
  --rate R        with --stream: the PCM's sample rate (default 16000; anything else is resampled on the way)
  --partial S     with --stream: a provisional transcript of the pending audio on stderr every S seconds
```

`P2_GEMM=exact|int8|int8w` picks the GEMM mode (default `int8w`; `int8` keeps the weights exact and quantises only
the activations, at half the speed). Audio over 35 s is cut at pauses into 25–35 s windows (see *Long audio*);
`P2_SEGMENT=0` hands the whole file to the encoder instead, where `P2_CHUNK=0` means one full-length attention
pass (quadratic memory: ~2 GB at 256 s) and `P2_CHUNK=1` the chunked encoder (`P2_CHUNK_SIZE` /
`P2_CHUNK_OVERLAP` in 80 ms frames, default 400 / 8) — both kept for comparison, neither a good idea on long files.

### Input

Any RIFF WAV: PCM 8, 16, 24 or 32-bit or IEEE float, plain or `WAVE_FORMAT_EXTENSIBLE`, any sample rate, any number
of channels. Channels are averaged to mono, and anything not at 16 kHz goes through the engine's windowed-sinc
polyphase resampler (Kaiser β = 9, cut off at 0.46 of the lower rate; 66–106 dB against analytic test signals,
0.03 s for 256 s of 44.1 or 48 kHz audio). The library takes mono float32 at any rate and does the same inside
`p2_transcribe`.

## Library

The CLI is a consumer of the library in `csrc/`; `make -C csrc` builds `libphonon2.a` next to it, and the whole API
is `csrc/phonon2.h`:

```c
#include "phonon2.h"

P2Engine *engine = p2_engine_open("c_weights");               /* weights.bin, weights.json, tokens.txt */
float *audio; int n, rate, channels;
p2_wav_read("recording.wav", &audio, &n, &rate, &channels);    /* any PCM / float WAV, mixed to mono */
P2Result r;
p2_transcribe(engine, audio, n, rate, &r);                     /* resample → mel → encoder → TDT → text */
printf("%s\n", r.text);
for (int i = 0; i < r.n_words; i++)
    printf("%.2f %.2f %s\n", r.words[i].start, r.words[i].end, r.words[i].text);
p2_result_free(&r); free(audio); p2_engine_close(engine);
```

`examples/transcribe.c` is that program (`make -C csrc example`, or the `transcribe` target of the CMake build).
`p2_transcribe` may be called from any number of threads over one engine.

For audio that arrives over time there is a stream: `p2_stream_open(engine, rate)`, `p2_stream_feed` (mono float32
at that rate, any chunk size — resampled on the way by a stateful `P2Resampler` whose output is bit for bit the
whole-file resampler's), `p2_stream_next` for each window as it completes, `p2_stream_partial` for a provisional
transcript of the audio pending since the last cut, `p2_stream_finish`, `p2_stream_result`. Windows are the model's
regime, so final text arrives at window granularity — up to 35 s after the speech — while partials (a decode of the
pending audio, ≤ 35 s, about 0.2 s of work) give provisional text as often as you like; a stream fed a whole file
emits exactly the windows `p2_transcribe` gives for it (`make check` holds it to that).
`phonon2 --stream [--rate R] [--partial S] [--json] c_weights < audio.pcm` is the CLI form: 16-bit little-endian
mono PCM in (16 kHz unless `--rate` says otherwise), one line per finished window out, with `--json` a segment
object carrying that window's words. `p2_pcm16le_to_float` is the library's converter for such raw PCM.

### Python

`python/phonon2.py` is the same API from Python, over the shared library `make -C csrc` builds next to the static
one (`csrc/libphonon2.dylib` / `.so`; found through `PHONON2_LIB` or its default location). One file, `ctypes`
only; numpy arrays are accepted when numpy is present but nothing is required:

```python
import phonon2
engine = phonon2.Engine("c_weights")                 # cache_codes=True pre-packs the int8 codes
result = engine.transcribe_file("recording.wav")     # or engine.transcribe(samples, rate)
print(result.text)
for w in result.words:
    print(f"{w.start:7.2f} {w.end:7.2f}  {w.text}")

with engine.stream(rate=48000) as s:                 # audio arriving over time, any rate
    for chunk in chunks:
        for seg in s.feed(chunk):
            print(seg.start, seg.end, seg.text)
    for seg in s.finish():
        print(seg.start, seg.end, seg.text)
```

`Result` carries `text`, `words`, `tokens`, `segments` (each with its own words and tokens), `encoder_frames`,
`duration_seconds` and the stage times; `phonon2.read_wav`, `phonon2.resample` and `phonon2.pcm16le_to_float` expose
the audio utilities. `python3 python/phonon2.py c_weights recording.wav` runs it as a script. `make check`
exercises the bindings against the CLI's output.

The stages behind it — `p2_resample`,
`p2_mel_spectrogram`, `p2_encoder`, `p2_greedy_decode`, `p2_tokens_text`, `p2_tokens_words` — are exported for
callers that want them one at a time, and `p2_transcribe_mel` starts from features. Thread count, GEMM mode and
chunking are read from `P2_THREADS`, `P2_GEMM`, `P2_CHUNK…` on first use; a server can instead give each of its
threads a pool of its own (`p2_pool_create`, `p2_set_pool` in `threadpool.h`). The library is compiled with
`-march=native`: build it on the machine that runs it.

## Speed

Apple M5 Max (6 performance + 12 efficiency cores), 18 threads, idle machine, best of several runs; one 256 s phone
call and 3 s / 10 s clips cut from it:

| Mode | 3 s utterance | 10 s clip | 256 s call, 25–35 s windows (default) | 256 s call, `P2_REDECODE=0` | 256 s, chunked encoder | 256 s, full-length attention |
| --- | --- | --- | --- | --- | --- | --- |
| `int8w` (default) | **0.032 s · 94×** | **0.072 s · 139×** | **3.0 s · 86× realtime** | **1.67 s · 153×** | 1.71 s · 150× | 3.5 s · 73× |
| `int8w` + `--cache-codes` | 0.023 s · 130× | 0.062 s · 161× | | 1.63 s | | |
| `exact` | 0.083 s · 36× | 0.21 s · 48× | 5.2 s · 50× | | 5.3 s · 48× | 6.6 s · 39× |

Of which the decoder is 0.3 s on the long file. The default's 3.0 s on this call against 1.67 s without the
re-decode is its 8 re-decodes (far-end speech the model misses and line noise, see *Long audio*); read speech pays
nothing for them — an 8.4 min chapter runs in about 3.5 s. Fermion's own CPU engine did the call in 2.25 s on this
machine (measured under the background load described below, when this engine took 2.7 s). `--cache-codes`
pre-packs the int8 weight codes once (604 MB) so that each pass skips expanding the sign and flag planes — a fixed
~10 ms per pass, which is a third of a 3 s utterance and nothing on long audio.

The int8 path is sensitive to background load in a way exact mode is not: with Docker's VM holding ~1.6 of the 18
cores the same call took 2.7 s and the 10 s clip 0.12 s, while exact mode was unchanged — its many short fork-joins
each wait for whichever worker was descheduled. An optional Accelerate / OpenBLAS build existed until 2026-10-06: it
matched the hand-rolled code in the default mode and was faster only in exact mode (3.6 s on the chunked call, the
AMX units); it was removed so that the library has one code path and no dependency.

The thread pool spins briefly between dispatches (`P2_SPIN`, default 8000 iterations ≈ 100 µs; `0` parks at once).
`--repeat N` runs the pipeline N times on one file and reports the fastest, for latency work.

## Accuracy and verification

### Word error rate

`scripts/eval_librispeech.py` fetches a LibriSpeech split from OpenSLR, runs the engine over it and scores with the
Whisper English normaliser (what the Open ASR Leaderboard uses):

```bash
python scripts/eval_librispeech.py c_weights                     # test-clean
python scripts/eval_librispeech.py c_weights --split test-other
```

| Set | `int8w` (default) | `exact` | Fermion's own engines, same 300-utterance subset | Model card |
| --- | --: | --: | --- | --: |
| test-clean (2,620 utterances, 5.4 h) | **2.16 %** | 2.16 % | MLX 2.13 %, CPU 2.09 % (this engine: exact 2.18 %, int8w 2.06 %) | 1.72 % |
| test-other (2,939 utterances, 5.3 h) | **4.34 %** | 4.33 % | | 3.92 % |

torch `ParakeetForTDT` scores the same as the exact mode on every utterance of the subset, and Fermion's shipped
engines land within noise of it, so the gap to the card is the leaderboard's scoring protocol (its dataset variant,
reference text and normaliser), not the engine. The errors are the model's: proper-noun spellings ("boolooroo" →
"booleroo"), "missus" written as "misses", function-word swaps.

The same subset fed as 22.05, 44.1 or 48 kHz, stereo, 24-bit or 8-bit WAVs (`--rate`, `--channels`, `--subtype`)
scores within noise of the 16 kHz float baseline (2.06–2.18 % against 2.06 %); 8 kHz input costs 0.2 points, which
is the missing 4–8 kHz band, not the resampler.

### Parity with transformers

`scripts/compare_reference.py` runs the C engine and `transformers` on the same recording:

```bash
python scripts/compare_reference.py c_weights recording.wav --phonon2-repo /path/to/FermionResearch/Phonon-2
```

- **Graph parity** (both sides fed the same log-mel features, exact mode, full-length attention): **identical token
  ids** — 52 of 52 on the 10 s clip, 910 of 910 on the 256 s call. This is the pass/fail criterion for the graph.
- **Default int8 mode**: ~50 token edits in 910 against the exact run on the 256 s call, 3 in 52 on the clip — the
  same scale as Fermion's bf16 and int8 engines show against torch. Its accuracy is the WER above, not parity.
- **End to end** with the C mel front end: 20 token edits in ~900 on the long call. Two feature extractors never agree
  exactly in silent mel bins, where per-feature normalisation amplifies float32 rounding to O(1); the numpy, torch and
  C extractors all differ from each other by that much, and the model's transcript of the surrounding words shifts
  slightly. The median feature difference is 1e-6.

### Long audio

Phonon-2 is a ≤ 35 s model with a sharp failure mode: given a window that begins mid-speech it can emit nothing
for 10–30 s of clear speech and resume later — torch does the same on the same samples, and the outcome flips with
a 0.5 s shift of the cut. `scripts/eval_longform.py` measures this on 5–8 minute recordings built by concatenating
LibriSpeech test-clean chapters (15 chapters, 116 min, exact references):

| Long-form strategy | test-clean (15 chapters, 116 min) | test-other (18 chapters, 121 min) |
| --- | --: | --: |
| the original utterances one by one | 1.76 % | 3.36 % |
| one full-length attention pass per recording | 1.82 % | 3.40 % |
| **pause-aligned 25–35 s windows with the dropout re-decode (the engine's default)** | **1.79 %** | **3.19 %** |
| the same windows without the re-decode (Fermion's runtime; `P2_REDECODE=0`) | 2.35 % | 3.87 % |
| cut at the quietest point every 15–25 s, pieces decoded alone | 1.78 % | |
| cut every 32 s regardless of content, pieces decoded alone | 26.0 % | |
| chunked encoder, 32 s chunks (the previous default; 16–64 s chunks, 0–10 s overlap all similar) | 40.4 % | 25.2 % |

So the engine cuts audio over 35 s into 25–35 s windows ending in the longest pause of each band — the same windows
Fermion's runtime uses — and decodes each on its own. Even then the model drops a window now and then (8 times in
those 237 minutes, 216 words), so every decoded window is checked for a stretch of ≥ 2 s without a token over
speech energy and re-decoded as two halves split at a pause when it has one; that brings the dropout count to zero
on both sets. `--json` lists the windows under `"segments"`. The check is energy-based, so on telephone audio with
line noise it also re-decodes the odd window for nothing, and far-end speech it recovers costs a re-decode too —
the 256 s call takes 3.0 s instead of 1.67 s (`P2_REDECODE=0`); clean read speech pays nothing (11 re-decodes in
237 minutes).

## Tests

`make -C csrc check` runs the built engine over three LibriSpeech test-clean utterances in `tests/data/` (CC BY 4.0,
~1 MB): exact-mode token ids must match `tests/expected/` byte for byte — they do on Apple silicon, x86-64 (AVX2 and
SSE2), generic ARM and the scalar-only build — and the default mode, `--json`, batched invocation, a 22.05 kHz
input and `examples/transcribe` must agree with the reference text and each other. Needs Python 3 and the converted
weights, nothing else. [`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs it on macOS (Apple silicon),
x86-64 Linux and ARM Linux on every push, with the CMake build and install and the scalar-only build.

`make -C csrc fuzz` rebuilds the library with AddressSanitizer and UBSan and throws a few thousand mutated WAV
files, weights manifests, truncated weight binaries and token files at the parsers (`tests/fuzz/`); CI runs a short
round on Linux. Non-finite samples (a NaN or infinity in a float WAV or a live feed) are zeroed on the way in — one
would otherwise silence a whole window through the mel normalisation.

## How the weights are converted

`convert_weights.py` reads the `model.fermion` container with Fermion's reference reader (`fermion_container.py`,
vendored unchanged) and writes `weights.bin` + `weights.json` + `tokens.txt`:

| Tensors | Count | Stored as |
| --- | --: | --- |
| Encoder linears (FF, Q/K/V fused, position, output, pointwise convs) | 216 | five-value: 2-bit signs + 1-bit lo/hi flags, `[in, out]`, fp32 `lo`/`hi` per output — 229 MB |
| Subsampling linear, encoder projector | 2 | int8 with one fp32 scale per output column (exact from the container's int6), used as stored |
| Embedding, LSTM, decoder projector, joint head | 7 | int8 with one fp32 scale per row (exact), expanded to fp32 when loaded — 17 MB on disk |
| Norms, biases, subsampling convs, depthwise conv (BatchNorm folded in) | 354 | fp32 |

Everything is exact except the BatchNorm fold (fp32 arithmetic). The vocabulary comes from the tarball's own
config, so no other download is needed.

`--archive` also packs the three files into `phonon-2-c-weights-v1.tar.gz`, built reproducibly (ustar, fixed
order, zero timestamps and owners, no gzip header name) so every run gives the same bytes. That file is what
[huggingface.co/eschmidbauer/phonon-2-c](https://huggingface.co/eschmidbauer/phonon-2-c) publishes, with its
sha256 and the three files' in `SHA256SUMS.txt` there. That repo carries verbatim copies of `convert_weights.py`
and `fermion_container.py` and a `build.sh` that rebuilds and checks the archive; a change to the converter here
is a change there too. `weights.json` records the layout's name under `_meta` (`phonon-2-c-weights-v1`); the loader
refuses any other, so a stale download or an old conversion fails with a message instead of running on garbage.

## Project layout

| File | Lines | Purpose |
| --- | --: | --- |
| `csrc/phonon2.h` | 425 | Model constants, `Weight` kinds, weight structs, the whole API (engine, stream, resampler, components) |
| `csrc/engine.c` | 474 | `P2Engine`, `p2_transcribe`: windows → mel → encoder → decoder → text, words, segments; the dropout re-decode; `P2Stream` |
| `csrc/segment.c` | 184 | Pause-aligned 25–35 s windows for long audio (Fermion's rule), pause and speech measures |
| `csrc/blas.c` | 1392 | fp32 sgemm (NEON/AVX2, packed, blocked) and the five-value / int8 GEMMs — no BLAS library |
| `csrc/encoder.c` | 762 | Subsampling, conformer blocks, chunking, projector |
| `csrc/decoder.c` | 141 | LSTM, joint, greedy TDT |
| `csrc/mel.c` | 217 | Log-mel features |
| `csrc/tensor_ops.c` | 649 | LayerNorm, softmax, SiLU, depthwise conv, row GEMV, producer-side int8 quantisation |
| `csrc/weights.c` | 527 | Manifest parser and strict loader, the int8 code cache |
| `csrc/wav.c`, `resample.c` | 346 | WAV reader (any PCM / float format, mixdown), stateful polyphase resampler |
| `csrc/main.c`, `threadpool.c`, `tokens.c` | 864 | CLI (files, JSON, `--stream`), spinning fork-join pool, detokeniser and word times |
| `examples/transcribe.c` | 47 | The library API end to end |
| `python/phonon2.py` | 400 | Python bindings (`ctypes` over `libphonon2`): `Engine`, `Stream`, the audio utilities |
| `csrc/bench_gemm.c` | 58 | GEMM microbenchmark (`make -C csrc bench_gemm`) |
| `scripts/get_weights.sh` | 37 | Downloads the published weights, checks the pinned sha256, unpacks them |
| `convert_weights.py` | 277 | Container → C weights, `--archive` packs them for the Hub |
| `scripts/compare_reference.py` | 177 | Parity check against transformers |
| `scripts/eval_librispeech.py` | 205 | WER on a LibriSpeech split, optionally through resampled / multi-channel / PCM variants |
| `scripts/eval_longform.py` | 247 | Long-form WER on concatenated chapters: windows, chunking and cutting strategies side by side |
| `tests/check.py`, `tests/fuzz/` | 533 | The regression check behind `make check`; the sanitizer fuzz harness behind `make fuzz` |

Derived from [parakeet-unified-en-0.6b-c](https://github.com/eschmidbauer/parakeet-unified-en-0.6b-c), which has
the same encoder family; the differences that matter for Phonon-2 (no linear biases, BatchNorm after the depthwise
conv, no input scaling, PyTorch LSTM gate order, TDT decoding, an 8193-token vocabulary) are all here.

## Licence

Engine code: MIT. `fermion_container.py`: Apache-2.0, Fermion Research. The weights the converter downloads are
CC-BY-4.0 (Fermion Research's derivation of NVIDIA's parakeet-tdt-0.6b-v3); see `NOTICE`.
