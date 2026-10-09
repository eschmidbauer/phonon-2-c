# tests

`make -C csrc check` (or `python3 tests/check.py`) runs the built engine over the three LibriSpeech test-clean
utterances in `data/` and checks:

- **exact mode** gives the token ids in `expected/` byte for byte — the same fp32 graph, the same argmax, on every CPU
  the engine has been built for;
- the **default mode**'s text is within one word edit of the reference (the int8 modes differ from exact by a token
  now and then);
- `--json` is consistent with the text, several files in one invocation give the same texts as one by one, a
  22.05 kHz copy gives the same text as the 16 kHz original, and `examples/transcribe` agrees with the CLI;
- `--stream` fed a 48 s concatenation (and the 22.05 kHz copy with `--rate`) reproduces the batch windows and words;
- the Python bindings (`python/phonon2.py` over `csrc/libphonon2.*`) give the CLI's text, tokens and windows.

It needs only Python 3, the built binary and the converted weights (`scripts/get_weights.sh`). `--update`
rewrites `expected/` after an intentional change to the exact path. The audio is CC BY 4.0 (see `NOTICE`).

`.github/workflows/ci.yml` runs the build and this check on macOS (Apple silicon), x86-64 Linux and ARM Linux, plus
the CMake build and install and, on ARM Linux, the scalar-only build (`-DP2_SCALAR`). A separate `weights-drift` job
converts Phonon-2's release with `convert_weights.py --archive` and fails if the result is not the archive
`scripts/get_weights.sh` pins, or if the Hub repo's copies of the converter differ from the ones here.

## fuzz/

`make -C csrc fuzz` (or `python3 tests/fuzz/fuzz.py --iterations N` after it has built the drivers) rebuilds the
library with AddressSanitizer + UBSan into `csrc/fuzz_obj/`, builds `fuzz_wav` (p2_wav_read, optionally
p2_resample) and `fuzz_weights` (p2_load_weights, p2_tokens_load), and runs the mutator: targeted header edits and
random corruptions of the WAVs in `data/`, structural and random
mutations of `c_weights/weights.json` against the real binary, truncated binaries against the real manifest, and
malformed token files. Any non-zero exit, hang or sanitizer report is saved under `fuzz/crashes/` and fails the
run. CI runs 150 mutations per family on x86-64 Linux. `make -C csrc fuzz_phonon2` builds the CLI itself under the
sanitizers, for running odd audio by hand.
