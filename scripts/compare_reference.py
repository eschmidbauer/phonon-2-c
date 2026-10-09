#!/usr/bin/env python3
"""Parity check: the C engine against transformers' ParakeetForTDT on one recording.

Graph parity feeds both sides the same log-mel features (computed here in numpy, handed to the C binary through
--mel-in), runs the engine in its exact mode (P2_GEMM=exact: fp32 arithmetic on the five-value weights) and expects
identical token ids. The end-to-end run with the C mel front end is reported separately: two feature extractors never
agree bit for bit in silent mel bins (per-feature normalisation blows float32 rounding up to O(1) there), so a few
tokens may differ. The default int8 GEMM mode and the chunked encoder are reported the same way: as drift, not
pass/fail — their accuracy is judged by scripts/eval_librispeech.py.

    python scripts/compare_reference.py c_weights alexi-16K.wav --phonon2-repo ~/src/huggingface.co/FermionResearch/Phonon-2

Needs numpy and soundfile; the torch side needs torch, transformers and the Phonon-2 repo's reference_transformers.py
(use that repo's .venv). The torch tokens are cached next to the audio as <audio>.ref_tokens.json so later runs need
only the C binary.
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
BINARY = HERE / "csrc" / "phonon2"
BASE_MODEL = "nvidia/parakeet-tdt-0.6b-v3"
BLANK = 8192

# transformers' ParakeetFeatureExtractor, numerically the same as csrc/mel.c
SR, N_FFT, WIN, HOP, N_MELS, PREEMPH = 16000, 512, 400, 160, 128, 0.97
LOG_GUARD, NORM_EPS = 2.0 ** -24, 1e-5


def slaney_mel_filters() -> np.ndarray:
    def hz2mel(hz): return hz * 3 / 200 if hz < 1000 else 15 + 27 * np.log(hz / 1000) / np.log(6.4)
    def mel2hz(m): return m * 200 / 3 if m < 15 else 1000 * np.exp((m - 15) * np.log(6.4) / 27)
    n_freqs = N_FFT // 2 + 1
    mels = np.array([mel2hz(hz2mel(0) + (hz2mel(SR / 2) - hz2mel(0)) * i / (N_MELS + 1)) for i in range(N_MELS + 2)])
    freqs = np.arange(n_freqs) * SR / N_FFT
    fb = np.zeros((N_MELS, n_freqs))
    for m in range(N_MELS):
        lo, c, hi = mels[m], mels[m + 1], mels[m + 2]
        norm = 2 / (hi - lo)
        for f in range(n_freqs):
            if lo <= freqs[f] <= c:
                fb[m, f] = (freqs[f] - lo) / (c - lo) * norm
            elif c < freqs[f] <= hi:
                fb[m, f] = (hi - freqs[f]) / (hi - c) * norm
    return fb.astype(np.float32)


def log_mel(wave: np.ndarray) -> np.ndarray:
    """[valid_frames × 128] float32, valid_frames = len // 160."""
    x = wave.astype(np.float32)
    x = np.concatenate([x[:1], x[1:] - np.float32(PREEMPH) * x[:-1]])
    pad = N_FFT // 2
    xp = np.pad(x, (pad, pad))
    valid = len(x) // HOP
    stride = xp.strides[0]
    frames = np.lib.stride_tricks.as_strided(xp, shape=(valid, N_FFT), strides=(stride * HOP, stride))
    window = np.zeros(N_FFT, dtype=np.float32)
    off = (N_FFT - WIN) // 2
    window[off:off + WIN] = np.hanning(WIN)
    spec = np.fft.rfft(frames * window, axis=1)
    power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)
    mel = np.log(power @ slaney_mel_filters().T + np.float32(LOG_GUARD))
    mean = mel.mean(axis=0)
    std = np.sqrt(((mel - mean) ** 2).sum(axis=0) / (valid - 1))
    return ((mel - mean) / (std + np.float32(NORM_EPS))).astype(np.float32)


def read_audio(path: str) -> np.ndarray:
    import soundfile as sf
    wave, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != SR:
        raise SystemExit(f"{path}: {sr} Hz, this check wants {SR} Hz")
    return wave.mean(axis=1) if wave.shape[1] > 1 else wave[:, 0]


def torch_tokens(feats: np.ndarray, repo: Path, container: Path) -> list[int]:
    """Non-blank token ids from ParakeetForTDT.generate on these features."""
    import torch
    sys.path.insert(0, str(repo))
    from reference_transformers import load_model
    model, _, _ = load_model(str(container), BASE_MODEL)
    x = torch.from_numpy(feats[None])
    mask = torch.ones(x.shape[:2], dtype=torch.long)
    with torch.no_grad():
        gen = model.generate(input_features=x, attention_mask=mask)
    return [t for t in gen.sequences[0].tolist() if t != BLANK]


def run_c(weights: str, audio: str, extra: list[str], env: dict) -> tuple[list[int], float, str]:
    """-> (token ids, wall seconds, stderr tail)"""
    t0 = time.perf_counter()
    proc = subprocess.run([str(BINARY), "--tokens", "--quiet", *extra, weights, audio],
                          capture_output=True, text=True, env={**os.environ, **env})
    if proc.returncode != 0:
        raise SystemExit(f"phonon2 failed:\n{proc.stderr}")
    ids = [int(t) for t in proc.stdout.split()]
    return ids, time.perf_counter() - t0, proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else ""


def token_edits(a: list[int], b: list[int]) -> int:
    ops = difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes()
    return sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in ops if tag != "equal")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("weights")
    ap.add_argument("audio")
    ap.add_argument("--phonon2-repo", type=Path, default=None, help="checkout of FermionResearch/Phonon-2 (for the torch side)")
    ap.add_argument("--container", type=Path, default=None, help="model.fermion; default <repo>/model_phonon2_c4c_int6/model.fermion")
    a = ap.parse_args()
    if not BINARY.is_file():
        raise SystemExit(f"{BINARY} not built; run make in csrc/")

    wave = read_audio(a.audio)
    feats = log_mel(wave)
    print(f"{a.audio}: {len(wave) / SR:.1f}s, {feats.shape[0]} feature frames")

    cache = Path(a.audio + ".ref_tokens.json")
    if cache.is_file():
        ref = json.loads(cache.read_text())
        print(f"torch tokens from {cache.name}: {len(ref)}")
    elif a.phonon2_repo:
        container = a.container or a.phonon2_repo / "model_phonon2_c4c_int6" / "model.fermion"
        t0 = time.perf_counter()
        ref = torch_tokens(feats, a.phonon2_repo, container)
        print(f"torch: {len(ref)} tokens in {time.perf_counter() - t0:.1f}s (cached to {cache.name})")
        cache.write_text(json.dumps(ref))
    else:
        raise SystemExit("no cached torch tokens; pass --phonon2-repo for the torch side")

    with tempfile.TemporaryDirectory() as tmp:
        feat_file = Path(tmp) / "features.f32"
        feats.tofile(feat_file)
        mel_file = Path(tmp) / "c_mel.f32"

        exact = {"P2_CHUNK": "0", "P2_GEMM": "exact"}
        ids, secs, line = run_c(a.weights, a.audio, ["--mel-in", str(feat_file)], exact)
        same = ids == ref
        print(f"graph parity (same features, exact GEMMs, full-length encoder): {len(ids)} tokens, "
              f"{'IDENTICAL' if same else f'{token_edits(ref, ids)} edits'}; {secs:.1f}s wall\n    {line}")

        ids_i8, secs, line = run_c(a.weights, a.audio, ["--mel-in", str(feat_file)], {"P2_CHUNK": "0"})
        print(f"default GEMM mode (same features, full-length encoder): {len(ids_i8)} tokens, "
              f"{token_edits(ids, ids_i8)} edits vs exact; {secs:.1f}s wall\n    {line}")

        ids_e2e, secs, line = run_c(a.weights, a.audio, ["--mel-out", str(mel_file)], exact)
        c_mel = np.fromfile(mel_file, dtype=np.float32).reshape(-1, N_MELS)
        n = min(len(c_mel), len(feats))
        diff = np.abs(c_mel[:n] - feats[:n])
        print(f"C mel vs numpy: {len(c_mel)} vs {len(feats)} frames; abs diff median {np.median(diff):.1e}, "
              f"99th pct {np.quantile(diff, 0.99):.1e}, max {diff.max():.1e}")
        print(f"end to end (C mel, exact GEMMs, full-length encoder): {len(ids_e2e)} tokens, {token_edits(ref, ids_e2e)} edits "
              f"vs torch; {secs:.1f}s wall\n    {line}")

        ids_chunk, secs, line = run_c(a.weights, a.audio, [], {})
        print(f"end to end (C mel, default GEMM mode and chunking): {len(ids_chunk)} tokens, {token_edits(ids_e2e, ids_chunk)} "
              f"edits vs the exact full-length run, {token_edits(ref, ids_chunk)} vs torch; {secs:.1f}s wall\n    {line}")

    if not same:
        raise SystemExit("graph parity FAILED")
    print("graph parity passed")


if __name__ == "__main__":
    main()
