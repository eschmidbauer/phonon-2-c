#!/usr/bin/env python3
"""Word error rate of the C engine on a LibriSpeech split.

    python scripts/eval_librispeech.py c_weights                       # test-clean, all 2,620 utterances
    python scripts/eval_librispeech.py c_weights --split test-other
    python scripts/eval_librispeech.py c_weights --limit 200 --seed 0  # a random subset

Fetches the split's tarball from OpenSLR into data/ when it is not there, converts the FLAC files to 16 kHz WAV once
(data/wav/<split>/), runs csrc/phonon2 over them in batches, normalises references and hypotheses with the Whisper
English normaliser (what the Open ASR Leaderboard scores with) and reports corpus WER. Per-utterance results go to
data/results/<split>.jsonl for inspection; --show N prints the N worst utterances.

Needs numpy, soundfile, jiwer and whisper-normalizer. The engine's own chunking never triggers here: every LibriSpeech
utterance is under 35 s.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent.parent
BINARY = HERE / "csrc" / "phonon2"
DATA = HERE / "data"
OPENSLR = "https://www.openslr.org/resources/12/{split}.tar.gz"   # OpenSLR hosts LibriSpeech (SLR12)
SAMPLE_RATE = 16000
BATCH = 16   # WAV files per phonon2 invocation; the CLI runs them concurrently over one loaded model


def fetch_split(split: str) -> Path:
    """-> data/LibriSpeech/<split>, downloading and unpacking the OpenSLR tarball when needed."""
    out = DATA / "LibriSpeech" / split
    if out.is_dir():
        return out
    DATA.mkdir(exist_ok=True)
    tarball = DATA / f"{split}.tar.gz"
    if not tarball.is_file():
        print(f"downloading {split}.tar.gz from OpenSLR …", flush=True)
        urllib.request.urlretrieve(OPENSLR.format(split=split), tarball)
    print(f"unpacking {tarball.name} …", flush=True)
    with tarfile.open(tarball) as tar:
        tar.extractall(DATA, filter="data")
    return out


def read_transcripts(split_dir: Path) -> dict[str, str]:
    """utterance id → reference text, from every *.trans.txt under the split."""
    refs = {}
    for trans in sorted(split_dir.rglob("*.trans.txt")):
        for line in trans.read_text().splitlines():
            utt, _, text = line.partition(" ")
            refs[utt] = text
    return refs


def flac_path(split_dir: Path, utt: str) -> Path:
    speaker, chapter, _ = utt.split("-")
    return split_dir / speaker / chapter / f"{utt}.flac"


def ensure_wav(split_dir: Path, split: str, utts: list[str], rate: int = SAMPLE_RATE, channels: int = 1,
               subtype: str = "FLOAT") -> dict[str, Path]:
    """FLAC → WAV, once per utterance. The default is the engine's native 16 kHz mono float; --rate / --channels /
    --subtype produce variants (scipy polyphase resampling, duplicated channels) that exercise the engine's own
    input conversion. -> utt → wav path"""
    tag = "" if (rate, channels, subtype) == (SAMPLE_RATE, 1, "FLOAT") else f"@{rate}Hz-{channels}ch-{subtype}"
    wav_dir = DATA / "wav" / (split + tag)
    wav_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    converted = 0
    for utt in utts:
        wav = wav_dir / f"{utt}.wav"
        if not wav.is_file():
            audio, sr = sf.read(flac_path(split_dir, utt), dtype="float32", always_2d=True)
            if sr != SAMPLE_RATE:
                raise SystemExit(f"{utt}: {sr} Hz; LibriSpeech is expected at {SAMPLE_RATE} Hz")
            mono = audio.mean(axis=1)
            if rate != SAMPLE_RATE:
                from math import gcd
                from scipy.signal import resample_poly
                g = gcd(rate, SAMPLE_RATE)
                mono = resample_poly(mono, rate // g, SAMPLE_RATE // g).astype(np.float32)
            out = np.stack([mono] * channels, axis=1) if channels > 1 else mono
            sf.write(wav, out, rate, subtype=subtype)
            converted += 1
        paths[utt] = wav
    if converted:
        print(f"converted {converted} FLAC files to WAV{' ' + tag if tag else ''}", flush=True)
    return paths


def run_engine(weights: str, wavs: list[Path], threads: int | None, extra_env: dict[str, str] | None = None) -> dict[Path, str]:
    """One phonon2 invocation over several files -> path → transcript. extra_env sets P2_* variables for the run."""
    env = dict(os.environ)
    if threads:
        env["P2_THREADS"] = str(threads)
    if extra_env:
        env.update(extra_env)
    proc = subprocess.run([str(BINARY), "--quiet", weights, *map(str, wavs)], capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise SystemExit(f"phonon2 failed:\n{proc.stderr[-2000:]}")
    out = {}
    if len(wavs) == 1:
        out[wavs[0]] = proc.stdout.rstrip("\n")
        return out
    by_name = {str(p): p for p in wavs}
    for line in proc.stdout.splitlines():
        name, sep, text = line.partition(": ")
        if sep and name in by_name:
            out[by_name[name]] = text
    missing = [p for p in wavs if p not in out]
    if missing:
        raise SystemExit(f"no transcript for {missing[:3]} …\nstderr: {proc.stderr[-1000:]}")
    return out


def word_errors(ref: str, hyp: str) -> tuple[int, int]:
    """(edits, reference words) for one utterance, both already normalised."""
    import jiwer
    r, h = ref.split(), hyp.split()
    if not r:
        return len(h), 0
    o = jiwer.process_words(" ".join(r), " ".join(h) if h else "")
    return o.substitutions + o.deletions + o.insertions, len(r)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("weights")
    ap.add_argument("--split", default="test-clean", choices=["test-clean", "test-other", "dev-clean", "dev-other"])
    ap.add_argument("--limit", type=int, default=None, help="evaluate a random subset of this many utterances")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=None, help="P2_THREADS for the engine")
    ap.add_argument("--show", type=int, default=10, help="print the N worst utterances")
    ap.add_argument("--rate", type=int, default=SAMPLE_RATE, help="feed the engine WAVs at this sample rate (tests its resampler)")
    ap.add_argument("--channels", type=int, default=1, help="feed the engine WAVs with this many (identical) channels")
    ap.add_argument("--subtype", default="FLOAT", help="WAV sample format: FLOAT, PCM_16, PCM_24, PCM_U8, ...")
    a = ap.parse_args()
    if not BINARY.is_file():
        raise SystemExit(f"{BINARY} not built; run make in csrc/")

    from whisper_normalizer.english import EnglishTextNormalizer
    normalize = EnglishTextNormalizer()

    split_dir = fetch_split(a.split)
    refs = read_transcripts(split_dir)
    utts = sorted(refs)
    if a.limit and a.limit < len(utts):
        utts = sorted(random.Random(a.seed).sample(utts, a.limit))
    wavs = ensure_wav(split_dir, a.split, utts, a.rate, a.channels, a.subtype)
    seconds = sum(sf.info(wavs[u]).duration for u in utts)
    variant = "" if (a.rate, a.channels, a.subtype) == (SAMPLE_RATE, 1, "FLOAT") else f" as {a.rate} Hz, {a.channels} ch, {a.subtype}"
    print(f"{a.split}: {len(utts)} utterances, {seconds / 3600:.2f} h{variant}", flush=True)

    results = []
    t0 = time.perf_counter()
    for i in range(0, len(utts), BATCH):
        batch = utts[i:i + BATCH]
        hyps = run_engine(a.weights, [wavs[u] for u in batch], a.threads)
        for u in batch:
            hyp = hyps[wavs[u]]
            ref_n, hyp_n = normalize(refs[u]), normalize(hyp)
            edits, n = word_errors(ref_n, hyp_n)
            results.append({"utt": u, "ref": refs[u], "hyp": hyp, "ref_norm": ref_n, "hyp_norm": hyp_n,
                            "edits": edits, "words": n})
        done = i + len(batch)
        e = sum(r["edits"] for r in results); w = sum(r["words"] for r in results)
        print(f"  {done:5d}/{len(utts)}  WER so far {100 * e / max(w, 1):.2f} %  ({time.perf_counter() - t0:.0f} s)",
              end="\r", flush=True)
    elapsed = time.perf_counter() - t0
    print()

    edits = sum(r["edits"] for r in results)
    words = sum(r["words"] for r in results)
    wer = 100 * edits / words
    exact = sum(1 for r in results if r["edits"] == 0)
    print(f"\n{a.split}: WER {wer:.2f} %  ({edits} edits / {words} words; {exact}/{len(results)} utterances exact)")
    print(f"engine time {elapsed:.0f} s for {seconds / 3600:.2f} h of audio = {seconds / elapsed:.0f}x realtime "
          f"including process start-up per batch of {BATCH}")

    out_dir = DATA / "results"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{a.split}{'' if not a.limit else f'-{a.limit}'}{variant.replace(' as ', '@').replace(', ', '-').replace(' ', '')}.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results))
    print(f"per-utterance results: {out}")

    if a.show:
        worst = sorted(results, key=lambda r: -r["edits"])[:a.show]
        print(f"\n{a.show} worst utterances (normalised):")
        for r in worst:
            print(f"  {r['utt']}  {r['edits']} edits / {r['words']} words\n    ref: {r['ref_norm']}\n    hyp: {r['hyp_norm']}")


if __name__ == "__main__":
    main()
