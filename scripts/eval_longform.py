#!/usr/bin/env python3
"""Long-form WER: how chunking and segmentation behave on 5–10 minute recordings.

LibriSpeech has no long recordings, but a chapter's utterances are consecutive cuts of one audiobook chapter, so
concatenating them in order — with a short pause between, as the reader left — gives 5–10 minutes of continuous
speech with an exact transcript. Every chapter of the split at or above --min-minutes becomes one file under
data/longform/.

Configurations, all scored with the Whisper normaliser against the concatenated reference:
  utterances     the engine on the original utterances one by one: the per-utterance WER on these words
  engine         the engine's own default on the whole file (pause-aligned 25–35 s windows, segment.c, with its
                 dropout re-decode); engine@<S> moves the window band to [S−10, S] seconds (P2_WINDOW_MAX);
                 engine-raw turns the re-decode off (P2_REDECODE=0) to measure the raw dropout rate
  fermion        the same windows planned by Fermion's segment.py (from the fermion-research package), each piece
                 transcribed on its own – the cross-check for `engine`
  full           one full-length attention pass (P2_SEGMENT=0 P2_CHUNK=0)
  chunk<S>/<O>   the encoder's chunking (P2_SEGMENT=0 P2_CHUNK=1): S frames of 80 ms per chunk, O frames of
                 context each side, the decoder running over the whole file – the pre-2026-10-06 default
  cut<lo>-<hi>   the audio cut at the quietest 100 ms between lo and hi seconds after the previous cut, each piece
                 transcribed on its own
  fixed<S>       the audio cut every S seconds regardless of content, pieces transcribed on their own

    python scripts/eval_longform.py c_weights                        # test-clean chapters ≥ 5 min, the default grid
    python scripts/eval_longform.py c_weights --split test-other
    python scripts/eval_longform.py c_weights --configs full chunk400/8 cut25-35

Needs what eval_librispeech.py needs. Results go to data/results/longform-<split>.json.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_librispeech as el  # noqa: E402

LONGFORM = el.DATA / "longform"
DEFAULT_CONFIGS = ["utterances", "engine", "fermion", "full",
                   "chunk200/8", "chunk300/8", "chunk400/8", "chunk600/8", "chunk800/8",
                   "chunk400/0", "chunk400/32", "chunk400/64", "chunk400/128",
                   "cut25-35", "cut15-25", "cut40-60", "fixed32"]
PIECE_BATCH = 16   # cut pieces per engine invocation
FILE_BATCH = 4     # long files per invocation in the chunked modes; full-length attention runs one at a time


def chapters(split_dir: Path, refs: dict[str, str], min_seconds: float) -> dict[str, list[str]]:
    """chapter id → its utterance ids in order, for chapters with at least min_seconds of audio."""
    by_chapter: dict[str, list[str]] = {}
    for utt in sorted(refs):
        spk, chap, _ = utt.split("-")
        by_chapter.setdefault(f"{spk}-{chap}", []).append(utt)
    out = {}
    for chap, utts in by_chapter.items():
        seconds = sum(sf.info(el.flac_path(split_dir, u)).duration for u in utts)
        if seconds >= min_seconds:
            out[chap] = utts
    return out


def build_longform(split_dir: Path, split: str, chaps: dict[str, list[str]], refs: dict[str, str],
                   gap: float) -> dict[str, Path]:
    """Concatenate each chapter's utterances with `gap` seconds of silence between → chapter → wav path.
    Writes <chapter>.txt (reference) and <chapter>.json (utterance boundaries) beside it."""
    out_dir = LONGFORM / f"{split}-gap{gap:g}"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    built = 0
    for chap, utts in chaps.items():
        wav = out_dir / f"{chap}.wav"
        paths[chap] = wav
        if wav.is_file() and (out_dir / f"{chap}.txt").is_file():
            continue
        pieces, bounds, t = [], [], 0.0
        silence = np.zeros(int(gap * el.SAMPLE_RATE), dtype=np.float32)
        for i, u in enumerate(utts):
            audio, sr = sf.read(el.flac_path(split_dir, u), dtype="float32", always_2d=True)
            assert sr == el.SAMPLE_RATE
            mono = audio.mean(axis=1)
            if i:
                pieces.append(silence)
                t += gap
            bounds.append({"utt": u, "start": round(t, 3), "end": round(t + len(mono) / sr, 3)})
            pieces.append(mono)
            t += len(mono) / sr
        sf.write(wav, np.concatenate(pieces), el.SAMPLE_RATE, subtype="FLOAT")
        (out_dir / f"{chap}.txt").write_text(" ".join(refs[u] for u in utts) + "\n")
        (out_dir / f"{chap}.json").write_text(json.dumps(bounds, indent=0) + "\n")
        built += 1
    if built:
        print(f"built {built} long recordings in {out_dir}", flush=True)
    return paths


def quiet_cuts(audio: np.ndarray, sr: int, lo: float, hi: float) -> list[float]:
    """Cut times: from each cut, the centre of the quietest 100 ms between lo and hi seconds later."""
    hop = sr // 100
    n = len(audio) // hop
    energy = 10 * np.log10(np.mean(audio[:n * hop].reshape(n, hop) ** 2, axis=1) + 1e-10)
    smooth = np.convolve(energy, np.ones(10) / 10, mode="same")
    cuts, t0, total = [], 0.0, len(audio) / sr
    while total - t0 > hi:
        a, b = int((t0 + lo) * 100), min(int((t0 + hi) * 100), n)
        i = a + int(np.argmin(smooth[a:b]))
        t0 = i / 100
        cuts.append(t0)
    return cuts


def fixed_cuts(audio: np.ndarray, sr: int, seconds: float) -> list[float]:
    total = len(audio) / sr
    return [t for t in np.arange(seconds, total, seconds) if total - t > 1.0]


def transcribe_pieces(weights: str, wav: Path, cuts: list[float], threads: int | None, tmp: Path) -> tuple[str, int]:
    """Cut the file at `cuts`, transcribe the pieces separately (no chunking inside a piece), join -> (text, pieces)."""
    audio, sr = sf.read(wav, dtype="float32")
    edges = [0.0, *cuts, len(audio) / sr]
    pieces = []
    for i in range(len(edges) - 1):
        p = tmp / f"{wav.stem}-{i:03d}.wav"
        sf.write(p, audio[int(edges[i] * sr):int(edges[i + 1] * sr)], sr, subtype="FLOAT")
        pieces.append(p)
    hyps = {}
    for i in range(0, len(pieces), PIECE_BATCH):
        batch = pieces[i:i + PIECE_BATCH]
        hyps.update(el.run_engine(weights, batch, threads, {"P2_CHUNK": "0"}))
    for p in pieces:
        p.unlink()
    return " ".join(hyps[p] for p in pieces), len(pieces)


def run_config(name: str, weights: str, wavs: dict[str, Path], threads: int | None,
               utt_wavs: dict[str, Path], chaps: dict[str, list[str]]) -> tuple[dict[str, str], str]:
    """-> chapter → hypothesis, plus a note for the table."""
    if name == "utterances":
        hyps = {}
        for chap, utts in chaps.items():
            paths = [utt_wavs[u] for u in utts]
            got = {}
            for i in range(0, len(paths), el.BATCH):
                got.update(el.run_engine(weights, paths[i:i + el.BATCH], threads))
            hyps[chap] = " ".join(got[p] for p in paths)
        return hyps, f"{sum(len(u) for u in chaps.values())} utterances"

    m = re.fullmatch(r"cut(\d+)-(\d+)", name) or re.fullmatch(r"fixed(\d+)", name)
    if m or name == "fermion":
        tmp = LONGFORM / "tmp"
        tmp.mkdir(exist_ok=True)
        hyps, n_pieces = {}, 0
        for chap, wav in wavs.items():
            audio, sr = sf.read(wav, dtype="float32")
            if name == "fermion":
                from fermion._speech.segment import plan   # the fermion-research package
                cuts = [a / sr for a, _ in plan(audio, sr)[1:]]
            elif name.startswith("cut"):
                cuts = quiet_cuts(audio, sr, float(m.group(1)), float(m.group(2)))
            else:
                cuts = fixed_cuts(audio, sr, float(m.group(1)))
            hyps[chap], k = transcribe_pieces(weights, wav, cuts, threads, tmp)
            n_pieces += k
        return hyps, f"{n_pieces} pieces"

    m = re.fullmatch(r"engine(-raw)?(?:@(\d+))?", name)
    if m:
        env = {"P2_WINDOW_MAX": m.group(2)} if m.group(2) else {}
        if m.group(1):
            env["P2_REDECODE"] = "0"
        batch = FILE_BATCH
        note = (f"the engine's pause-aligned windows{' of up to ' + m.group(2) + ' s' if m.group(2) else ''}"
                f"{', no dropout re-decode' if m.group(1) else ''}")
    elif name == "full":
        env, batch, note = {"P2_SEGMENT": "0", "P2_CHUNK": "0"}, 1, "full-length attention"
    else:
        m = re.fullmatch(r"chunk(\d+)/(\d+)", name)
        if not m:
            raise SystemExit(f"unknown config {name!r}")
        env = {"P2_SEGMENT": "0", "P2_CHUNK": "1", "P2_CHUNK_SIZE": m.group(1), "P2_CHUNK_OVERLAP": m.group(2)}
        batch, note = FILE_BATCH, f"{int(m.group(1)) * 0.08:.0f} s chunks, {int(m.group(2)) * 0.08:.1f} s context"
    paths = list(wavs.values())
    got = {}
    for i in range(0, len(paths), batch):
        got.update(el.run_engine(weights, paths[i:i + batch], threads, env))
    return {chap: got[wav] for chap, wav in wavs.items()}, note


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("weights")
    ap.add_argument("--split", default="test-clean", choices=["test-clean", "test-other", "dev-clean", "dev-other"])
    ap.add_argument("--min-minutes", type=float, default=5.0, help="chapters with at least this much audio")
    ap.add_argument("--gap", type=float, default=0.3, help="seconds of silence between concatenated utterances")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    ap.add_argument("--limit", type=int, default=None, help="only the first N chapters (for a quick look)")
    a = ap.parse_args()
    if not el.BINARY.is_file():
        raise SystemExit(f"{el.BINARY} not built; run make in csrc/")

    from whisper_normalizer.english import EnglishTextNormalizer
    normalize = EnglishTextNormalizer()

    split_dir = el.fetch_split(a.split)
    refs = el.read_transcripts(split_dir)
    chaps = chapters(split_dir, refs, a.min_minutes * 60)
    if a.limit:
        chaps = dict(list(chaps.items())[:a.limit])
    wavs = build_longform(split_dir, a.split, chaps, refs, a.gap)
    seconds = {c: sf.info(w).duration for c, w in wavs.items()}
    print(f"{a.split}: {len(chaps)} chapters ≥ {a.min_minutes:g} min, {sum(seconds.values()) / 60:.0f} min of audio, "
          f"{min(seconds.values()) / 60:.1f}–{max(seconds.values()) / 60:.1f} min each", flush=True)
    ref_norm = {c: normalize((LONGFORM / f"{a.split}-gap{a.gap:g}" / f"{c}.txt").read_text()) for c in chaps}
    words_total = sum(len(r.split()) for r in ref_norm.values())
    utt_wavs = el.ensure_wav(split_dir, a.split, [u for us in chaps.values() for u in us]) if "utterances" in a.configs else {}

    results = {}
    print(f"\n{'config':<14} {'WER':>7} {'edits':>6} {'words':>6} {'time':>7}  note")
    for name in a.configs:
        t0 = time.perf_counter()
        hyps, note = run_config(name, a.weights, wavs, a.threads, utt_wavs, chaps)
        elapsed = time.perf_counter() - t0
        per_file = {}
        edits = 0
        for c in chaps:
            e, n = el.word_errors(ref_norm[c], normalize(hyps[c]))
            per_file[c] = {"edits": e, "words": n, "hyp": hyps[c]}
            edits += e
        wer = 100 * edits / words_total
        results[name] = {"wer": wer, "edits": edits, "words": words_total, "seconds": elapsed, "note": note,
                         "files": per_file}
        print(f"{name:<14} {wer:6.2f}% {edits:6d} {words_total:6d} {elapsed:6.0f}s  {note}", flush=True)

    out = el.DATA / "results" / f"longform-{a.split}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"split": a.split, "gap": a.gap, "min_minutes": a.min_minutes,
                               "chapters": {c: seconds[c] for c in chaps}, "results": results}, indent=1))
    print(f"\nper-file results: {out}")


if __name__ == "__main__":
    main()
