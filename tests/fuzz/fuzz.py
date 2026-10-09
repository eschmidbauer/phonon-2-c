#!/usr/bin/env python3
"""Mutation fuzzing of the engine's file parsers, through sanitizer-instrumented drivers.

    make -C csrc fuzz                      # builds csrc/fuzz_wav and csrc/fuzz_weights with ASan + UBSan, runs this
    python3 tests/fuzz/fuzz.py --iterations 2000 --seed 1

WAV: every tests/data/*.wav is mutated by random byte flips, truncation at random points, and targeted header
edits (chunk sizes 0 / odd / 0xFFFFFFFF, channels 0 / 1 / 7 / 65535, bits 0 / 7 / 12 / 24 / 32 / 64, rates 0 / 1 /
7 / 192000 / 2^31−1, format tags PCM / float / EXTENSIBLE / 0 / 0xFFFF, truncated fmt chunks, data before fmt,
nested RIFF, a LIST chunk of odd size …). Weights: c_weights/weights.json is mutated structurally (fields dropped,
values negated / zeroed / huge, offsets past the file, shapes swapped, deep nesting, unterminated strings, random
flips) against the real weights.bin, and weights.bin itself truncated against the real manifest; tokens.txt gets
fewer / more / longer lines. A driver that exits non-zero, or whose stderr carries a sanitizer report, is a finding:
the input is saved under tests/fuzz/crashes/ and the run fails.

Nothing here needs the model to run – the parsers are the target – so a few thousand cases take a minute.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DATA = HERE.parent / "data"
CRASHES = HERE / "crashes"


def run(cmd: list[str]) -> tuple[bool, str]:
    """-> (clean, detail). Clean = exit 0 within the time limit and no sanitizer report. A hang is a finding too:
    the first run of this harness caught one (a 4 GHz sample rate in a header made the resampler design a tap table
    of tens of gigabytes)."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT after 60 s (hang)"
    bad = p.returncode != 0 or "ERROR: AddressSanitizer" in p.stderr or "runtime error:" in p.stderr or \
        "LeakSanitizer" in p.stderr
    return (not bad), (p.stderr[-1500:] if bad else p.stdout.strip())


# ── WAV mutations ─────────────────────────────────────────────────────────────

def wav_header(fmt_tag=1, channels=1, rate=16000, bits=16, data: bytes = b"", fmt_extra: bytes = b"",
               fmt_size=None, data_size=None, riff_size=None, pre_chunks: bytes = b"", data_first=False) -> bytes:
    block = max(1, channels * max(bits, 1) // 8) & 0xFFFF           # the header fields are 16 / 32-bit: wrap like a
    fmt = struct.pack("<HHIIHH", fmt_tag & 0xFFFF, channels & 0xFFFF, rate & 0xFFFFFFFF, (rate * block) & 0xFFFFFFFF,
                      block, bits & 0xFFFF) + fmt_extra               # careless writer would
    fmt_chunk = b"fmt " + struct.pack("<I", len(fmt) if fmt_size is None else fmt_size) + fmt
    data_chunk = b"data" + struct.pack("<I", len(data) if data_size is None else data_size) + data
    body = (data_chunk + pre_chunks + fmt_chunk) if data_first else (pre_chunks + fmt_chunk + data_chunk)
    return b"RIFF" + struct.pack("<I", 4 + len(body) if riff_size is None else riff_size) + b"WAVE" + body


def wav_cases(rng: random.Random, base: bytes, n: int) -> list[bytes]:
    out = []
    pcm = base[44:44 + 4000]   # some real sample bytes
    targeted = [
        wav_header(channels=0, data=pcm), wav_header(channels=65535, data=pcm), wav_header(channels=7, data=pcm),
        wav_header(bits=0, data=pcm), wav_header(bits=7, data=pcm), wav_header(bits=12, data=pcm),
        wav_header(bits=24, data=pcm), wav_header(bits=32, data=pcm), wav_header(bits=64, data=pcm),
        wav_header(fmt_tag=3, bits=32, data=pcm), wav_header(fmt_tag=3, bits=16, data=pcm),
        wav_header(fmt_tag=0, data=pcm), wav_header(fmt_tag=0xFFFF, data=pcm),
        wav_header(rate=0, data=pcm), wav_header(rate=1, data=pcm), wav_header(rate=7, data=pcm),
        wav_header(rate=192000, data=pcm), wav_header(rate=2**31 - 1, data=pcm), wav_header(rate=2**32 - 1, data=pcm),
        wav_header(rate=4000000007, data=pcm), wav_header(rate=16000 * 257, data=pcm), wav_header(rate=16000 * 255 + 1, data=pcm),
        wav_header(rate=1000003, data=pcm),   # prime: L = 16000 phases, the tap table at its largest legitimate size
        wav_header(data=pcm, data_size=0), wav_header(data=pcm, data_size=0xFFFFFFFF), wav_header(data=pcm, data_size=3),
        wav_header(data=pcm, data_size=len(pcm) + 100), wav_header(data=pcm[:1]), wav_header(data=b""),
        wav_header(data=pcm, fmt_size=0), wav_header(data=pcm, fmt_size=8), wav_header(data=pcm, fmt_size=15),
        wav_header(data=pcm, fmt_size=17), wav_header(data=pcm, fmt_size=0xFFFFFFFF), wav_header(data=pcm, fmt_size=1000),
        wav_header(data=pcm, riff_size=0), wav_header(data=pcm, riff_size=0xFFFFFFFF),
        wav_header(data=pcm, data_first=True),
        wav_header(data=pcm, pre_chunks=b"LIST" + struct.pack("<I", 5) + b"abcde" + b"\x00"),
        wav_header(data=pcm, pre_chunks=b"LIST" + struct.pack("<I", 0xFFFFFFF0)),
        wav_header(data=pcm, pre_chunks=b"junk" + struct.pack("<I", 0)),
        wav_header(data=pcm, pre_chunks=b"RIFF" + struct.pack("<I", 100) + b"WAVE"),
        # EXTENSIBLE with full, short and zero extension
        wav_header(fmt_tag=0xFFFE, data=pcm, fmt_extra=struct.pack("<HHI", 22, 16, 4) + b"\x01\x00" + b"\x00" * 14),
        wav_header(fmt_tag=0xFFFE, data=pcm, fmt_extra=struct.pack("<HHI", 22, 16, 4) + b"\x03\x00" + b"\x00" * 14),
        wav_header(fmt_tag=0xFFFE, data=pcm, fmt_extra=struct.pack("<HHI", 22, 16, 4) + b"\xff\xff" + b"\x00" * 14),
        wav_header(fmt_tag=0xFFFE, data=pcm, fmt_extra=struct.pack("<H", 0)),
        wav_header(fmt_tag=0xFFFE, data=pcm),
        wav_header(fmt_tag=0xFFFE, data=pcm, fmt_extra=b"\x00" * 100),
        b"RIFF", b"RIFF\x00\x00\x00\x00WAVE", b"", b"\x00" * 44, b"RIFX" + base[4:], base[:12], base[:20], base[:43],
        base + b"\x00" * 7,   # trailing garbage
    ]
    out += targeted
    for _ in range(n):
        b = bytearray(base)
        kind = rng.random()
        if kind < 0.4:      # byte flips in the header region
            for _ in range(rng.randint(1, 6)):
                i = rng.randrange(0, min(len(b), 64))
                b[i] = rng.randrange(256)
        elif kind < 0.7:    # truncation
            b = b[:rng.randrange(0, len(b))]
        elif kind < 0.85:   # flips anywhere
            for _ in range(rng.randint(1, 20)):
                b[rng.randrange(len(b))] = rng.randrange(256)
        else:               # header rewrite with random fields on real data
            b = bytearray(wav_header(fmt_tag=rng.choice([1, 3, 0xFFFE, rng.randrange(65536)]),
                                     channels=rng.choice([1, 2, 6, 0, rng.randrange(65536)]),
                                     rate=rng.choice([16000, 8000, 44100, 0, rng.randrange(2**32)]),
                                     bits=rng.choice([8, 16, 24, 32, 0, rng.randrange(256)]), data=bytes(pcm),
                                     data_size=rng.choice([None, 0, 0xFFFFFFFF, rng.randrange(2**32)])))
        out.append(bytes(b))
    return out


# ── Manifest / weights / tokens mutations ──────────────────────────────────────

def manifest_cases(rng: random.Random, text: str, n: int) -> list[str]:
    try:
        doc = json.loads(text)
    except json.JSONDecodeError:
        doc = None
    out = ["", "{}", "[]", "null", "{", "{\"tensors\":[", "\x00" * 100, text[:len(text) // 2], text + "}}}}",
           text.replace('"', ""), "{" * 100000 + "}" * 100000, "[" * 50000, '{"a":"' + "x" * 1000000 + '"}',
           text.replace(":", ":-"), text.replace("0", "99999999999999999999"), text.replace("0", "-1"),
           text.replace(",", ",,"), text.replace("[", "[["), text.encode("utf-16").decode("latin-1")]
    for _ in range(n):
        kind = rng.random()
        if kind < 0.35 or doc is None:
            b = bytearray(text.encode())
            for _ in range(rng.randint(1, 30)):
                b[rng.randrange(len(b))] = rng.randrange(32, 127)
            out.append(b.decode("latin-1"))
        elif kind < 0.5:
            out.append(text[:rng.randrange(len(text))])
        else:
            d = json.loads(json.dumps(doc))   # deep copy, then structural edits

            def mutate(node, depth=0):
                if isinstance(node, dict):
                    for k in list(node):
                        if rng.random() < 0.03: del node[k]; continue
                        if rng.random() < 0.02: node[k + "_x"] = node[k]
                        node[k] = mutate(node[k], depth + 1)
                    return node
                if isinstance(node, list):
                    if rng.random() < 0.02: return node[::-1]
                    if rng.random() < 0.02 and node: return node[:1]
                    return [mutate(v, depth + 1) for v in node]
                if isinstance(node, int) and not isinstance(node, bool):
                    r = rng.random()
                    if r < 0.03: return -node
                    if r < 0.05: return 0
                    if r < 0.07: return node * 1000003
                    if r < 0.08: return 2**63 - 1
                    if r < 0.09: return node + rng.randrange(-64, 64)
                    if r < 0.10: return str(node)
                    return node
                if isinstance(node, str):
                    r = rng.random()
                    if r < 0.02: return node + "x"
                    if r < 0.03: return ""
                    if r < 0.035: return node * 50
                    return node
                return node
            out.append(json.dumps(mutate(d)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iterations", type=int, default=800, help="random mutations per input family")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fuzz-wav", default=str(ROOT / "csrc" / "fuzz_wav"))
    ap.add_argument("--fuzz-weights", default=str(ROOT / "csrc" / "fuzz_weights"))
    ap.add_argument("--weights", default=str(ROOT / "c_weights"))
    a = ap.parse_args()
    rng = random.Random(a.seed)
    for b in (a.fuzz_wav, a.fuzz_weights):
        if not Path(b).is_file():
            raise SystemExit(f"{b} not built; run `make -C csrc fuzz`")
    findings = []
    tmp = Path(tempfile.mkdtemp(prefix="p2fuzz-"))
    try:
        # WAV
        total = 0
        for wav in sorted(DATA.glob("*.wav")):
            base = wav.read_bytes()
            for i, case in enumerate(wav_cases(rng, base, a.iterations)):
                p = tmp / f"w{i}.wav"
                p.write_bytes(case)
                for extra in ([], ["8000"], ["44100"]):
                    ok, detail = run([a.fuzz_wav, str(p), *extra])
                    total += 1
                    if not ok:
                        findings.append(("wav", case, detail, extra))
        print(f"wav: {total} runs, {len([f for f in findings if f[0] == 'wav'])} findings")

        # Manifest against the real binary
        wdir = Path(a.weights)
        text = (wdir / "weights.json").read_text()
        total = 0
        for i, case in enumerate(manifest_cases(rng, text, a.iterations)):
            p = tmp / f"m{i}.json"
            p.write_text(case, encoding="latin-1")
            ok, detail = run([a.fuzz_weights, str(wdir / "weights.bin"), str(p)])
            total += 1
            if not ok:
                findings.append(("manifest", case.encode("latin-1"), detail, []))
        print(f"manifest: {total} runs, {len([f for f in findings if f[0] == 'manifest'])} findings")

        # Truncated / tiny binaries against the real manifest, and mutated tokens.txt
        total = 0
        full = (wdir / "weights.bin").stat().st_size
        for size in [0, 1, 63, 64, 4096, 1 << 20, full // 2, full - 1]:
            p = tmp / f"b{size}.bin"
            with open(wdir / "weights.bin", "rb") as src, open(p, "wb") as dst:
                dst.write(src.read(size))
            ok, detail = run([a.fuzz_weights, str(p), str(wdir / "weights.json")])
            total += 1
            if not ok:
                findings.append(("binary", str(size).encode(), detail, []))
        tokens = (wdir / "tokens.txt").read_text().splitlines()
        for i, lines in enumerate([tokens[:-1], tokens + ["extra"], ["x" * 10000] * len(tokens), [], ["a"],
                                   [t + "\r" for t in tokens], ["\x00" * 10] * len(tokens)]):
            p = tmp / f"t{i}.txt"
            p.write_text("\n".join(lines) + ("\n" if lines else ""))
            ok, detail = run([a.fuzz_weights, str(wdir / "weights.bin"), str(wdir / "weights.json"), str(p)])
            total += 1
            if not ok:
                findings.append(("tokens", f"case {i}".encode(), detail, []))
        print(f"binary/tokens: {total} runs, {len([f for f in findings if f[0] in ('binary', 'tokens')])} findings")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if findings:
        CRASHES.mkdir(exist_ok=True)
        for k, (kind, data, detail, extra) in enumerate(findings):
            p = CRASHES / f"{kind}-{a.seed}-{k}.bin"
            p.write_bytes(data)
            print(f"\nFINDING {kind} {extra} → {p}\n{detail[-800:]}")
        sys.exit(1)
    print("no findings")


if __name__ == "__main__":
    main()
