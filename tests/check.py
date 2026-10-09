#!/usr/bin/env python3
"""Regression test for the built engine: `make -C csrc check`, or

    python3 tests/check.py [--binary csrc/phonon2] [--weights c_weights] [--example examples/transcribe]

For every tests/data/<utt>.wav with a <utt>.txt reference (LibriSpeech test-clean, CC BY 4.0, see NOTICE):

  exact      P2_GEMM=exact token ids must equal tests/expected/<utt>.tokens byte for byte – the same fp32 graph
             gives the same argmax on every CPU tried (Apple silicon, x86-64 AVX2 and SSE2, generic ARM, scalar)
  default    the default mode's text must be within --edits word edits of the reference (int8 modes differ from
             exact by a token now and then; on CPUs without the dot-product extension the default *is* exact)
  json       --json parses, carries one word per text word and the file's duration
  batch      all files in one invocation give the same texts as one by one (the concurrent path)
  resampled  <utt>.<rate>Hz.wav gives the same text as the 16 kHz original, within --edits
  example    examples/transcribe (when built) prints the default text on its first line

Exit status 1 on any failure. Needs only Python 3 and the built binary; refresh the expectations with --update
after an intentional change to the exact path (and say why in the commit).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import struct
import subprocess
import sys
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = HERE / "data"
EXPECTED = HERE / "expected"


def run(binary: str, weights: str, files: list[str], opts: list[str] = (), env_extra: dict | None = None) -> str:
    """phonon2 [opts] <weights> <files…> → stdout."""
    env = dict(os.environ, **(env_extra or {}))
    cmd = [binary, "--quiet", *opts, weights, *files]
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if p.returncode != 0:
        raise SystemExit(f"{' '.join(cmd)} failed:\n{p.stderr[-2000:]}")
    return p.stdout.rstrip("\n")


def normalise(text: str) -> list[str]:
    """Lower-case words, punctuation dropped (apostrophes kept): enough for read speech with spelled-out numbers."""
    return re.findall(r"[a-z0-9']+", text.lower())


def word_edits(ref: list[str], hyp: list[str]) -> int:
    """Levenshtein distance over words."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def wav_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", default=str(ROOT / "csrc" / "phonon2"))
    ap.add_argument("--weights", default=str(ROOT / "c_weights"))
    ap.add_argument("--example", default=str(ROOT / "examples" / "transcribe"))
    ap.add_argument("--edits", type=int, default=1, help="word edits allowed per utterance in the default mode")
    ap.add_argument("--update", action="store_true", help="rewrite tests/expected from the exact mode's output")
    a = ap.parse_args()
    if not Path(a.binary).is_file():
        raise SystemExit(f"{a.binary} not built")
    if not (Path(a.weights) / "weights.bin").is_file():
        raise SystemExit(f"no weights in {a.weights}; run convert_weights.py")

    utts = sorted(p.stem for p in DATA.glob("*.wav") if "Hz" not in p.stem and (DATA / f"{p.stem}.txt").is_file())
    if not utts:
        raise SystemExit(f"no test utterances in {DATA}")
    EXPECTED.mkdir(exist_ok=True)
    failures: list[str] = []
    texts: dict[str, str] = {}

    def check(ok: bool, what: str, detail: str = "") -> None:
        print(f"  {'ok  ' if ok else 'FAIL'} {what}" + (f"\n       {detail}" if detail and not ok else ""))
        if not ok:
            failures.append(what)

    for utt in utts:
        wav = DATA / f"{utt}.wav"
        ref = normalise((DATA / f"{utt}.txt").read_text())
        print(f"{utt} ({wav_seconds(wav):.1f} s, {len(ref)} words)")

        tokens = run(a.binary, a.weights, [str(wav)], ["--tokens"], {"P2_GEMM": "exact"})
        exp_path = EXPECTED / f"{utt}.tokens"
        if a.update or not exp_path.is_file():
            exp_path.write_text(tokens + "\n")
            print(f"  wrote {exp_path.relative_to(ROOT)} ({len(tokens.split())} tokens)")
        expected = exp_path.read_text().strip()
        check(tokens == expected, "exact-mode tokens match tests/expected",
              f"{len(tokens.split())} tokens vs {len(expected.split())} expected; "
              f"{word_edits(expected.split(), tokens.split())} token edits")

        text = run(a.binary, a.weights, [str(wav)])
        texts[utt] = text
        e = word_edits(ref, normalise(text))
        check(e <= a.edits, f"default-mode text within {a.edits} edit(s) of the reference ({e})", text)

        js = json.loads(run(a.binary, a.weights, [str(wav)], ["--json"]))
        check(js["text"] == text and len(js["words"]) == len(text.split()) and
              abs(js["duration_seconds"] - wav_seconds(wav)) < 0.01,
              "--json: same text, one word per text word, duration", json.dumps(js)[:200])

    out = run(a.binary, a.weights, [str(DATA / f"{u}.wav") for u in utts])
    batch = {Path(line.partition(": ")[0]).stem: line.partition(": ")[2] for line in out.splitlines()}
    print("batch")
    check(all(batch.get(u) == texts[u] for u in utts), f"{len(utts)} files in one invocation give the same texts",
          json.dumps(batch)[:300])

    for wav in sorted(DATA.glob("*Hz.wav")):
        utt = wav.stem.split(".")[0]
        print(wav.name)
        text = run(a.binary, a.weights, [str(wav)])
        e = word_edits(normalise(texts[utt]), normalise(text))
        check(e <= a.edits, f"resampled input within {a.edits} edit(s) of the 16 kHz text ({e})", text)

    if Path(a.example).is_file():
        print("example")
        utt = utts[0]
        p = subprocess.run([a.example, a.weights, str(DATA / f"{utt}.wav")], capture_output=True, text=True)
        first = p.stdout.split("\n", 1)[0]
        check(p.returncode == 0 and first == texts[utt], "examples/transcribe prints the default text",
              first or p.stderr[-300:])

    # --stream must reproduce the batch result: the three utterances concatenated twice (~48 s) make one window cut
    # inside the stream plus a final window, fed as 16-bit PCM in 100 ms pieces.
    print("stream")
    import tempfile
    frames = b""
    for u in utts * 2:
        with wave.open(str(DATA / f"{u}.wav"), "rb") as w:
            assert w.getsampwidth() == 2 and w.getnchannels() == 1 and w.getframerate() == 16000
            frames += w.readframes(w.getnframes())
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        with wave.open(tmp, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(frames)
        long_wav = tmp.name
    try:
        batch = json.loads(run(a.binary, a.weights, [long_wav], ["--json"]))
        p = subprocess.Popen([a.binary, "--quiet", "--stream", "--json", a.weights], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        for i in range(0, len(frames), 3200):
            p.stdin.write(frames[i:i + 3200])
        out, err = p.communicate()
        streamed = [json.loads(line) for line in out.decode().splitlines()]
        same = (p.returncode == 0 and len(streamed) == len(batch["segments"]) >= 2 and
                all(abs(s["start"] - b["start"]) < 1e-3 and abs(s["end"] - b["end"]) < 1e-3 and s["text"] == b["text"]
                    for s, b in zip(streamed, batch["segments"])))
        check(same, f"--stream gives the batch result's {len(batch['segments'])} windows ({len(frames) / 32000:.0f} s of audio)",
              f"stream {[ (s['start'], s['end']) for s in streamed ]} vs batch {[ (s['start'], s['end']) for s in batch['segments'] ]}; "
              f"{err.decode()[-300:]}")
        # word times come from 80 ms frames offset to the window start, so a boundary word may sit up to a frame
        # outside its window's [start, end); the windows' word lists must still partition the batch word list
        words_ok = all(all(s["start"] - 0.1 <= w["start"] < s["end"] + 0.1 for w in s["words"]) for s in streamed) and \
            [w["text"] for s in streamed for w in s["words"]] == [w["text"] for w in batch["words"]]
        check(words_ok, "--stream --json carries each window's words", json.dumps(streamed[:1])[:300])
    finally:
        os.unlink(long_wav)

    # a stream at another rate: the 22.05 kHz copy fed with --rate must give the batch text of the same file, since
    # the stream's chunked resampling is the whole-file resampler fed in pieces
    for wav in sorted(DATA.glob("*Hz.wav")):
        rate = int(wav.stem.split(".")[1].rstrip("Hz"))
        with wave.open(str(wav), "rb") as w:
            assert w.getframerate() == rate and w.getsampwidth() == 2
            pcm = w.readframes(w.getnframes())
        p = subprocess.Popen([a.binary, "--quiet", "--stream", "--rate", str(rate), a.weights], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate(pcm)
        lines = out.decode().splitlines()
        streamed_text = lines[0].split("] ", 1)[1] if lines else ""
        batch_text = run(a.binary, a.weights, [str(wav)])
        check(p.returncode == 0 and streamed_text == batch_text, f"--stream --rate {rate} gives the batch text",
              f"{streamed_text!r} vs {batch_text!r}; {err.decode()[-200:]}")

    # a float WAV with a NaN or an infinity every ~1,000 samples: the engine zeroes them (a click), so the text must
    # stay within reach of the clean one; without that a single NaN silenced the whole window
    print("non-finite samples")
    utt = utts[0]
    with wave.open(str(DATA / f"{utt}.wav"), "rb") as w:
        pcm = w.readframes(w.getnframes())
    floats = [v / 32768.0 for v in struct.unpack("<%dh" % (len(pcm) // 2), pcm)]
    for k, v in enumerate(floats):
        if k % 997 == 0: floats[k] = float("nan")
        elif k % 991 == 0: floats[k] = float("inf")
    body = struct.pack("<%df" % len(floats), *floats)
    hdr = (b"RIFF" + struct.pack("<I", 36 + len(body)) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 3, 1, 16000, 64000, 4, 32)
           + b"data" + struct.pack("<I", len(body)))
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(hdr + body)
        nan_wav = tmp.name
    try:
        text = run(a.binary, a.weights, [nan_wav])
        e = word_edits(normalise(texts[utt]), normalise(text))
        check(e <= 2, f"NaN/inf samples zeroed: text within 2 edits of the clean file ({e})", text)
    finally:
        os.unlink(nan_wav)

    # a malformed manifest – a string inside a shape array, a huge offset, a negative shape – must be refused, not
    # hang or crash (the fuzzer found the shape loop spinning until it overran the entry)
    print("malformed manifest")
    import shutil
    good = Path(a.weights)
    manifest = (good / "weights.json").read_text()
    bad_dir = Path(tempfile.mkdtemp(prefix="p2bad-"))
    try:
        os.symlink(good.resolve() / "weights.bin", bad_dir / "weights.bin")
        shutil.copy(good / "tokens.txt", bad_dir / "tokens.txt")
        doc = json.loads(manifest)
        f32 = next(k for k, v in doc.items() if isinstance(v, dict) and v.get("dtype") == "float32")
        fv = next(k for k, v in doc.items() if isinstance(v, dict) and v.get("dtype") == "five_value")

        def with_entry(name, **fields) -> str:
            d = json.loads(manifest)
            d[name].update(fields)
            return json.dumps(d)
        cases = {"string in shape": with_entry(fv, shape=["1024", 4096]),
                 "huge offset": with_entry(f32, offset=99999999999999999999999),
                 "odd float offset": with_entry(f32, offset=doc[f32]["offset"] + 1),
                 "negative shape": with_entry(fv, shape=[-1024, 4096]),
                 "huge shape": with_entry(fv, shape=[2**62, 2**62]),
                 "cut after dtype": manifest[: manifest.index('"dtype":') + 8],
                 "truncated": manifest[: len(manifest) // 2],
                 "nested": "{" * 5000 + "}" * 5000,
                 "wrong format": with_entry("_meta", format="phonon-2-c-weights-v0"),
                 "format not a string": with_entry("_meta", format=1),
                 "no _meta": json.dumps({k: v for k, v in doc.items() if k != "_meta"})}
        for name, text in cases.items():
            (bad_dir / "weights.json").write_text(text)
            p = subprocess.run([a.binary, "--quiet", str(bad_dir), str(DATA / f"{utts[0]}.wav")],
                               capture_output=True, text=True, timeout=60)
            check(p.returncode != 0 and "Sanitizer" not in p.stderr, f"manifest refused: {name}",
                  f"exit {p.returncode}: {p.stderr[-300:]}")
            if "format" in name or "_meta" in name:
                check("weights format" in p.stderr, f"manifest refused for its format: {name}", p.stderr[-300:])
    finally:
        shutil.rmtree(bad_dir, ignore_errors=True)

    # the format name is defined in the converter (FORMAT), the loader (P2_WEIGHTS_FORMAT) and, through the archive's
    # name, the fetch script; the weights in use carry it too
    print("weights format")
    import re
    root = HERE.parent
    converter = re.search(r'^FORMAT = "([^"]+)"', (root / "convert_weights.py").read_text(), re.M).group(1)
    loader = re.search(r'^#define P2_WEIGHTS_FORMAT "([^"]+)"', (root / "csrc/phonon2.h").read_text(), re.M).group(1)
    archive = re.search(r"^WEIGHTS_FILE=(\S+)", (root / "scripts/get_weights.sh").read_text(), re.M).group(1)
    check(converter == loader, "converter and loader agree on the format", f"{converter} vs {loader}")
    check(archive == f"{converter}.tar.gz", "the archive is named after the format", f"{archive} vs {converter}.tar.gz")
    check(doc["_meta"]["format"] == loader, "the weights under test carry the format", str(doc["_meta"]))

    # the Python bindings over the shared library (PHONON2_LIB, or csrc/libphonon2.*): same engine, same results
    print("python")
    sys.path.insert(0, str(ROOT / "python"))
    try:
        import phonon2 as pybind
        lib_path = next(c for c in pybind._candidates() if os.path.exists(c))
    except (ImportError, StopIteration) as exc:
        check(False, "python bindings load libphonon2", str(exc))
    else:
        eng = pybind.Engine(a.weights)
        utt = utts[0]
        r = eng.transcribe_file(DATA / f"{utt}.wav")
        check(r.text == texts[utt] and len(r.words) == len(texts[utt].split()) and len(r.segments) == 1,
              f"Engine.transcribe_file matches the CLI ({Path(lib_path).name})", r.text)
        samples, rate = pybind.read_wav(DATA / f"{utt}.wav")
        r2 = eng.transcribe(samples, rate)
        check(r2.text == r.text and [t.id for t in r2.tokens] == [t.id for t in r.tokens],
              "Engine.transcribe on read_wav samples matches", r2.text)
        with eng.stream(16000) as s:
            segs = []
            for i in range(0, len(frames), 3200):
                segs += s.feed(pybind.pcm16le_to_float(frames[i:i + 3200]))
            segs += s.finish()
        check([(round(g.start, 3), round(g.end, 3), g.text) for g in segs] ==
              [(round(b["start"], 3), round(b["end"], 3), b["text"]) for b in batch["segments"]] and
              sum(len(g.words) for g in segs) == len(batch["words"]),
              "Stream.feed/finish reproduce the batch windows", str([(g.start, g.end) for g in segs]))
        eng.close()

    print(f"\n{len(failures)} failure(s)" if failures else "\nall checks passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
