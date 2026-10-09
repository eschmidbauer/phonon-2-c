#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Convert Phonon-2's weights into the flat files csrc/weights.c loads.

    python convert_weights.py                                  # fetch phonon-2.bps.tar.zst from the Hub, unpack, convert
    python convert_weights.py --tarball phonon-2.bps.tar.zst
    python convert_weights.py --container phonon-2/model.fermion --config phonon-2/config.json
    python convert_weights.py --archive [phonon-2-c-weights-v1.tar.gz]   # also pack c_weights/ for the Hub
    → c_weights/weights.bin, c_weights/weights.json, c_weights/tokens.txt

The container (`fermion-five-value-parakeet-v1`, read by the vendored fermion_container.py) holds 723 tensors under
transformers' ParakeetForTDT names. On the way to weights.bin:

  five-value linears  stay five values: a 2-bit sign plane and a 1-bit lo/hi plane in [in, out] layout, with the fp32
                      lo/hi per output column; q_proj | k_proj | v_proj are concatenated into one [1024, 3072] tensor
  int6 tables         → int8 with one fp32 scale per output feature; exact, the 6-bit codes fit int8 and the fp16
                      row scale is kept. GEMM tables (subsampling.linear, encoder_projector) are stored [in, out] with
                      the scale per column; decoder tables (embedding, LSTM, projector, joint head) stay [out, in] with
                      the scale per row, as the decoder reads them row by row
  depthwise conv      int6 → fp32 with the following BatchNorm folded in (the one step that rounds: fp32 arithmetic)
  fp16 tensors        → fp32

Tensor names in weights.json are the ones csrc/weights.c looks up; change them in both places.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import tarfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fermion_container import read_container  # noqa: E402

HUB_REPO = "FermionResearch/Phonon-2"
TARBALL = "phonon-2.bps.tar.zst"
BASE_MODEL = "nvidia/parakeet-tdt-0.6b-v3"
FORMAT = "phonon-2-c-weights-v1"
C_WEIGHTS = "c_weights"            # output directory, and the directory inside the archive
ARCHIVE = f"{FORMAT}.tar.gz"       # what --archive writes; the file published at eschmidbauer/phonon-2-c on the Hub
BN_EPS = 1e-5        # nn.BatchNorm1d default, as in transformers' ParakeetEncoderConvolutionModule
ALIGN = 64           # byte alignment of every tensor in weights.bin
N_BLOCKS = 24


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_tarball(dest_dir: Path) -> Path:
    """Download the release tarball from the Hub (huggingface_hub caches it; the local copy lands in dest_dir)."""
    from huggingface_hub import hf_hub_download
    print(f"downloading {TARBALL} from {HUB_REPO} …")
    return Path(hf_hub_download(HUB_REPO, TARBALL, local_dir=str(dest_dir)))


def unpack_tarball(tarball: Path, out_dir: Path) -> tuple[Path, Path]:
    """zstd tar → out_dir/{model.fermion, config.json, manifests}; the container's hash is checked. -> (container, config)"""
    import zstandard
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"unpacking {tarball.name} → {out_dir}/")
    with open(tarball, "rb") as fh, zstandard.ZstdDecompressor().stream_reader(fh) as zr, \
            tarfile.open(fileobj=zr, mode="r|") as tar:
        tar.extractall(out_dir, filter="data")
    manifest = json.loads((out_dir / "packed_manifest.json").read_text())
    container = out_dir / manifest["container"]
    digest = sha256(container)
    if digest != manifest["container_sha256"]:
        raise SystemExit(f"{container}: sha256 {digest} does not match the manifest's {manifest['container_sha256']}")
    return container, out_dir / "config.json"


class BinWriter:
    """weights.bin plus its manifest; every tensor starts on an ALIGN-byte boundary."""

    def __init__(self, path: Path):
        self.f = open(path, "wb")
        self.off = 0
        self.entries: dict = {}

    def put(self, arr: np.ndarray) -> int:
        pad = (-self.off) % ALIGN
        self.f.write(b"\0" * pad)
        self.off += pad
        off = self.off
        data = np.ascontiguousarray(arr).tobytes()
        self.f.write(data)
        self.off += len(data)
        return off

    def f32(self, name: str, arr) -> None:
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        self.entries[name] = {"dtype": "float32", "shape": list(arr.shape), "numel": int(arr.size), "offset": self.put(arr)}

    def i8(self, name: str, q: np.ndarray, scales: np.ndarray) -> None:
        """q int8 [rows, cols]; `scales` is fp32 per output feature – per column for a GEMM table ([in, out]), per row
        for a decoder table ([out, in]). Both dimensions are multiples of 16 for every table in the model."""
        rows, cols = q.shape
        assert cols % 16 == 0 and scales.shape in ((rows,), (cols,)), (name, q.shape, scales.shape)
        self.entries[name] = {"dtype": "int8", "shape": [rows, cols],
                              "offset": self.put(q.astype(np.int8)),
                              "scales_offset": self.put(scales.astype(np.float32))}

    def fv(self, name: str, sign: np.ndarray, is_hi: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> None:
        """sign int8 [K, N] in {-1, 0, 1}; is_hi bool [K, N]; lo, hi fp32 [N]. Signs pack 4 per byte (weight n at bits
        2*(n%4), code = sign + 1), flags 8 per byte (bit n%8), both along N."""
        K, N = sign.shape
        assert N % 16 == 0, (name, sign.shape)
        codes = (sign.astype(np.int16) + 1).astype(np.uint8).reshape(K, N // 4, 4)
        signs = (codes[..., 0] | (codes[..., 1] << 2) | (codes[..., 2] << 4) | (codes[..., 3] << 6)).astype(np.uint8)
        flags = np.packbits(np.ascontiguousarray(is_hi, dtype=bool), axis=1, bitorder="little")
        assert flags.shape == (K, N // 8)
        self.entries[name] = {"dtype": "five_value", "shape": [K, N],
                              "signs_offset": self.put(signs), "flags_offset": self.put(flags),
                              "lo_offset": self.put(lo.astype(np.float32)), "hi_offset": self.put(hi.astype(np.float32))}

    def close(self) -> int:
        self.f.close()
        return self.off


def convert(container: Path, config: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"reading {container} …")
    tensors, index, raw = read_container(str(container), with_raw=True)

    def f32(name):
        return tensors[name].astype(np.float32)

    def five(module):
        """A five-value module → (sign [K, N], is_hi [K, N], lo [N], hi [N]) in [in, out] layout."""
        r = raw[module]
        return r["sign"].T, r["is_hi"].T, r["lo"].astype(np.float32), r["hi"].astype(np.float32)

    bw = BinWriter(out_dir / "weights.bin")

    def put_fv(out_name, modules):
        parts = [five(m) for m in modules]
        bw.fv(out_name,
              np.concatenate([p[0] for p in parts], axis=1), np.concatenate([p[1] for p in parts], axis=1),
              np.concatenate([p[2] for p in parts]), np.concatenate([p[3] for p in parts]))

    def put_i8(out_name, hf_name):
        """GEMM table: [out, in] in the container → [in, out] with the per-output scale per column."""
        r = raw[hf_name]
        assert r["bits"] == 6, (hf_name, r["bits"])
        bw.i8(out_name, r["q"].T, r["scale"])

    def put_i8_rows(out_name, hf_name):
        """Decoder table: kept [out, in], per-output scale per row."""
        r = raw[hf_name]
        assert r["bits"] == 6, (hf_name, r["bits"])
        bw.i8(out_name, r["q"], r["scale"])

    # ── subsampling ──
    sub = "encoder.subsampling"
    for layer in (0, 2, 5):   # 3×3 convs: full (1→256) then two depthwise
        bw.f32(f"subsampling.conv.{layer}.weight", f32(f"{sub}.layers.{layer}.weight").reshape(256, 9))
        bw.f32(f"subsampling.conv.{layer}.bias", f32(f"{sub}.layers.{layer}.bias"))
    for layer in (3, 6):      # pointwise 1×1 convs, int6 in the container, small → fp32
        bw.f32(f"subsampling.conv.{layer}.weight", f32(f"{sub}.layers.{layer}.weight").reshape(256, 256))
        bw.f32(f"subsampling.conv.{layer}.bias", f32(f"{sub}.layers.{layer}.bias"))
    put_i8("subsampling.linear.weight", f"{sub}.linear.weight")
    bw.f32("subsampling.linear.bias", f32(f"{sub}.linear.bias"))

    # ── conformer blocks ──
    for i in range(N_BLOCKS):
        p, o = f"encoder.layers.{i}", f"layers.{i}"
        for norm in ("norm_feed_forward1", "norm_self_att", "norm_conv", "norm_feed_forward2", "norm_out"):
            bw.f32(f"{o}.{norm}.weight", f32(f"{p}.{norm}.weight"))
            bw.f32(f"{o}.{norm}.bias", f32(f"{p}.{norm}.bias"))
        for ff in ("feed_forward1", "feed_forward2"):
            put_fv(f"{o}.{ff}.linear1.weight", [f"{p}.{ff}.linear1"])
            put_fv(f"{o}.{ff}.linear2.weight", [f"{p}.{ff}.linear2"])
        put_fv(f"{o}.self_attn.qkv_proj.weight", [f"{p}.self_attn.q_proj", f"{p}.self_attn.k_proj", f"{p}.self_attn.v_proj"])
        put_fv(f"{o}.self_attn.relative_k_proj.weight", [f"{p}.self_attn.relative_k_proj"])
        put_fv(f"{o}.self_attn.o_proj.weight", [f"{p}.self_attn.o_proj"])
        bw.f32(f"{o}.self_attn.bias_u", f32(f"{p}.self_attn.bias_u"))
        bw.f32(f"{o}.self_attn.bias_v", f32(f"{p}.self_attn.bias_v"))
        put_fv(f"{o}.conv.pointwise_conv1.weight", [f"{p}.conv.pointwise_conv1"])
        put_fv(f"{o}.conv.pointwise_conv2.weight", [f"{p}.conv.pointwise_conv2"])
        # depthwise conv with its BatchNorm folded in: w' = w · γ/√(σ²+ε), b' = β − μ · γ/√(σ²+ε)
        w = f32(f"{p}.conv.depthwise_conv.weight").reshape(1024, 9).astype(np.float64)
        gamma, beta = f32(f"{p}.conv.norm.weight").astype(np.float64), f32(f"{p}.conv.norm.bias").astype(np.float64)
        mean, var = f32(f"{p}.conv.norm.running_mean").astype(np.float64), f32(f"{p}.conv.norm.running_var").astype(np.float64)
        s = gamma / np.sqrt(var + BN_EPS)
        bw.f32(f"{o}.conv.depthwise_conv.weight", w * s[:, None])
        bw.f32(f"{o}.conv.depthwise_conv.bias", beta - mean * s)

    put_i8("encoder_projector.weight", "encoder_projector.weight")
    bw.f32("encoder_projector.bias", f32("encoder_projector.bias"))

    # ── decoder: int8 rows, read row by row at decode time ──
    emb_q = raw["decoder.embedding.weight"]["q"]
    assert emb_q.shape == (8193, 640) and not np.any(emb_q[8192]), "blank embedding row is expected to be zero"
    put_i8_rows("decoder.embedding.weight", "decoder.embedding.weight")
    for layer in range(2):
        put_i8_rows(f"decoder.lstm.weight_ih_l{layer}", f"decoder.lstm.weight_ih_l{layer}")
        put_i8_rows(f"decoder.lstm.weight_hh_l{layer}", f"decoder.lstm.weight_hh_l{layer}")
        bw.f32(f"decoder.lstm.bias_ih_l{layer}", f32(f"decoder.lstm.bias_ih_l{layer}"))
        bw.f32(f"decoder.lstm.bias_hh_l{layer}", f32(f"decoder.lstm.bias_hh_l{layer}"))
    put_i8_rows("decoder.decoder_projector.weight", "decoder.decoder_projector.weight")
    bw.f32("decoder.decoder_projector.bias", f32("decoder.decoder_projector.bias"))
    put_i8_rows("joint.head.weight", "joint.head.weight")
    bw.f32("joint.head.bias", f32("joint.head.bias"))

    total = bw.close()
    bw.entries["_meta"] = {"format": FORMAT, "source_container_sha256": sha256(container),
                           "source_records": len(index), "base_model": BASE_MODEL, "license": "cc-by-4.0",
                           "converter": "convert_weights.py"}
    (out_dir / "weights.json").write_text(json.dumps(bw.entries, indent=1) + "\n")

    labels = json.loads(config.read_text())["labels"]
    assert len(labels) == 8192, len(labels)
    (out_dir / "tokens.txt").write_text("\n".join(labels + ["<blank>"]) + "\n", encoding="utf-8")

    by_dtype = {}
    for e in bw.entries.values():
        if "dtype" in e:
            by_dtype[e["dtype"]] = by_dtype.get(e["dtype"], 0) + 1
    print(f"wrote {out_dir}/weights.bin ({total / 1e6:,.0f} MB), {sum(by_dtype.values())} tensors {by_dtype}, "
          f"tokens.txt ({len(labels) + 1} ids)")


def write_archive(out_dir: Path, archive: Path) -> None:
    """Pack out_dir's three files as C_WEIGHTS/ into a reproducible .tar.gz: ustar, fixed order, zero mtimes, ids and
    names, no gzip header name or time — the same weights give the same bytes on any machine."""
    def clean(info: tarfile.TarInfo) -> tarfile.TarInfo:
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        info.mtime = 0
        info.mode = 0o755 if info.isdir() else 0o644
        return info

    with open(archive, "wb") as raw, \
            gzip.GzipFile(filename="", mode="wb", compresslevel=9, fileobj=raw, mtime=0) as gz, \
            tarfile.open(fileobj=gz, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        tar.add(out_dir, arcname=C_WEIGHTS, recursive=False, filter=clean)
        for name in ("tokens.txt", "weights.bin", "weights.json"):
            tar.add(out_dir / name, arcname=f"{C_WEIGHTS}/{name}", filter=clean)
    print(f"wrote {archive} ({archive.stat().st_size / 1e6:,.0f} MB), sha256 {sha256(archive)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tarball", type=Path, default=None, help=f"{TARBALL}; default: download it from the Hub")
    ap.add_argument("--container", type=Path, default=None, help="an already unpacked model.fermion (with --config)")
    ap.add_argument("--config", type=Path, default=None, help="the tarball's config.json (the vocabulary comes from it)")
    ap.add_argument("--unpack-dir", type=Path, default=HERE / "phonon-2", help="where the tarball is unpacked")
    ap.add_argument("--out", type=Path, default=HERE / C_WEIGHTS)
    ap.add_argument("--archive", type=Path, nargs="?", const=HERE / ARCHIVE, default=None,
                    help=f"also pack the output as a reproducible .tar.gz (default name {ARCHIVE})")
    a = ap.parse_args()

    if a.container:
        if not a.config:
            ap.error("--container needs --config")
        container, config = a.container, a.config
    else:
        tarball = a.tarball or (HERE / TARBALL if (HERE / TARBALL).is_file() else fetch_tarball(HERE))
        container, config = unpack_tarball(tarball, a.unpack_dir)
    convert(container, config, a.out)
    if a.archive:
        write_archive(a.out, a.archive)


if __name__ == "__main__":
    main()
