#!/bin/sh
# SPDX-License-Identifier: MIT
# Fetch the converted weights into c_weights/ (255 MB) without Python.
#
# Downloads the archive from huggingface.co/<WEIGHTS_REPO>, served by Hugging Face (that repo also carries the
# converter and a build.sh that reproduces the archive byte for byte), checks its sha256 against the value pinned
# here and unpacks it. These three constants are the engine's one definition of the archive; the CI drift job reads
# them from this file. `--force` replaces an existing c_weights/.
set -eu
WEIGHTS_REPO=eschmidbauer/phonon-2-c
WEIGHTS_FILE=phonon-2-c-weights-v1.tar.gz
WEIGHTS_SHA256=48137a58011c4b0bc0a97087ea7e868daabf1525ca8447414be230b21faf99e6

cd "$(dirname "$0")/.."
if [ -f c_weights/weights.bin ] && [ "${1:-}" != "--force" ]; then
    echo "c_weights/ is already here; scripts/get_weights.sh --force replaces it"
    exit 0
fi

url="https://huggingface.co/$WEIGHTS_REPO/resolve/main/$WEIGHTS_FILE"
echo "downloading $url"
if [ -t 1 ]; then progress=--progress-bar; else progress=-sS; fi   # a bar on a terminal, quiet in logs
curl -fL $progress -o "$WEIGHTS_FILE.part" "$url"
if command -v sha256sum >/dev/null 2>&1; then
    got=$(sha256sum "$WEIGHTS_FILE.part" | cut -d' ' -f1)
else
    got=$(shasum -a 256 "$WEIGHTS_FILE.part" | cut -d' ' -f1)
fi
if [ "$got" != "$WEIGHTS_SHA256" ]; then
    rm -f "$WEIGHTS_FILE.part"
    echo "$WEIGHTS_FILE: sha256 $got, expected $WEIGHTS_SHA256; nothing unpacked" >&2
    exit 1
fi
rm -rf c_weights
tar xzf "$WEIGHTS_FILE.part"
rm -f "$WEIGHTS_FILE.part"
echo "c_weights/ ready ($(du -sh c_weights | cut -f1)), sha256 of the archive verified"
