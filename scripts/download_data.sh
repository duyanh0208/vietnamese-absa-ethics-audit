#!/usr/bin/env bash
# Tai UIT-ViSFD tu repository chinh thuc cua nhom tac gia.
# Nguon: https://github.com/LuongPhan/UIT-ViSFD
# Giay phep: free for research purposes (xem DATA.md muc 1.2).
set -euo pipefail

RAW_DIR="${1:-data/raw}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

mkdir -p "$RAW_DIR"
echo "Dang tai UIT-ViSFD ..."
curl -sSL -o "$TMP/repo.zip" \
  https://codeload.github.com/LuongPhan/UIT-ViSFD/zip/refs/heads/main
unzip -qo "$TMP/repo.zip" -d "$TMP"
unzip -qo "$TMP/UIT-ViSFD-main/UIT-ViSFD.zip" -d "$TMP/data"
cp "$TMP/data/Train.csv" "$TMP/data/Dev.csv" "$TMP/data/Test.csv" "$RAW_DIR/"

echo "Da luu vao $RAW_DIR:"
ls -1 "$RAW_DIR"
sha256sum "$RAW_DIR"/*.csv
