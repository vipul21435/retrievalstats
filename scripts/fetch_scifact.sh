#!/usr/bin/env bash
# Download BEIR SciFact (5,183 abstracts, 300 test queries) into DIR (default: data).
# The data is not part of this repository; see the README for its licence.
set -euo pipefail
dir="${1:-data}"
url="https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
sha256="536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165"
mkdir -p "$dir"
if [ ! -f "$dir/scifact.zip" ]; then
  curl -sSL -o "$dir/scifact.zip" "$url"
fi
if command -v sha256sum >/dev/null; then
  got=$(sha256sum "$dir/scifact.zip" | cut -d' ' -f1)
else
  got=$(shasum -a 256 "$dir/scifact.zip" | cut -d' ' -f1)
fi
if [ "$got" != "$sha256" ]; then
  echo "checksum mismatch for $dir/scifact.zip: $got" >&2
  exit 1
fi
unzip -q -o "$dir/scifact.zip" -d "$dir"
echo "SciFact ready in $dir/scifact"
