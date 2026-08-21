#!/usr/bin/env bash
# Fetch LLaDA's own generation loop, for the origin row.
#
# Stock LLaDA keeps no cache at all: every step runs the whole sequence again.
# That is the reference the cached variants are measured against, so the code has
# to be theirs. Lands in eval/origin/, which is gitignored.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$REPO/eval/origin"
URL="${LLADA_GENERATE_URL:-https://raw.githubusercontent.com/ML-GSAI/LLaDA/main/generate.py}"
mkdir -p "$DEST"
curl -fsSL "$URL" -o "$DEST/llada_generate.py"
: > "$DEST/__init__.py"
echo "$DEST/llada_generate.py <- $URL ($(wc -l < "$DEST/llada_generate.py") lines)"
