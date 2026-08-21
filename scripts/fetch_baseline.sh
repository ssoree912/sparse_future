#!/usr/bin/env bash
# Fetch Sparse-dLLM's eviction code, for the baseline row.
#
# Not vendored here: it is someone else's code and the point of the baseline is
# that it is theirs. This pulls the pinned commit into eval/baseline/, which is
# gitignored.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$REPO/eval/baseline/sparse_dllm"
COMMIT="${SPARSE_DLLM_COMMIT:-3fd8986}"
UPSTREAM="${SPARSE_DLLM_REPO:-https://github.com/OpenMOSS/Sparse-dLLM}"
LOCAL="${SPARSE_DLLM_LOCAL:-$REPO/../Sparse-dLLM}"

mkdir -p "$DEST"
files="modeling_llada.py llada_generate.py configuration_llada.py"

if [ -d "$LOCAL/.git" ]; then
  # A clone is already here. Take the files out of the pinned commit rather than
  # the working tree - the copy on this machine has local modifications, and the
  # baseline has to be the authors' code, not someone's edit of it.
  for f in $files; do
    git -C "$LOCAL" show "$COMMIT:opencompass/models/sparse_dllm/$f" > "$DEST/$f"
  done
  echo "extracted $COMMIT from $LOCAL"
else
  tmp=$(mktemp -d)
  git clone --quiet "$UPSTREAM" "$tmp/repo"
  git -C "$tmp/repo" checkout --quiet "$COMMIT"
  for f in $files; do
    cp "$tmp/repo/opencompass/models/sparse_dllm/$f" "$DEST/$f"
  done
  rm -rf "$tmp"
  echo "cloned $UPSTREAM at $COMMIT"
fi

cat > "$DEST/__init__.py" <<'PY'
"""Sparse-dLLM's eviction code, fetched by scripts/fetch_baseline.sh. Not edited."""
from .modeling_llada import LLaDAModelLM, CustomCache
from .llada_generate import generate

__all__ = ["LLaDAModelLM", "CustomCache", "generate"]
PY
echo "$DEST: $(ls "$DEST" | tr '\n' ' ')"
