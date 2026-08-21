#!/usr/bin/env bash
# Re-measure a list of rows. Stops at the first failure rather than retrying:
# run_eval.sh keeps every finished answer, so re-running this picks up where it
# stopped instead of starting over.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
CK=artifacts/ckpts/1ds_300_e6_lr2e-4_6a5fc6/checkpoint-best

progress () {   # answers already on disk for this row
  local n=0 f
  for f in results/.resume/$1_keep$2*.jsonl; do
    [ -f "$f" ] && n=$((n + $(wc -l < "$f")))
  done
  echo "$n"
}

run () {   # $1=dataset  $2=keep_ratio  $3=checkpoint | "baseline" | ""
  echo -e "\n=== $1 keep$2 ${3:-none} $(date +%H:%M) ==="
  local before after log status
  # Restart only while the run is getting somewhere. Every answer is durable, so
  # a crash costs one item, not the run - but a restart that adds nothing means
  # something is actually wrong, and then we stop.
  while :; do
    while [ "$(nvidia-smi --id=2 --query-compute-apps=pid --format=csv,noheader|wc -l)" -ne 0 ]; do sleep 30; done
    before=$(progress "$1" "$2")
    log=$(mktemp)
    CUDA_VISIBLE_DEVICES=2 conda run --no-capture-output -n dllm \
        bash scripts/run_eval.sh "$1" "$2" $3 > "$log" 2>&1
    status=$?
    grep -vE "it/s\]|Left truncation|Loading|^ *$" "$log" | tail -5
    rm -f "$log"
    [ $status -eq 0 ] && { echo "  완료"; return 0; }
    after=$(progress "$1" "$2")
    echo "  세그폴트 (exit $status) — $before → $after 문항"
    if [ "$after" -le "$before" ]; then
      echo
      echo "중단: $1 keep$2 가 전진하지 못함 ($after 문항에서 멈춤)."
      echo "  이어서 하려면: bash scripts/rerun_queue.sh"
      exit 1
    fi
  done
}

run samsum 1.0 ""          # independent re-measure: ours reads above the ceiling
run samsum 0.1 baseline    # Sparse-dLLM criterion, same data and prompt
run gsm8k  1.0 ""
run gsm8k  0.1 baseline
run gsm8k  0.1 "$CK"
echo -e "\n=== 전체 완료 $(date +%H:%M) ==="
