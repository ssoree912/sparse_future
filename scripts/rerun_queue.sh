#!/usr/bin/env bash
# Re-measure a list of rows. Stops at the first failure rather than retrying:
# run_eval.sh keeps every finished answer, so re-running this picks up where it
# stopped instead of starting over.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
CK=artifacts/ckpts/1ds_300_e6_lr2e-4_6a5fc6/checkpoint-best

progress () {   # answers already on disk for this row: dataset, keep, method
  local n=0 f
  for f in results/.resume/*_$1_keep$2_$3_*.jsonl; do
    [ -f "$f" ] && n=$((n + $(wc -l < "$f")))
  done
  echo "$n"
}

run () {   # $1=dataset  $2=keep_ratio  $3=checkpoint | "baseline" | ""
  local method model done_n
  case "${3:-}" in
    "")         method=none ;;
    baseline)   method=baseline ;;
    *)          method=$(basename "$(dirname "$3")") ;;
  esac
  model="${FUTURE_DLLM_MODEL_TAG:-$(basename "${FUTURE_DLLM_MODEL:-/workspace/dllm/model/LLaDA-8B-Instruct}")}"
  done_n=$(ls results/"$model"/keep"$2"/"$1"/"$1"_keep"$2"_"$method"_*.json 2>/dev/null | wc -l)
  if [ "$done_n" -gt 0 ]; then
    echo -e "\n=== $1 keep$2 $method — 결과 있음, 건너뜀 ==="
    return 0
  fi
  echo -e "\n=== $1 keep$2 ${3:-none} $(date +%H:%M) ==="
  local before after log status
  # Restart only while the run is getting somewhere. Every answer is durable, so
  # a crash costs one item, not the run - but a restart that adds nothing means
  # something is actually wrong, and then we stop.
  while :; do
    while [ "$(nvidia-smi --id=2 --query-compute-apps=pid --format=csv,noheader|wc -l)" -ne 0 ]; do sleep 30; done
    before=$(progress "$1" "$2" "$method")
    log=$(mktemp)
    CUDA_VISIBLE_DEVICES=2 conda run --no-capture-output -n dllm \
        bash scripts/run_eval.sh "$1" "$2" $3 > "$log" 2>&1
    status=$?
    grep -vE "it/s\]|Left truncation|Loading|^ *$" "$log" | tail -5
    rm -f "$log"
    [ $status -eq 0 ] && { echo "  완료"; return 0; }
    after=$(progress "$1" "$2" "$method")
    echo "  세그폴트 (exit $status) — $before → $after 문항"
    if [ "$after" -le "$before" ]; then
      echo
      echo "중단: $1 keep$2 가 전진하지 못함 ($after 문항에서 멈춤)."
      echo "  이어서 하려면: bash scripts/rerun_queue.sh"
      exit 1
    fi
  done
}

run gsm8k 0.1 baseline     # 200 answers already on disk, replays
run gsm8k 0.1 "$CK"

echo -e "\n=== 전체 완료 $(date +%H:%M) ==="
