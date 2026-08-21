#!/usr/bin/env bash
# Re-measure a list of rows. Stops at the first failure rather than retrying:
# run_eval.sh keeps every finished answer, so re-running this picks up where it
# stopped instead of starting over.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
CK=artifacts/ckpts/1ds_300_e6_lr2e-4_6a5fc6/checkpoint-best

run () {   # $1=dataset  $2=keep_ratio  $3=checkpoint | "baseline" | ""
  echo -e "\n=== $1 keep$2 ${3:-none} $(date +%H:%M) ==="
  while [ "$(nvidia-smi --id=2 --query-compute-apps=pid --format=csv,noheader|wc -l)" -ne 0 ]; do sleep 30; done
  local log status
  log=$(mktemp)
  # The status has to be taken from the command itself: piping it into grep
  # hands back grep's status, which made every failure read as success.
  CUDA_VISIBLE_DEVICES=2 conda run --no-capture-output -n dllm \
      bash scripts/run_eval.sh "$1" "$2" $3 > "$log" 2>&1
  status=$?
  grep -vE "it/s\]|Left truncation|Loading|^ *$" "$log" | tail -6
  rm -f "$log"
  if [ $status -ne 0 ]; then
    echo
    echo "중단: $1 keep$2 실패 (exit $status, 세그폴트로 보임)."
    for r in results/.resume/$1_keep$2*.jsonl; do
      [ -f "$r" ] && echo "  완료된 답변: $(wc -l < "$r")개 보존됨 ($r)"
    done
    echo "  이어서 하려면: bash scripts/rerun_queue.sh"
    exit 1
  fi
  echo "  완료"
}

run samsum 1.0 ""          # independent re-measure: ours reads above the ceiling
run samsum 0.1 baseline    # Sparse-dLLM criterion, same data and prompt
run gsm8k  1.0 ""
run gsm8k  0.1 baseline
run gsm8k  0.1 "$CK"
echo -e "\n=== 전체 완료 $(date +%H:%M) ==="
