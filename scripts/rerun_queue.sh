#!/usr/bin/env bash
# Re-measure a list of rows. Stops at the first failure rather than retrying:
# run_eval.sh keeps every finished answer, so re-running this picks up where it
# stopped instead of starting over.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
CK=artifacts/ckpts/1ds_300_e6_lr2e-4_6a5fc6/checkpoint-best

run () {   # $1=dataset  $2=keep_ratio  $3=checkpoint or ""
  local resume
  resume="results/.resume/$1_keep$2"*.jsonl
  echo -e "\n=== $1 keep$2 $(date +%H:%M) ==="
  while [ "$(nvidia-smi --id=2 --query-compute-apps=pid --format=csv,noheader|wc -l)" -ne 0 ]; do sleep 30; done
  if ! CUDA_VISIBLE_DEVICES=2 conda run --no-capture-output -n dllm \
        bash scripts/run_eval.sh "$1" "$2" $3 2>&1 \
        | grep -vE "it/s\]|Left truncation|Loading|^ *$" | tail -6; then
    echo
    echo "중단: $1 keep$2 실패 (세그폴트로 보임)."
    echo "  완료된 답변: $(cat $resume 2>/dev/null | wc -l)개 보존됨"
    echo "  이어서 하려면: bash scripts/rerun_queue.sh"
    exit 1
  fi
  echo "  완료"
}

run samsum 1.0 ""          # independent re-measure: ours reads above the ceiling
run gsm8k  1.0 ""
run gsm8k  0.1 "$CK"
echo -e "\n=== 전체 완료 $(date +%H:%M) ==="
