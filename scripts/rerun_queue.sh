#!/usr/bin/env bash
# Re-measure a list of rows, resuming through segfaults. Uses run_eval.sh, which
# caches finished requests, so a retry picks up where the crash left off.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
CK=artifacts/ckpts/1ds_300_e6_lr2e-4_6a5fc6/checkpoint-best
run () {   # $1=dataset  $2=keep_ratio  $3=checkpoint or ""
  echo -e "\n=== $1 keep$2 $(date +%H:%M) ==="
  for a in $(seq 1 25); do
    while [ "$(nvidia-smi --id=2 --query-compute-apps=pid --format=csv,noheader|wc -l)" -ne 0 ]; do sleep 30; done
    if CUDA_VISIBLE_DEVICES=2 conda run --no-capture-output -n dllm \
         bash scripts/run_eval.sh "$1" "$2" $3 2>&1 \
         | grep -vE "it/s\]|Left truncation|Loading|^ *$" | tail -4; then
      echo "  성공 (시도 $a)"; return 0
    fi
    echo "  (세그폴트 — 캐시에서 재개, 시도 $a)"
  done
  echo "  25회 실패"
}
run samsum 0.1 "$CK"
run gsm8k  1.0 ""
run gsm8k  0.1 "$CK"
echo "=== 전체 완료 $(date +%H:%M) ==="
