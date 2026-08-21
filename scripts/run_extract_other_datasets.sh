#!/bin/bash
# gsm8k / mmlu / mbpp 프롬프트로 final x row-max teacher 추출. 각 단계 이어받기+재시도.
set -u
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2
export HF_HOME=/workspace/dllm/.hf_cache HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
LOG=/workspace/dllm/dLLM_f/results/budget/extract_other.log
cd /workspace/dllm/dLLM_f

run() {   # dataset  gen_length  steps  n
  local ds=$1 gen=$2 steps=$3 n=$4
  local out=results/budget/teacher_final_rowmax_$ds
  echo -e "\n=== $ds (gen=$gen, n=$n) $(date +%H:%M) ===" | tee -a "$LOG"
  for attempt in 1 2 3 4 5 6; do
    conda run --no-capture-output -n sparse-dllm python -m dllm_cache.budget.extract_final_rowmax_teacher \
      --source-glob "results/budget/prompt_shards/$ds/*.pt" \
      --output-root "$out" --n-samples "$n" --gen-length "$gen" --steps "$steps" \
      2>&1 | grep -E "s/sample|done|Error|Traceback" | tee -a "$LOG"
    have=$(ls "$out"/*.pt 2>/dev/null | wc -l)
    echo "  shards=$have/$n" | tee -a "$LOG"
    [ "$have" -ge "$n" ] && break
    echo "  (재시도 $attempt)" | tee -a "$LOG"
  done
}

run gsm8k 128 128 300
run mmlu   64  64 300
run mbpp  128 128 100
echo -e "\n=== 전체 완료 $(date +%H:%M) ===" | tee -a "$LOG"
