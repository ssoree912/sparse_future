#!/usr/bin/env bash
# baseline(원래 Sparse-dLLM scorer) vs ours(도메인 혼합 student) — keep 0.1.
#   SAMSum : OpenCompass LongBench (baseline 33.89는 기존 결과 재사용)
#   GSM8K  : lm-eval 표준 하네스, 5-shot, 200 문항
# GPU 2 하나만 쓴다.
set -u
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2
export HF_HOME=/workspace/dllm/.hf_cache HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
unset DATASET_SOURCE COMPASS_DATA_CACHE

M=/workspace/dllm/model/LLaDA-8B-Instruct
STU=/workspace/dllm/dLLM_f/results/budget/student_final_rowmax_mixed/checkpoint-best
LOG=/workspace/dllm/dLLM_f/results/budget/baseline_vs_ours.log
OUT=/workspace/dllm/dLLM_f/results/budget/lmeval
mkdir -p "$OUT"
say() { echo -e "\n=== $* ($(date +%H:%M)) ===" | tee -a "$LOG"; }
wait_gpu () { while [ "$(nvidia-smi --id=2 --query-compute-apps=pid --format=csv,noheader | wc -l)" -ne 0 ]; do sleep 30; done; }

# ---------- 1) SAMSum, ours ----------
say "1/3 SAMSum keep0.1 — ours (혼합 student)"
wait_gpu
/tmp/claude-0/-workspace-dllm/4a27d45a-7287-4963-bd14-cbe2a09f4e0c/scratchpad/run_samsum_with_retry.sh \
  sparse_llada_student_mixed \
  /workspace/dllm/Sparse-dLLM/outputs/samsum_2048_student_mixed 2 2>&1 \
  | grep -vE "^\[|it/s\]$" | tail -25 | tee -a "$LOG"

# ---------- 2·3) GSM8K ----------
run_gsm8k () {   # $1=태그  $2=추가 model_args
  say "GSM8K keep0.1 — $1"
  for attempt in 1 2 3 4 5 6; do
    wait_gpu
    cd /workspace/dllm/dLLM_f
    conda run --no-capture-output -n dllm python evaluation_script.py \
      --model LLaDA_sparse \
      --model_args "pretrained=$M,keep_ratio=0.1,block_len=32${2}" \
      --tasks gsm8k --num_fewshot 5 --limit 200 --batch_size 1 \
      --gen_kwargs "block_length=32,gen_length=256,steps=256,cfg_scale=0.0" \
      --use_cache "$OUT/cache_$1" \
      --output_path "$OUT/$1" 2>&1 | grep -vE "it/s\]$|^\s*$" | tail -30 | tee -a "$LOG"
    ls "$OUT/$1"/**/*.json >/dev/null 2>&1 && { echo "  OK" | tee -a "$LOG"; return 0; }
    echo "  (재시도 $attempt)" | tee -a "$LOG"
  done
  echo "  실패" | tee -a "$LOG"
}

run_gsm8k baseline ",scorer=sparse_dllm"
run_gsm8k ours     ",student_cache_path=$STU,student_score_head=score,student_evict_suffix=True"

say "완료"
