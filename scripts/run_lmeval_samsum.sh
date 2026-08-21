#!/usr/bin/env bash
# Usage: run_lmeval_samsum.sh <tag> <student_ckpt|"">
set -u
TAG="$1"; STU="${2:-}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2
export HF_HOME=/workspace/dllm/.hf_cache HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
OUT=/workspace/dllm/dLLM_f/results/budget/lmeval
LOG=/workspace/dllm/dLLM_f/results/budget/lmeval_samsum.log
ARGS="pretrained=/workspace/dllm/model/LLaDA-8B-Instruct,keep_ratio=0.1,block_len=32,max_length=2176"
[ -n "$STU" ] && ARGS="$ARGS,student_cache_path=$STU,student_score_head=score,student_evict_suffix=True"
echo -e "\n=== lm-eval SAMSum: $TAG ($(date +%H:%M)) ===" | tee -a $LOG
for a in 1 2 3 4 5 6; do
  while [ "$(nvidia-smi --id=2 --query-compute-apps=pid --format=csv,noheader|wc -l)" -ne 0 ]; do sleep 30; done
  cd /workspace/dllm/dLLM_f
  conda run --no-capture-output -n dllm python evaluation_script.py \
    --model LLaDA_sparse --model_args "$ARGS" \
    --tasks longbench_samsum \
    --include_path experiment/345/2026-08-13/tasks/longbench_full_local \
    --limit 200 --batch_size 1 \
    --gen_kwargs "block_length=32,gen_length=128,steps=128,cfg_scale=0.0,temperature=0.0" \
    --use_cache "$OUT/cache_samsum_$TAG" --output_path "$OUT/samsum_$TAG" 2>&1 \
    | grep -vE "it/s\]$|DBG|Left truncation" | tail -10 | tee -a $LOG
  find "$OUT/samsum_$TAG" -name "results*.json" | grep -q . && { echo "  OK" | tee -a $LOG; exit 0; }
  echo "  (재시도 $a)" | tee -a $LOG
done
