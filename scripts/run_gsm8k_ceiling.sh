#!/usr/bin/env bash
set -u
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2
export HF_HOME=/workspace/dllm/.hf_cache HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
LOG=/workspace/dllm/dLLM_f/results/budget/baseline_vs_ours.log
OUT=/workspace/dllm/dLLM_f/results/budget/lmeval
echo -e "\n=== GSM8K keep1.0 (캐시 안 자름, 상한) ($(date +%H:%M)) ===" | tee -a $LOG
for a in 1 2 3 4; do
  while [ "$(nvidia-smi --id=2 --query-compute-apps=pid --format=csv,noheader|wc -l)" -ne 0 ]; do sleep 30; done
  cd /workspace/dllm/dLLM_f
  conda run --no-capture-output -n dllm python evaluation_script.py \
    --model LLaDA_sparse \
    --model_args "pretrained=/workspace/dllm/model/LLaDA-8B-Instruct,keep_ratio=1.0,block_len=32" \
    --tasks gsm8k --num_fewshot 5 --limit 200 --batch_size 1 \
    --gen_kwargs "block_length=32,gen_length=256,steps=256,cfg_scale=0.0" \
    --use_cache "$OUT/cache_keep1.0" --output_path "$OUT/keep1.0" 2>&1 \
    | grep -vE "it/s\]$" | tail -12 | tee -a $LOG
  find "$OUT/keep1.0" -name "results*.json" | grep -q . && { echo "  OK" | tee -a $LOG; break; }
done
echo "=== 상한 측정 완료 ($(date +%H:%M)) ===" | tee -a $LOG
