#!/bin/bash
# 추출 → 학습 → 평가를 한 번에. tmux 세션 안에서 돌아 세션이 끊겨도 살아남는다.
set -u
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2
export HF_HOME=/workspace/dllm/.hf_cache HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
LOG=/workspace/dllm/dLLM_f/results/budget/pipeline.log
mkdir -p "$(dirname "$LOG")"
say() { echo -e "\n=== $* ($(date +%H:%M)) ===" | tee -a "$LOG"; }

say "1/3 teacher 추출 (final x row-max, samsum 300)"
for attempt in 1 2 3 4 5 6; do
  cd /workspace/dllm/dLLM_f
  conda run --no-capture-output -n sparse-dllm python -m dllm_cache.budget.extract_final_rowmax_teacher \
    --n-samples 300 2>&1 | grep -E "s/sample|done|Error|Traceback" | tee -a "$LOG"
  n=$(ls /workspace/dllm/dLLM_f/results/budget/teacher_final_rowmax_samsum300/*.pt 2>/dev/null | wc -l)
  echo "  shards=$n" | tee -a "$LOG"
  [ "$n" -ge 300 ] && break
  echo "  (재시도 $attempt — segfault 등으로 중단, 남은 shard부터 이어받음)" | tee -a "$LOG"
done

say "2/3 student 학습"
for attempt in 1 2 3; do
  cd /workspace/dllm/dLLM_f
  conda run --no-capture-output -n sparse-dllm python -m dllm_cache.budget.train_final_rowmax_student \
    2>&1 | grep -E "epoch|saved|done|train .* val|Error|Traceback" | tee -a "$LOG"
  [ -f /workspace/dllm/dLLM_f/results/budget/student_final_rowmax_samsum300/checkpoint-best/pytorch_model.bin ] && break
  echo "  (학습 재시도 $attempt)" | tee -a "$LOG"
done

say "3/3 SAMSum 평가 (학습된 student로 블록당 1회 선택)"
/tmp/claude-0/-workspace-dllm/4a27d45a-7287-4963-bd14-cbe2a09f4e0c/scratchpad/run_samsum_with_retry.sh \
  sparse_llada_student_final_rowmax \
  /workspace/dllm/Sparse-dLLM/outputs/samsum_2048_student_final_rowmax 2 2>&1 | tee -a "$LOG"

say "완료"
cat /workspace/dllm/Sparse-dLLM/outputs/samsum_2048_student_final_rowmax/*/results/*/*.json 2>/dev/null | tee -a "$LOG"
