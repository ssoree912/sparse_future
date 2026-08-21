#!/usr/bin/env bash
# 도메인 혼합 student 학습 + 두 축 평가 (SAMSum = 학습 도메인 안, GSM8K = 밖).
# GPU 2 하나만 쓴다.
set -u
export CUDA_VISIBLE_DEVICES=2
R=/workspace/dllm/dLLM_f/results/budget
LOG=$R/mixed_student.log
OUT=$R/student_final_rowmax_mixed
CONDA="conda run -n sparse-dllm --no-capture-output"

wait_gpu () { while [ "$(nvidia-smi --id=2 --query-compute-apps=pid --format=csv,noheader | wc -l)" -ne 0 ]; do sleep 30; done; }

echo "=== 1/2 혼합 학습 ($(date +%H:%M)) ===" | tee -a $LOG
wait_gpu
$CONDA python /workspace/dllm/dLLM_f/dllm_cache/budget/train_final_rowmax_student.py \
  --teacher-root "$R/teacher_final_rowmax_samsum300,$R/teacher_final_rowmax_gsm8k,$R/teacher_final_rowmax_mmlu,$R/teacher_final_rowmax_mbpp" \
  --output-dir "$OUT" --epochs 6 2>&1 | tee -a $LOG

if [ ! -f "$OUT/checkpoint-best/pytorch_model.bin" ]; then
  echo "=== 학습 실패, 중단 ===" | tee -a $LOG; exit 1
fi

echo "=== 2/2 도메인 밖 recall 점검 ($(date +%H:%M)) ===" | tee -a $LOG
wait_gpu
$CONDA python /workspace/dllm/dLLM_f/dllm_cache/budget/eval_student_cross_domain.py \
  --student "$OUT/checkpoint-best" --shards 20 2>&1 | tee -a $LOG

echo "=== 완료 ($(date +%H:%M)) ===" | tee -a $LOG
