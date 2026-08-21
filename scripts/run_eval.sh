#!/usr/bin/env bash
# lm-eval, writing one json into results/ per run.
#
#   scripts/run_eval.sh <dataset> <keep_ratio> [checkpoint]
#
#   scripts/run_eval.sh samsum 0.1 artifacts/ckpts/1ds_300_e6_lr2e-4_6a5fc6/checkpoint-best
#   scripts/run_eval.sh gsm8k  1.0            # no eviction, no scorer needed
#
# Generation length belongs to the dataset, not the caller - it has to match what
# the task was measured with, because it sets how much of the cache is prompt
# rather than the model's own output.
set -euo pipefail

DATASET="${1:?usage: run_eval.sh <dataset> <keep_ratio> [checkpoint]}"
KEEP="${2:?usage: run_eval.sh <dataset> <keep_ratio> [checkpoint]}"
CKPT="${3:-}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HARNESS="${FUTURE_DLLM_HARNESS:-/workspace/dllm/dLLM_f}"
MODEL="${FUTURE_DLLM_MODEL:-/workspace/dllm/model/LLaDA-8B-Instruct}"
TASKS_DIR="$HARNESS/experiment/345/2026-08-13/tasks/longbench_full_local"

case "$DATASET" in
  samsum) TASK=longbench_samsum; GEN=128; SHOTS=""            ; MAXLEN=2176 ;;
  gsm8k)  TASK=gsm8k;            GEN=256; SHOTS="--num_fewshot 5"; MAXLEN=4096 ;;
  mmlu)   TASK=mmlu_generative;  GEN=64;  SHOTS="--num_fewshot 5"; MAXLEN=4096 ;;
  mbpp)   TASK=mbpp;             GEN=256; SHOTS=""            ; MAXLEN=4096 ;;
  math)   TASK=minerva_math;     GEN=256; SHOTS=""            ; MAXLEN=4096 ;;
  *) echo "unknown dataset: $DATASET" >&2; exit 1 ;;
esac

STAMP="$(date +%Y%m%d_%H%M%S)"
RESULT="$REPO/results/${DATASET}_keep${KEEP}_${STAMP}.json"
TMP="$REPO/results/.run_${DATASET}_${STAMP}"
ARGS="pretrained=$MODEL,block_len=32,max_length=$MAXLEN,keep_ratio=$KEEP"
[ -n "$CKPT" ] && ARGS="$ARGS,student_path=$(cd "$(dirname "$CKPT")" && pwd)/$(basename "$CKPT")"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME=/workspace/dllm/.hf_cache HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "$DATASET keep=$KEEP -> $RESULT"
cd "$HARNESS"
python evaluation_script.py \
  --model LLaDA_future \
  --model_args "$ARGS" \
  --tasks "$TASK" ${SHOTS} \
  --include_path "$TASKS_DIR" \
  --limit "${LIMIT:-200}" --batch_size 1 \
  --gen_kwargs "block_length=32,gen_length=$GEN,steps=$GEN,cfg_scale=0.0,temperature=0.0" \
  --output_path "$TMP"

# lm-eval buries its json under <path>/<model>/results_<iso>.json; one run is one
# result, so it comes out flat and the directory goes away.
mv "$(find "$TMP" -name 'results_*.json' | sort | tail -1)" "$RESULT"
rm -rf "$TMP"
echo "wrote $RESULT"
