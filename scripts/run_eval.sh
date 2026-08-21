#!/usr/bin/env bash
# lm-eval, writing one json into results/ per run.
#
#   scripts/run_eval.sh <dataset> <keep_ratio> [checkpoint]
#
#   scripts/run_eval.sh samsum 0.1 artifacts/ckpts/1ds_300_e6_lr2e-4_6a5fc6/checkpoint-best
#   scripts/run_eval.sh gsm8k  1.0        # no eviction, no scorer needed
#
# Generation length, stop strings and shot count come from the lm-eval task, not
# from here.
set -euo pipefail

DATASET="${1:?usage: run_eval.sh <dataset> <keep_ratio> [checkpoint]}"
KEEP="${2:?usage: run_eval.sh <dataset> <keep_ratio> [checkpoint]}"
CKPT="${3:-}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${FUTURE_DLLM_MODEL:-$REPO/../model/LLaDA-8B-Instruct}"
LONGBENCH_DATA="${LONGBENCH_DATA:-$REPO/../data/eval/longbench}"

case "$DATASET" in
  samsum|2wikimqa|trec|triviaqa|qasper|gov_report)
          TASK="longbench_$DATASET"; SHOTS="" ;;
  gsm8k)  TASK=gsm8k;               SHOTS="--num_fewshot 5" ;;
  mmlu)   TASK=mmlu_generative;     SHOTS="--num_fewshot 5" ;;
  mbpp)   TASK=mbpp;                SHOTS="" ;;
  math)   TASK=minerva_math;        SHOTS="" ;;
  *) echo "unknown dataset: $DATASET" >&2; exit 1 ;;
esac

STAMP="$(date +%Y%m%d_%H%M%S)"
RESULT="$REPO/results/${DATASET}_keep${KEEP}_${STAMP}.json"
TMP="$REPO/results/.run_${DATASET}_${STAMP}"

# The LongBench task files carry a placeholder for the data directory, so the
# repo does not hard-code where the parquets live.
TASKS_DIR="$TMP/tasks"
mkdir -p "$TASKS_DIR"
cp "$REPO"/eval/tasks/longbench/*.py "$TASKS_DIR/"
for y in "$REPO"/eval/tasks/longbench/*.yaml; do
  sed "s|LONGBENCH_DATA_DIR|$LONGBENCH_DATA|" "$y" > "$TASKS_DIR/$(basename "$y")"
done

ARGS="pretrained=$MODEL,block_len=32,keep_ratio=$KEEP"
[ -n "$CKPT" ] && ARGS="$ARGS,student_path=$(cd "$(dirname "$CKPT")" && pwd)/$(basename "$CKPT")"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-$REPO/../.hf_cache}"
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

echo "$DATASET keep=$KEEP -> $RESULT"
cd "$REPO"
python eval/run.py \
  --model LLaDA_future \
  --model_args "$ARGS" \
  --tasks "$TASK" ${SHOTS} \
  --include_path "$TASKS_DIR" \
  --limit "${LIMIT:-200}" --batch_size 1 \
  --output_path "$TMP/out"

# lm-eval buries its json under <path>/<model>/results_<iso>.json; one run is one
# result, so it comes out flat and the scratch directory goes away.
mv "$(find "$TMP/out" -name 'results_*.json' | sort | tail -1)" "$RESULT"
rm -rf "$TMP"
echo "wrote $RESULT"
