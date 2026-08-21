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
LONGBENCH_DATA="${LONGBENCH_DATA:-$REPO/../data/longbench/data}"

case "$DATASET" in
  # LongBench - task files in eval/tasks/longbench/, data from LONGBENCH_DATA
  samsum|trec|triviaqa|2wikimqa|hotpotqa|musique|qasper|narrativeqa|multifieldqa_en|\
  gov_report|qmsum|multi_news|lcc|repobench-p|passage_retrieval_en|passage_count)
              TASK="longbench_$DATASET";              SHOTS="" ;;
  # lm-eval's own
  mmlu)       TASK=mmlu_generative;                   SHOTS="--num_fewshot 5" ;;
  arc_c)      TASK=arc_challenge;                     SHOTS="--num_fewshot 25" ;;
  piqa)       TASK=piqa;                              SHOTS="" ;;
  gpqa)       TASK=gpqa_main_generative_n_shot;       SHOTS="--num_fewshot 5" ;;
  gsm8k)      TASK=gsm8k;                             SHOTS="--num_fewshot 5" ;;
  math)       TASK=minerva_math;                      SHOTS="" ;;
  humaneval)  TASK=humaneval;                         SHOTS="" ;;
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
