#!/bin/bash
# Usage: run_samsum_with_retry.sh <model_config_name> <work_dir> <gpu_id>
# Runs LongBench SAMSum via OpenCompass; on segfault, resumes with -r <timestamp>.
set -u
MODEL="$1"; WORKDIR="$2"; GPU="$3"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HOME=/workspace/dllm/.hf_cache
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
unset DATASET_SOURCE COMPASS_DATA_CACHE

cd /workspace/dllm/opencompass

for attempt in 1 2 3 4 5 6 7 8; do
  TS=$(ls -1 "$WORKDIR" 2>/dev/null | sort | tail -1)
  if [ -n "${TS:-}" ]; then RESUME=(-r "$TS"); else RESUME=(); fi
  echo "=== attempt $attempt (resume: ${TS:-none}) ==="
  conda run --no-capture-output -n sparse-dllm python run.py \
    --config-dir myeval \
    --models "$MODEL" \
    --datasets longbench_samsum_gen \
    --work-dir "$WORKDIR" \
    --max-workers-per-gpu 1 \
    "${RESUME[@]}"
  if ls "$WORKDIR"/*/results/*/LongBench_samsum.json >/dev/null 2>&1; then
    echo "=== SUCCESS after attempt $attempt ==="
    exit 0
  fi
  echo "=== attempt $attempt incomplete (likely segfault); retrying with resume ==="
done
echo "=== FAILED after 8 attempts ==="
exit 1
