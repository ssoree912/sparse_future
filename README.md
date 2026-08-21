# future_dllm

KV-cache eviction for diffusion LLMs, ranked by what the finished answer will need.

Model: `GSAI-ML/LLaDA-8B-Instruct`

## Install

```bash
conda env create -f environment.yml
conda activate dllm
```

Commands run from the repo root. Two paths are expected beside the repo and can be overridden:

| | default | override |
|---|---|---|
| model | `../model/LLaDA-8B-Instruct` | `FUTURE_DLLM_MODEL`, or `--model` |
| LongBench parquets | `../data/eval/longbench` | `LONGBENCH_DATA` |

## Datasets

| `--dataset` | source, non-test split | `--gen-length` | eval task |
|---|---|---|---|
| `samsum` | LongBench samsum dialogues | 128 | `longbench_samsum` |
| `gsm8k` | gsm8k train | 128 | `gsm8k`, 5-shot |
| `mmlu` | mmlu validation + dev | 64 | `mmlu_generative` |
| `mbpp` | mbpp validation + prompt | 128 | `mbpp` |
| `math` | hendrycks_math train | 256 | `minerva_math` |
| `mbpp_full` | mbpp full train | 256 | `mbpp` |
| `samsum_lb` / `trec_lb` / `wiki2_lb` | LongBench format, ~2048-token prompts | 128 / 96 / 64 | `longbench_*` |

`--gen-length` should match the task's `max_gen_toks`, since it decides how much of the cache is
prompt rather than model output. Add a dataset with a builder in `teacher/build_prompt_shards.py`.

## Teacher labels

```bash
python teacher/build_prompt_shards.py --dataset samsum --limit 300
python teacher/extract_teacher.py     --dataset samsum --n-samples 300 --gen-length 128
```

→ `artifacts/prompt_shards/<dataset>/`, `artifacts/teacher/<dataset>/`. Both resume; raising the
count extracts only what is missing.

## Training

```bash
python student/train_student.py --teacher-root artifacts/teacher/samsum
```

Comma-separate roots for mixed-domain training. `--epochs`, `--lr`, `--name`.
→ `artifacts/ckpts/<n>ds_<samples>_e<epochs>_lr<lr>_<hash>/`, with `meta.json`.

## Inference

```bash
scripts/run_eval.sh <dataset> <keep_ratio> [checkpoint]

scripts/run_eval.sh samsum 0.1 artifacts/ckpts/1ds_300_e6_lr2e-4_6a5fc6/checkpoint-best
scripts/run_eval.sh gsm8k  1.0        # no eviction, no checkpoint
```

→ `results/<dataset>_keep<ratio>_<timestamp>.json`. `keep_ratio` under 1.0 needs a checkpoint.
`LIMIT` sets the number of items (default 200). Generation length, stop strings and shot count come
from the lm-eval task definition; LongBench tasks are in `eval/tasks/longbench/`, the rest are
lm-eval's own.

Recall against held-out labels, no generation:

```bash
python student/eval_recall.py --student artifacts/ckpts/<name>/checkpoint-best
```
