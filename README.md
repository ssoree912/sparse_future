# future_dllm

KV-cache eviction for diffusion LLMs, ranked by what the finished answer will need.

## Setup

```bash
conda activate dllm          # torch 2.5.1+cu124, transformers 4.46.3, lm-eval 0.4.12
```

Model: [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct), local copy at
`../model/LLaDA-8B-Instruct`. One GPU, batch size 1 — the cache state is per sequence.

Every command below runs from the repo root, and every path is relative to it. Two things live
beside the repo rather than inside it: the model at `../model/`, and the lm-eval harness at
`../dLLM_f/`. Override either with `FUTURE_DLLM_MODEL` / `FUTURE_DLLM_HARNESS`, or `--model`.

Register the lm-eval model once. The package stays in this repo, so there is a single copy:

```bash
ln -sf $(pwd)/eval/LLaDA_future.py ../dLLM_f/eval_model/LLaDA_future.py
echo "from .LLaDA_future import LLaDAFuture" >> ../dLLM_f/eval_model/__init__.py
```

## Datasets

`--dataset` picks the prompt source. Generation length belongs to the dataset — it has to match
what the evaluation runs, because it sets how much of the cache is prompt rather than model output.

| `--dataset` | source, non-test split | `--gen-length` | eval task |
|---|---|---|---|
| `samsum` | LongBench samsum dialogues | 128 | `longbench_samsum` |
| `gsm8k` | gsm8k train | 128 | `gsm8k`, 5-shot |
| `mmlu` | mmlu validation + dev | 64 | `mmlu_generative` |
| `mbpp` | mbpp validation + prompt | 128 | `mbpp` |
| `math` | hendrycks_math train | 256 | `minerva_math` |
| `mbpp_full` | mbpp full train | 256 | `mbpp` |
| `samsum_lb` / `trec_lb` / `wiki2_lb` | LongBench-format, ~2048-token prompts | 128 / 96 / 64 | `longbench_*` |

Add one by writing a builder in `teacher/build_prompt_shards.py` and listing it in `BUILDERS`.

## 1. Teacher labels

```bash
python teacher/build_prompt_shards.py --dataset samsum --limit 300
python teacher/extract_teacher.py     --dataset samsum --n-samples 300 --gen-length 128
```

Writes `artifacts/prompt_shards/<dataset>/` and `artifacts/teacher/<dataset>/`. Both skip samples
already extracted, so an interrupted run resumes and raising the count only adds what is missing.

## 2. Train the scorer

```bash
python student/train_student.py --teacher-root artifacts/teacher/samsum
```

Comma-separate several roots for mixed-domain training. The checkpoint lands in
`artifacts/ckpts/<name>`, named for what distinguishes it — `1ds_300_e6_lr2e-4_6a5fc6` is one
domain, 300 samples, 6 epochs, lr 2e-4, and a hash of the domain names. `--epochs`, `--lr` and
`--name` are there if needed; `meta.json` beside the weights records everything.

## 3. Inference

```bash
scripts/run_eval.sh samsum 0.1 artifacts/ckpts/1ds_300_e6_lr2e-4_6a5fc6/checkpoint-best
scripts/run_eval.sh gsm8k  1.0        # no eviction, no scorer needed
```

Arguments are dataset, `keep_ratio`, and the checkpoint. Each run writes one json to
`results/<dataset>_keep<ratio>_<timestamp>.json`. `keep_ratio` below 1.0 requires a checkpoint;
`1.0` disables eviction. Set `LIMIT` to change the number of items (default 200).

Recall against held-out labels, without generating:

```bash
python student/eval_recall.py --student artifacts/ckpts/<name>/checkpoint-best
```
