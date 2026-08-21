# future_dllm

KV-cache eviction for diffusion LLMs, decided by **what the answer will need** rather than by what the block is attending to right now.

A diffusion LLM decodes a block of tokens over many denoising steps. Sparse-dLLM prunes the KV cache once per block, ranking cache entries by the attention the block pays *at the moment of pruning* — when every token in the block is still `[MASK]`. That ranking is made before the block knows what it is going to say.

`future_dllm` learns the ranking the finished answer would have given. A teacher pass fills each block with the full cache, runs one extra forward over the completed block, and records what those answer tokens actually looked at. A small scorer is trained to predict that from the state available at pruning time, so at deployment it makes the same single selection per block with no extra forward and no extra memory.

## The label

For a block with rows `r` (the answer tokens, once written) and cache candidates `j`:

```
a_rj = softmax_j ( q_r · k_j / sqrt(d) )          head-averaged attention
I_j  = max_r a_rj                                 per-candidate importance
```

The maximum, not the sum. A cache entry matters if **any** finished token depended on it strongly; summing averages that token away among 31 others that did not care. On SAMSum at keep 0.1 the sum form scores 34.72 against the max form's 36.63, which is why only the max form remains in this repo.

The scorer is trained per layer to rank candidates by `I_j`, with a listwise KL term against the normalised label plus a pairwise ordering term, and checkpoints are selected on recall averaged over k ∈ {5, 10, 20, 30, 50}% so the result is a scorer rather than a fixed budget.

## Results

`LLaDA-8B-Instruct`, keep ratio 0.1, block length 32. Everything is lm-eval, 200 items, one scorer across both columns. Budget verified identical between rows — both keep 95/100/96 entries per block, so only the selection criterion differs.

| | SAMSum ROUGE-L | GSM8K flex | GSM8K strict |
|---|---|---|---|
| no eviction (keep 1.0) | 35.16 | 0.765 | 0.480 |
| Sparse-dLLM baseline | 28.97 | 0.455 | 0.140 |
| **future_dllm** | **31.45** | **0.745** | **0.470** |

Of what eviction costs the baseline, GSM8K recovers 93% and SAMSum 40%. The split follows what the cache holds — a maths block's cache is the model's own reasoning chain, where losing one intermediate result destroys everything after it, while a summarisation block's cache is input text that no ranking can reconstruct from 10% of the tokens.

One scorer covers all four training domains. Trained on summarisation alone it reaches recall 0.752 in domain but 0.625–0.692 outside; trained on the mixture it holds 0.752 in domain and reaches 0.760–0.771 outside. The weakness was missing data, not scorer capacity. That mixture costs nothing where it might have — on SAMSum the summarisation-only scorer scores 31.01 against the mixture's 31.45, a difference well inside the ±1.7 standard error.

Earlier runs measured SAMSum under OpenCompass, where the same three rows read 33.89 / 35.86 / 40.02. The two harnesses truncate long prompts from opposite ends — OpenCompass keeps the first 2048 tokens, lm-eval the last — so the absolute numbers differ by about five points while the share of the gap recovered does not.

## Environment

```bash
conda activate dllm          # torch 2.5.1+cu124, transformers 4.46.3, lm-eval 0.4.12
```

Model: `LLaDA-8B-Instruct` at `/workspace/dllm/model/LLaDA-8B-Instruct`. One GPU, batch size 1 — the cache state is per sequence.

Install the lm-eval model once. The package stays in this repo, so there is a single copy to keep current:

```bash
HARNESS=/workspace/dllm/dLLM_f
ln -sf $(pwd)/eval/LLaDA_future.py $HARNESS/eval_model/LLaDA_future.py
echo "from .LLaDA_future import LLaDAFuture" >> $HARNESS/eval_model/__init__.py
```

## Running it

Three steps. Everything else is fixed: block length 32, greedy decoding, one selection per block.

**1. Prompts → teacher labels.** Prompt shards come from non-test splits only, so the teacher never sees an evaluation item.

```bash
python teacher/build_prompt_shards.py --dataset samsum --limit 300
python teacher/extract_teacher.py --dataset samsum --n-samples 300 --gen-length 128
```

**2. Train the scorer.** Pass several teacher roots comma-separated for mixed-domain training; validation is split per domain and the checkpoint is chosen on the domain macro average.

```bash
python student/train_student.py \
  --teacher-root results/budget/teacher/samsum \
  --output-dir  results/budget/student/samsum
```

**3. Evaluate.** lm-eval is the only harness here.

```bash
cd $HARNESS
python evaluation_script.py --model LLaDA_future \
  --model_args "pretrained=$MODEL,keep_ratio=0.1,student_path=$CKPT" \
  --tasks gsm8k --num_fewshot 5 --limit 200 --batch_size 1 \
  --gen_kwargs "block_length=32,gen_length=256,steps=256,temperature=0.0"
```

Optionally check the scorer against held-out labels before spending a generation run:

```bash
python student/eval_recall.py --student results/budget/student/samsum/checkpoint-best
```

## Options

Two knobs are meant to be changed.

| option | where | meaning |
|---|---|---|
| `keep_ratio` | `--model_args` | share of cache candidates kept per block. `1.0` disables eviction and needs no scorer; below `1.0` a `student_path` is required |
| `--dataset` | `build_prompt_shards.py`, `extract_teacher.py` | which prompt source to build labels from |

Generation length belongs to the dataset, not to the user — it has to match what the evaluation runs, because it decides how much of the cache is prompt versus the model's own output:

| dataset | prompt | `--gen-length` | eval task | eval `gen_length` |
|---|---|---|---|---|
| `samsum` | dialogue | 128 | `longbench_samsum` | 128 |
| `gsm8k` | train split | 128 | `gsm8k` (5-shot) | 256 |
| `mmlu` | validation + dev | 64 | `mmlu_generative` | 64 |
| `mbpp` | validation + prompt | 128 | `mbpp` | 256 |
| `math` | `hendrycks_math` train | 256 | `minerva_math` | 256 |

`samsum_lb`, `trec_lb` and `wiki2_lb` build the long LongBench-format prompts (~2048 tokens) if you want that regime; nothing in the reported results uses them.

## Layout

```
future_dllm/     the model: CustomCache (eviction), generate (block-wise decoding), the scorer module
teacher/         prompt shards, and the final × row-max label extractor
student/         scorer training, and a recall check against held-out labels
eval/            the lm-eval model, registered as LLaDA_future
```

## Notes

- The scorer sees `x_at_block_start` replayed at training time rather than stored hidden states, so its features match deployment exactly. Two things had to line up for that: the forward is the same shape (an earlier version trained on a prompt-only forward and was deployed on a prompt+generation forward), and the snapshot is taken *entering* step 1 rather than after it, because step 1 prunes before it reveals.
- Prompt and suffix compete in one top-k, so the budget is exactly `candidates × keep_ratio` and matches the baseline's accounting.
- `--use_cache <dir>` on lm-eval makes a run resumable, which matters on long jobs.
