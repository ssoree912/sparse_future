# results

Not tracked by git. One lm-eval run, one json:

```
<model>/keep<ratio>/<dataset>/<dataset>_keep<ratio>_<method>_<YYYYmmdd_HHMMSS>.json
origin/<dataset>/<dataset>_origin_<YYYYmmdd_HHMMSS>.json
```

`<method>` is `none` (no eviction), `baseline` (Sparse-dLLM's criterion) or the
checkpoint's name. Stock LLaDA keeps no cache to take a fraction of, so it gets
its own tree rather than a keep ratio.

Written by `scripts/run_eval.sh <dataset> <keep_ratio> [checkpoint|baseline]`,
which also takes the generation length from the lm-eval task.

`.resume/` holds answers from a run still in progress — one json line each,
fsynced as they are produced, so a segfault costs the item in flight rather than
the run. Removed once the run finishes; a file left there means a crash.

## SAMSum, keep 0.1, 200 items

Official LongBench data, same prompt and generation settings across rows, cache
budget verified identical at 214 entries per block.

| | ROUGE-L |
|---|---|
| no eviction (keep 1.0) | 36.05 |
| Sparse-dLLM baseline | 35.13 |
| future_dllm | 38.61 |

Eviction costs the baseline 0.92 and gains ours 2.56 over keeping everything.
Worth explaining before it is published: samsum fills 2048 tokens with few-shot
examples around one dialogue to summarise, and dropping the ones the scorer
judges irrelevant appears to help. Whether that holds anywhere else is what the
GSM8K rows are for.

Runs from before the LongBench data and prompt were fixed are in
`../v1_results/sparse_future/results/`.
