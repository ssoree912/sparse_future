# results

Not tracked by git. One lm-eval run, one json:

```
<dataset>_keep<ratio>_<YYYYmmdd_HHMMSS>.json
```

Written by `scripts/run_eval.sh <dataset> <keep_ratio> [checkpoint]`, which also
picks the generation length the task was measured with.

Current numbers, keep 0.1, 200 items:

| | SAMSum ROUGE-L | GSM8K flex |
|---|---|---|
| no eviction (keep 1.0) | 35.16 | 0.765 |
| Sparse-dLLM baseline | 28.97 | 0.455 |
| future_dllm | 31.91¹ | 0.745² |

¹ summarisation-only scorer, trained under the step-1 snapshot fix.
² mixed-domain scorer, trained before the fix — its labels for gsm8k, mmlu and
mbpp have not been re-extracted yet, so this row is one generation behind the code.

Runs from before the fix are in `../v1_results/sparse_future/results/`.
