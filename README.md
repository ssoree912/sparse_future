# sparse_future

Cache-eviction experiments on top of [Sparse-dLLM](https://github.com/OpenMOSS/Sparse-dLLM)
with LLaDA-8B-Instruct, aimed at answering one question: **when the KV cache is cut to 10%,
how much of the lost quality can a better token-selection scorer recover?**

Everything here is evaluated with OpenCompass on LongBench SAMSum, 2048-token context,
200 samples, `block_length=32`, `steps=128` (4 blocks x 32 steps), greedy (`temperature=0`).

## Results so far (ROUGE-L)

| setting | keep 0.5 | keep 0.1 |
|---|---|---|
| origin (no cache) | 38.14 | — |
| heuristic eviction (Sparse-dLLM) | 39.75 | 33.89 |
| student scorer, suffix always kept | 39.13 | 16.59 |
| student scorer, suffix also evictable | — | 19.00 |
| **block oracle** (perfect future attention, one choice per block) | — | **34.60** |
| per-step oracle (re-select every step) | — | running |
| no eviction, frozen cache (`keep=1.0`) | — | queued |

The block oracle is the headroom gate: it selects the kept 10% using the attention the
block's queries *actually* pay over its remaining steps. At **34.60 vs the heuristic's
33.89** it says that under one-choice-per-block eviction, no scorer — however well
trained — buys more than ~0.7 ROUGE-L. That is what motivates the two runs still in
flight: `keep=1.0` separates cache staleness from eviction capacity, and the per-step
oracle separates *what* is kept from *how often* the choice is revisited.

## What is in here

### `sparse_dllm/` — instrumented model code

Drop-in replacements for `opencompass/models/sparse_dllm/`. New model kwargs:

- `student_evict_suffix=True` — let the student scorer rank prompt **and** suffix columns
  in a single top-k with the baseline's budget (`int(n_candidates * keep_ratio)`), instead
  of unconditionally retaining every suffix token.
- `oracle_eviction=True` — two passes per block. Pass 1 keeps the whole frozen cache and
  records, per layer, the attention the block's still-masked rows pay to each candidate
  (mean over heads, summed over steps 2..S). Then `x` is reset to the post-step-1 snapshot
  and pass 2 replays those steps against the top-k pruned cache.
- `oracle_per_step=True` — same, but the kept set is re-chosen at every step from that
  step's own attention: run the step on the full cache, undo the commit, prune, re-run.
- `oracle_pool=True/False` — whether the baseline's `max_pool1d(kernel_size=3)` smoothing
  is applied to the recorded importance before top-k.

Both oracle modes are validated by construction: at `keep_ratio=1.0` they reproduce the
unmodified generation path token-for-token (`scripts/verify_oracle.py`,
`scripts/verify_perstep.py`). In record mode `filter_cache` keeps candidates in their
original order, because `topk` otherwise permutes cache columns and recorded attention
column *j* would no longer correspond to candidate *j*.

### `teacher/extract_block_cache_teacher.py` — block-wise teacher collection

Collects eviction labels under the deployment protocol rather than the legacy one. Per
(sample, block) it stores the exact model input needed to replay inference-time features
(`x_at_block_start`), the candidate index set (prompt + suffix, current block excluded),
and four per-layer targets: attention summed over still-masked rows, summed over all rows,
max over steps and rows, and the deployment heuristic's own score for reference.

This differs from the legacy teacher in ways that each matter:

| | legacy | here |
|---|---|---|
| columns scored | prompt only | prompt + suffix |
| aggregation unit | whole answer | one block |
| cache during collection | recomputed each step | frozen at step 1, as at inference |
| feature source | prompt-only forward | full prompt+generation forward, replayed |
| block length | 8 | 32 |

### `configs/` — OpenCompass model configs

One per row of the results table. Use with
`python run.py --config-dir myeval --models <name> --datasets longbench_samsum_gen`.

### `scripts/`

`run_samsum_with_retry.sh` reruns an eval with `-r <timestamp>` until it completes; the
student/oracle code paths hit an intermittent segfault inside `libcuda.so` (same faulting
offset every time, no NVRM Xid, not data-dependent) and OpenCompass resumes cleanly from
its partial predictions.

## Gotcha worth knowing

If the environment has a non-editable `opencompass` install in `site-packages`, the
inference **subprocess** imports that copy, not the working tree. A new model kwarg then
surfaces only as `WARNING - Unused argument <name>=True` in the infer log while the old
code path runs and produces byte-identical predictions. Copy changed files into
site-packages, and verify a code change by a debug print in the infer log or by differing
predictions — never by the smoke score, which can match the old path exactly.
