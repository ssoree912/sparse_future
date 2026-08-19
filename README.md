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
| **per-step oracle** (re-select every step) | — | **36.70** |
| **no eviction, frozen cache** (`keep=1.0`) | — | **40.02** |

Those three oracles decompose the whole keep-0.1 gap. Freezing the cache without evicting
anything scores **40.02**, above the uncompressed model, so cache staleness costs nothing
and 40.02 — not 38.14 — is the ceiling eviction is measured against. From there:

| slice | cost | recoverable by |
|---|---|---|
| capacity at 10% (40.02 → 36.70) | **3.32** | nothing at this budget — even perfect per-step selection pays it |
| choosing once per block (36.70 → 34.60) | **2.10** | re-selecting more often |
| imperfect scoring (34.60 → 33.89) | **0.71** | a better scorer |

A better-trained scorer is aimed at the smallest of the three. Re-selection frequency is
worth three times as much, which is why `oracle_reselect_every` exists.

One caveat on that lever: bringing an evicted entry back requires its K/V to still exist,
so per-step re-selection preserves the attention-compute saving but not the memory saving,
unless the full cache is rebuilt periodically by an extra full-sequence forward (the
block already does two, at steps 0 and 1).

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

This differs from the pipeline under diagnosis in ways that each matter:

| | `teacher/attention_delta/` | `teacher/extract_block_cache_teacher.py` |
|---|---|---|
| columns scored | prompt only | prompt + suffix |
| aggregation unit | whole answer | one block |
| cache during collection | recomputed each step | frozen at step 1, as at inference |
| feature source | prompt-only forward | full prompt+generation forward, replayed |
| block length | 8 | 32 |

### `student/` — scorer training pipeline (as it stands today)

`student_model.py` is the per-layer scorer, `training_loop.py` builds features and targets
and runs the loss, `train_student.py` / `train_config.py` are the CLI. This is the code
that produced the deployed checkpoint, kept here because the numbers above came from it —
not because it is the design being kept. Two definitions in it are candidate reasons the
scorer collapses at keep 0.1:

- `compute_prompt_hidden_states` (`training_loop.py`) runs the model on **prompt tokens
  only**, while at inference the scorer consumes hidden states from a full
  prompt+generation forward captured at `cache_state==1`. LLaDA attends bidirectionally,
  so the mask region changes even the prompt representations — the scorer is deployed on
  inputs it never saw.
- targets come from the teacher below, which scores prompt columns only.

### `teacher/attention_delta/` — the teacher pipeline currently under diagnosis

This is the extraction chain that produced the deployed checkpoint, not dead code.
`extract_offline_hybrid_from_shards.py` drives generation from prompt-source shards and
`offline_hybrid_teacher.py` collects the two targets: commit-time attention and cumulative
prompt K/V movement (the delta head, unused at inference today).

Three definitions in it are what the block-wise collector is meant to replace.
`_capture_reference` computes attention over the whole sequence and then discards the
suffix half at the last moment (`attention[:, :, :, :prompt_length]`), so suffix columns
have no labels at all. Aggregation runs over the entire answer rather than per block, so a
label cannot express what one block needs next. And the checkpoint's teacher was extracted
at `block_length=8` while deployment decodes at 32.

### `configs/` — OpenCompass model configs

One per row of the results table. Use with
`python run.py --config-dir myeval --models <name> --datasets longbench_samsum_gen`.

### `scripts/`

`run_samsum_with_retry.sh` reruns an eval with `-r <timestamp>` until it completes; the
student/oracle code paths hit an intermittent segfault inside `libcuda.so` (same faulting
offset every time, no NVRM Xid, not data-dependent) and OpenCompass resumes cleanly from
its partial predictions.

## Status — what is and is not done

| piece | state |
|---|---|
| oracle harness (block and per-step) | implemented, validated at `keep=1.0`, block oracle measured |
| block-wise teacher collector | written, statically checked, **not run yet** |
| ratio-agnostic scorer training | **not written** — the oracle verdict says it is worth ≤0.71 |
| current student + teacher pipeline | included as-is, under diagnosis |

The scorer redesign was deliberately blocked on the oracle numbers, and they came back
against it: a perfect one-shot-per-block scorer reaches 34.60, so retraining buys at most
0.71 at this budget. The structural lever — how often the kept set is re-chosen — is worth
2.10, and the remaining 3.32 is a hard capacity cost of keeping only 10%.

When it is written, the scorer is to be trained as a **continuous, budget-independent
ranker** — no `keep_ratio` baked into the objective, listwise targets in rank or log space
so the loss does not collapse onto the head of the attention distribution, pairwise terms
sampled across the whole range, and checkpoint selection on a k-grid (0.05/0.1/0.2/0.3/0.5)
rather than any single operating point.

## Gotcha worth knowing

If the environment has a non-editable `opencompass` install in `site-packages`, the
inference **subprocess** imports that copy, not the working tree. A new model kwarg then
surfaces only as `WARNING - Unused argument <name>=True` in the infer log while the old
code path runs and produces byte-identical predictions. Copy changed files into
site-packages, and verify a code change by a debug print in the infer log or by differing
predictions — never by the smoke score, which can match the old path exactly.
