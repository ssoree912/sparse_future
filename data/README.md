# Datasets

Where every dataset came from, and how it was checked. Data lives outside the repo, under
`../data/` — `../data/train/<name>/` and `../data/eval/<name>/`.

Provenance column:

- **recorded** — a `meta.json` / `SOURCE.json` written at download time names the repo and split
- **row-count** — no download record survives; the repo is inferred and the row counts match the
  official release exactly
- **card** — identified from the dataset card shipped alongside the files

## Evaluation

| dataset | rows | source | provenance | `run_eval.sh` |
|---|---|---|---|---|
| MMLU | test 14,042 · validation 1,531 · dev 285 | [cais/mmlu](https://huggingface.co/datasets/cais/mmlu) | row-count | `mmlu` |
| ARC-Challenge | test 1,172 · validation 299 | [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc) `ARC-Challenge` | recorded | `arc_c` |
| PIQA | validation 1,838 | [ybisk/piqa](https://huggingface.co/datasets/ybisk/piqa) | recorded | `piqa` |
| GPQA | main 448 | [Idavidrein/gpqa](https://huggingface.co/datasets/Idavidrein/gpqa) | card | `gpqa` |
| GSM8K | test 1,319 | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) `main` | recorded | `gsm8k` |
| MATH | test 5,000 | [EleutherAI/hendrycks_math](https://huggingface.co/datasets/EleutherAI/hendrycks_math) | row-count | `math` |
| MATH-500 | 500 | [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) | card | — |
| HumanEval | 164 | [openai/openai_humaneval](https://huggingface.co/datasets/openai/openai_humaneval) | recorded | `humaneval` |
| LongBench | 34 tasks, 200 rows each (500 for lcc / repobench-p) | [zai-org/LongBench](https://huggingface.co/datasets/zai-org/LongBench) | recorded | task name |

PIQA's test split is unlabelled, so validation is the evaluation set — lm-eval does the same.

LongBench data is the official release: `input` plus a raw `context`, with the prompt in
`config/dataset2prompt.json`. The eval task files in `eval/tasks/longbench/` are generated from
that config. Repackaged copies that bake the prompt into `context` render the instruction twice.

`../data/eval/` also holds MBPP, MMLU-Pro, BBH, LongProc and a clone of the Hendrycks MATH
repository. Nothing here evaluates on them.

## Training

Only non-test splits, so a teacher label never sees an evaluation item.

| dataset | rows | source | provenance |
|---|---|---|---|
| MMLU | auxiliary_train 99,842 | [cais/mmlu](https://huggingface.co/datasets/cais/mmlu) | row-count |
| ARC-Challenge | train 1,119 | [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc) | recorded |
| PIQA | train 16,113 | [ybisk/piqa](https://huggingface.co/datasets/ybisk/piqa) | recorded |
| GSM8K | train 7,473 | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) `main` | row-count |
| MATH | train 7,500 | [EleutherAI/hendrycks_math](https://huggingface.co/datasets/EleutherAI/hendrycks_math) | row-count |
| MBPP | full train 374 · sanitized 120 | [google-research-datasets/mbpp](https://huggingface.co/datasets/google-research-datasets/mbpp) | row-count |

GPQA and HumanEval ship no training split. MBPP stands in for the code domain; GPQA has no stand-in.

### LongBench source datasets

LongBench itself is test-only. These are the upstream datasets each task was built from, per
[`LongBench/task.md`](https://github.com/THUDM/LongBench/blob/main/LongBench/task.md), kept raw —
the conversion into LongBench prompt format is a separate step.

| task | rows | source | provenance |
|---|---|---|---|
| samsum | train 14,732 | official SAMSum release, [`corpus.7z`](https://arxiv.org/src/1911.12237v2/anc/corpus.7z) | card |
| 2wikimqa | train 167,454 | [2WikiMultihopQA](https://github.com/Alab-NII/2wikimultihop), mirror `xanhho/2WikiMultihopQA` | card |
| trec | `train_5500.label` | TREC original | card |
| triviaqa | 64,916 | [mandarjoshi/trivia_qa](https://huggingface.co/datasets/mandarjoshi/trivia_qa) `rc.web` | card |
| narrativeqa | train 32,747 | [deepmind/narrativeqa](https://huggingface.co/datasets/deepmind/narrativeqa) | recorded |
| qasper | train 888 · validation 281 | [allenai/qasper](https://huggingface.co/datasets/allenai/qasper) | recorded |
| gov_report | train 17,517 | [ccdv/govreport-summarization](https://huggingface.co/datasets/ccdv/govreport-summarization) | recorded |
| multi_news | train 44,972 | [alexfabbri/multi_news](https://huggingface.co/datasets/alexfabbri/multi_news) | recorded |
| musique | train 19,938 | [dgslibisey/MuSiQue](https://huggingface.co/datasets/dgslibisey/MuSiQue) | recorded |
| repobench-p | cross_file_first 8,033 | [tianyang/repobench_python_v1.1](https://huggingface.co/datasets/tianyang/repobench_python_v1.1) | recorded |
| qmsum | — | [Yale-LILY/QMSum](https://github.com/Yale-LILY/QMSum) | not downloaded |

Two notes on sources that are easy to get wrong. SAMSum: the popular re-upload
[`knkarthick/samsum`](https://huggingface.co/datasets/knkarthick/samsum) normalises CRLF to LF,
while LongBench's dialogues keep `\r\n`, so the official corpus is the one to use. GovReport,
QMSum, NarrativeQA, Qasper and MultiNews: `task.md` says LongBench takes the original paper data
and applies the [ZeroSCROLLS](https://www.zero.scrolls-benchmark.com/) template, which is not the
same as taking [`tau/scrolls`](https://huggingface.co/datasets/tau/scrolls) directly — the row
counts differ (gov_report 17,517 here against scrolls' 17,457).

## Contamination

Checked, all zero unless noted:

| training data | against | overlap |
|---|---|---|
| samsum / trec / 2wikimqa LongBench-format train | LongBench test | 0 |
| MATH train | MATH-500 | 0 / 100 |
| MMLU auxiliary_train | ARC-Challenge test | 0 / 1,172 |
| teacher shards (samsum, gsm8k, mmlu, mbpp) | their eval sets | 0 |

MMLU's `auxiliary_train` bundles ARC data — `auxiliary_train_raw/arc_hard.csv` alone shares 6 rows
with ARC-Challenge test — but the distributed `auxiliary_train` parquet those rows were dropped
from is what gets used, and it overlaps by none.

LongBench builds most of its test sets from the upstream *validation* split (2wikimqa, hotpotqa,
musique, samsum, trec), which is why the upstream train splits are safe to train on.
