"""Turn evaluation datasets into prompt shards the teacher extractor can read.

Only non-test splits are used so the teacher never sees an evaluation item.

카테고리별 대표 하나씩. 토픽보다 중요한 축은 캐시에서 프롬프트가 차지하는 비율
P/(P+32b)이라, 장문(0.9대)과 자기추론(0.2~0.3)의 양 끝을 모두 덮도록 골랐다.

  samsum_lb   장문 요약        LongBench 형식 train, 프롬프트 2048
  trec_lb     장문 few-shot 분류 LongBench 형식 train, 프롬프트 2048
  wiki2_lb    장문 멀티홉 QA    LongBench 형식 train, 프롬프트 ~1000
  math        수학 추론        hendrycks_math train (MATH500은 test에서 나온다)
  mbpp_full   코드            mbpp full train
  gsm8k/mmlu/mbpp  기존 버킷

LongBench 계열은 평가와 같이 chat template 없이 오른쪽 절단한다 (llada_wrapper의
drop_middle=False 경로와 동일). 나머지는 평가와 같이 chat template을 씌운다.
"""

from __future__ import annotations

import argparse, glob, json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT.parent / "data"          # dataset sources live beside the repo
HF_CACHE = REPO_ROOT.parent / ".hf_cache"

import torch
from transformers import AutoTokenizer

MATH_INSTRUCTION = ("Please reason step by step, and put your final answer within "
                    "\\boxed{}.")


def _drop_test_overlap(table, column, test_glob):
    """MMLU and MBPP each repeat a few items across splits upstream - 43 of MMLU's
    1816 validation+dev questions appear verbatim in test, and three MBPP task
    descriptions do under a different task_id. Drop them before sampling."""
    import pandas as pd, re
    frames = [pd.read_parquet(f) for f in sorted(glob.glob(str(test_glob)))]
    if not frames:
        return table
    norm = lambda s: re.sub(r"\s+", " ", str(s)).strip().lower()
    seen = {norm(v) for v in pd.concat(frames)[column]}
    keep = [norm(v) not in seen for v in table[column]]
    dropped = len(keep) - sum(keep)
    if dropped:
        print(f"  dropped {dropped} rows that also appear in test", flush=True)
    return table[keep]


def _balanced(table, columns, limit, seed=0):
    """Take `limit` rows spread evenly over the sub-categories.

    Several of these datasets arrive sorted by category - MMLU's validation split
    is in subject order - so head(limit) picks the first few subjects and nothing
    else. Taking one row from each group in turn covers all of them, and a group
    that runs out simply drops out of the rotation.
    """
    import pandas as pd
    if not columns or not all(c in table.columns for c in columns):
        return table.sample(frac=1.0, random_state=seed).head(limit)
    groups = [g.sample(frac=1.0, random_state=seed).to_dict("records")
              for _, g in table.groupby(list(columns), sort=True)]
    out, i = [], 0
    while len(out) < limit and any(groups):
        g = groups[i % len(groups)]
        if g:
            out.append(g.pop())
        i += 1
        if i % max(1, len(groups)) == 0:
            groups = [g for g in groups if g]
            i = 0
            if not groups:
                break
    return pd.DataFrame(out[:limit])


def samsum(limit):
    """LongBench-style single dialogue, byte-for-byte the format the first 300
    teacher shards were built with (no chat template, no few-shot block)."""
    rows = []
    with open(DATA / "train/samsum/a100_source_train.jsonl") as fh:
        for i, line in enumerate(fh):
            if i >= limit:
                break
            rows.append(json.loads(line))
    return [(f"{r['_id']}-1",
             "Summarize the dialogue into a few short sentences. "
             "The following are some examples.\n\n"
             f"Dialogue: {r['context']}\n\n"
             "Summarize the dialogue into a short summary.")
            for r in rows]


def gsm8k(limit):
    from datasets import Dataset
    path = glob.glob(str(HF_CACHE / "datasets/openai___gsm8k/main/*/*/gsm8k-train.arrow"))
    rows = Dataset.from_file(path[0]).select(range(limit))
    return [(f"gsm8k-{i}", f"{r}\n\n{MATH_INSTRUCTION}")
            for i, r in enumerate(rows["question"])]


def mmlu(limit):
    import pandas as pd
    frames = [pd.read_parquet(f) for split in ("validation", "dev")
              for f in glob.glob(str(DATA / f"eval/mmlu/all/{split}-*.parquet"))]
    table = _drop_test_overlap(pd.concat(frames), "question",
                               DATA / "eval/mmlu/all/test-*.parquet")
    table = _balanced(table, ["subject"], limit)               # 57 subjects
    out = []
    for i, row in enumerate(table.itertuples()):
        options = "\n".join(f"{chr(65 + j)}. {c}" for j, c in enumerate(row.choices))
        out.append((f"mmlu-{i}",
                    f"The following is a multiple choice question about "
                    f"{row.subject.replace('_', ' ')}.\n\n{row.question}\n{options}\n\n"
                    f"Answer with the letter of the correct option.\nAnswer:"))
    return out


def mbpp(limit):
    import pandas as pd
    frames = [pd.read_parquet(f) for split in ("validation", "prompt")
              for f in glob.glob(str(DATA / f"eval/mbpp/full/{split}-*.parquet"))]
    table = _balanced(pd.concat(frames), [], limit)
    return [(f"mbpp-{i}",
             f"You are an expert Python programmer. {row.text}\n"
             f"Your code should pass these tests:\n" + "\n".join(row.test_list) + "\n")
            for i, row in enumerate(table.itertuples())]


def _longbench(path, limit):
    """LongBench train 파일은 평가 parquet과 필드가 바이트 단위로 같다.
    평가에서 쓰는 프롬프트는 context + "\n\n" + question 이다."""
    rows, out = [], []
    with open(path) as fh:
        for i, line in enumerate(fh):
            if i >= limit:
                break
            rows.append(json.loads(line))
    for i, r in enumerate(rows):
        out.append((f"{Path(path).parent.name}-{i}", f"{r['context']}\n\n{r['question']}"))
    return out


def samsum_lb(limit):
    return _longbench(DATA / "train/samsum/train.jsonl", limit)


def trec_lb(limit):
    return _longbench(DATA / "train/trec/train.jsonl", limit)


def wiki2_lb(limit):
    return _longbench(DATA / "train/2wikimqa/train.jsonl", limit)


def math(limit):
    """MATH500은 test에서 뽑은 것이라 train split은 겹치지 않는다."""
    import pandas as pd
    frames = [pd.read_parquet(f) for f
              in sorted(glob.glob(str(DATA / "train/hendrycks_math/*/train-*.parquet")))]
    table = _balanced(pd.concat(frames), ["type", "level"], limit)  # 7 x 5 cells
    return [(f"math-{i}", f"{row.problem}\n\n{MATH_INSTRUCTION}")
            for i, row in enumerate(table.itertuples())]


def mbpp_full(limit):
    import pandas as pd
    table = _drop_test_overlap(
        pd.read_parquet(DATA / "train/mbpp/full/train-00000-of-00001.parquet"),
        "text", DATA / "eval/mbpp/full/test-*.parquet")
    table = _balanced(table, [], limit)
    return [(f"mbppfull-{i}",
             f"You are an expert Python programmer. {row.text}\n"
             f"Your code should pass these tests:\n" + "\n".join(row.test_list) + "\n")
            for i, row in enumerate(table.itertuples())]


BUILDERS = {"samsum": samsum, "gsm8k": gsm8k, "mmlu": mmlu, "mbpp": mbpp,
            "samsum_lb": samsum_lb, "trec_lb": trec_lb, "wiki2_lb": wiki2_lb,
            "math": math, "mbpp_full": mbpp_full}

# LongBench 계열은 평가 경로가 chat template을 쓰지 않는다.
RAW_TEXT = {"samsum", "samsum_lb", "trec_lb", "wiki2_lb"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=sorted(BUILDERS), required=True)
    p.add_argument("--limit", type=int, default=300)
    p.add_argument("--model", default=str(REPO_ROOT / ".." / "model" / "LLaDA-8B-Instruct"))
    p.add_argument("--chat-template", type=int, default=-1,
                   help="-1이면 데이터셋 기본값(LongBench 계열은 끔)")
    p.add_argument("--out-root", default=str(REPO_ROOT / "artifacts" / "prompt_shards"))
    args = p.parse_args()

    chat = (args.dataset not in RAW_TEXT) if args.chat_template < 0 else bool(args.chat_template)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    out = Path(args.out_root) / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    added = 0
    for sid, text in BUILDERS[args.dataset](args.limit):
        if (out / f"{sid}.pt").exists():      # raising --limit only adds new samples
            continue
        added += 1
        if chat:
            text = tok.apply_chat_template([{"role": "user", "content": text}],
                                           add_generation_prompt=True, tokenize=False)
        ids = tok(text, return_tensors="pt", add_special_tokens=not chat,
                  truncation=True, max_length=2048).input_ids[0]
        torch.save({"sample_id": sid, "dataset": args.dataset,
                    "prompt_input_ids": ids.to(torch.long)}, out / f"{sid}.pt")
    n = len(list(out.glob("*.pt")))
    print(f"{args.dataset}: {n} shards total, {added} new "
          f"(chat_template={chat}) -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
