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

import torch
from transformers import AutoTokenizer

MATH_INSTRUCTION = ("Please reason step by step, and put your final answer within "
                    "\\boxed{}.")


def samsum(limit):
    """LongBench-style single dialogue, byte-for-byte the format the first 300
    teacher shards were built with (no chat template, no few-shot block)."""
    rows = []
    with open("/workspace/dllm/data/train/samsum/a100_source_train.jsonl") as fh:
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
    path = glob.glob("/workspace/dllm/.hf_cache/datasets/openai___gsm8k/main/*/*/gsm8k-train.arrow")
    rows = Dataset.from_file(path[0]).select(range(limit))
    return [(f"gsm8k-{i}", f"{r}\n\n{MATH_INSTRUCTION}")
            for i, r in enumerate(rows["question"])]


def mmlu(limit):
    import pandas as pd
    frames = [pd.read_parquet(f) for split in ("validation", "dev")
              for f in glob.glob(f"/workspace/dllm/data/eval/mmlu/all/{split}-*.parquet")]
    table = pd.concat(frames).head(limit)
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
              for f in glob.glob(f"/workspace/dllm/data/eval/mbpp/full/{split}-*.parquet")]
    table = pd.concat(frames).head(limit)
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
    return _longbench("/workspace/dllm/data/train/samsum/train.jsonl", limit)


def trec_lb(limit):
    return _longbench("/workspace/dllm/data/train/trec/train.jsonl", limit)


def wiki2_lb(limit):
    return _longbench("/workspace/dllm/data/train/2wikimqa/train.jsonl", limit)


def math(limit):
    """MATH500은 test에서 뽑은 것이라 train split은 겹치지 않는다."""
    import pandas as pd
    frames = [pd.read_parquet(f) for f
              in sorted(glob.glob("/workspace/dllm/data/train/hendrycks_math/*/train-*.parquet"))]
    table = pd.concat(frames).sample(frac=1.0, random_state=0).head(limit)
    return [(f"math-{i}", f"{row.problem}\n\n{MATH_INSTRUCTION}")
            for i, row in enumerate(table.itertuples())]


def mbpp_full(limit):
    import pandas as pd
    table = pd.read_parquet(
        "/workspace/dllm/data/train/mbpp/full/train-00000-of-00001.parquet").head(limit)
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
    p.add_argument("--model", default="/workspace/dllm/model/LLaDA-8B-Instruct")
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
