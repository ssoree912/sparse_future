"""Teacher labels from the finished answer: final x row-max.

For each block the collector lets the block fill in with the full cache, then runs
one extra forward on the completed block and reads what those answer tokens look
at. The label is the per-candidate maximum over the 32 answer rows — a candidate
survives if *any* finished token needed it strongly, which is what separates this
from the sum-style labels that average such tokens away.

Measured as an oracle it scores 36.63 on SAMSum keep 0.1, above both the sum
variant (34.72) and the training-free current-attention criterion (36.19).

Stored per (sample, block): the label [n_layers, n_candidates], the exact model
input at the block's step-1 (so the student's features can be replayed without
keeping hidden states), and the candidate index set.
"""

from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

import torch

MASK_ID = 126336


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="/workspace/dllm/model/LLaDA-8B-Instruct")
    p.add_argument("--source-glob",
                   default="/tmp/claude-0/-workspace-dllm/4a27d45a-7287-4963-bd14-cbe2a09f4e0c/"
                           "scratchpad/src_shards/*/samsum/*.pt")
    p.add_argument("--output-root", default="/workspace/dllm/dLLM_f/results/budget/"
                                            "teacher_final_rowmax_samsum300")
    p.add_argument("--n-samples", type=int, default=300)
    p.add_argument("--gen-length", type=int, default=128)
    p.add_argument("--block-length", type=int, default=32)
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--max-prompt-len", type=int, default=2048)
    p.add_argument("--opencompass-root", default="/workspace/dllm/opencompass")
    return p.parse_args()


@torch.no_grad()
def collect(model, prompt_ids, args):
    from opencompass.models.sparse_dllm.modeling_llada import CustomCache
    from opencompass.models.sparse_dllm.llada_generate import (
        add_gumbel_noise, get_num_transfer_tokens)
    import torch.nn.functional as F

    device = model.device
    prompt_ids = prompt_ids[: args.max_prompt_len].to(device).unsqueeze(0)
    P = prompt_ids.shape[1]
    G, B = args.gen_length, args.block_length
    n_blocks = G // B
    S = args.steps // n_blocks
    L = model.config.n_layers

    x = torch.full((1, P + G), MASK_ID, dtype=torch.long, device=device)
    x[:, :P] = prompt_ids
    records = []

    for block in range(n_blocks):
        cache = CustomCache(n_layers=L, device=device, kernel_size=3, keep_ratio=1.0)
        cache.oracle_mode = "record"          # keep the whole pool, in candidate order
        bs, be = P + block * B, P + (block + 1) * B
        ntt = get_num_transfer_tokens(x[:, bs:be] == MASK_ID, S)

        def step(i):
            state = 2 if i > 1 else i
            inp = x if state != 2 else x[:, bs:be]
            m = (inp == MASK_ID)
            logits = model(inp, bs, state, cache).logits
            x0 = torch.argmax(add_gumbel_noise(logits, 0.0), dim=-1)
            conf = torch.squeeze(torch.gather(F.softmax(logits, -1), -1,
                                              x0.unsqueeze(-1)), -1)
            tgt = x if state != 2 else x[:, bs:be]
            if state != 2:
                conf[:, be:] = -float("inf")
            x0 = torch.where(m, x0, tgt)
            conf = torch.where(m, conf, torch.full_like(conf, -float("inf")))
            keep = torch.topk(conf[0], k=ntt[0, i]).indices
            tgt[0, keep] = x0[0, keep]

        step(0)
        step(1)
        x_at_block_start = x.clone()
        for i in range(2, S):
            step(i)

        # 완성된 블록에서 한 번 더 — 32행 전부 실제 토큰
        cache.capture_rows = True
        cache.set_row_mask(None)
        step(S - 1)
        label = torch.stack([cache.pending_rows[l].max(dim=0).values for l in range(L)])
        cache.pending_rows.clear()
        cache.capture_rows = False

        candidates = torch.cat([torch.arange(bs, device=device),
                                torch.arange(be, x.shape[1], device=device)])
        records.append({
            "block_index": block,
            "block_start": int(bs),
            "block_length": B,
            "prompt_length": int(P),
            "gen_length": G,
            "steps_per_block": S,
            "x_at_block_start": x_at_block_start[0].cpu(),
            "candidate_indices": candidates.cpu(),
            "label_final_rowmax": label.to(torch.float16).cpu(),
        })
    return records


def main():
    args = parse_args()
    sys.path.insert(0, args.opencompass_root)
    from transformers import AutoConfig
    from opencompass.models.sparse_dllm.modeling_llada import LLaDAModelLM

    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    cfg.block_len, cfg.kernel_size, cfg.keep_ratio = args.block_length, 3, 1.0
    model = LLaDAModelLM.from_pretrained(args.model, config=cfg, device_map="auto",
                                         torch_dtype=torch.bfloat16,
                                         trust_remote_code=True).eval()

    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    shards = sorted(glob.glob(args.source_glob))[: args.n_samples]
    started = time.time()
    for i, path in enumerate(shards):
        target = out / Path(path).name
        if target.exists():
            continue
        src = torch.load(path, map_location="cpu", weights_only=False)
        records = collect(model, src["prompt_input_ids"].to(torch.long), args)
        torch.save({"sample_id": src.get("sample_id"),
                    "prompt_input_ids": src["prompt_input_ids"].to(torch.long),
                    "teacher_kind": "final_rowmax",
                    "blocks": records}, target)
        if (i + 1) % 10 == 0:
            print(f"{i + 1}/{len(shards)}  {(time.time() - started) / (i + 1):.1f}s/sample",
                  flush=True)
    print(f"done: {len(list(out.glob('*.pt')))} shards", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
