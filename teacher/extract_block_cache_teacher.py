"""Block-wise cache-eviction teacher, collected under the deployment protocol.

For every (sample, block) the collector reproduces exactly what Sparse-dLLM does
at inference time: the cache is snapshotted at ``cache_state==1`` and frozen, and
the block's 32 queries then run steps 2..S-1 against it.  The attention those
queries pay to each cached candidate is what eviction destroys when it drops a
column, so that is what we record.

Unlike the legacy teacher this covers **prompt and suffix columns alike**, is
aggregated per block rather than over the whole answer, and stores the exact
model input needed to replay the student's inference-time features.

Targets stored per (layer, candidate):
  sum_masked  sum over steps of attention from still-masked block rows
  sum_all     same, over all 32 block rows
  max_any     max over steps and rows
  heuristic   the deployment's own score (mean-head avg-query dot product)
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import torch


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/workspace/dllm/model/LLaDA-8B-Instruct"))
    parser.add_argument("--source-root", type=Path, required=True,
                        help="root holding <dataset>/*.pt shards with prompt_input_ids")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=["samsum"])
    parser.add_argument("--n-samples", type=int, default=0, help="per dataset; 0 = all")
    parser.add_argument("--opencompass-root", type=Path,
                        default=Path("/workspace/dllm/opencompass"))
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--max-prompt-len", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def load_model(args: argparse.Namespace):
    sys.path.insert(0, str(args.opencompass_root))
    from transformers import AutoConfig
    from opencompass.models.sparse_dllm.modeling_llada import LLaDAModelLM

    config = AutoConfig.from_pretrained(str(args.model), trust_remote_code=True)
    config.block_len = args.block_length
    config.kernel_size = 3
    config.keep_ratio = 1.0  # teacher observes the full cache
    model = LLaDAModelLM.from_pretrained(
        str(args.model), config=config, device_map="auto",
        torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).eval()
    return model


@torch.no_grad()
def collect_sample(model, prompt_ids: torch.Tensor, args: argparse.Namespace) -> list[dict]:
    """Run one generation and emit one teacher record per block."""
    from opencompass.models.sparse_dllm.modeling_llada import CustomCache
    from opencompass.models.sparse_dllm.llada_generate import (
        add_gumbel_noise, get_num_transfer_tokens,
    )
    import torch.nn.functional as F

    mask_id = 126336
    device = model.device
    prompt_ids = prompt_ids[: args.max_prompt_len].to(device).unsqueeze(0)
    prompt_len = prompt_ids.shape[1]
    gen_length, block_length = args.gen_length, args.block_length
    num_blocks = gen_length // block_length
    steps = args.steps // num_blocks

    x = torch.full((1, prompt_len + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = prompt_ids.clone()

    records = []
    for num_block in range(num_blocks):
        cache = CustomCache(
            n_layers=model.config.n_layers, device=device,
            kernel_size=3, keep_ratio=1.0, prompt_length=prompt_len,
            generation_length=gen_length,
        )
        cache.oracle_mode = "record"
        cache.teacher_mode = True  # accumulate every target variant

        block_start = prompt_len + num_block * block_length
        block_end = block_start + block_length
        num_transfer_tokens = get_num_transfer_tokens(
            x[:, block_start:block_end] == mask_id, steps
        )

        def run_step(i: int) -> None:
            cache_state = 2 if i > 1 else i
            if cache_state != 2:
                model_input, mask_index = x, (x == mask_id)
            else:
                model_input = x[:, block_start:block_end]
                mask_index = (model_input == mask_id)
            logits = model(model_input, block_start, cache_state, cache).logits
            x0 = torch.argmax(add_gumbel_noise(logits, temperature=0.0), dim=-1)
            confidence = torch.squeeze(
                torch.gather(F.softmax(logits, dim=-1), dim=-1,
                             index=torch.unsqueeze(x0, -1)), -1)
            if cache_state != 2:
                confidence[:, block_end:] = -float("inf")
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, confidence, -float("inf"))
                keep = torch.topk(confidence[0], k=num_transfer_tokens[0, i]).indices
                x[0, keep] = x0[0, keep]
            else:
                block_x = x[:, block_start:block_end]
                x0 = torch.where(mask_index, x0, block_x)
                confidence = torch.where(mask_index, confidence, -float("inf"))
                keep = torch.topk(confidence[0], k=num_transfer_tokens[0, i]).indices
                block_x[0, keep] = x0[0, keep]

        run_step(0)
        run_step(1)
        x_at_block_start = x.clone()
        for i in range(2, steps):
            cache.set_row_mask(x[0, block_start:block_end] == mask_id)
            run_step(i)

        candidate_indices = torch.cat([
            torch.arange(block_start, device=device),
            torch.arange(block_end, x.shape[1], device=device),
        ])
        n_layers = model.config.n_layers
        stacked = {
            name: torch.stack([cache.teacher_accum[name][l] for l in range(n_layers)])
            for name in ("sum_masked", "sum_all", "max_any", "heuristic")
        }
        records.append({
            "block_index": num_block,
            "block_start": int(block_start),
            "block_length": block_length,
            "prompt_length": int(prompt_len),
            "gen_length": gen_length,
            "steps_per_block": steps,
            "x_at_block_start": x_at_block_start[0].to(torch.long).cpu(),
            "candidate_indices": candidate_indices.to(torch.long).cpu(),
            **{f"target_{k}": v.to(torch.float16).cpu() for k, v in stacked.items()},
        })
    return records


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    model = load_model(args)
    args.output_root.mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        shards = sorted((args.source_root / dataset).glob("*.pt"))
        if args.n_samples > 0:
            shards = shards[: args.n_samples]
        if not shards:
            raise RuntimeError(f"no source shards for {dataset}")
        out_dir = args.output_root / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        for index, path in enumerate(shards):
            out_path = out_dir / path.name
            if out_path.exists():
                continue
            source = torch.load(path, map_location="cpu", weights_only=False)
            records = collect_sample(model, source["prompt_input_ids"].to(torch.long), args)
            torch.save({
                "sample_id": source.get("sample_id"),
                "dataset": dataset,
                "prompt_input_ids": source["prompt_input_ids"].to(torch.long),
                "teacher_kind": "block_frozen_cache_future_attention",
                "blocks": records,
            }, out_path)
            print(f"[{dataset}] {index + 1}/{len(shards)} {path.name} "
                  f"blocks={len(records)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
