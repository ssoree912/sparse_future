"""Measure what each cache strategy actually costs in GPU memory.

Re-selection only helps if the entries it re-ranks are still around, so it trades
the memory eviction was supposed to save. This measures that trade directly:
peak allocation, bytes held by the cache itself, latency, and whether the
generated tokens change.

Run it on the two LongBench tasks already evaluated (SAMSum at 2048, passage
retrieval at 4096) so the numbers line up with the accuracy results.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import torch

sys.path.insert(0, '/workspace/dllm/opencompass')

VARIANTS = {
    'keep1.0':            dict(keep_ratio=1.0),
    'baseline-keep0.1':   dict(keep_ratio=0.1),
    'reselect8':          dict(keep_ratio=0.1, reselect_every=8),
    'reselect8-offloadV': dict(keep_ratio=0.1, reselect_every=8, reselect_offload_v=True),
    'reselect8-int8K':    dict(keep_ratio=0.1, reselect_every=8, reselect_offload_v=True,
                               reselect_k_bits=8),
    'reselect8-int4K':    dict(keep_ratio=0.1, reselect_every=8, reselect_offload_v=True,
                               reselect_k_bits=4),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='/workspace/dllm/model/LLaDA-8B-Instruct')
    p.add_argument('--task', choices=['samsum', 'passage_retrieval'], default='samsum')
    p.add_argument('--n-samples', type=int, default=5)
    p.add_argument('--variants', nargs='+', default=list(VARIANTS))
    p.add_argument('--out', default='/workspace/dllm/Sparse-dLLM/results/memory')
    return p.parse_args()


def cache_bytes(cache) -> int:
    """Bytes the cache holds on the GPU, ignoring anything paged out to host."""
    total = 0
    for store in (cache.cache, cache.full_cache):
        for entry in store.values():
            for tensor in entry.values():
                if hasattr(tensor, 'nbytes') and not torch.is_tensor(tensor):
                    total += tensor.nbytes            # quantized keys
                elif torch.is_tensor(tensor) and tensor.is_cuda:
                    total += tensor.numel() * tensor.element_size()
    return total


def load_prompts(task: str, n: int, tokenizer):
    import pandas as pd
    if task == 'samsum':
        frame = pd.read_parquet(
            '/workspace/dllm/opencompass/data/Longbench/samsum/test-00000-of-00001.parquet')
        texts = [f"{row.context}\n\n{row.question}\nSummary:" for row in frame.itertuples()][:n]
        return texts, 2048, 128
    frame = pd.read_parquet(
        '/workspace/dllm/opencompass/data/Longbench/passage_retrieval_short/test-00000-of-00001.parquet')
    texts = [
        'Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine '
        f'which paragraph the abstract is from.\n\n{row.context}\n\nThe following is an '
        f'abstract.\n\n{row.question}\n\nPlease enter the number of the paragraph that the '
        'abstract is from. The answer format must be like "Paragraph 1", "Paragraph 2", '
        'etc.\n\nThe answer is: ' for row in frame.itertuples()][:n]
    return texts, 4096, 32


def main() -> int:
    args = parse_args()
    from transformers import AutoConfig, AutoTokenizer
    from opencompass.models.sparse_dllm.modeling_llada import LLaDAModelLM
    from opencompass.models.sparse_dllm import modeling_llada, llada_generate

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    config.block_len, config.kernel_size, config.keep_ratio = 32, 3, 1.0
    model = LLaDAModelLM.from_pretrained(
        args.model, config=config, device_map='auto',
        torch_dtype=torch.bfloat16, trust_remote_code=True).eval()

    prompts, max_len, gen_length = load_prompts(args.task, args.n_samples, tokenizer)

    # Watch every cache the run creates so we can report the largest.
    peak_cache = {'bytes': 0}
    original_filter = modeling_llada.CustomCache.filter_cache

    def watched_filter(self, *a, **kw):
        out = original_filter(self, *a, **kw)
        peak_cache['bytes'] = max(peak_cache['bytes'], cache_bytes(self))
        return out

    modeling_llada.CustomCache.filter_cache = watched_filter

    baseline_tokens, results = None, []
    for name in args.variants:
        cfg = dict(VARIANTS[name])
        model.config.keep_ratio = cfg.pop('keep_ratio')
        peak_cache['bytes'] = 0
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        outputs, started = [], time.time()
        for text in prompts:
            ids = tokenizer(text, return_tensors='pt', truncation=True,
                            max_length=max_len).input_ids.to(model.device)
            out = llada_generate.generate(
                model, ids, steps=128, gen_length=gen_length, block_length=32,
                temperature=0.0, cfg_scale=0.0, remasking='low_confidence', **cfg)
            outputs.append(out[0, ids.shape[1]:].tolist())
        elapsed = (time.time() - started) / len(prompts)
        if baseline_tokens is None:
            baseline_tokens = outputs   # first variant listed is the reference
        identical = (sum(a == b for a, b in zip(outputs, baseline_tokens))
                     if baseline_tokens is not None else None)
        row = {
            'variant': name,
            'cache_mb': peak_cache['bytes'] / 1e6,
            'peak_total_gb': torch.cuda.max_memory_allocated() / 1e9,
            'over_weights_mb': (torch.cuda.max_memory_allocated() - base) / 1e6,
            'seconds_per_sample': elapsed,
            'same_output_as_reference': identical,
        }
        results.append(row)
        print(f"{name:22s} cache {row['cache_mb']:7.1f} MB | peak {row['peak_total_gb']:5.2f} GB "
              f"| +{row['over_weights_mb']:7.1f} MB over weights | {elapsed:5.2f}s/sample "
              f"| same-as-reference {identical}/{len(prompts)}", flush=True)

    modeling_llada.CustomCache.filter_cache = original_filter
    import pathlib
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out_dir / f'{args.task}.json', 'w'), indent=1)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
