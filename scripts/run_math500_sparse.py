"""MATH-500 under Sparse-dLLM cache eviction.

Unlike the LongBench tasks this one inverts the cache: the prompt is ~40 tokens
and the answer is hundreds, so almost every eviction candidate is the model's own
partially written reasoning rather than input text. Keeping 10% of the cache here
means throwing away most of what the model has just derived.

Runs several cache variants in one process so the model is loaded once.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, '/workspace/dllm/opencompass')

INSTRUCTION = ("Please reason step by step, and put your final answer within "
               "\\boxed{}.")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='/workspace/dllm/model/LLaDA-8B-Instruct')
    p.add_argument('--data', default='/workspace/dllm/data/eval/math500/test_balanced100.jsonl')
    p.add_argument('--n-samples', type=int, default=100)
    p.add_argument('--gen-length', type=int, default=256)
    p.add_argument('--block-length', type=int, default=32)
    p.add_argument('--steps', type=int, default=256)
    p.add_argument('--out', default='/workspace/dllm/Sparse-dLLM/results/math500')
    p.add_argument('--variants', nargs='+',
                   default=['keep1.0', 'keep0.1', 'keep0.1-oracle', 'keep0.1-reselect8'])
    return p.parse_args()


def last_boxed(text: str) -> str | None:
    idx = text.rfind('\\boxed')
    if idx < 0:
        return None
    i = text.find('{', idx)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == '{':
            depth += 1
        elif text[j] == '}':
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
    return None


def normalize(ans: str) -> str:
    if ans is None:
        return ''
    s = ans.strip()
    for a, b in (('\\left', ''), ('\\right', ''), ('\\!', ''), ('\\,', ''),
                 ('\\;', ''), ('\\ ', ' '), ('dfrac', 'frac'), ('tfrac', 'frac'),
                 ('^{\\circ}', ''), ('^\\circ', ''), ('\\$', ''), ('$', ''),
                 ('\\%', ''), ('%', ''), (' ', '')):
        s = s.replace(a, b)
    s = re.sub(r'\\text\{([^}]*)\}', r'\1', s)
    s = re.sub(r'\\mbox\{([^}]*)\}', r'\1', s)
    s = s.rstrip('.').strip()
    if s.startswith('{') and s.endswith('}'):
        s = s[1:-1]
    return s


def equivalent(pred: str, gold: str) -> bool:
    p, g = normalize(pred), normalize(gold)
    if not p:
        return False
    if p == g:
        return True
    try:
        from sympy.parsing.latex import parse_latex
        return bool(abs(float(parse_latex(p).evalf() - parse_latex(g).evalf())) < 1e-6)
    except Exception:
        return False


def build_variant(name: str) -> dict:
    """Map a variant name to the knobs generate() takes."""
    cfg = {'keep_ratio': 1.0, 'oracle_eviction': False, 'oracle_pool': True,
           'oracle_per_step': False, 'oracle_reselect_every': 1, 'reselect_every': 0,
           'reselect_offload_v': False, 'reselect_k_bits': 0}
    ratio = re.match(r'keep([0-9.]+)', name)
    if ratio:
        cfg['keep_ratio'] = float(ratio.group(1))
    if 'oracle' in name:
        cfg['oracle_eviction'] = True
    m = re.search(r'reselect(\d+)', name)
    if m:
        cfg['reselect_every'] = int(m.group(1))
    if 'offloadv' in name.lower():
        cfg['reselect_offload_v'] = True
    bits = re.search(r'int(\d+)k', name.lower())
    if bits:
        cfg['reselect_k_bits'] = int(bits.group(1))
        cfg['reselect_offload_v'] = True
    return cfg


def main() -> int:
    args = parse_args()
    from transformers import AutoConfig, AutoTokenizer
    from opencompass.models.sparse_dllm.modeling_llada import LLaDAModelLM
    from opencompass.models.sparse_dllm.llada_generate import generate

    rows = [json.loads(l) for l in open(args.data)][: args.n_samples]
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    config.block_len = args.block_length
    config.kernel_size = 3
    config.keep_ratio = 1.0
    model = LLaDAModelLM.from_pretrained(
        args.model, config=config, device_map='auto',
        torch_dtype=torch.bfloat16, trust_remote_code=True).eval()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for variant in args.variants:
        cfg = build_variant(variant)
        model.config.keep_ratio = cfg.pop('keep_ratio')
        correct, records, started = 0, [], time.time()
        torch.cuda.reset_peak_memory_stats()
        for i, row in enumerate(rows):
            text = tok.apply_chat_template(
                [{'role': 'user', 'content': f"{row['problem']}\n\n{INSTRUCTION}"}],
                add_generation_prompt=True, tokenize=False)
            ids = tok(text, return_tensors='pt', add_special_tokens=False).input_ids.to(model.device)
            out = generate(model, ids, steps=args.steps, gen_length=args.gen_length,
                           block_length=args.block_length, temperature=0.0,
                           cfg_scale=0.0, remasking='low_confidence', **cfg)
            answer = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
            hit = equivalent(last_boxed(answer) or '', row['answer'])
            correct += hit
            records.append({'problem': row['problem'], 'gold': row['answer'],
                            'prediction': answer, 'boxed': last_boxed(answer),
                            'correct': bool(hit)})
            if (i + 1) % 10 == 0:
                print(f'[{variant}] {i + 1}/{len(rows)} acc={correct / (i + 1):.3f} '
                      f'{(time.time() - started) / (i + 1):.1f}s/q', flush=True)
        summary = {'variant': variant, 'n': len(rows), 'accuracy': correct / len(rows),
                   'seconds_per_sample': (time.time() - started) / len(rows),
                   'peak_gpu_gb': torch.cuda.max_memory_allocated() / 1e9,
                   'gen_length': args.gen_length, 'block_length': args.block_length,
                   'steps': args.steps}
        json.dump({'summary': summary, 'records': records},
                  open(out_dir / f'{variant}.json', 'w'), indent=1)
        print(f'== {variant}: acc {summary["accuracy"]:.3f} | '
              f'{summary["seconds_per_sample"]:.1f}s/q | '
              f'peak {summary["peak_gpu_gb"]:.2f} GB', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
