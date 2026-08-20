"""lm-eval wrapper for Sparse-dLLM cache eviction on LLaDA.

The existing `LLaDA` wrapper drives dLLM-Cache's feature caching, which is a
different mechanism entirely; this one loads the Sparse-dLLM `LLaDAModelLM`
(cache-aware attention) and generates through its `generate()`, so the eviction
knobs — the scoring criterion, the keep ratio, the oracle label definitions —
are reachable from `--model_args` and scored by lm-eval's own task metrics.

    accelerate launch evaluation_script.py -m lm_eval \\
      --model LLaDA_sparse --tasks minerva_math --batch_size 1 \\
      --model_args "pretrained=/path/LLaDA-8B-Instruct,keep_ratio=0.25,scorer=masked_row" \\
      --gen_kwargs "block_length=32,gen_length=256,steps=256,cfg_scale=0.0" \\
      --num_fewshot 0 --apply_chat_template --fewshot_as_multiturn
"""

from __future__ import annotations

import sys
from typing import List

import torch
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model

from .LLaDA import LLaDA

SPARSE_DLLM_ROOT = "/workspace/dllm/dLLM_f"   # standalone copy, no OpenCompass import


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


@register_model("LLaDA_sparse")
class LLaDASparse(LLaDA):
    def __init__(
        self,
        pretrained: str,
        keep_ratio: float = 1.0,
        kernel_size: int = 3,
        block_len: int = 32,
        scorer: str = "sparse_dllm",
        reselect_every: int = 0,
        reselect_offload_v: bool = False,
        reselect_k_bits: int = 0,
        oracle_eviction: bool = False,
        oracle_pool: bool = True,
        oracle_per_step: bool = False,
        oracle_reselect_every: int = 1,
        oracle_future_window: bool = False,
        oracle_rows: str = "masked",
        sparse_dllm_root: str = SPARSE_DLLM_ROOT,
        dtype: str = "bfloat16",
        **kwargs,
    ):
        if sparse_dllm_root not in sys.path:
            sys.path.insert(0, sparse_dllm_root)
        from transformers import AutoConfig
        from sparse_dllm import LLaDAModelLM
        from sparse_dllm import generate as sparse_generate

        self._sparse_generate = sparse_generate
        self.sparse_cfg = dict(
            scorer=str(scorer),
            reselect_every=int(reselect_every),
            reselect_offload_v=_as_bool(reselect_offload_v),
            reselect_k_bits=int(reselect_k_bits),
            oracle_eviction=_as_bool(oracle_eviction),
            oracle_pool=_as_bool(oracle_pool),
            oracle_per_step=_as_bool(oracle_per_step),
            oracle_reselect_every=int(oracle_reselect_every),
            oracle_future_window=_as_bool(oracle_future_window),
            oracle_rows=str(oracle_rows),
        )

        config = AutoConfig.from_pretrained(str(pretrained), trust_remote_code=True)
        config.block_len = int(block_len)
        config.kernel_size = int(kernel_size)
        config.keep_ratio = float(keep_ratio)
        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                       "float32": torch.float32}.get(str(dtype), torch.bfloat16)
        model = LLaDAModelLM.from_pretrained(
            str(pretrained), config=config, device_map="auto",
            torch_dtype=torch_dtype, trust_remote_code=True).eval()

        # The parent skips its own loading when handed a live model, but it still
        # needs the path to find the tokenizer.
        kwargs.setdefault("tokenizer", str(pretrained))
        super().__init__(pretrained=model, **kwargs)
        print(f"[LLaDA_sparse] keep_ratio={keep_ratio} block_len={block_len} "
              f"{self.sparse_cfg}", flush=True)

    @torch.no_grad()
    def generate_until(self, requests: List[Instance]) -> List[str]:
        from datasets import Dataset
        from tqdm import tqdm
        from dllm_cache.budget.prune_cache import resolve_generation_kwargs

        results, bar = [], tqdm(total=len(requests), disable=(self.rank != 0),
                                desc="Sparse-dLLM generate_until")
        gen_kwargs = resolve_generation_kwargs(requests[0].args[1])
        gen_length = int(gen_kwargs["gen_length"])
        left_truncate_len = max(1, self.max_length - gen_length)
        dataset = Dataset.from_list([{"text": r.args[0]} for r in requests])

        for batch in dataset.iter(1):          # eviction state is per sequence
            contexts = batch["text"]
            if self.add_bos_token:
                contexts = [self.tokenizer.bos_token + p for p in contexts]
            context_enc, _ = self.tok_batch_encode(
                contexts, truncation=self.truncation,
                left_truncate_len=left_truncate_len,
                truncation_strategy=self.truncation_strategy)
            out = self._sparse_generate(
                self.model, context_enc.to(self.device),
                steps=int(gen_kwargs.get("steps")),
                gen_length=gen_length,
                block_length=int(gen_kwargs.get("block_length")),
                temperature=float(gen_kwargs.get("temperature", 0.0)),
                cfg_scale=float(gen_kwargs.get("cfg_scale", 0.0)),
                remasking=gen_kwargs.get("remasking") or "low_confidence",
                **self.sparse_cfg,
            )
            text = self.tokenizer.batch_decode(
                out[:, context_enc.shape[1]:], skip_special_tokens=True)
            for s in text:
                if not self.escape_until:
                    for term in gen_kwargs.get("until") or []:
                        if term:
                            s = s.split(term)[0]
                results.append(s)
                bar.update(1)
        bar.close()
        return results
