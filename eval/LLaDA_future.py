"""lm-eval model for future_dllm: LLaDA with future-attention cache eviction.

Registered as ``LLaDA_future``. It reuses the repo's ``LLaDA`` wrapper for
tokenisation and lm-eval plumbing, and replaces generation with future_dllm's
block-wise ``generate()`` so the cache knobs are reachable from ``--model_args``.

Install once (single source of truth stays in this repo):

    ln -s <repo>/eval/LLaDA_future.py <harness>/eval_model/LLaDA_future.py
    echo "from .LLaDA_future import LLaDAFuture" >> <harness>/eval_model/__init__.py
"""

from __future__ import annotations

import os
import sys
from typing import List

import torch
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model

from .LLaDA import LLaDA

# The package lives in this repo; keep one copy so a stale duplicate cannot be
# imported by accident.
FUTURE_DLLM_ROOT = os.environ.get(
    "FUTURE_DLLM_ROOT",
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
)


@register_model("LLaDA_future")
class LLaDAFuture(LLaDA):
    def __init__(
        self,
        pretrained: str,
        keep_ratio: float = 1.0,
        block_len: int = 32,
        student_path: str = "",
        question_window: int = 128,
        dtype: str = "bfloat16",
        **kwargs,
    ):
        if FUTURE_DLLM_ROOT not in sys.path:
            sys.path.insert(0, FUTURE_DLLM_ROOT)
        from transformers import AutoConfig
        from future_dllm import LLaDAModelLM, generate, load_prompt_utility_student

        self._generate = generate
        self._question_window = int(question_window)

        config = AutoConfig.from_pretrained(str(pretrained), trust_remote_code=True)
        config.block_len = int(block_len)
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

        self._scorer = None
        if student_path:
            self._scorer = load_prompt_utility_student(
                student_path, next(model.parameters()).device)
        elif float(keep_ratio) < 1.0:
            raise ValueError(
                "eviction needs a trained scorer: pass student_path=<checkpoint>, "
                "or keep_ratio=1.0 to run without eviction")
        print(f"[LLaDA_future] keep_ratio={keep_ratio} block_len={block_len} "
              f"scorer={student_path or 'none (no eviction)'}", flush=True)

    @torch.no_grad()
    def generate_until(self, requests: List[Instance]) -> List[str]:
        from datasets import Dataset
        from tqdm import tqdm
        from dllm_cache.budget.prune_cache import resolve_generation_kwargs

        results, bar = [], tqdm(total=len(requests), disable=(self.rank != 0),
                                desc="future_dllm generate_until")
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
            out = self._generate(
                self.model, context_enc.to(self.device),
                steps=int(gen_kwargs.get("steps")),
                gen_length=gen_length,
                block_length=int(gen_kwargs.get("block_length")),
                temperature=float(gen_kwargs.get("temperature", 0.0)),
                cfg_scale=float(gen_kwargs.get("cfg_scale", 0.0)),
                remasking=gen_kwargs.get("remasking") or "low_confidence",
                cache_scorer=self._scorer,
                question_window=self._question_window,
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
