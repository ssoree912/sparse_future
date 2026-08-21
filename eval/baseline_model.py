"""lm-eval model for the Sparse-dLLM baseline, registered as ``LLaDA_sparse``.

The eviction code is the authors' own, fetched by ``scripts/fetch_baseline.sh``
into ``eval/baseline/`` and not kept in this repository; only the harness around
it is ours, so the baseline row and our row come out of the same tasks, data,
prompts and scoring.

    --model LLaDA_sparse --model_args "pretrained=<model>,keep_ratio=0.1"
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import torch
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model

from .lm_eval_model import LLaDAFuture, _generation_kwargs

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "eval" / "baseline") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "eval" / "baseline"))


@register_model("LLaDA_sparse")
class LLaDASparseBaseline(LLaDAFuture):
    def __init__(self, pretrained: str = str(REPO_ROOT.parent / "model" / "LLaDA-8B-Instruct"),
                 keep_ratio: float = 1.0, kernel_size: int = 3, block_len: int = 32,
                 max_prompt_len: int = 2048, dtype: str = "bfloat16", **kwargs):
        from transformers import AutoConfig
        from lm_eval.models.huggingface import HFLM
        try:
            from sparse_dllm import LLaDAModelLM, generate
        except ImportError as exc:
            raise RuntimeError(
                "Sparse-dLLM is not here: run scripts/fetch_baseline.sh") from exc

        self._generate = generate
        self._block_len = int(block_len)
        self._max_prompt_len = int(max_prompt_len)
        self._scorer = None

        config = AutoConfig.from_pretrained(str(pretrained), trust_remote_code=True)
        config.block_len = int(block_len)
        config.kernel_size = int(kernel_size)
        config.keep_ratio = float(keep_ratio)
        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                       "float32": torch.float32}.get(str(dtype), torch.bfloat16)
        model = LLaDAModelLM.from_pretrained(
            str(pretrained), config=config, device_map="auto",
            torch_dtype=torch_dtype, trust_remote_code=True).eval()

        kwargs.setdefault("tokenizer", str(pretrained))
        kwargs.setdefault("batch_size", 1)
        kwargs.setdefault("trust_remote_code", True)
        HFLM.__init__(self, pretrained=model, **kwargs)

        device = next(model.parameters()).device
        if device.type != "cuda":
            raise RuntimeError(f"model landed on {device}, not CUDA - rerun")
        print(f"[LLaDA_sparse] baseline, keep_ratio={keep_ratio} "
              f"block_len={block_len} kernel_size={kernel_size}", flush=True)

    @torch.no_grad()
    def generate_until(self, requests: List[Instance], disable_tqdm: bool = False) -> List[str]:
        # Same loop and resume store as ours; the upstream generate() takes no
        # scorer arguments, so they are simply absent.
        return super().generate_until(requests, disable_tqdm)

    def _call_generate(self, context_enc, gen_kwargs, gen_length):
        return self._generate(
            self.model, context_enc.to(self.device),
            steps=int(gen_kwargs["steps"]), gen_length=gen_length,
            block_length=self._block_len,
            temperature=float(gen_kwargs.get("temperature", 0.0)),
            cfg_scale=float(gen_kwargs.get("cfg_scale", 0.0)),
            remasking=gen_kwargs.get("remasking") or "low_confidence")
