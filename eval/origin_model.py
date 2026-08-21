"""lm-eval model for stock LLaDA, registered as ``LLaDA_origin``.

No cache at all: LLaDA's own generation loop runs the whole sequence again at
every denoising step. This is the reference the cached variants are measured
against, so both the model and the loop are the authors' - the model through
``trust_remote_code``, the loop fetched by ``scripts/fetch_origin.sh``.

    --model LLaDA_origin --model_args "pretrained=<model>"
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from lm_eval.api.registry import register_model

from .lm_eval_model import LLaDAFuture

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "eval") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "eval"))


@register_model("LLaDA_origin")
class LLaDAOrigin(LLaDAFuture):
    def __init__(self, pretrained: str = str(REPO_ROOT.parent / "model" / "LLaDA-8B-Instruct"),
                 block_len: int = 32, max_prompt_len: int = 2048,
                 dtype: str = "bfloat16", **kwargs):
        from transformers import AutoModel
        from lm_eval.models.huggingface import HFLM
        try:
            from origin.llada_generate import generate
        except ImportError as exc:
            raise RuntimeError(
                "LLaDA's generate.py is not here: run scripts/fetch_origin.sh") from exc

        self._generate = generate
        self._block_len = int(block_len)
        self._max_prompt_len = int(max_prompt_len)
        self._scorer = None

        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                       "float32": torch.float32}.get(str(dtype), torch.bfloat16)
        model = AutoModel.from_pretrained(
            str(pretrained), device_map="auto", torch_dtype=torch_dtype,
            trust_remote_code=True).eval()

        kwargs.setdefault("tokenizer", str(pretrained))
        kwargs.setdefault("batch_size", 1)
        kwargs.setdefault("trust_remote_code", True)
        HFLM.__init__(self, pretrained=model, **kwargs)

        device = next(model.parameters()).device
        if device.type != "cuda":
            raise RuntimeError(f"model landed on {device}, not CUDA - rerun")
        print(f"[LLaDA_origin] stock LLaDA, no cache, block_len={block_len}", flush=True)

    def _call_generate(self, context_enc, gen_kwargs, gen_length):
        return self._generate(
            self.model, context_enc.to(self.device),
            steps=int(gen_kwargs["steps"]), gen_length=gen_length,
            block_length=self._block_len,
            temperature=float(gen_kwargs.get("temperature", 0.0)),
            cfg_scale=float(gen_kwargs.get("cfg_scale", 0.0)),
            remasking=gen_kwargs.get("remasking") or "low_confidence")
