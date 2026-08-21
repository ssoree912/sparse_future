"""lm-eval model for future_dllm, registered as ``LLaDA_future``.

Self-contained: it subclasses lm-eval's own ``HFLM`` for tokenisation and
plumbing, and replaces generation with future_dllm's block-wise ``generate()``,
so the cache knobs are reachable from ``--model_args``:

    --model_args "pretrained=<model>,keep_ratio=0.1,student_path=<checkpoint>"

``keep_ratio`` below 1.0 needs a trained scorer; 1.0 disables eviction.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import torch
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _generation_kwargs(raw: dict, default_max_gen_toks: int) -> dict:
    """Take the task's own generation settings and read them as diffusion ones.

    Diffusion decoding needs its token budget up front, so the task's
    ``max_gen_toks`` — or lm-eval's default when the task does not set one —
    becomes the block schedule, one denoising step per token.

    ``do_sample: false`` means greedy, which for Gumbel-max sampling is
    temperature 0. Tasks pair it with ``temperature: 1``, meaning "unused";
    taking that literally would sample.
    """
    out = dict(raw)
    gen_length = int(out.get("gen_length", out.get("max_gen_toks", default_max_gen_toks)))
    out["gen_length"] = gen_length
    out.setdefault("steps", gen_length)
    if not out.get("do_sample", False):
        out["temperature"] = 0.0
    return out


@register_model("LLaDA_future")
class LLaDAFuture(HFLM):
    def __init__(
        self,
        pretrained: str = str(REPO_ROOT.parent / "model" / "LLaDA-8B-Instruct"),
        keep_ratio: float = 1.0,
        block_len: int = 32,
        max_prompt_len: int = 2048,
        student_path: str = "",
        question_window: int = 128,
        dtype: str = "bfloat16",
        **kwargs,
    ):
        from transformers import AutoConfig
        from future_dllm import LLaDAModelLM, generate, load_prompt_utility_student

        self._generate = generate
        self._question_window = int(question_window)
        self._block_len = int(block_len)
        self._max_prompt_len = int(max_prompt_len)

        config = AutoConfig.from_pretrained(str(pretrained), trust_remote_code=True)
        config.block_len = int(block_len)
        config.keep_ratio = float(keep_ratio)
        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                       "float32": torch.float32}.get(str(dtype), torch.bfloat16)
        model = LLaDAModelLM.from_pretrained(
            str(pretrained), config=config, device_map="auto",
            torch_dtype=torch_dtype, trust_remote_code=True).eval()

        # HFLM skips its own loading when handed a live model, but still needs
        # the path to find the tokenizer.
        kwargs.setdefault("tokenizer", str(pretrained))
        kwargs.setdefault("batch_size", 1)      # cache state is per sequence
        kwargs.setdefault("trust_remote_code", True)
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
              f"max_prompt_len={max_prompt_len} "
              f"scorer={student_path or 'none (no eviction)'}", flush=True)

    @torch.no_grad()
    def generate_until(self, requests: List[Instance], disable_tqdm: bool = False) -> List[str]:
        from tqdm import tqdm

        results = []
        bar = tqdm(total=len(requests), disable=(disable_tqdm or self.rank != 0),
                   desc="future_dllm generate_until")
        for request in requests:
            context, raw_kwargs = request.args
            gen_kwargs = _generation_kwargs(raw_kwargs, self.max_gen_toks)
            gen_length = int(gen_kwargs["gen_length"])
            if gen_length % self._block_len:      # blocks have to divide the budget
                gen_length += self._block_len - gen_length % self._block_len

            if self.add_bos_token:
                context = self.tokenizer.bos_token + context
            context_enc, _ = self.tok_batch_encode(
                [context], truncation=self.truncation,
                left_truncate_len=self._max_prompt_len)

            out = self._generate(
                self.model, context_enc.to(self.device),
                steps=int(gen_kwargs["steps"]),
                gen_length=gen_length,
                block_length=self._block_len,
                temperature=float(gen_kwargs.get("temperature", 0.0)),
                cfg_scale=float(gen_kwargs.get("cfg_scale", 0.0)),
                remasking=gen_kwargs.get("remasking") or "low_confidence",
                cache_scorer=self._scorer,
                question_window=self._question_window,
            )
            text = self.tokenizer.decode(out[0, context_enc.shape[1]:],
                                         skip_special_tokens=True)
            for term in gen_kwargs.get("until") or []:
                if term:
                    text = text.split(term)[0]
            results.append(text)
            bar.update(1)
        bar.close()
        return results
