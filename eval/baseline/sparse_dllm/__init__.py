"""Sparse-dLLM, verbatim from the official repository.

Extracted at commit 3fd8986 of https://github.com/OpenMOSS/Sparse-dLLM, from
opencompass/models/sparse_dllm/. Nothing here is edited: it is the baseline this
project compares against, so it runs the authors' own eviction criterion.
"""
from .modeling_llada import LLaDAModelLM, CustomCache
from .llada_generate import generate

__all__ = ["LLaDAModelLM", "CustomCache", "generate"]
