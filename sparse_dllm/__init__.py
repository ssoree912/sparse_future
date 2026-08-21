"""Sparse-dLLM cache eviction, standalone copy so lm-eval need not import OpenCompass."""
from .modeling_llada import LLaDAModelLM, CustomCache
from .llada_generate import generate

__all__ = ["LLaDAModelLM", "CustomCache", "generate"]
