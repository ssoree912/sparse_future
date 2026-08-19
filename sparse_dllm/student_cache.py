"""Load and run dLLM_f prompt-utility checkpoints for cache selection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn


@dataclass(frozen=True, slots=True)
class StudentConfig:
    layer_count: int = 32
    hidden_dim: int = 4096
    proj_dim: int = 256
    mlp_dim: int = 512
    heads: tuple[str, ...] = ("score",)


def _normalize_heads(heads: tuple[str, ...] | list[str] | str) -> tuple[str, ...]:
    if isinstance(heads, str):
        normalized = (heads,)
    else:
        normalized = tuple(heads)
    if not normalized or len(set(normalized)) != len(normalized):
        raise RuntimeError(f"invalid student heads: {normalized}")
    return normalized


def _build_score_head(config: StudentConfig) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(config.proj_dim * 3, config.mlp_dim),
        nn.GELU(),
        nn.Linear(config.mlp_dim, 1),
    )


class PromptUtilityStudentLayer(nn.Module):
    def __init__(self, config: StudentConfig) -> None:
        super().__init__()
        self.heads = _normalize_heads(config.heads)
        self.token_proj = nn.Linear(config.hidden_dim, config.proj_dim)
        self.question_proj = nn.Linear(config.hidden_dim, config.proj_dim)
        if self.heads == ("score",):
            self.score_head = _build_score_head(config)
        else:
            self.score_heads = nn.ModuleDict(
                {head: _build_score_head(config) for head in self.heads}
            )

    def _head(self, head: str | None) -> nn.Module:
        if hasattr(self, "score_heads"):
            selected = head or self.heads[0]
            if selected not in self.score_heads:
                raise RuntimeError(
                    f"student head {selected!r} not found in {self.heads}"
                )
            return self.score_heads[selected]
        return self.score_head

    def forward(
        self,
        hidden_states: torch.Tensor,
        candidate_indices: torch.Tensor,
        question_indices: torch.Tensor,
        head: str | None = None,
    ) -> torch.Tensor:
        token_hidden = hidden_states.index_select(dim=1, index=candidate_indices)
        question_hidden = hidden_states.index_select(dim=1, index=question_indices)
        question_pool = question_hidden.mean(dim=1)
        token_proj = self.token_proj(token_hidden)
        question_proj = self.question_proj(question_pool).unsqueeze(1)
        question_proj = question_proj.expand(-1, token_proj.shape[1], -1)
        fused = torch.cat(
            [token_proj, question_proj, token_proj * question_proj], dim=-1
        )
        return self._head(head)(fused).squeeze(-1)


class PromptUtilityStudent(nn.Module):
    def __init__(self, config: StudentConfig) -> None:
        super().__init__()
        heads = _normalize_heads(config.heads)
        self.config = StudentConfig(
            layer_count=config.layer_count,
            hidden_dim=config.hidden_dim,
            proj_dim=config.proj_dim,
            mlp_dim=config.mlp_dim,
            heads=heads,
        )
        self.layer_indices = tuple(range(self.config.layer_count))
        self.layers = nn.ModuleDict(
            {
                str(layer_id): PromptUtilityStudentLayer(self.config)
                for layer_id in self.layer_indices
            }
        )

    def forward_layer(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        candidate_indices: torch.Tensor,
        question_indices: torch.Tensor,
        head: str | None = None,
    ) -> torch.Tensor:
        return self.layers[str(layer_id)](
            hidden_states, candidate_indices, question_indices, head=head
        )


def load_prompt_utility_student(
    checkpoint_dir: str | Path, device: torch.device
) -> PromptUtilityStudent:
    checkpoint_dir = Path(checkpoint_dir).resolve()
    config_path = checkpoint_dir / "config.json"
    state_path = checkpoint_dir / "pytorch_model.bin"
    if not config_path.is_file() or not state_path.is_file():
        raise FileNotFoundError(
            f"invalid student checkpoint directory: {checkpoint_dir}"
        )
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    raw_config["heads"] = tuple(raw_config.get("heads", ("score",)))
    student = PromptUtilityStudent(StudentConfig(**raw_config))
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    student.load_state_dict(state, strict=True)
    student.to(device)
    student.eval()
    return student
