from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class AuditReader:
    """Read-only diagnostics. It owns no parameters and never changes Writer output."""

    records: list[dict[str, Any]] = field(default_factory=list)

    def reset(self) -> None:
        self.records.clear()

    def observe(
        self,
        *,
        step_index: int,
        layer_id: int,
        semantic: torch.Tensor,
        candidate: torch.Tensor,
        applied: torch.Tensor,
        token_cap_fraction: float,
        global_cap_coefficient: float,
    ) -> None:
        with torch.no_grad():
            semantic_fp32 = semantic.detach().float()
            candidate_fp32 = candidate.detach().float()
            applied_fp32 = applied.detach().float()
            self.records.append(
                {
                    "step_index": int(step_index),
                    "layer_id": int(layer_id),
                    "semantic_norm": float(torch.linalg.vector_norm(semantic_fp32).item()),
                    "candidate_norm": float(torch.linalg.vector_norm(candidate_fp32).item()),
                    "applied_norm": float(torch.linalg.vector_norm(applied_fp32).item()),
                    "token_cap_fraction": float(token_cap_fraction),
                    "global_cap_coefficient": float(global_cap_coefficient),
                }
            )

