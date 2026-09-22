from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class AuditReader:
    """Read-only diagnostics; enabling it must never change Writer tensors."""

    records: list[dict[str, Any]] = field(default_factory=list)
    cfg_records: list[dict[str, Any]] = field(default_factory=list)

    def reset(self) -> None:
        self.records.clear()
        self.cfg_records.clear()

    def observe_operator(
        self,
        *,
        step_index: int,
        layer_id: int,
        stage_id: str,
        region: str,
        positive: torch.Tensor,
        negative: torch.Tensor,
        difference: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            p = positive.detach().float()
            n = negative.detach().float()
            d = difference.detach().float()
            denominator = torch.linalg.vector_norm(p) * torch.linalg.vector_norm(n)
            cosine = float((p * n).sum().div(denominator.clamp_min(1e-12)).item())
            self.records.append(
                {
                    "type": "stage_operator",
                    "step_index": int(step_index),
                    "layer_id": int(layer_id),
                    "stage_id": stage_id,
                    "region": region,
                    "positive_norm": float(torch.linalg.vector_norm(p).item()),
                    "negative_norm": float(torch.linalg.vector_norm(n).item()),
                    "difference_norm": float(torch.linalg.vector_norm(d).item()),
                    "positive_negative_cosine": cosine,
                }
            )

    def observe_layer(self, **record: Any) -> None:
        self.records.append({"type": "layer_cap", **record})

    def observe_boundary(self, **record: Any) -> None:
        self.records.append({"type": "boundary", **record})

    def observe_cfg(self, **record: Any) -> None:
        self.cfg_records.append({"type": "cfg_cap", **record})

