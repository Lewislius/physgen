from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch


def masked_energy(tensor: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 3 or support.shape != tensor.shape[:2] + (1,):
        raise ValueError("masked_energy expects tensor [B,L,C] and support [B,L,1]")
    return (tensor.float().square() * support.float().clamp(0, 1)).sum((1, 2), keepdim=True)


@dataclass
class CapLedger:
    ratio: float
    applied_energy: torch.Tensor | None = None
    semantic_energy: torch.Tensor | None = None
    records: list[dict[str, Any]] = field(default_factory=list)

    def reset(self) -> None:
        self.applied_energy = None
        self.semantic_energy = None

    def consume(
        self,
        candidate: torch.Tensor,
        semantic: torch.Tensor,
        support: torch.Tensor,
        *,
        layer_id: int,
    ) -> tuple[torch.Tensor, float]:
        current = masked_energy(candidate, support)
        semantic_current = masked_energy(semantic, support)
        if self.applied_energy is None:
            self.applied_energy = torch.zeros_like(current)
            self.semantic_energy = torch.zeros_like(semantic_current)
        assert self.semantic_energy is not None
        self.semantic_energy = self.semantic_energy + semantic_current
        budget = (float(self.ratio) ** 2) * self.semantic_energy
        remaining = (budget - self.applied_energy).clamp_min(0.0)
        coefficient = torch.sqrt(remaining / current.clamp_min(1e-12)).clamp(max=1.0)
        coefficient = torch.where(current > 0, coefficient, torch.ones_like(coefficient))
        applied = candidate.float() * coefficient
        consumed = masked_energy(applied, support)
        self.applied_energy = self.applied_energy + consumed
        value = float(coefficient.min().item())
        self.records.append(
            {
                "layer_id": int(layer_id),
                "group_cap_coefficient": value,
                "group_applied_energy": float(self.applied_energy.sum().item()),
                "group_semantic_energy": float(self.semantic_energy.sum().item()),
            }
        )
        return applied.to(candidate.dtype), value


def cap_prediction_delta(
    delta: torch.Tensor,
    reference: torch.Tensor,
    support: torch.Tensor,
    *,
    inside_ratio: float,
    outside_ratio: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Cap CFG-space delta separately inside and outside a spatiotemporal ROI."""

    if delta.shape != reference.shape or delta.ndim != 4:
        raise ValueError("prediction tensors must share [C,F,H,W]")
    if support.shape != (1, delta.shape[1], delta.shape[2], delta.shape[3]):
        raise ValueError("prediction support must be [1,F,H,W]")
    mask = support.to(device=delta.device, dtype=torch.float32).clamp(0, 1)
    inside = mask
    outside = 1.0 - mask

    def cap(part: torch.Tensor, ratio: float) -> tuple[torch.Tensor, float]:
        delta_norm = torch.linalg.vector_norm((delta.float() * part).reshape(-1))
        reference_norm = torch.linalg.vector_norm((reference.float() * part).reshape(-1))
        if float(delta_norm.item()) == 0.0:
            return delta.float() * part, 1.0
        coefficient = min(1.0, float(ratio) * float(reference_norm.item()) / (float(delta_norm.item()) + 1e-12))
        return delta.float() * part * coefficient, coefficient

    inside_delta, inside_coefficient = cap(inside, inside_ratio)
    outside_delta, outside_coefficient = cap(outside, outside_ratio)
    return (inside_delta + outside_delta).to(delta.dtype), {
        "cfg_inside_coefficient": inside_coefficient,
        "cfg_outside_coefficient": outside_coefficient,
        "cfg_delta_norm_before": float(torch.linalg.vector_norm(delta.float()).item()),
        "cfg_delta_norm_after": float(
            torch.linalg.vector_norm(inside_delta + outside_delta).item()
        ),
    }
