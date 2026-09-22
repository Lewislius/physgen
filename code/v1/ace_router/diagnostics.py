from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ResidualDiagnostics:
    mode: str = "summary"
    records: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.mode not in {"off", "summary", "full"}:
            raise ValueError("diagnostic mode must be off, summary, or full")

    def should_collect(
        self,
        *,
        step_index: int,
        total_steps: int,
        layer_id: int,
        active_layer_ids: tuple[int, ...],
    ) -> bool:
        if self.mode == "off":
            return False
        if self.mode == "full":
            return True
        step_boundaries = {
            0,
            max(0, math.ceil(0.30 * total_steps) - 1),
            min(total_steps - 1, math.ceil(0.30 * total_steps)),
            max(0, math.ceil(0.60 * total_steps) - 1),
            min(total_steps - 1, math.ceil(0.60 * total_steps)),
            max(0, math.ceil(0.80 * total_steps) - 1),
        }
        if not active_layer_ids:
            return False
        layer_boundaries = {
            active_layer_ids[0],
            active_layer_ids[-1],
        }
        for previous, current in zip(active_layer_ids, active_layer_ids[1:]):
            if current != previous + 1:
                layer_boundaries.update((previous, current))
        return step_index in step_boundaries and layer_id in layer_boundaries

    def record(
        self,
        *,
        residual_mode: str,
        step_index: int,
        total_steps: int,
        layer_id: int,
        layer_gate: float,
        step_gate: float,
        lambda0: float,
        semantic: Any,
        positive: Any,
        counterfactual: Any | None,
        candidate_delta: Any,
        applied_delta: Any,
        clip_coefficient: Any,
        residual_cap_ratio: float | None,
    ) -> None:
        semantic_fp32 = semantic.detach().float()
        positive_fp32 = positive.detach().float()
        if residual_mode == "positive_only":
            reference_name = "semantic"
            reference_fp32 = semantic_fp32
            counterfactual_fp32 = None
        elif residual_mode == "positive_minus_counterfactual":
            if counterfactual is None:
                raise ValueError(
                    "positive_minus_counterfactual diagnostics require counterfactual"
                )
            reference_name = "counterfactual"
            counterfactual_fp32 = counterfactual.detach().float()
            reference_fp32 = counterfactual_fp32
        else:
            raise ValueError(f"unsupported residual mode: {residual_mode!r}")
        raw_delta = positive_fp32 - reference_fp32
        candidate_fp32 = candidate_delta.detach().float()
        applied_fp32 = applied_delta.detach().float()
        coefficient_fp32 = clip_coefficient.detach().float()
        semantic_norm = semantic_fp32.norm()
        positive_norm = positive_fp32.norm()
        reference_norm = reference_fp32.norm()
        raw_delta_norm = raw_delta.norm()
        candidate_norm = candidate_fp32.norm()
        applied_norm = applied_fp32.norm()
        dot = (positive_fp32 * reference_fp32).sum()
        cosine = dot / (positive_norm * reference_norm + 1e-12)
        finite = bool(
            semantic_fp32.isfinite().all().item()
            and positive_fp32.isfinite().all().item()
            and reference_fp32.isfinite().all().item()
            and candidate_fp32.isfinite().all().item()
            and applied_fp32.isfinite().all().item()
            and coefficient_fp32.isfinite().all().item()
        )
        coefficient_min = float(coefficient_fp32.min().item())
        coefficient_max = float(coefficient_fp32.max().item())
        was_clipped = coefficient_min < 1.0 - 1e-7
        self.records.append(
            {
                "residual_mode": residual_mode,
                "residual_reference": reference_name,
                "step_index": step_index,
                "total_steps": total_steps,
                "layer_id": layer_id,
                "layer_gate": layer_gate,
                "step_gate": step_gate,
                "lambda0": lambda0,
                "semantic_l2": float(semantic_norm.item()),
                "positive_l2": float(positive_norm.item()),
                "counterfactual_l2": (
                    float(reference_norm.item())
                    if counterfactual_fp32 is not None
                    else None
                ),
                "reference_l2": float(reference_norm.item()),
                "raw_delta_l2": float(raw_delta_norm.item()),
                "candidate_delta_l2": float(candidate_norm.item()),
                "applied_delta_l2": float(applied_norm.item()),
                "scaled_delta_l2": float(applied_norm.item()),
                "candidate_to_semantic": float(
                    (candidate_norm / (semantic_norm + 1e-12)).item()
                ),
                "applied_to_semantic": float(
                    (applied_norm / (semantic_norm + 1e-12)).item()
                ),
                "scaled_to_semantic": float(
                    (applied_norm / (semantic_norm + 1e-12)).item()
                ),
                "residual_cap_ratio": residual_cap_ratio,
                "clip_coefficient_min": coefficient_min,
                "clip_coefficient_max": coefficient_max,
                "was_clipped": was_clipped,
                "lambda_effective_min": lambda0 * coefficient_min,
                "positive_reference_cosine": float(cosine.item()),
                "positive_counterfactual_cosine": (
                    float(cosine.item())
                    if counterfactual_fp32 is not None
                    else None
                ),
                "isfinite": finite,
            }
        )
        if not finite:
            raise FloatingPointError(
                f"non-finite ACE residual at step={step_index}, layer={layer_id}"
            )

    def summary(self) -> dict[str, Any]:
        if not self.records:
            return {"mode": self.mode, "record_count": 0}
        ratios = [record["scaled_to_semantic"] for record in self.records]
        candidate_ratios = [
            record["candidate_to_semantic"] for record in self.records
        ]
        raw_norms = [record["raw_delta_l2"] for record in self.records]
        clipped_count = sum(bool(record["was_clipped"]) for record in self.records)
        return {
            "mode": self.mode,
            "record_count": len(self.records),
            "all_finite": all(record["isfinite"] for record in self.records),
            "scaled_to_semantic_min": min(ratios),
            "scaled_to_semantic_max": max(ratios),
            "applied_to_semantic_min": min(ratios),
            "applied_to_semantic_max": max(ratios),
            "candidate_to_semantic_min": min(candidate_ratios),
            "candidate_to_semantic_max": max(candidate_ratios),
            "clipped_record_count": clipped_count,
            "clip_fraction": clipped_count / len(self.records),
            "clip_coefficient_min": min(
                record["clip_coefficient_min"] for record in self.records
            ),
            "raw_delta_l2_min": min(raw_norms),
            "raw_delta_l2_max": max(raw_norms),
        }
