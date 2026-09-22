from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch

from v2.ace_router.trace_schema import TracePlan

from .control_schema import STAGE_IDS


_CORE_LAYOUTS = {
    "fixed": {1: (4, 3, 7, 4, 3), 2: (3, 2, 6, 3, 3)},
    "triggered": {1: (4, 3, 7, 4, 3), 2: (3, 2, 6, 3, 3)},
    "continuous": {1: (3, 3, 8, 4, 3), 2: (2, 2, 8, 2, 3)},
    "multi_transition": {1: (4, 4, 6, 4, 3), 2: (3, 3, 6, 3, 2)},
}


def _profile(plan: TracePlan, mode: str) -> str:
    if mode == "fixed":
        return "fixed"
    if len(plan.transitions) > 1:
        return "multi_transition"
    if plan.transitions and plan.transitions[0].kind == "continuous":
        return "continuous"
    return "triggered"


def _smoothstep(value: float) -> float:
    return value * value * (3.0 - 2.0 * value)


def compile_temporal_route(
    plan: TracePlan,
    *,
    frames: int = 25,
    crossfade_tokens: int = 2,
    mode: str = "plan_prior",
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if frames != 25:
        raise ValueError("TRACE v3 currently requires F=25")
    if crossfade_tokens not in {1, 2}:
        raise ValueError("crossfade_tokens must be 1 or 2")
    if mode not in {"fixed", "plan_prior", "reader_hysteresis"}:
        raise ValueError("invalid event clock mode")
    profile = _profile(plan, "fixed" if mode == "fixed" else "plan_prior")
    cores = _CORE_LAYOUTS[profile][crossfade_tokens]
    if sum(cores) + 4 * crossfade_tokens != frames:
        raise AssertionError("temporal layout does not cover F=25")
    weights = torch.zeros(len(STAGE_IDS), frames, dtype=torch.float32, device=device)
    cursor = 0
    spans: list[dict[str, Any]] = []
    boundaries: list[dict[str, Any]] = []
    for stage_index, (stage_id, core_length) in enumerate(zip(STAGE_IDS, cores)):
        core = tuple(range(cursor, cursor + core_length))
        weights[stage_index, list(core)] = 1.0
        cursor += core_length
        outgoing: tuple[int, ...] = ()
        if stage_index < len(STAGE_IDS) - 1:
            outgoing = tuple(range(cursor, cursor + crossfade_tokens))
            values: list[float] = []
            for local_index, frame in enumerate(outgoing):
                u = (local_index + 1) / (crossfade_tokens + 1)
                new_weight = _smoothstep(u)
                weights[stage_index, frame] = 1.0 - new_weight
                weights[stage_index + 1, frame] = new_weight
                values.append(new_weight)
            boundaries.append(
                {
                    "left": stage_id,
                    "right": STAGE_IDS[stage_index + 1],
                    "frames": list(outgoing),
                    "right_weights": values,
                }
            )
            cursor += crossfade_tokens
        spans.append({"stage_id": stage_id, "core": list(core), "outgoing": list(outgoing)})
    if cursor != frames:
        raise AssertionError(f"route consumed {cursor} frames")
    if not torch.allclose(weights.sum(0), torch.ones(frames, device=device), atol=1e-6):
        raise AssertionError("temporal weights are not a partition of unity")
    if bool(((weights > 0).sum(0) > 2).any().item()):
        raise AssertionError("non-adjacent temporal overlap detected")
    return weights, {
        "mode": mode,
        "profile": profile,
        "crossfade_tokens": crossfade_tokens,
        "interpolation": "smoothstep_c1",
        "core_lengths": list(cores),
        "spans": spans,
        "boundaries": boundaries,
        "weights": [[round(float(x), 6) for x in row] for row in weights.cpu()],
    }


@dataclass
class MonotonicEventClock:
    """Low-freedom hysteresis clock; it can advance but never regress."""

    stage_index: int = 0
    dwell: int = 0
    high: float = 0.70
    low: float = 0.45
    min_dwell: int = 1

    def update(self, next_stage_confidence: float) -> int:
        value = float(next_stage_confidence)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("event confidence must be in [0,1]")
        self.dwell += 1
        if (
            self.stage_index < len(STAGE_IDS) - 1
            and self.dwell >= self.min_dwell
            and value >= self.high
        ):
            self.stage_index += 1
            self.dwell = 0
        return self.stage_index
