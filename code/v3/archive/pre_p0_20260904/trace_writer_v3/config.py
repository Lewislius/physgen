from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class TraceWriterConfigV3:
    """Versioned safe defaults for the HD97 / 25-token TRACE path."""

    frames: int = 25
    crossfade_tokens: int = 2
    lambda0: float = 0.05
    token_cap_ratio: float = 0.05
    layer_cap_ratio: float = 0.01
    group_cap_ratio: float = 0.01
    cfg_cap_ratio: float = 0.01
    cfg_outside_cap_ratio: float = 0.0025
    temporal_derivative_ratio: float = 0.05
    mask_epsilon: float = 0.03
    writer_block_start: int = 14
    writer_block_stop: int = 24
    cap_mode: str = "aggregate_strict"
    boundary_morph: str = "cosine_trust"
    event_clock_mode: str = "plan_prior"
    dynamic_support_mode: str = "planned"
    cosine_low: float = -0.20
    cosine_high: float = 0.30
    saliency_min_area: float = 0.03
    saliency_max_area: float = 0.35
    saliency_ema: float = 0.80

    def __post_init__(self) -> None:
        if self.frames != 25:
            raise ValueError("TRACE v3 requires the HD97 25-token temporal grid")
        if self.crossfade_tokens not in {1, 2}:
            raise ValueError("crossfade_tokens must be 1 or 2")
        for name in (
            "lambda0",
            "token_cap_ratio",
            "layer_cap_ratio",
            "group_cap_ratio",
            "cfg_cap_ratio",
            "cfg_outside_cap_ratio",
            "temporal_derivative_ratio",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.cap_mode not in {"legacy", "aggregate_estimated", "aggregate_strict"}:
            raise ValueError("invalid cap_mode")
        if self.boundary_morph not in {"linear", "smoothstep", "cosine_trust"}:
            raise ValueError("invalid boundary_morph")
        if self.event_clock_mode not in {"fixed", "plan_prior", "reader_hysteresis"}:
            raise ValueError("invalid event_clock_mode")
        if self.dynamic_support_mode not in {"off", "planned", "planned_saliency"}:
            raise ValueError("invalid dynamic_support_mode")
        if not 0 <= self.writer_block_start < self.writer_block_stop <= 30:
            raise ValueError("writer block interval must lie inside [0,30]")
        if not 0.0 <= self.mask_epsilon < 1.0:
            raise ValueError("mask_epsilon must be in [0,1)")
        if not -1.0 <= self.cosine_low < self.cosine_high <= 1.0:
            raise ValueError("invalid cosine trust thresholds")
        if not 0.0 < self.saliency_min_area <= self.saliency_max_area < 1.0:
            raise ValueError("invalid saliency area bounds")
        if not 0.0 <= self.saliency_ema < 1.0:
            raise ValueError("saliency_ema must be in [0,1)")

    def layer_gates(self, *, num_layers: int = 30) -> list[float]:
        if num_layers != 30:
            raise ValueError("TRACE v3 preset is pinned to 30 Wan blocks")
        return [
            1.0 if self.writer_block_start <= i < self.writer_block_stop else 0.0
            for i in range(num_layers)
        ]

    def step_gates(self, total_steps: int) -> list[float]:
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        gates: list[float] = []
        for index in range(total_steps):
            ratio = index / total_steps
            if ratio < 0.30:
                gates.append(0.5)
            elif ratio < 0.70:
                gates.append(1.0)
            elif ratio < 0.90:
                gates.append(0.3)
            else:
                gates.append(0.1)
        return gates

    def manifest(self) -> dict[str, object]:
        return asdict(self)
