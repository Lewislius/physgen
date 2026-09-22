from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TraceWriterConfig:
    frames: int = 25
    crossfade_tokens: int = 1
    lambda0: float = 0.10
    token_cap_ratio: float = 0.10
    global_cap_ratio: float = 0.02
    mask_epsilon: float = 0.03
    writer_block_start: int = 14
    writer_block_stop: int = 24

    def __post_init__(self) -> None:
        if self.frames != 25:
            raise ValueError("TRACE v1 is pinned to the HD97 F=25 temporal grid")
        if self.crossfade_tokens not in {1, 2}:
            raise ValueError("crossfade_tokens must be 1 or 2")
        lambda_value = float(self.lambda0)
        if not math.isfinite(lambda_value) or lambda_value < 0.0:
            raise ValueError("lambda0 must be finite and non-negative")
        for name in ("token_cap_ratio", "global_cap_ratio"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0.0 <= float(self.mask_epsilon) < 1.0:
            raise ValueError("mask_epsilon must be in [0,1)")
        if not 0 <= self.writer_block_start < self.writer_block_stop <= 30:
            raise ValueError("writer block interval must lie inside [0,30]")

    def layer_gates(self, *, num_layers: int = 30) -> list[float]:
        if num_layers != 30:
            raise ValueError("TRACE v1 layer preset is pinned to 30 Wan blocks")
        return [
            1.0 if self.writer_block_start <= index < self.writer_block_stop else 0.0
            for index in range(num_layers)
        ]

    def step_gates(self, total_steps: int) -> list[float]:
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        result = []
        for index in range(total_steps):
            ratio = index / total_steps
            if ratio < 0.30:
                result.append(0.5)
            elif ratio < 0.70:
                result.append(1.0)
            elif ratio < 0.90:
                result.append(0.3)
            else:
                result.append(0.1)
        return result
