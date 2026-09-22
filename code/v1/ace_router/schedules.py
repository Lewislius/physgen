from __future__ import annotations

import math
from collections.abc import Iterable


LAYER_PRESETS = {
    "mvp_mid16": ([0.0] * 8) + ([0.4] * 6) + ([1.0] * 10) + ([0.0] * 6),
    "paper_soft30": ([0.1] * 8) + ([0.4] * 6) + ([1.0] * 10) + ([0.1] * 6),
    "early": ([1.0] * 10) + ([0.0] * 20),
    "mid_a": ([0.0] * 10) + ([1.0] * 10) + ([0.0] * 10),
    "mid_b": ([0.0] * 14) + ([1.0] * 10) + ([0.0] * 6),
    "late": ([0.0] * 20) + ([1.0] * 10),
    "all": [1.0] * 30,
}

STEP_PRESETS = {
    "legacy": ((0.30, 1.0), (0.60, 0.7), (0.80, 0.3), (1.00, 0.0)),
    "safe_i2v_v1": ((0.30, 0.5), (0.70, 1.0), (0.90, 0.3), (1.00, 0.1)),
    "safe_t2v_v1": ((0.30, 0.25), (0.70, 0.8), (0.90, 0.3), (1.00, 0.1)),
    "safe_shared_v1": ((0.30, 0.5), (0.70, 1.0), (0.90, 0.3), (1.00, 0.1)),
}


def _validate_gates(values: Iterable[float], expected_layers: int) -> list[float]:
    gates = [float(value) for value in values]
    if len(gates) != expected_layers:
        raise ValueError(
            f"layer gate count must be {expected_layers}, got {len(gates)}"
        )
    if any((not math.isfinite(value)) or value < 0.0 for value in gates):
        raise ValueError("layer gates must be finite and non-negative")
    return gates


def layer_gates_for_preset(name: str, *, num_layers: int = 30) -> list[float]:
    if name not in LAYER_PRESETS:
        raise ValueError(
            f"unknown layer preset {name!r}; choose from {sorted(LAYER_PRESETS)}"
        )
    if num_layers != 30:
        raise ValueError(
            f"ACE V1 presets are defined for the 30-layer TI2V-5B, got {num_layers}"
        )
    return _validate_gates(LAYER_PRESETS[name], num_layers)


def parse_layer_gates(value: str, *, num_layers: int = 30) -> list[float]:
    return _validate_gates((part.strip() for part in value.split(",")), num_layers)


def step_gate(step_index: int, total_steps: int, *, preset: str = "legacy") -> float:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if step_index < 0 or step_index >= total_steps:
        raise ValueError(
            f"step_index must be in [0, {total_steps}), got {step_index}"
        )
    if preset not in STEP_PRESETS:
        raise ValueError(
            f"unknown step preset {preset!r}; choose from {sorted(STEP_PRESETS)}"
        )
    ratio = step_index / total_steps
    for upper_bound, value in STEP_PRESETS[preset]:
        if ratio < upper_bound:
            return value
    raise AssertionError("step preset must cover the complete [0, 1) interval")


def step_gates(total_steps: int, *, preset: str = "legacy") -> list[float]:
    return [
        step_gate(index, total_steps, preset=preset) for index in range(total_steps)
    ]
