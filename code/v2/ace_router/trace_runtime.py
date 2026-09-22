from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from .controllers import AuditReader
from .trace_compile import CompiledStage, CompiledTrace
from .trace_masks import active_stage_indices, flatten_stage_gates


@dataclass(frozen=True)
class StageContexts:
    positive: torch.Tensor
    violations: tuple[torch.Tensor, ...]
    violation_weights: tuple[float, ...]


@dataclass(frozen=True)
class StageRuntime:
    stage: CompiledStage
    contexts: StageContexts
    token_gate: torch.Tensor  # [1, seq_len, 1]
    active_indices: torch.Tensor  # [N]


@dataclass
class RoutingRuntime:
    compiled: CompiledTrace
    stages: tuple[StageRuntime, ...]
    frames: int
    height: int
    width: int
    seq_len: int
    spatial_mode: str
    reader: AuditReader | None = None


def _check_context(name: str, value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor) or value.ndim != 3:
        raise ValueError(f"{name} must be a tensor with shape [B,T,C]")
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point")


def build_routing_runtime(
    compiled: CompiledTrace,
    encoded_contexts: Mapping[str, torch.Tensor],
    *,
    height: int,
    width: int,
    seq_len: int,
    spatial_mode: str = "time",
    spatial_masks: torch.Tensor | None = None,
    reader: AuditReader | None = None,
) -> RoutingRuntime:
    """Build executable gates. Invalid spacetime inputs raise; no mode fallback occurs."""

    if spatial_mode not in {"time", "spacetime"}:
        raise ValueError("spatial_mode must be 'time' or 'spacetime'")
    if height <= 0 or width <= 0 or seq_len <= 0:
        raise ValueError("height, width, and seq_len must be positive")
    valid_tokens = compiled.frames * height * width
    if valid_tokens > seq_len:
        raise ValueError(f"valid tokens {valid_tokens} exceed seq_len {seq_len}")
    device = next(iter(encoded_contexts.values())).device if encoded_contexts else None
    temporal = compiled.temporal_weights.to(device=device)
    if spatial_mode == "time":
        if spatial_masks is not None:
            raise ValueError("time mode does not accept spatial_masks")
        masks = temporal.new_ones(len(compiled.stages), height, width)
    else:
        if spatial_masks is None:
            raise ValueError("spacetime mode requires spatial_masks; no fallback was selected")
        masks = spatial_masks.to(device=device, dtype=temporal.dtype)
        if masks.shape != (len(compiled.stages), height, width):
            raise ValueError(
                "spatial_masks must have shape "
                f"{(len(compiled.stages), height, width)}, got {tuple(masks.shape)}"
            )
    token_gates = flatten_stage_gates(temporal, masks, seq_len=seq_len)
    indices = active_stage_indices(token_gates)

    stage_runtimes = []
    expected_batch: int | None = None
    expected_dim: int | None = None
    for stage_index, stage in enumerate(compiled.stages):
        positive_key = f"stage.{stage.stage_id}.positive"
        if positive_key not in encoded_contexts:
            raise ValueError(f"missing encoded context {positive_key!r}")
        positive = encoded_contexts[positive_key]
        _check_context(positive_key, positive)
        violations = []
        for violation_index, _ in enumerate(stage.violations):
            key = f"stage.{stage.stage_id}.negative.{violation_index}"
            if key not in encoded_contexts:
                raise ValueError(f"missing encoded context {key!r}")
            context = encoded_contexts[key]
            _check_context(key, context)
            if context.shape[0] != positive.shape[0] or context.shape[2] != positive.shape[2]:
                raise ValueError(f"{key} batch/feature dimensions differ from {positive_key}")
            violations.append(context)
        if expected_batch is None:
            expected_batch = int(positive.shape[0])
            expected_dim = int(positive.shape[2])
        elif positive.shape[0] != expected_batch or positive.shape[2] != expected_dim:
            raise ValueError("all stage contexts must share batch and feature dimensions")
        stage_runtimes.append(
            StageRuntime(
                stage=stage,
                contexts=StageContexts(
                    positive=positive,
                    violations=tuple(violations),
                    violation_weights=tuple(
                        violation.weight for violation in stage.violations
                    ),
                ),
                token_gate=token_gates[stage_index].unsqueeze(0),
                active_indices=indices[stage_index],
            )
        )
    return RoutingRuntime(
        compiled=compiled,
        stages=tuple(stage_runtimes),
        frames=compiled.frames,
        height=height,
        width=width,
        seq_len=seq_len,
        spatial_mode=spatial_mode,
        reader=reader,
    )


def required_context_names(compiled: CompiledTrace) -> tuple[str, ...]:
    return tuple(compiled.context_texts().keys())

