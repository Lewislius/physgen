from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from .cap_ledger import CapLedger
from .compile import CompiledTraceV3
from .config import TraceWriterConfigV3
from .control_schema import ContextPair
from .controllers import AuditReader
from .dynamic_support import OperatorSaliencyState, build_dynamic_support


@dataclass(frozen=True)
class StageContexts:
    positive: torch.Tensor
    violations: tuple[torch.Tensor, ...]
    violation_weights: tuple[float, ...]


@dataclass(frozen=True)
class StageRuntime:
    stage_id: str
    pair: ContextPair
    contexts: StageContexts
    full_gate: torch.Tensor
    core_gate: torch.Tensor
    core_indices: torch.Tensor


@dataclass(frozen=True)
class BoundaryRuntime:
    left_index: int
    right_index: int
    temporal_frames: tuple[int, ...]
    indices: torch.Tensor
    left_gate: torch.Tensor
    right_gate: torch.Tensor


@dataclass
class RoutingRuntimeV3:
    compiled: CompiledTraceV3
    stages: tuple[StageRuntime, ...]
    boundaries: tuple[BoundaryRuntime, ...]
    frames: int
    height: int
    width: int
    seq_len: int
    spatiotemporal_support: torch.Tensor
    active_support: torch.Tensor
    support_manifest: Mapping[str, Any]
    config: TraceWriterConfigV3
    cap_ledger: CapLedger
    saliency_state: OperatorSaliencyState | None
    reader: AuditReader | None = None

    def prediction_support(self) -> torch.Tensor:
        valid = self.active_support[0, : self.frames * self.height * self.width, 0]
        planned = valid.reshape(1, self.frames, self.height, self.width)
        if self.saliency_state is None:
            return planned
        selected = self.saliency_state.combined_support(device=planned.device)
        return planned if selected is None else planned * selected


def _flatten(value: torch.Tensor, seq_len: int) -> torch.Tensor:
    flattened = value.reshape(value.size(0), -1, 1)
    if flattened.size(1) > seq_len:
        raise ValueError("route exceeds model seq_len")
    padding = flattened.new_zeros(flattened.size(0), seq_len - flattened.size(1), 1)
    return torch.cat([flattened, padding], dim=1)


def build_routing_runtime_v3(
    compiled: CompiledTraceV3,
    encoded_contexts: Mapping[str, torch.Tensor],
    *,
    height: int,
    width: int,
    seq_len: int,
    config: TraceWriterConfigV3,
    reader: AuditReader | None = None,
) -> RoutingRuntimeV3:
    if compiled.frames * height * width > seq_len:
        raise ValueError("valid route tokens exceed padded sequence")
    device = next(iter(encoded_contexts.values())).device
    support, support_manifest = build_dynamic_support(
        compiled,
        height=height,
        width=width,
        support_mode=config.dynamic_support_mode,
        epsilon=config.mask_epsilon,
        device=device,
    )
    temporal = compiled.temporal_weights.to(device)
    full = temporal[:, :, None, None] * support
    full_gate = _flatten(full, seq_len)
    core_temporal = (temporal == 1.0).to(temporal.dtype)
    core_gate = _flatten(core_temporal[:, :, None, None] * support, seq_len)

    stages: list[StageRuntime] = []
    for index, pair in enumerate(compiled.contexts.stage_pairs):
        positive_key = f"stage.{pair.stage_id}.positive"
        positive = encoded_contexts[positive_key]
        violations = tuple(
            encoded_contexts[f"stage.{pair.stage_id}.negative.{negative_index}"]
            for negative_index in range(len(pair.violation_texts))
        )
        if positive.ndim != 3 or any(item.ndim != 3 for item in violations):
            raise ValueError("projected contexts must be [B,T,C]")
        indices = torch.nonzero(core_gate[index, :, 0] > 0, as_tuple=False).flatten()
        stages.append(
            StageRuntime(
                stage_id=pair.stage_id,
                pair=pair,
                contexts=StageContexts(positive, violations, pair.violation_weights),
                full_gate=full_gate[index].unsqueeze(0),
                core_gate=core_gate[index].unsqueeze(0),
                core_indices=indices,
            )
        )

    boundaries: list[BoundaryRuntime] = []
    for left_index in range(len(stages) - 1):
        right_index = left_index + 1
        frame_mask = (temporal[left_index] > 0) & (temporal[right_index] > 0)
        temporal_frames = tuple(torch.nonzero(frame_mask, as_tuple=False).flatten().tolist())
        left = full_gate[left_index].clone()
        right = full_gate[right_index].clone()
        valid_frame_gate = torch.zeros(compiled.frames, device=device)
        if temporal_frames:
            valid_frame_gate[list(temporal_frames)] = 1.0
        allowed = _flatten(
            valid_frame_gate[None, :, None, None].expand(1, -1, height, width),
            seq_len,
        )[0]
        left *= allowed
        right *= allowed
        union = (left[:, 0] > 0) | (right[:, 0] > 0)
        indices = torch.nonzero(union, as_tuple=False).flatten()
        boundaries.append(
            BoundaryRuntime(
                left_index=left_index,
                right_index=right_index,
                temporal_frames=temporal_frames,
                indices=indices,
                left_gate=left.unsqueeze(0),
                right_gate=right.unsqueeze(0),
            )
        )
    active = full_gate.max(0).values.unsqueeze(0).clamp(0, 1)
    saliency = None
    if config.dynamic_support_mode == "planned_saliency":
        saliency = OperatorSaliencyState(
            compiled.frames,
            height,
            width,
            max_area=config.saliency_max_area,
            ema=config.saliency_ema,
        )
    return RoutingRuntimeV3(
        compiled=compiled,
        stages=tuple(stages),
        boundaries=tuple(boundaries),
        frames=compiled.frames,
        height=height,
        width=width,
        seq_len=seq_len,
        spatiotemporal_support=support,
        active_support=active,
        support_manifest=support_manifest,
        config=config,
        cap_ledger=CapLedger(config.group_cap_ratio),
        saliency_state=saliency,
        reader=reader,
    )
