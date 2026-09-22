from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch

from .trace_schema import FIXED_STAGE_IDS, TracePlan


CORE_LENGTHS_BY_CROSSFADE = {
    1: (4, 3, 7, 4, 3),
    2: (3, 2, 6, 3, 3),
}
NEW_STAGE_WEIGHTS = {
    1: (0.5,),
    2: (0.25, 0.75),
}


def fixed5_stage_spans(
    *,
    frames: int = 25,
    crossfade_tokens: int = 1,
) -> tuple[dict[str, object], ...]:
    if frames != 25:
        raise ValueError("fixed5-causal-r3 currently requires F=25")
    if crossfade_tokens not in CORE_LENGTHS_BY_CROSSFADE:
        raise ValueError("crossfade_tokens must be 1 or 2")
    core_lengths = CORE_LENGTHS_BY_CROSSFADE[crossfade_tokens]
    cursor = 0
    spans: list[dict[str, object]] = []
    for index, (stage_id, core_length) in enumerate(
        zip(FIXED_STAGE_IDS, core_lengths)
    ):
        core = tuple(range(cursor, cursor + core_length))
        cursor += core_length
        outgoing: tuple[int, ...] = ()
        if index + 1 < len(FIXED_STAGE_IDS):
            outgoing = tuple(range(cursor, cursor + crossfade_tokens))
            cursor += crossfade_tokens
        spans.append(
            {
                "stage_id": stage_id,
                "core": core,
                "outgoing_crossfade": outgoing,
            }
        )
    if cursor != frames:
        raise AssertionError(f"fixed stage allocation consumed {cursor}, expected {frames}")
    return tuple(spans)


def fixed5_temporal_weights(
    *,
    frames: int = 25,
    crossfade_tokens: int = 1,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return fixed stage weights [5, F] with exact compact support."""

    spans = fixed5_stage_spans(
        frames=frames,
        crossfade_tokens=crossfade_tokens,
    )
    weights = torch.zeros(
        len(FIXED_STAGE_IDS), frames, device=device, dtype=dtype
    )
    new_stage_weights = NEW_STAGE_WEIGHTS[crossfade_tokens]
    for stage_index, span in enumerate(spans):
        core = list(span["core"])
        weights[stage_index, core] = 1.0
        boundary = list(span["outgoing_crossfade"])
        for token_index, new_weight in zip(boundary, new_stage_weights):
            weights[stage_index, token_index] = 1.0 - new_weight
            weights[stage_index + 1, token_index] = new_weight
    if not bool(((weights > 0).sum(0) <= 2).all().item()):
        raise AssertionError("more than two stages overlap")
    if not torch.allclose(
        weights.sum(0), torch.ones(frames, device=device, dtype=dtype), atol=1e-6
    ):
        raise AssertionError("temporal stage weights are not a partition of unity")
    return weights


def normalized_grid(
    height: int,
    width: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    y, x = torch.meshgrid(
        torch.linspace(0.0, 1.0, height, device=device, dtype=dtype),
        torch.linspace(0.0, 1.0, width, device=device, dtype=dtype),
        indexing="ij",
    )
    return y, x


def _validate_box(box: Sequence[float]) -> tuple[float, float, float, float]:
    if len(box) != 4:
        raise ValueError("normalized box must be [x1, y1, x2, y2]")
    x1, y1, x2, y2 = (float(value) for value in box)
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        raise ValueError("box coordinates must be finite")
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise ValueError("box must satisfy 0<=x1<x2<=1 and 0<=y1<y2<=1")
    return x1, y1, x2, y2


def soft_box_mask(
    box: Sequence[float],
    height: int,
    width: int,
    *,
    softness: float = 0.03,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    if not math.isfinite(softness) or softness <= 0.0:
        raise ValueError("softness must be finite and positive")
    x1, y1, x2, y2 = _validate_box(box)
    y, x = normalized_grid(height, width, device=device)
    inside_x = torch.sigmoid((x - x1) / softness) * torch.sigmoid(
        (x2 - x) / softness
    )
    inside_y = torch.sigmoid((y - y1) / softness) * torch.sigmoid(
        (y2 - y) / softness
    )
    return (inside_x * inside_y).clamp(0.0, 1.0)


def soft_corridor_mask(
    source_box: Sequence[float],
    target_box: Sequence[float],
    height: int,
    width: int,
    *,
    radius: float = 0.10,
    softness: float = 0.03,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be finite and positive")
    source = _validate_box(source_box)
    target = _validate_box(target_box)
    y, x = normalized_grid(height, width, device=device)
    points = torch.stack([x, y], dim=-1)
    source_center = points.new_tensor(
        [(source[0] + source[2]) / 2.0, (source[1] + source[3]) / 2.0]
    )
    target_center = points.new_tensor(
        [(target[0] + target[2]) / 2.0, (target[1] + target[3]) / 2.0]
    )
    direction = target_center - source_center
    denominator = direction.square().sum()
    if float(denominator.item()) <= 1e-12:
        raise ValueError("corridor endpoints have coincident centers")
    projection = ((points - source_center) * direction).sum(-1) / denominator
    projection = projection.clamp(0.0, 1.0)
    closest = source_center + projection[..., None] * direction
    distance = torch.linalg.vector_norm(points - closest, dim=-1)
    tube = torch.sigmoid((radius - distance) / softness)
    return torch.maximum(
        tube,
        torch.maximum(
            soft_box_mask(source, height, width, softness=softness, device=device),
            soft_box_mask(target, height, width, softness=softness, device=device),
        ),
    ).clamp(0.0, 1.0)


def threshold_and_protect(
    mask: torch.Tensor,
    *,
    epsilon: float = 0.03,
    protected_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if not 0.0 <= float(epsilon) < 1.0:
        raise ValueError("epsilon must be in [0, 1)")
    result = mask.float().clamp(0.0, 1.0)
    result = torch.where(result >= float(epsilon), result, torch.zeros_like(result))
    if protected_mask is not None:
        if protected_mask.shape != result.shape:
            raise ValueError("protected mask shape must match spatial mask")
        result = result * (1.0 - protected_mask.float().clamp(0.0, 1.0))
    return result.clamp(0.0, 1.0)


def _union_masks(masks: Sequence[torch.Tensor]) -> torch.Tensor:
    if not masks:
        raise ValueError("cannot build a spatial support without boxes")
    result = masks[0]
    for mask in masks[1:]:
        result = torch.maximum(result, mask)
    return result


def build_stage_spatial_masks(
    plan: TracePlan,
    *,
    height: int,
    width: int,
    epsilon: float = 0.03,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build [5,H,W] masks or raise; this function never selects a fallback mode."""

    raw_boxes = plan.grounding.get("entity_boxes", [])
    if not isinstance(raw_boxes, list):
        raise ValueError("grounding.entity_boxes must be an array")
    box_by_entity: dict[str, tuple[float, float, float, float]] = {}
    for item in raw_boxes:
        if not isinstance(item, Mapping):
            raise ValueError("each grounding box must be an object")
        entity_id = str(item.get("entity_id", "")).strip()
        box_by_entity[entity_id] = _validate_box(item.get("box", []))
    role_to_entities: dict[str, list[str]] = {}
    for entity in plan.entities:
        role_to_entities.setdefault(entity.role, []).append(entity.entity_id)

    protected_parts = []
    for item in plan.grounding.get("protected_boxes", []):
        if not isinstance(item, Mapping):
            raise ValueError("each protected box must be an object")
        protected_parts.append(
            soft_box_mask(item.get("box", []), height, width, device=device)
        )
    protected = _union_masks(protected_parts) if protected_parts else None

    stage_masks = []
    for stage in plan.stages:
        if stage.support_kind == "global":
            raw_mask = torch.ones(height, width, device=device)
        else:
            entity_ids = [
                entity_id
                for role in stage.affected_roles
                for entity_id in role_to_entities.get(role, [])
                if entity_id in box_by_entity
            ]
            boxes = [box_by_entity[entity_id] for entity_id in entity_ids]
            if stage.support_kind == "role_union":
                raw_mask = _union_masks(
                    [soft_box_mask(box, height, width, device=device) for box in boxes]
                )
            elif stage.support_kind in {
                "motion_corridor",
                "motion_contact_corridor",
                "source_stream_sink",
            }:
                if len(boxes) < 2:
                    raise ValueError(
                        f"stage {stage.stage_id} support {stage.support_kind} needs two boxes"
                    )
                radius = 0.12 if stage.support_kind == "motion_contact_corridor" else 0.10
                raw_mask = soft_corridor_mask(
                    boxes[0], boxes[1], height, width, radius=radius, device=device
                )
            elif stage.support_kind == "contact_interface":
                if len(boxes) < 2:
                    raise ValueError("contact_interface needs two boxes")
                raw_mask = soft_corridor_mask(
                    boxes[0], boxes[1], height, width, radius=0.06, device=device
                )
            else:
                raise ValueError(f"unsupported support kind {stage.support_kind!r}")
        stage_masks.append(
            threshold_and_protect(
                raw_mask,
                epsilon=epsilon,
                protected_mask=protected,
            )
        )
    return torch.stack(stage_masks)


def flatten_stage_gates(
    temporal_weights: torch.Tensor,
    spatial_masks: torch.Tensor,
    *,
    seq_len: int,
) -> torch.Tensor:
    """Combine f/y/x gates and return [stage, seq_len, 1]."""

    if temporal_weights.ndim != 2:
        raise ValueError("temporal_weights must have shape [stage, F]")
    if spatial_masks.ndim == 2:
        spatial_masks = spatial_masks.unsqueeze(0).expand(
            temporal_weights.size(0), -1, -1
        )
    if spatial_masks.ndim != 3:
        raise ValueError("spatial_masks must have shape [stage,H,W] or [H,W]")
    if spatial_masks.size(0) != temporal_weights.size(0):
        raise ValueError("temporal and spatial stage counts must match")
    valid = (
        temporal_weights[:, :, None, None] * spatial_masks[:, None, :, :]
    ).reshape(temporal_weights.size(0), -1, 1)
    if valid.size(1) > seq_len:
        raise ValueError("stage gate exceeds padded Wan sequence length")
    padding = valid.new_zeros(valid.size(0), seq_len - valid.size(1), 1)
    return torch.cat([valid, padding], dim=1)


def active_stage_indices(token_gates: torch.Tensor) -> tuple[torch.Tensor, ...]:
    if token_gates.ndim != 3 or token_gates.size(-1) != 1:
        raise ValueError("token_gates must have shape [stage,seq_len,1]")
    return tuple(
        torch.nonzero(stage_gate[:, 0] > 0, as_tuple=False).flatten()
        for stage_gate in token_gates
    )

