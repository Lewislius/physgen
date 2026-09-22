from __future__ import annotations

import math
import re
from typing import Any, Mapping, Sequence

import torch

from v2.ace_router.trace_masks import soft_box_mask, soft_corridor_mask, threshold_and_protect

from .compile import CompiledTraceV3


_DIRECTION = {
    "right": (0.18, 0.0),
    "rightward": (0.18, 0.0),
    "left": (-0.18, 0.0),
    "leftward": (-0.18, 0.0),
    "down": (0.0, 0.18),
    "downward": (0.0, 0.18),
    "fall": (0.0, 0.18),
    "falls": (0.0, 0.18),
    "descent": (0.0, 0.18),
    "descend": (0.0, 0.18),
    "descends": (0.0, 0.18),
    "descending": (0.0, 0.18),
    "up": (0.0, -0.18),
    "upward": (0.0, -0.18),
    "rise": (0.0, -0.18),
}


def _validate_box(value: Sequence[float]) -> tuple[float, float, float, float]:
    if len(value) != 4:
        raise ValueError("box must have four normalized coordinates")
    x1, y1, x2, y2 = (float(x) for x in value)
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError("invalid normalized box")
    return x1, y1, x2, y2


def _transform_box(
    box: tuple[float, float, float, float],
    dx: float,
    dy: float,
    scale: float = 1.0,
) -> tuple[float, float, float, float]:
    width = min((box[2] - box[0]) * scale, 1.0)
    height = min((box[3] - box[1]) * scale, 1.0)
    cx = min(max((box[0] + box[2]) / 2 + dx, width / 2), 1 - width / 2)
    cy = min(max((box[1] + box[3]) / 2 + dy, height / 2), 1 - height / 2)
    return cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2


def _lerp_box(a: Sequence[float], b: Sequence[float], ratio: float) -> tuple[float, ...]:
    return tuple(float(x) + ratio * (float(y) - float(x)) for x, y in zip(a, b))


def _centers_differ(a: Sequence[float], b: Sequence[float]) -> bool:
    ax, ay = (float(a[0]) + float(a[2])) / 2, (float(a[1]) + float(a[3])) / 2
    bx, by = (float(b[0]) + float(b[2])) / 2, (float(b[1]) + float(b[3])) / 2
    return abs(ax - bx) + abs(ay - by) > 1e-8


def _entity_direction(compiled: CompiledTraceV3, entity_id: str) -> tuple[float, float]:
    predicate_ids = {
        item.predicate_id for item in compiled.plan.predicates if item.subject == entity_id
    }
    text = " ".join(
        [
            *(item.attribute for item in compiled.plan.predicates if item.subject == entity_id),
            *(
                effect.direction
                for transition in compiled.plan.transitions
                for effect in transition.effects
                if effect.predicate in predicate_ids
            ),
        ]
    ).casefold()
    dx = dy = 0.0
    for word, vector in _DIRECTION.items():
        if re.search(rf"\b{re.escape(word)}\b", text):
            if vector[0]:
                dx = vector[0]
            if vector[1]:
                dy = vector[1]
    return dx, dy


def _entity_envelope_scale(compiled: CompiledTraceV3, entity_id: str) -> float:
    """Estimate a conservative support envelope, not a generated object scale."""

    predicate_ids = {
        item.predicate_id for item in compiled.plan.predicates if item.subject == entity_id
    }
    text = " ".join(
        [
            *(item.attribute for item in compiled.plan.predicates if item.subject == entity_id),
            *(
                f"{effect.from_value} {effect.to_value} {effect.direction}"
                for transition in compiled.plan.transitions
                for effect in transition.effects
                if effect.predicate in predicate_ids
            ),
        ]
    ).casefold()
    strong = ("inflate", "expands", "large_taut", "scattered", "spread over")
    moderate = ("expand", "spread", "fragment", "shatter", "burst", "pieces")
    if any(word in text for word in strong):
        return 2.2
    if any(word in text for word in moderate):
        return 1.7
    return 1.0


def _explicit_tracks(raw: Mapping[str, Any]) -> dict[str, tuple[float, float, float, float]]:
    control = raw.get("control", {})
    tracks = control.get("entity_tracks", {}) if isinstance(control, Mapping) else {}
    if not isinstance(tracks, Mapping):
        raise ValueError("control.entity_tracks must be an object")
    result: dict[str, tuple[float, float, float, float]] = {}
    for entity_id, spec in tracks.items():
        if not isinstance(spec, Mapping):
            raise ValueError(f"track {entity_id!r} must be an object")
        keyframes = spec.get("keyframes", [])
        if not isinstance(keyframes, list) or not keyframes:
            continue
        last = keyframes[-1]
        if not isinstance(last, Mapping):
            raise ValueError(f"track {entity_id!r} keyframe must be an object")
        result[str(entity_id)] = _validate_box(last.get("box", []))
    return result


def build_dynamic_support(
    compiled: CompiledTraceV3,
    *,
    height: int,
    width: int,
    support_mode: str,
    epsilon: float = 0.03,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return stage spatial support [5,F,H,W], never a free HD dense mask."""

    if support_mode not in {"off", "planned", "planned_saliency"}:
        raise ValueError("invalid dynamic support mode")
    stage_count, frames = compiled.temporal_weights.shape
    if support_mode == "off":
        support = torch.ones(stage_count, frames, height, width, device=device)
        return support, {"mode": "off", "source": "full_frame_explicit"}
    raw_boxes = compiled.plan.grounding.get("entity_boxes", [])
    if not isinstance(raw_boxes, list):
        raise ValueError("grounding.entity_boxes must be an array")
    initial: dict[str, tuple[float, float, float, float]] = {}
    for item in raw_boxes:
        if not isinstance(item, Mapping):
            raise ValueError("grounding entity box must be an object")
        initial[str(item.get("entity_id", ""))] = _validate_box(item.get("box", []))
    if not initial:
        support = torch.ones(stage_count, frames, height, width, device=device)
        source = "operator_saliency_seed" if support_mode == "planned_saliency" else "layout_unresolved"
        return support, {
            "mode": support_mode,
            "source": source,
            "requires_runtime_saliency": support_mode == "planned_saliency",
            "issues": ["no spatial boxes; no image-grounded support was invented"],
        }
    explicit_targets = _explicit_tracks(compiled.raw_plan)
    target: dict[str, tuple[float, float, float, float]] = {}
    provenance: dict[str, str] = {}
    for entity_id, box in initial.items():
        if entity_id in explicit_targets:
            target[entity_id] = explicit_targets[entity_id]
            provenance[entity_id] = "explicit"
        else:
            dx, dy = _entity_direction(compiled, entity_id)
            envelope_scale = _entity_envelope_scale(compiled, entity_id)
            target[entity_id] = _transform_box(box, dx, dy, envelope_scale)
            provenance[entity_id] = (
                "derived_motion_envelope"
                if dx or dy or envelope_scale != 1.0
                else "unresolved_static_envelope"
            )

    role_to_entities: dict[str, list[str]] = {}
    for entity in compiled.plan.entities:
        role_to_entities.setdefault(entity.role, []).append(entity.entity_id)
    protected_parts: list[torch.Tensor] = []
    for item in compiled.plan.grounding.get("protected_boxes", []):
        if isinstance(item, Mapping):
            protected_parts.append(soft_box_mask(item.get("box", []), height, width, device=device))
    protected = None
    if protected_parts:
        protected = protected_parts[0]
        for part in protected_parts[1:]:
            protected = torch.maximum(protected, part)

    result = torch.zeros(stage_count, frames, height, width, device=device)
    stage_areas: list[dict[str, Any]] = []
    for stage_index, stage in enumerate(compiled.plan.stages):
        entity_ids = [
            entity_id
            for role in stage.affected_roles
            for entity_id in role_to_entities.get(role, [])
            if entity_id in initial
        ]
        if not entity_ids:
            entity_ids = list(initial)
        for frame in range(frames):
            progress = frame / max(frames - 1, 1)
            boxes = [_lerp_box(initial[x], target[x], progress) for x in entity_ids]
            parts = [soft_box_mask(box, height, width, softness=0.04, device=device) for box in boxes]
            if stage.support_kind in {
                "motion_corridor", "motion_contact_corridor", "contact_interface",
                "source_stream_sink",
            } and len(boxes) >= 2:
                radius = 0.07 if stage.support_kind == "contact_interface" else 0.12
                parts.append(
                    soft_corridor_mask(boxes[0], boxes[1], height, width, radius=radius, device=device)
                )
            for entity_id in entity_ids:
                if _centers_differ(initial[entity_id], target[entity_id]):
                    parts.append(
                        soft_corridor_mask(
                            initial[entity_id], target[entity_id], height, width,
                            radius=0.10, device=device,
                        )
                    )
            mask = parts[0]
            for part in parts[1:]:
                mask = torch.maximum(mask, part)
            result[stage_index, frame] = threshold_and_protect(
                mask, epsilon=epsilon, protected_mask=protected
            )
        stage_areas.append(
            {
                "stage_id": stage.stage_id,
                "mean_soft_area": float(result[stage_index].sum((-1, -2)).mean().item()),
                "max_soft_area": float(result[stage_index].sum((-1, -2)).max().item()),
            }
        )
    return result, {
        "mode": support_mode,
        "source": "plan_boxes_and_tracks",
        "grid": [frames, height, width],
        "target_provenance": provenance,
        "stage_areas": stage_areas,
        "requires_runtime_saliency": support_mode == "planned_saliency",
    }


class OperatorSaliencyState:
    """Runtime top-area support derived from positive cross-attention energy."""

    def __init__(self, frames: int, height: int, width: int, *, max_area: float, ema: float):
        self.frames = frames
        self.height = height
        self.width = width
        self.max_area = float(max_area)
        self.ema = float(ema)
        self._state: dict[str, torch.Tensor] = {}
        self._selected: dict[str, torch.Tensor] = {}

    def local_mask(
        self,
        stage_id: str,
        positive: torch.Tensor,
        indices: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        energy = torch.linalg.vector_norm(positive.detach().float(), dim=-1).mean(0)
        full = energy.new_zeros(seq_len)
        full.index_copy_(0, indices, energy)
        valid = full[: self.frames * self.height * self.width].reshape(
            self.frames, self.height * self.width
        )
        prior = self._state.get(stage_id)
        if prior is not None:
            valid = self.ema * prior.to(valid) + (1.0 - self.ema) * valid
        self._state[stage_id] = valid.detach().cpu()
        selected = torch.zeros_like(valid)
        k = max(1, min(valid.size(1), math.ceil(self.max_area * valid.size(1))))
        active_frames = torch.nonzero(valid.max(1).values > 0, as_tuple=False).flatten()
        for frame in active_frames.tolist():
            values, positions = torch.topk(valid[frame], k=k, largest=True)
            denominator = values.max().clamp_min(1e-12)
            selected[frame, positions] = (values / denominator).clamp(0.05, 1.0)
        self._selected[stage_id] = selected.detach().cpu()
        flattened = torch.cat(
            [selected.reshape(-1), selected.new_zeros(seq_len - selected.numel())]
        )
        return flattened.index_select(0, indices).view(1, -1, 1)

    def combined_support(self, *, device: torch.device | str) -> torch.Tensor | None:
        if not self._selected:
            return None
        values = [item.to(device=device, dtype=torch.float32) for item in self._selected.values()]
        result = values[0]
        for item in values[1:]:
            result = torch.maximum(result, item)
        return result.reshape(1, self.frames, self.height, self.width).clamp(0, 1)
