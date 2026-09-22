from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping


COMPILED_SCHEMA_VERSION = "trace-writer-v2"
DESIGN_REVISION = "fixed5-causal-r4"
STAGE_IDS = ("setup", "onset", "evolution", "completion", "terminal")
Provenance = Literal["explicit", "derived", "default", "unresolved"]


@dataclass(frozen=True)
class Cardinality:
    kind: Literal["exact", "range", "group", "unknown"]
    exact: int | None = None
    minimum: int | None = None
    maximum: int | None = None

    def label(self) -> str:
        if self.kind == "exact" and self.exact is not None:
            return f"exactly {self.exact}"
        if self.kind == "range" and self.minimum is not None:
            upper = self.maximum if self.maximum is not None else "unbounded"
            return f"between {self.minimum} and {upper}"
        if self.kind == "group":
            return "one declared collective group"
        return "declared but not assigned an exact visual count"


@dataclass(frozen=True)
class EntitySlot:
    entity_id: str
    canonical_mention: str
    role: str
    cardinality: Cardinality
    persistent_identity: bool
    mutable_attributes: tuple[str, ...]
    invariant_attributes: tuple[str, ...]
    aliases: tuple[str, ...]
    provenance: Provenance


@dataclass(frozen=True)
class StagePresence:
    entity_id: str
    stage_id: str
    logical_count: Cardinality
    visible: Literal["required", "allowed", "forbidden", "transitioning"]
    state_facts: tuple[str, ...]


@dataclass(frozen=True)
class ClosedWorldSpec:
    allowed_entity_ids: tuple[str, ...]
    forbidden_new_categories: tuple[str, ...]
    empty_regions: tuple[str, ...]
    external_source_allowed: bool
    provenance: Provenance


@dataclass(frozen=True)
class LifecycleTransition:
    entity_id: str
    from_stage: str
    to_stage: str
    from_state: str
    to_state: str
    identity_preserved: bool
    mutually_exclusive_states: tuple[tuple[str, str], ...]
    provenance: Provenance


@dataclass(frozen=True)
class ConservationRule:
    rule_id: str
    quantity: Literal[
        "instance_count", "material_proxy", "area_proxy", "volume_proxy",
        "motion_transfer",
    ]
    sources: tuple[str, ...]
    sinks: tuple[str, ...]
    relation: Literal["constant", "transfer", "coupled_monotonic"]
    tolerance: float
    observable_proxy: str
    provenance: Provenance


@dataclass(frozen=True)
class EntityLedger:
    slots: tuple[EntitySlot, ...]
    stage_presence: tuple[StagePresence, ...]
    closed_world: ClosedWorldSpec
    lifecycles: tuple[LifecycleTransition, ...]
    conservation: tuple[ConservationRule, ...]
    issues: tuple[str, ...]

    def stage(self, stage_id: str) -> tuple[StagePresence, ...]:
        return tuple(item for item in self.stage_presence if item.stage_id == stage_id)

    def slot(self, entity_id: str) -> EntitySlot:
        for item in self.slots:
            if item.entity_id == entity_id:
                return item
        raise KeyError(entity_id)

    def manifest(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ContextPair:
    stage_id: str
    shared_anchor: str
    positive_relation: str
    violation_relations: tuple[str, ...]
    positive_text: str
    violation_texts: tuple[str, ...]
    violation_kinds: tuple[str, ...]
    violation_constraint_ids: tuple[tuple[str, ...], ...]
    violation_weights: tuple[float, ...]
    pair_quality: str


@dataclass(frozen=True)
class ContextBundle:
    global_anchor: str
    stage_pairs: tuple[ContextPair, ...]
    protected_coverage: Mapping[str, tuple[str, ...]]
    token_budget_priority: Mapping[str, int]
    issues: tuple[str, ...]

    def texts(self) -> dict[str, str]:
        values: dict[str, str] = {"global_semantic": self.global_anchor}
        for pair in self.stage_pairs:
            values[f"stage.{pair.stage_id}.positive"] = pair.positive_text
            for index, text in enumerate(pair.violation_texts):
                values[f"stage.{pair.stage_id}.negative.{index}"] = text
        return values

    def manifest(self) -> dict[str, Any]:
        result = asdict(self)
        result["texts"] = self.texts()
        return result


def merge_overlay(raw: Mapping[str, Any], overlay: Mapping[str, Any] | None) -> dict[str, Any]:
    """Merge only the optional control namespace; source graph facts stay immutable."""

    result = dict(raw)
    base_control = result.get("control", {})
    if not isinstance(base_control, Mapping):
        raise ValueError("plan.control must be an object")
    merged = dict(base_control)
    if overlay is not None:
        overlay_control = overlay.get("control", overlay)
        if not isinstance(overlay_control, Mapping):
            raise ValueError("control overlay must be an object")
        for key, value in overlay_control.items():
            if key in merged and merged[key] != value:
                raise ValueError(f"control overlay conflicts with plan control at {key!r}")
            merged[key] = value
    result["control"] = merged
    return result
