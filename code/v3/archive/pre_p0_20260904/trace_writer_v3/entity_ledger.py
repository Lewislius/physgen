from __future__ import annotations

import re
from typing import Any, Mapping

from v2.ace_router.trace_schema import TracePlan

from .control_schema import (
    Cardinality,
    ClosedWorldSpec,
    ConservationRule,
    EntityLedger,
    EntitySlot,
    LifecycleTransition,
    STAGE_IDS,
    StagePresence,
)


_COLLECTIVE = re.compile(
    r"\b(row|fragments?|seeds?|stream|liquid|water|sand|coffee|juice|cloth|"
    r"floor|surface|sky|fire|flames?|pile|region|support|background)\b",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"\bexactly\s+(one|two|three|\d+)\b", re.IGNORECASE)
_NUMBER_VALUE = {"one": 1, "two": 2, "three": 3}


def _explicit_cardinality(control: Mapping[str, Any], entity_id: str) -> Cardinality | None:
    counts = control.get("entity_counts", {})
    if not isinstance(counts, Mapping) or entity_id not in counts:
        return None
    value = counts[entity_id]
    if isinstance(value, int) and value >= 0:
        return Cardinality("exact", exact=value)
    if isinstance(value, Mapping):
        kind = str(value.get("kind", "unknown"))
        if kind == "exact":
            return Cardinality("exact", exact=int(value["exact"]))
        if kind == "range":
            return Cardinality(
                "range", minimum=int(value["minimum"]),
                maximum=int(value["maximum"]) if value.get("maximum") is not None else None,
            )
        if kind in {"group", "unknown"}:
            return Cardinality(kind)
    raise ValueError(f"invalid cardinality for {entity_id!r}")


def _derive_cardinality(entity_id: str, mention: str, protected: tuple[str, ...]) -> tuple[Cardinality, str]:
    explicit_text = " ".join(protected)
    if entity_id.replace("_", " ") in explicit_text.casefold() or any(
        token in mention.casefold() for token in ("ball", "ice piece")
    ):
        match = _NUMBER.search(explicit_text)
        if match:
            raw = match.group(1).casefold()
            value = _NUMBER_VALUE.get(raw, int(raw) if raw.isdigit() else 1)
            return Cardinality("exact", exact=value), "derived"
    folded_mention = mention.casefold()
    if " and its " in folded_mention and any(
        word in folded_mention for word in ("fragment", "piece", "debris")
    ):
        # One persistent logical object slot may change into a collective visual
        # state.  The fragments are not treated as extra copies of the parent.
        return Cardinality("exact", exact=1), "derived"
    if _COLLECTIVE.search(mention):
        return Cardinality("group"), "derived"
    return Cardinality("exact", exact=1), "default"


def _state_fact_maps(plan: TracePlan) -> tuple[dict[str, str], ...]:
    if not plan.states:
        return ({},) * len(STAGE_IDS)
    source = dict(plan.states[0].facts)
    target = dict(plan.states[-1].facts)
    middle = dict(plan.states[min(1, len(plan.states) - 1)].facts)
    return source, source, middle, target, target


def _facts_by_entity(plan: TracePlan, facts: Mapping[str, str]) -> dict[str, list[str]]:
    subject_by_predicate = {item.predicate_id: item.subject for item in plan.predicates}
    result: dict[str, list[str]] = {item.entity_id: [] for item in plan.entities}
    for predicate_id, value in facts.items():
        subject = subject_by_predicate.get(predicate_id)
        if subject in result:
            result[subject].append(f"{predicate_id}={value}")
    return result


def _forbidden_categories(plan: TracePlan, control: Mapping[str, Any]) -> tuple[tuple[str, ...], str]:
    explicit = control.get("forbidden_new_categories")
    if explicit is not None:
        if not isinstance(explicit, list) or not all(isinstance(x, str) for x in explicit):
            raise ValueError("control.forbidden_new_categories must be a string array")
        requested = [item.strip().casefold() for item in explicit if item.strip()]
        provenance = "explicit"
    else:
        text = " ".join(
            [plan.global_semantic, *(x.mention for x in plan.entities), *plan.protected_predicates]
        ).casefold()
        requested = ["extra copy", "unlisted object", "external source"]
        if any(word in text for word in ("billiard", "domino", "fall", "ice", "melt", "balloon")):
            requested.extend(["person", "hand", "tool"])
        if "billiard" in text:
            requested.extend(["cue", "cue stick", "extra ball"])
        if any(word in text for word in ("ice", "melt", "butter")):
            requested.extend(["dropper", "pipette", "external liquid stream"])
        provenance = "derived"
    allowed = " ".join(item.mention.casefold() for item in plan.entities)
    filtered = sorted({item for item in requested if item and item not in allowed})
    return tuple(filtered), provenance


def _lifecycles(plan: TracePlan) -> tuple[LifecycleTransition, ...]:
    predicate_by_id = {item.predicate_id: item for item in plan.predicates}
    result: list[LifecycleTransition] = []
    for transition in plan.transitions:
        by_entity: dict[str, list[tuple[str, str]]] = {}
        for effect in transition.effects:
            predicate = predicate_by_id.get(effect.predicate)
            if predicate is None:
                continue
            by_entity.setdefault(predicate.subject, []).append(
                (effect.from_value, effect.to_value)
            )
        for entity_id, pairs in by_entity.items():
            result.append(
                LifecycleTransition(
                    entity_id=entity_id,
                    from_stage="setup",
                    to_stage="terminal",
                    from_state="; ".join(x for x, _ in pairs),
                    to_state="; ".join(y for _, y in pairs),
                    identity_preserved=True,
                    mutually_exclusive_states=tuple(pairs),
                    provenance="derived",
                )
            )
    return tuple(result)


def _conservation(plan: TracePlan, slots: tuple[EntitySlot, ...]) -> tuple[ConservationRule, ...]:
    slot_ids = tuple(item.entity_id for item in slots)
    rules: list[ConservationRule] = [
        ConservationRule(
            rule_id="entity_slots_constant",
            quantity="instance_count",
            sources=slot_ids,
            sinks=slot_ids,
            relation="constant",
            tolerance=0.0,
            observable_proxy="tracked logical entity slots remain consistent",
            provenance="derived",
        )
    ]
    effect_owner: dict[str, str] = {}
    effect_desc: dict[str, str] = {}
    for transition in plan.transitions:
        for effect in transition.effects:
            effect_owner[effect.effect_id] = effect.predicate
            effect_desc[effect.effect_id] = effect.direction
    for constraint in plan.constraints:
        if constraint.kind != "couples":
            continue
        effects = tuple(str(x) for x in constraint.arguments.get("effects", []))
        text = " ".join(effect_desc.get(x, x) for x in effects).casefold()
        quantity = "motion_transfer" if any(
            word in text for word in ("motion", "speed", "displacement", "velocity")
        ) else "material_proxy"
        predicates = tuple(effect_owner.get(x, x) for x in effects)
        split = max(1, len(predicates) // 2)
        rules.append(
            ConservationRule(
                rule_id=f"conservation_{constraint.constraint_id}",
                quantity=quantity,
                sources=predicates[:split],
                sinks=predicates[split:] or predicates[:split],
                relation="coupled_monotonic",
                tolerance=0.10,
                observable_proxy=str(constraint.arguments.get("relation", text)),
                provenance="derived",
            )
        )
    return tuple(rules)


def build_entity_ledger(
    plan: TracePlan,
    raw_plan: Mapping[str, Any],
) -> EntityLedger:
    control = raw_plan.get("control", {})
    if not isinstance(control, Mapping):
        raise ValueError("plan.control must be an object")
    issues: list[str] = []
    slots: list[EntitySlot] = []
    for entity in plan.entities:
        cardinality = _explicit_cardinality(control, entity.entity_id)
        if cardinality is None:
            cardinality, provenance = _derive_cardinality(
                entity.entity_id, entity.mention, plan.protected_predicates
            )
        else:
            provenance = "explicit"
        if cardinality.kind in {"group", "unknown"}:
            issues.append(
                f"{entity.entity_id}: exact visual count unresolved; retained as {cardinality.kind}"
            )
        slots.append(
            EntitySlot(
                entity_id=entity.entity_id,
                canonical_mention=entity.mention,
                role=entity.role,
                cardinality=cardinality,
                persistent_identity=True,
                mutable_attributes=tuple(
                    p.attribute for p in plan.predicates if p.subject == entity.entity_id
                ),
                invariant_attributes=tuple(plan.protected_predicates),
                aliases=(),
                provenance=provenance,  # type: ignore[arg-type]
            )
        )
    state_maps = _state_fact_maps(plan)
    stage_presence: list[StagePresence] = []
    stage_by_id = {stage.stage_id: stage for stage in plan.stages}
    for stage_id, state_facts in zip(STAGE_IDS, state_maps):
        per_entity = _facts_by_entity(plan, state_facts)
        for slot in slots:
            stage = stage_by_id[stage_id]
            affected = slot.role in stage.affected_roles
            if slot.role == "carrier" and not affected:
                visible = "forbidden"
                stage_count = Cardinality("exact", exact=0)
            elif stage_id in {"onset", "completion"} and affected:
                visible = "transitioning"
                stage_count = slot.cardinality
            else:
                visible = "required"
                stage_count = slot.cardinality
            stage_presence.append(
                StagePresence(
                    entity_id=slot.entity_id,
                    stage_id=stage_id,
                    logical_count=stage_count,
                    visible=visible,
                    state_facts=tuple(per_entity.get(slot.entity_id, ())),
                )
            )
    forbidden, provenance = _forbidden_categories(plan, control)
    empty_regions = control.get("empty_regions", [])
    if not isinstance(empty_regions, list) or not all(isinstance(x, str) for x in empty_regions):
        raise ValueError("control.empty_regions must be a string array")
    closed = ClosedWorldSpec(
        allowed_entity_ids=tuple(slot.entity_id for slot in slots),
        forbidden_new_categories=forbidden,
        empty_regions=tuple(x.strip() for x in empty_regions if x.strip()),
        external_source_allowed=bool(control.get("external_source_allowed", False)),
        provenance=provenance,  # type: ignore[arg-type]
    )
    slot_tuple = tuple(slots)
    return EntityLedger(
        slots=slot_tuple,
        stage_presence=tuple(stage_presence),
        closed_world=closed,
        lifecycles=_lifecycles(plan),
        conservation=_conservation(plan, slot_tuple),
        issues=tuple(issues),
    )
