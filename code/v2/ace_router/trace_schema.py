from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "trace-writer-v1"
DESIGN_REVISION = "fixed5-causal-r3"
FIXED_STAGE_IDS = ("setup", "onset", "evolution", "completion", "terminal")
TRANSITION_KINDS = {"triggered", "continuous"}
CONSTRAINT_TYPES = {
    "requires",
    "precedes",
    "changes",
    "persists",
    "excludes",
    "couples",
}
VIOLATION_TYPES = {
    "initial_state_broken",
    "outcome_preset",
    "precondition_missing",
    "early_effect",
    "trigger_missing",
    "wrong_order",
    "effect_missing",
    "wrong_direction",
    "coupling_broken",
    "envelope_broken",
    "overshoot",
    "incomplete_transition",
    "reversal",
    "exclusion_broken",
}
SUPPORT_KINDS = {
    "role_union",
    "motion_corridor",
    "contact_interface",
    "motion_contact_corridor",
    "source_stream_sink",
    "global",
}


class TraceSchemaError(ValueError):
    """Raised only when data cannot be represented as a TracePlan."""


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TraceSchemaError(f"{path} must be an object")
    return value


def _sequence(value: Any, path: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise TraceSchemaError(f"{path} must be an array")
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str):
        raise TraceSchemaError(f"{path} must be a string")
    return value.strip()


def _strings(value: Any, path: str) -> tuple[str, ...]:
    return tuple(
        _string(item, f"{path}[{index}]")
        for index, item in enumerate(_sequence(value, path))
    )


@dataclass(frozen=True)
class Entity:
    entity_id: str
    mention: str
    role: str


@dataclass(frozen=True)
class Predicate:
    predicate_id: str
    subject: str
    attribute: str
    values: tuple[str, ...]
    observable: str


@dataclass(frozen=True)
class State:
    state_id: str
    facts: Mapping[str, str]


@dataclass(frozen=True)
class Event:
    event_id: str
    description: str
    observable: str


@dataclass(frozen=True)
class Effect:
    effect_id: str
    predicate: str
    from_value: str
    to_value: str
    direction: str


@dataclass(frozen=True)
class Transition:
    transition_id: str
    kind: str
    source_state: str
    target_state: str
    cause_event: str | None
    preconditions: tuple[str, ...]
    effects: tuple[Effect, ...]
    support_kind: str


@dataclass(frozen=True)
class Constraint:
    constraint_id: str
    kind: str
    arguments: Mapping[str, Any]


@dataclass(frozen=True)
class Violation:
    kind: str
    text: str
    constraint_ids: tuple[str, ...]


@dataclass(frozen=True)
class CausalStage:
    stage_id: str
    positive: str
    violations: tuple[Violation, ...]
    observable_check: str
    affected_roles: tuple[str, ...]
    support_kind: str
    constraint_ids: tuple[str, ...]


@dataclass(frozen=True)
class TracePlan:
    schema_version: str
    design_revision: str
    sample_id: str
    global_semantic: str
    entities: tuple[Entity, ...]
    predicates: tuple[Predicate, ...]
    events: tuple[Event, ...]
    states: tuple[State, ...]
    transitions: tuple[Transition, ...]
    constraints: tuple[Constraint, ...]
    stages: tuple[CausalStage, ...]
    protected_predicates: tuple[str, ...]
    grounding: Mapping[str, Any]
    assumptions: tuple[str, ...]
    planner_warnings: tuple[str, ...]

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "TracePlan":
        root = _mapping(raw, "$")

        entities = []
        for index, item in enumerate(_sequence(root.get("entities"), "$.entities")):
            data = _mapping(item, f"$.entities[{index}]")
            entities.append(
                Entity(
                    entity_id=_string(data.get("id"), f"$.entities[{index}].id"),
                    mention=_string(data.get("mention"), f"$.entities[{index}].mention"),
                    role=_string(data.get("role"), f"$.entities[{index}].role"),
                )
            )

        predicates = []
        for index, item in enumerate(_sequence(root.get("predicates"), "$.predicates")):
            data = _mapping(item, f"$.predicates[{index}]")
            predicates.append(
                Predicate(
                    predicate_id=_string(data.get("id"), f"$.predicates[{index}].id"),
                    subject=_string(data.get("subject"), f"$.predicates[{index}].subject"),
                    attribute=_string(data.get("attribute"), f"$.predicates[{index}].attribute"),
                    values=_strings(data.get("values"), f"$.predicates[{index}].values"),
                    observable=_string(data.get("observable"), f"$.predicates[{index}].observable"),
                )
            )

        events = []
        for index, item in enumerate(_sequence(root.get("events", []), "$.events")):
            data = _mapping(item, f"$.events[{index}]")
            events.append(
                Event(
                    event_id=_string(data.get("id"), f"$.events[{index}].id"),
                    description=_string(data.get("description"), f"$.events[{index}].description"),
                    observable=_string(data.get("observable"), f"$.events[{index}].observable"),
                )
            )

        states = []
        for index, item in enumerate(_sequence(root.get("states"), "$.states")):
            data = _mapping(item, f"$.states[{index}]")
            facts_raw = _mapping(data.get("facts"), f"$.states[{index}].facts")
            facts = {
                _string(key, f"$.states[{index}].facts key"): _string(
                    value, f"$.states[{index}].facts.{key}"
                )
                for key, value in facts_raw.items()
            }
            states.append(
                State(
                    state_id=_string(data.get("id"), f"$.states[{index}].id"),
                    facts=facts,
                )
            )

        transitions = []
        for index, item in enumerate(_sequence(root.get("transitions"), "$.transitions")):
            data = _mapping(item, f"$.transitions[{index}]")
            effects = []
            for effect_index, effect_item in enumerate(
                _sequence(data.get("effects"), f"$.transitions[{index}].effects")
            ):
                effect = _mapping(
                    effect_item,
                    f"$.transitions[{index}].effects[{effect_index}]",
                )
                base = f"$.transitions[{index}].effects[{effect_index}]"
                effects.append(
                    Effect(
                        effect_id=_string(effect.get("id"), f"{base}.id"),
                        predicate=_string(effect.get("predicate"), f"{base}.predicate"),
                        from_value=_string(effect.get("from"), f"{base}.from"),
                        to_value=_string(effect.get("to"), f"{base}.to"),
                        direction=_string(effect.get("direction"), f"{base}.direction"),
                    )
                )
            cause_raw = data.get("cause_event")
            transitions.append(
                Transition(
                    transition_id=_string(data.get("id"), f"$.transitions[{index}].id"),
                    kind=_string(data.get("kind"), f"$.transitions[{index}].kind"),
                    source_state=_string(data.get("from"), f"$.transitions[{index}].from"),
                    target_state=_string(data.get("to"), f"$.transitions[{index}].to"),
                    cause_event=(
                        None
                        if cause_raw is None
                        else _string(cause_raw, f"$.transitions[{index}].cause_event")
                    ),
                    preconditions=_strings(
                        data.get("preconditions", []),
                        f"$.transitions[{index}].preconditions",
                    ),
                    effects=tuple(effects),
                    support_kind=_string(
                        data.get("support_kind", "global"),
                        f"$.transitions[{index}].support_kind",
                    ),
                )
            )

        constraints = []
        for index, item in enumerate(_sequence(root.get("constraints"), "$.constraints")):
            data = dict(_mapping(item, f"$.constraints[{index}]"))
            constraint_id = _string(data.pop("id", None), f"$.constraints[{index}].id")
            kind = _string(data.pop("type", None), f"$.constraints[{index}].type")
            data.pop("source_refs", None)
            constraints.append(Constraint(constraint_id, kind, data))

        stages = []
        for index, item in enumerate(_sequence(root.get("stages"), "$.stages")):
            data = _mapping(item, f"$.stages[{index}]")
            violations = []
            for violation_index, violation_item in enumerate(
                _sequence(data.get("violations"), f"$.stages[{index}].violations")
            ):
                violation = _mapping(
                    violation_item,
                    f"$.stages[{index}].violations[{violation_index}]",
                )
                base = f"$.stages[{index}].violations[{violation_index}]"
                violations.append(
                    Violation(
                        kind=_string(violation.get("type"), f"{base}.type"),
                        text=_string(violation.get("text"), f"{base}.text"),
                        constraint_ids=_strings(
                            violation.get("constraint_ids", []),
                            f"{base}.constraint_ids",
                        ),
                    )
                )
            stages.append(
                CausalStage(
                    stage_id=_string(data.get("id"), f"$.stages[{index}].id"),
                    positive=_string(data.get("positive"), f"$.stages[{index}].positive"),
                    violations=tuple(violations),
                    observable_check=_string(
                        data.get("observable_check"),
                        f"$.stages[{index}].observable_check",
                    ),
                    affected_roles=_strings(
                        data.get("affected_roles", []),
                        f"$.stages[{index}].affected_roles",
                    ),
                    support_kind=_string(
                        data.get("support_kind", "global"),
                        f"$.stages[{index}].support_kind",
                    ),
                    constraint_ids=_strings(
                        data.get("constraint_ids", []),
                        f"$.stages[{index}].constraint_ids",
                    ),
                )
            )

        grounding = dict(_mapping(root.get("grounding", {}), "$.grounding"))
        return cls(
            schema_version=_string(root.get("schema_version"), "$.schema_version"),
            design_revision=_string(root.get("design_revision"), "$.design_revision"),
            sample_id=_string(root.get("sample_id"), "$.sample_id"),
            global_semantic=_string(root.get("global_semantic"), "$.global_semantic"),
            entities=tuple(entities),
            predicates=tuple(predicates),
            events=tuple(events),
            states=tuple(states),
            transitions=tuple(transitions),
            constraints=tuple(constraints),
            stages=tuple(stages),
            protected_predicates=_strings(
                root.get("protected_predicates", []), "$.protected_predicates"
            ),
            grounding=grounding,
            assumptions=_strings(root.get("assumptions", []), "$.assumptions"),
            planner_warnings=_strings(root.get("warnings", []), "$.warnings"),
        )


def load_trace_plan(path: str | Path) -> TracePlan:
    source = Path(path).expanduser().resolve()
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TraceSchemaError(f"failed to load trace plan {source}: {exc}") from exc
    return TracePlan.from_dict(_mapping(raw, "$"))
