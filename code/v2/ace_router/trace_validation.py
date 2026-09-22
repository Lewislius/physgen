from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Mapping, TextIO

from .trace_schema import (
    CONSTRAINT_TYPES,
    DESIGN_REVISION,
    FIXED_STAGE_IDS,
    SCHEMA_VERSION,
    SUPPORT_KINDS,
    TRANSITION_KINDS,
    VIOLATION_TYPES,
    TracePlan,
    TraceSchemaError,
)


STAGE_VIOLATIONS = {
    "setup": {"initial_state_broken", "outcome_preset", "precondition_missing"},
    "onset": {"early_effect", "trigger_missing", "wrong_order"},
    "evolution": {"effect_missing", "wrong_direction", "coupling_broken"},
    "completion": {"envelope_broken", "overshoot", "incomplete_transition"},
    "terminal": {"reversal", "exclusion_broken"},
}


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    code: str
    path: str
    message: str


@dataclass(frozen=True)
class ValidationReport:
    issues: tuple[ValidationIssue, ...]

    @property
    def error_count(self) -> int:
        return sum(issue.severity == "error" for issue in self.issues)

    @property
    def warning_count(self) -> int:
        return sum(issue.severity == "warning" for issue in self.issues)

    @property
    def has_errors(self) -> bool:
        return self.error_count > 0

    def emit(self, stream: TextIO | None = None) -> None:
        target = stream or sys.stderr
        if not self.issues:
            print("[TRACE VALIDATION] no issues found", file=target)
            return
        for issue in self.issues:
            print(
                f"[TRACE VALIDATION][{issue.severity.upper()}] "
                f"{issue.code} {issue.path}: {issue.message}",
                file=target,
            )
        print(
            f"[TRACE VALIDATION] errors={self.error_count} "
            f"warnings={self.warning_count}; no fallback was selected",
            file=target,
        )


class _Issues:
    def __init__(self) -> None:
        self.items: list[ValidationIssue] = []

    def error(self, code: str, path: str, message: str) -> None:
        self.items.append(ValidationIssue("error", code, path, message))

    def warning(self, code: str, path: str, message: str) -> None:
        self.items.append(ValidationIssue("warning", code, path, message))


def _ids(issues: _Issues, values: list[str], path: str) -> set[str]:
    seen: set[str] = set()
    for index, value in enumerate(values):
        if not value:
            issues.error("empty_id", f"{path}[{index}].id", "id is empty")
        elif value in seen:
            issues.error("duplicate_id", path, f"duplicate id {value!r}")
        seen.add(value)
    return seen


def _fact(value: str) -> tuple[str, str] | None:
    if "=" not in value:
        return None
    predicate, state = (part.strip() for part in value.split("=", 1))
    return (predicate, state) if predicate and state else None


def _validate_plan(plan: TracePlan, issues: _Issues) -> None:
    if plan.schema_version != SCHEMA_VERSION:
        issues.error("schema_version", "$.schema_version", f"expected {SCHEMA_VERSION!r}")
    if plan.design_revision != DESIGN_REVISION:
        issues.warning("design_revision", "$.design_revision", f"runtime uses {DESIGN_REVISION!r}")
    if not plan.sample_id:
        issues.error("empty_sample_id", "$.sample_id", "sample_id is empty")
    if not plan.global_semantic:
        issues.error("empty_global_semantic", "$.global_semantic", "text is empty")

    for index, message in enumerate(plan.assumptions):
        issues.warning("planner_assumption", f"$.assumptions[{index}]", message)
    for index, message in enumerate(plan.planner_warnings):
        issues.warning("planner_warning", f"$.warnings[{index}]", message)

    entity_ids = _ids(issues, [item.entity_id for item in plan.entities], "$.entities")
    roles = {item.role for item in plan.entities}
    predicate_ids = _ids(
        issues, [item.predicate_id for item in plan.predicates], "$.predicates"
    )
    predicate_by_id = {item.predicate_id: item for item in plan.predicates}
    for index, predicate in enumerate(plan.predicates):
        if predicate.subject not in entity_ids:
            issues.error("unknown_entity", f"$.predicates[{index}].subject", predicate.subject)
        if not predicate.observable or len(predicate.values) < 2:
            issues.error(
                "invalid_predicate",
                f"$.predicates[{index}]",
                "predicate needs an observable and at least two values",
            )

    event_ids = _ids(issues, [item.event_id for item in plan.events], "$.events")
    state_ids = _ids(issues, [item.state_id for item in plan.states], "$.states")
    state_by_id = {item.state_id: item for item in plan.states}
    for state_index, state in enumerate(plan.states):
        for predicate_id, value in state.facts.items():
            predicate = predicate_by_id.get(predicate_id)
            if predicate is None:
                issues.error("unknown_predicate", f"$.states[{state_index}].facts", predicate_id)
            elif value not in predicate.values:
                issues.error("invalid_state_value", f"$.states[{state_index}].facts", value)

    transition_ids = _ids(
        issues, [item.transition_id for item in plan.transitions], "$.transitions"
    )
    effects = [effect for transition in plan.transitions for effect in transition.effects]
    effect_ids = _ids(issues, [item.effect_id for item in effects], "$.transitions[*].effects")
    effect_owner = {
        effect.effect_id: transition.transition_id
        for transition in plan.transitions
        for effect in transition.effects
    }
    if not 1 <= len(plan.transitions) <= 2:
        issues.error("transition_count", "$.transitions", "expected one or two transitions")
    for index, transition in enumerate(plan.transitions):
        base = f"$.transitions[{index}]"
        if transition.kind not in TRANSITION_KINDS:
            issues.error("transition_kind", f"{base}.kind", transition.kind)
        if transition.source_state not in state_ids or transition.target_state not in state_ids:
            issues.error("unknown_state", base, "transition references an unknown state")
        if transition.kind == "triggered" and transition.cause_event not in event_ids:
            issues.error("missing_visible_cause", f"{base}.cause_event", "unknown event")
        if transition.support_kind not in SUPPORT_KINDS:
            issues.error("support_kind", f"{base}.support_kind", transition.support_kind)
        source = state_by_id.get(transition.source_state)
        target = state_by_id.get(transition.target_state)
        for precondition_index, precondition in enumerate(transition.preconditions):
            parsed = _fact(precondition)
            if parsed is None:
                issues.error("fact_syntax", f"{base}.preconditions[{precondition_index}]", precondition)
            elif source is not None and source.facts.get(parsed[0]) != parsed[1]:
                issues.error(
                    "precondition_state_mismatch",
                    f"{base}.preconditions[{precondition_index}]",
                    precondition,
                )
        if not transition.effects:
            issues.error("missing_effect", f"{base}.effects", "transition has no effects")
        for effect_index, effect in enumerate(transition.effects):
            path = f"{base}.effects[{effect_index}]"
            predicate = predicate_by_id.get(effect.predicate)
            if predicate is None:
                issues.error("unknown_predicate", f"{path}.predicate", effect.predicate)
                continue
            if source is not None and source.facts.get(effect.predicate) != effect.from_value:
                issues.error("source_state_mismatch", f"{path}.from", effect.from_value)
            if target is not None and target.facts.get(effect.predicate) != effect.to_value:
                issues.error("target_state_mismatch", f"{path}.to", effect.to_value)
            if effect.from_value == effect.to_value or not effect.direction:
                issues.error("invalid_effect", path, "effect needs a change and direction")
    for index in range(len(plan.transitions) - 1):
        if plan.transitions[index].target_state != plan.transitions[index + 1].source_state:
            issues.error("broken_transition_chain", "$.transitions", "transitions are not contiguous")

    constraint_ids = _ids(
        issues, [item.constraint_id for item in plan.constraints], "$.constraints"
    )
    constraint_kinds = {item.kind for item in plan.constraints}
    causal_ids = event_ids | effect_ids
    for index, constraint in enumerate(plan.constraints):
        base = f"$.constraints[{index}]"
        kind = constraint.kind
        args = constraint.arguments
        if kind not in CONSTRAINT_TYPES:
            issues.error("constraint_type", f"{base}.type", kind)
            continue
        if kind in {"requires", "precedes"}:
            keys = ("cause", "effect") if kind == "requires" else ("before", "after")
            values = [args.get(key) for key in keys]
            if any(not isinstance(value, str) or value not in causal_ids for value in values):
                issues.error("causal_reference", base, "unknown event/effect reference")
        elif kind == "changes":
            transition_id, effect_id = args.get("transition"), args.get("effect")
            if transition_id not in transition_ids or effect_owner.get(effect_id) != transition_id:
                issues.error("change_reference", base, "transition/effect reference mismatch")
        elif kind == "couples":
            coupled = args.get("effects")
            if not isinstance(coupled, list) or len(coupled) < 2 or any(
                effect_id not in effect_ids for effect_id in coupled
            ):
                issues.error("coupling_reference", base, "couples needs at least two known effects")
        elif kind == "persists":
            if args.get("state") not in state_ids or not args.get("facts"):
                issues.error("persistence_reference", base, "persists needs a known state and facts")
        elif kind == "excludes":
            facts = args.get("facts")
            if not isinstance(facts, list) or len(facts) < 2:
                issues.error("exclusion_reference", base, "excludes needs at least two facts")
    if any(item.kind == "triggered" for item in plan.transitions):
        for kind in ("requires", "precedes"):
            if kind not in constraint_kinds:
                issues.error("missing_causal_constraint", "$.constraints", kind)
    if not ({"changes", "couples"} & constraint_kinds):
        issues.error("missing_change_constraint", "$.constraints", "changes or couples is required")

    if tuple(stage.stage_id for stage in plan.stages) != FIXED_STAGE_IDS:
        issues.error("stage_order", "$.stages", f"expected {FIXED_STAGE_IDS}")
    covered_constraints: set[str] = set()
    previous_positive = ""
    for index, stage in enumerate(plan.stages):
        base = f"$.stages[{index}]"
        if not stage.positive or not stage.observable_check:
            issues.error("incomplete_stage", base, "positive/check is empty")
        if stage.positive.casefold() == previous_positive:
            issues.error("duplicate_stage_semantics", f"{base}.positive", "same as previous stage")
        previous_positive = stage.positive.casefold()
        if stage.support_kind not in SUPPORT_KINDS:
            issues.error("support_kind", f"{base}.support_kind", stage.support_kind)
        if not 1 <= len(stage.violations) <= 2:
            issues.error("violation_count", f"{base}.violations", "expected one or two violations")
        covered_constraints.update(stage.constraint_ids)
        for constraint_id in stage.constraint_ids:
            if constraint_id not in constraint_ids:
                issues.error("unknown_constraint", f"{base}.constraint_ids", constraint_id)
        allowed = STAGE_VIOLATIONS.get(stage.stage_id, set())
        for violation_index, violation in enumerate(stage.violations):
            path = f"{base}.violations[{violation_index}]"
            if violation.kind not in VIOLATION_TYPES or violation.kind not in allowed:
                issues.error("violation_type", f"{path}.type", violation.kind)
            if not violation.text or not violation.constraint_ids:
                issues.error("incomplete_violation", path, "text/constraint_ids is empty")
            covered_constraints.update(violation.constraint_ids)
            for constraint_id in violation.constraint_ids:
                if constraint_id not in constraint_ids:
                    issues.error("unknown_constraint", f"{path}.constraint_ids", constraint_id)
        for role in stage.affected_roles:
            if role not in roles:
                issues.warning("unknown_role", f"{base}.affected_roles", role)
    for constraint_id in sorted(constraint_ids - covered_constraints):
        issues.error("constraint_uncovered", "$.constraints", constraint_id)

    boxes = plan.grounding.get("entity_boxes", [])
    if not isinstance(boxes, list):
        issues.error("grounding_boxes", "$.grounding.entity_boxes", "must be an array")
    else:
        for index, item in enumerate(boxes):
            path = f"$.grounding.entity_boxes[{index}]"
            if not isinstance(item, Mapping) or item.get("entity_id") not in entity_ids:
                issues.error("grounding_entity", path, "unknown entity")
                continue
            box = item.get("box")
            if not isinstance(box, list) or len(box) != 4:
                issues.error("grounding_box", f"{path}.box", "expected [x1,y1,x2,y2]")
                continue
            try:
                x1, y1, x2, y2 = (float(value) for value in box)
            except (TypeError, ValueError):
                issues.error("grounding_box", f"{path}.box", "coordinates must be numeric")
                continue
            if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                issues.error("grounding_box", f"{path}.box", "coordinates must be normalized")


def validate_trace_plan(
    raw_or_plan: Mapping[str, Any] | TracePlan,
    *,
    emit: bool = True,
    stream: TextIO | None = None,
) -> ValidationReport:
    """Print plan problems without mutating the plan or selecting a fallback."""

    issues = _Issues()
    if isinstance(raw_or_plan, TracePlan):
        plan = raw_or_plan
    else:
        try:
            plan = TracePlan.from_dict(raw_or_plan)
        except TraceSchemaError as exc:
            issues.error("schema_parse", "$", str(exc))
            report = ValidationReport(tuple(issues.items))
            if emit:
                report.emit(stream)
            return report
    _validate_plan(plan, issues)
    report = ValidationReport(tuple(issues.items))
    if emit:
        report.emit(stream)
    return report
