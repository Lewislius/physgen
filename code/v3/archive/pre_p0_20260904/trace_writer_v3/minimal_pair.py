from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from v2.ace_router.trace_schema import Constraint, CausalStage, TracePlan, Violation

from .control_schema import ContextPair


@dataclass(frozen=True)
class PairCompileResult:
    pairs: tuple[ContextPair, ...]
    issues: tuple[str, ...]


def _effect_phrases(plan: TracePlan) -> dict[str, str]:
    result: dict[str, str] = {}
    for transition in plan.transitions:
        for effect in transition.effects:
            result[effect.effect_id] = effect.direction or (
                f"{effect.predicate} changes from {effect.from_value} to {effect.to_value}"
            )
    return result


def _reference_phrases(plan: TracePlan) -> dict[str, str]:
    result = _effect_phrases(plan)
    result.update({event.event_id: event.description for event in plan.events})
    for state in plan.states:
        result[state.state_id] = ", ".join(f"{k}={v}" for k, v in state.facts.items())
    return result


def _describe(refs: Mapping[str, str], value: object) -> str:
    text = str(value)
    return refs.get(text, text.replace("_", " "))


def _positive_relation(constraint: Constraint, refs: Mapping[str, str]) -> str:
    args = constraint.arguments
    if constraint.kind == "precedes":
        return f"{_describe(refs, args.get('before'))} happens BEFORE {_describe(refs, args.get('after'))}."
    if constraint.kind == "requires":
        return f"{_describe(refs, args.get('effect'))} starts ONLY AFTER {_describe(refs, args.get('cause'))}."
    if constraint.kind == "changes":
        return f"The declared change {_describe(refs, args.get('effect'))} occurs in its stated direction."
    if constraint.kind == "couples":
        effects = [_describe(refs, item) for item in args.get("effects", [])]
        return "The coupled changes occur together: " + " WHILE ".join(effects) + "."
    if constraint.kind == "persists":
        facts = ", ".join(str(x) for x in args.get("facts", []))
        return f"The terminal facts {facts} REMAIN through the final frames."
    if constraint.kind == "excludes":
        facts = ", ".join(str(x) for x in args.get("facts", []))
        return f"The same logical slot has only its valid state; mutually exclusive facts {facts} do not coexist."
    raise ValueError(f"unsupported constraint kind {constraint.kind!r}")


def _negative_relation(
    violation: Violation,
    constraint: Constraint,
    refs: Mapping[str, str],
) -> str:
    args = constraint.arguments
    kind = violation.kind
    if kind == "wrong_order" and constraint.kind == "precedes":
        return f"{_describe(refs, args.get('after'))} happens BEFORE {_describe(refs, args.get('before'))}."
    if kind in {"early_effect", "outcome_preset"}:
        cause = args.get("before", args.get("cause", "declared cause"))
        effect = args.get("after", args.get("effect", "declared effect"))
        return f"{_describe(refs, effect)} is already present BEFORE {_describe(refs, cause)}."
    if kind in {"trigger_missing", "precondition_missing"}:
        cause = args.get("cause", args.get("before", "declared cause"))
        effect = args.get("effect", args.get("after", "declared effect"))
        return f"{_describe(refs, effect)} occurs WITHOUT {_describe(refs, cause)}."
    if kind in {"effect_missing", "incomplete_transition"}:
        return "The declared cause occurs BUT the required target change does not become visible."
    if kind in {"wrong_direction", "overshoot", "envelope_broken"}:
        return "The same declared property changes in the opposite or out-of-range direction."
    if kind == "coupling_broken":
        effects = [_describe(refs, item) for item in args.get("effects", [])]
        return "Only one coupled change occurs while the other remains unchanged: " + "; ".join(effects) + "."
    if kind == "reversal":
        return "The valid terminal state returns to its mutually exclusive earlier state."
    if kind == "exclusion_broken":
        return "Copies of the same logical slot appear in mutually exclusive states at the same time."
    if kind == "initial_state_broken":
        return "The same entities begin outside the declared initial state."
    return violation.text


def _select_constraint(
    stage: CausalStage,
    violation: Violation,
    by_id: Mapping[str, Constraint],
) -> Constraint | None:
    for constraint_id in violation.constraint_ids:
        if constraint_id in by_id:
            return by_id[constraint_id]
    for constraint_id in stage.constraint_ids:
        if constraint_id in by_id:
            return by_id[constraint_id]
    return None


def compile_minimal_pairs(
    plan: TracePlan,
    stage_anchors: Mapping[str, str],
    *,
    strict: bool,
) -> PairCompileResult:
    by_id = {item.constraint_id: item for item in plan.constraints}
    refs = _reference_phrases(plan)
    issues: list[str] = []
    pairs: list[ContextPair] = []
    for stage in plan.stages:
        negative_relations: list[str] = []
        positive_candidates: list[str] = []
        quality = "minimal_verified"
        for violation in stage.violations:
            constraint = _select_constraint(stage, violation, by_id)
            if constraint is None:
                quality = "legacy_nonminimal"
                positive_candidates.append(stage.positive)
                negative_relations.append(violation.text)
                issues.append(f"{stage.stage_id}/{violation.kind}: no referenced constraint")
                continue
            try:
                positive_candidates.append(_positive_relation(constraint, refs))
                negative_relations.append(_negative_relation(violation, constraint, refs))
            except ValueError as exc:
                quality = "legacy_nonminimal"
                positive_candidates.append(stage.positive)
                negative_relations.append(violation.text)
                issues.append(f"{stage.stage_id}/{violation.kind}: {exc}")
        if not negative_relations:
            raise ValueError(f"stage {stage.stage_id} has no violations")
        if quality != "minimal_verified" and strict:
            raise ValueError(f"stage {stage.stage_id} cannot compile a strict minimal pair")
        positive_relation = " [AND] ".join(positive_candidates)
        shared = stage_anchors[stage.stage_id]
        positive_text = f"{shared} [VALID RELATION] {positive_relation}"
        violation_texts_list: list[str] = []
        for changed_index, changed_relation in enumerate(negative_relations):
            clauses = list(positive_candidates)
            clauses[changed_index] = changed_relation
            violation_texts_list.append(
                f"{shared} [VIOLATED RELATION] " + " [AND] ".join(clauses)
            )
        violation_texts = tuple(violation_texts_list)
        weight = 1.0 / len(violation_texts)
        pairs.append(
            ContextPair(
                stage_id=stage.stage_id,
                shared_anchor=shared,
                positive_relation=positive_relation,
                violation_relations=tuple(negative_relations),
                positive_text=positive_text,
                violation_texts=violation_texts,
                violation_kinds=tuple(item.kind for item in stage.violations),
                violation_constraint_ids=tuple(item.constraint_ids for item in stage.violations),
                violation_weights=tuple(weight for _ in violation_texts),
                pair_quality=quality,
            )
        )
    return PairCompileResult(tuple(pairs), tuple(issues))
