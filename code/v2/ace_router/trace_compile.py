from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch

from .trace_masks import fixed5_stage_spans, fixed5_temporal_weights
from .trace_schema import FIXED_STAGE_IDS, CausalStage, TracePlan, Violation
from .trace_validation import ValidationReport, validate_trace_plan


class TraceCompileError(ValueError):
    """Compilation stopped because executable structure is missing; no fallback is used."""


@dataclass(frozen=True)
class CompiledViolation:
    kind: str
    text: str
    constraint_ids: tuple[str, ...]
    weight: float


@dataclass(frozen=True)
class CompiledStage:
    stage_id: str
    positive: str
    violations: tuple[CompiledViolation, ...]
    observable_check: str
    affected_roles: tuple[str, ...]
    support_kind: str
    constraint_ids: tuple[str, ...]


@dataclass(frozen=True)
class CompiledTrace:
    plan: TracePlan
    stages: tuple[CompiledStage, ...]
    frames: int
    crossfade_tokens: int
    temporal_weights: torch.Tensor
    validation: ValidationReport

    def context_texts(self) -> Mapping[str, str]:
        texts: dict[str, str] = {"global_semantic": self.plan.global_semantic}
        for stage in self.stages:
            texts[f"stage.{stage.stage_id}.positive"] = stage.positive
            for index, violation in enumerate(stage.violations):
                texts[f"stage.{stage.stage_id}.negative.{index}"] = violation.text
        return texts

    def manifest(self) -> dict[str, object]:
        stage_spans = fixed5_stage_spans(
            frames=self.frames,
            crossfade_tokens=self.crossfade_tokens,
        )
        weights = [
            [round(float(value), 6) for value in row]
            for row in self.temporal_weights.detach().cpu()
        ]
        constraint_coverage: dict[str, dict[str, list[str]]] = {
            constraint.constraint_id: {"stages": [], "violations": []}
            for constraint in self.plan.constraints
        }
        for stage in self.stages:
            for constraint_id in stage.constraint_ids:
                if constraint_id in constraint_coverage:
                    constraint_coverage[constraint_id]["stages"].append(stage.stage_id)
            for violation in stage.violations:
                for constraint_id in violation.constraint_ids:
                    if constraint_id in constraint_coverage:
                        constraint_coverage[constraint_id]["violations"].append(
                            f"{stage.stage_id}:{violation.kind}"
                        )
        return {
            "schema_version": self.plan.schema_version,
            "design_revision": self.plan.design_revision,
            "sample_id": self.plan.sample_id,
            "frames": self.frames,
            "crossfade_tokens": self.crossfade_tokens,
            "stage_ids": list(FIXED_STAGE_IDS),
            "stage_spans": list(stage_spans),
            "temporal_weights": weights,
            "constraint_coverage": constraint_coverage,
            "validation": {
                "errors": self.validation.error_count,
                "warnings": self.validation.warning_count,
                "fallback_selected": False,
                "issues": [
                    {
                        "severity": issue.severity,
                        "code": issue.code,
                        "path": issue.path,
                        "message": issue.message,
                    }
                    for issue in self.validation.issues
                ],
            },
            "stages": [
                {
                    "id": stage.stage_id,
                    "positive": stage.positive,
                    "observable_check": stage.observable_check,
                    "affected_roles": list(stage.affected_roles),
                    "support_kind": stage.support_kind,
                    "constraint_ids": list(stage.constraint_ids),
                    "violations": [
                        {
                            "type": violation.kind,
                            "text": violation.text,
                            "constraint_ids": list(violation.constraint_ids),
                            "weight": violation.weight,
                        }
                        for violation in stage.violations
                    ],
                }
                for stage in self.stages
            ],
        }

    def write_manifest(self, path: str | Path) -> Path:
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.manifest(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return target


def _compile_stage(stage: CausalStage) -> CompiledStage:
    if not stage.violations:
        raise TraceCompileError(
            f"stage {stage.stage_id!r} has no counterfactual; no fallback was selected"
        )
    weight = 1.0 / len(stage.violations)
    violations = tuple(
        CompiledViolation(
            kind=violation.kind,
            text=violation.text,
            constraint_ids=violation.constraint_ids,
            weight=weight,
        )
        for violation in stage.violations
    )
    return CompiledStage(
        stage_id=stage.stage_id,
        positive=stage.positive,
        violations=violations,
        observable_check=stage.observable_check,
        affected_roles=stage.affected_roles,
        support_kind=stage.support_kind,
        constraint_ids=stage.constraint_ids,
    )


def compile_trace_plan(
    plan: TracePlan,
    *,
    frames: int = 25,
    crossfade_tokens: int = 1,
    emit_validation: bool = True,
) -> CompiledTrace:
    """Compile a plan while reporting issues and never changing router mode."""

    report = validate_trace_plan(plan, emit=emit_validation)
    stage_ids = tuple(stage.stage_id for stage in plan.stages)
    if stage_ids != FIXED_STAGE_IDS:
        raise TraceCompileError(
            f"cannot execute stage order {stage_ids}; expected {FIXED_STAGE_IDS}; "
            "no fallback was selected"
        )
    stages = tuple(_compile_stage(stage) for stage in plan.stages)
    weights = fixed5_temporal_weights(
        frames=frames,
        crossfade_tokens=crossfade_tokens,
    )
    return CompiledTrace(
        plan=plan,
        stages=stages,
        frames=frames,
        crossfade_tokens=crossfade_tokens,
        temporal_weights=weights,
        validation=report,
    )
