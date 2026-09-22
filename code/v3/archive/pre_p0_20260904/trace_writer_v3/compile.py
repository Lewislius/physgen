from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping

import torch

from v2.ace_router.trace_schema import FIXED_STAGE_IDS, TracePlan
from v2.ace_router.trace_validation import ValidationReport, validate_trace_plan

from .control_schema import (
    COMPILED_SCHEMA_VERSION,
    DESIGN_REVISION,
    ContextBundle,
    EntityLedger,
)
from .entity_ledger import build_entity_ledger
from .minimal_pair import compile_minimal_pairs
from .semantic_anchor import compile_semantic_anchor
from .temporal import compile_temporal_route


@dataclass(frozen=True)
class CompiledTraceV3:
    plan: TracePlan
    raw_plan: Mapping[str, Any]
    mode: str
    entity_ledger: EntityLedger
    contexts: ContextBundle
    temporal_weights: torch.Tensor
    temporal_manifest: Mapping[str, Any]
    validation: ValidationReport
    issues: tuple[str, ...]
    frames: int = 25

    @property
    def stages(self):
        return self.plan.stages

    def context_texts(self) -> dict[str, str]:
        return self.contexts.texts()

    def manifest(self) -> dict[str, Any]:
        raw_digest = hashlib.sha256(
            json.dumps(self.raw_plan, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return {
            "input_schema_version": self.plan.schema_version,
            "compiled_schema_version": COMPILED_SCHEMA_VERSION,
            "design_revision": DESIGN_REVISION,
            "sample_id": self.plan.sample_id,
            "mode": self.mode,
            "frames": self.frames,
            "stage_ids": list(FIXED_STAGE_IDS),
            "raw_plan_sha256": raw_digest,
            "temporal": dict(self.temporal_manifest),
            "entity_ledger": self.entity_ledger.manifest(),
            "contexts": self.contexts.manifest(),
            "issues": list(self.issues),
            "validation": {
                "errors": self.validation.error_count,
                "warnings": self.validation.warning_count,
                "issues": [asdict(item) for item in self.validation.issues],
            },
        }


def compile_trace_plan_v3(
    plan: TracePlan,
    raw_plan: Mapping[str, Any],
    *,
    mode: str,
    crossfade_tokens: int = 2,
    event_clock_mode: str = "plan_prior",
    minimal_pair_mode: str = "compat",
    emit_validation: bool = False,
) -> CompiledTraceV3:
    if mode not in {"i2v", "t2v"}:
        raise ValueError("mode must be i2v or t2v")
    if tuple(stage.stage_id for stage in plan.stages) != tuple(FIXED_STAGE_IDS):
        raise ValueError("TRACE v3 requires setup/onset/evolution/completion/terminal")
    if minimal_pair_mode not in {"compat", "strict"}:
        raise ValueError("minimal_pair_mode must be compat or strict")
    validation = validate_trace_plan(plan, emit=emit_validation)
    ledger = build_entity_ledger(plan, raw_plan)
    anchor = compile_semantic_anchor(plan, ledger, mode=mode)
    pair_result = compile_minimal_pairs(
        plan,
        anchor.stage_anchors,
        strict=minimal_pair_mode == "strict",
    )
    contexts = ContextBundle(
        global_anchor=anchor.global_anchor,
        stage_pairs=pair_result.pairs,
        protected_coverage=anchor.protected_coverage,
        token_budget_priority={"entity_count": 0, "protected": 1, "causal": 2, "style": 3},
        issues=tuple((*anchor.issues, *pair_result.issues)),
    )
    temporal, temporal_manifest = compile_temporal_route(
        plan,
        frames=25,
        crossfade_tokens=crossfade_tokens,
        mode=event_clock_mode,
    )
    issues = tuple((*ledger.issues, *contexts.issues))
    return CompiledTraceV3(
        plan=plan,
        raw_plan=raw_plan,
        mode=mode,
        entity_ledger=ledger,
        contexts=contexts,
        temporal_weights=temporal,
        temporal_manifest=temporal_manifest,
        validation=validation,
        issues=issues,
    )
