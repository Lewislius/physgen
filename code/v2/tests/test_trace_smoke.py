from __future__ import annotations

import copy
import io
import json
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from ace_router.trace_compile import compile_trace_plan
from ace_router.trace_config import TraceWriterConfig
from ace_router.trace_masks import (
    build_stage_spatial_masks,
    fixed5_stage_spans,
    fixed5_temporal_weights,
)
from ace_router.trace_model_adapter import TraceModelProxy
from ace_router.trace_schema import load_trace_plan
from ace_router.trace_validation import validate_trace_plan
from ace_router.trace_writer import cap_global, cap_per_token


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "P01-trace-plan.json"


class _FrozenWan(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model_type = "ti2v"
        self.num_layers = 30
        self.text_len = 512
        self.blocks = nn.ModuleList([nn.Identity() for _ in range(30)])
        self.marker = nn.Parameter(torch.tensor(0.0), requires_grad=False)

    def forward(self, *, x, t, context, seq_len, y=None):
        return [item + 1 for item in x]


class TraceSmokeTests(unittest.TestCase):
    def test_fixed_five_schedules(self) -> None:
        expected = {1: (4, 3, 7, 4, 3), 2: (3, 2, 6, 3, 3)}
        for width, core_lengths in expected.items():
            spans = fixed5_stage_spans(crossfade_tokens=width)
            self.assertEqual(tuple(len(item["core"]) for item in spans), core_lengths)
            weights = fixed5_temporal_weights(crossfade_tokens=width)
            self.assertTrue(torch.equal(weights.sum(0), torch.ones(25)))
            self.assertLessEqual(int((weights > 0).sum(0).max()), 2)

    def test_plan_compiles_and_spatial_masks_build(self) -> None:
        plan = load_trace_plan(EXAMPLE)
        compiled = compile_trace_plan(plan, emit_validation=False)
        self.assertEqual(len(compiled.stages), 5)
        for stage in compiled.stages:
            self.assertAlmostEqual(
                sum(violation.weight for violation in stage.violations),
                1.0,
            )
        coverage = compiled.manifest()["constraint_coverage"]
        self.assertEqual(set(coverage), {item.constraint_id for item in plan.constraints})
        self.assertTrue(
            all(item["stages"] or item["violations"] for item in coverage.values())
        )
        masks = build_stage_spatial_masks(plan, height=22, width=40)
        self.assertEqual(tuple(masks.shape), (5, 22, 40))

    def test_validator_reports_but_does_not_select_fallback(self) -> None:
        raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))
        raw["stages"][1]["positive"] = raw["stages"][0]["positive"]
        stream = io.StringIO()
        report = validate_trace_plan(raw, stream=stream)
        self.assertTrue(report.has_errors)
        self.assertIn("no fallback was selected", stream.getvalue())
        self.assertFalse(hasattr(report, "effective_mode"))
        compile_trace_plan(load_trace_plan(EXAMPLE), emit_validation=False)

    def test_lambda_zero_uses_original_wan(self) -> None:
        self.assertEqual(TraceWriterConfig(lambda0=0.0).lambda0, 0.0)
        proxy = TraceModelProxy(_FrozenWan())
        sample = torch.zeros(2, 3)
        output = proxy(
            x=[sample],
            t=torch.tensor([1.0]),
            context=[torch.zeros(2, 4)],
            seq_len=25,
            lambda0=0.0,
        )
        self.assertTrue(torch.equal(output[0], sample + 1))

    def test_two_caps_are_bounded(self) -> None:
        semantic = torch.ones(1, 3, 4)
        candidate = torch.full_like(semantic, 100.0)
        token_capped, _ = cap_per_token(candidate, semantic, ratio=0.10)
        applied, _ = cap_global(token_capped, semantic, ratio=0.02)
        ratio = torch.linalg.vector_norm(applied) / torch.linalg.vector_norm(semantic)
        self.assertLessEqual(float(ratio), 0.02 + 1e-6)


if __name__ == "__main__":
    unittest.main()
