from __future__ import annotations

from pathlib import Path
import unittest

import torch

from ace_router.trace_compile import compile_trace_plan
from ace_router.trace_runtime import build_routing_runtime
from ace_router.trace_schema import load_trace_plan
from ace_router.trace_writer import build_trace_residual


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "P01-trace-plan.json"


class _ContextValueCrossAttention:
    def cross_attn(self, query, context, context_lens):
        del context_lens
        value = context.mean(dim=1, keepdim=True)
        return value.expand(query.size(0), query.size(1), query.size(2))


class TraceRuntimeWriterTests(unittest.TestCase):
    def _runtime(self, *, seq_len: int = 30):
        compiled = compile_trace_plan(load_trace_plan(EXAMPLE), emit_validation=False)
        contexts = {}
        for stage in compiled.stages:
            contexts[f"stage.{stage.stage_id}.positive"] = torch.full((1, 1, 2), 2.0)
            for index, _ in enumerate(stage.violations):
                contexts[f"stage.{stage.stage_id}.negative.{index}"] = torch.zeros(1, 1, 2)
        return build_routing_runtime(
            compiled,
            contexts,
            height=1,
            width=1,
            seq_len=seq_len,
            spatial_mode="time",
        )

    def test_sliced_writer_covers_valid_tokens_and_not_padding(self) -> None:
        runtime = self._runtime()
        query = torch.zeros(1, 30, 2)
        semantic = torch.full_like(query, 100.0)
        result = build_trace_residual(
            _ContextValueCrossAttention(),
            query,
            semantic,
            runtime,
            scale=1.0,
            token_cap_ratio=1.0,
            global_cap_ratio=1.0,
        )
        self.assertTrue(torch.equal(result.candidate[:, :25], torch.full((1, 25, 2), 2.0)))
        self.assertTrue(torch.equal(result.candidate[:, 25:], torch.zeros(1, 5, 2)))

    def test_active_indices_match_exact_temporal_support(self) -> None:
        runtime = self._runtime(seq_len=25)
        for stage_index, stage in enumerate(runtime.stages):
            expected = torch.nonzero(
                runtime.compiled.temporal_weights[stage_index] > 0,
                as_tuple=False,
            ).flatten()
            self.assertTrue(torch.equal(stage.active_indices.cpu(), expected.cpu()))


if __name__ == "__main__":
    unittest.main()
