from __future__ import annotations

import json
from pathlib import Path
import unittest

import torch
import torch.nn as nn

from v2.ace_router.trace_schema import TracePlan

from trace_writer_v3.cap_ledger import CapLedger, cap_prediction_delta, masked_energy
from trace_writer_v3.compile import compile_trace_plan_v3
from trace_writer_v3.config import TraceWriterConfigV3
from trace_writer_v3.runtime import build_routing_runtime_v3
from trace_writer_v3.writer import build_trace_residual_v3


class MeanContextAttention(nn.Module):
    def cross_attn(self, query, context, context_lens):
        return context.mean(1, keepdim=True).expand(query.size(0), query.size(1), -1)


class SafeCapWriterTests(unittest.TestCase):
    def test_group_cap_is_cumulative(self) -> None:
        ledger = CapLedger(0.1)
        candidate = torch.ones(1, 4, 2)
        semantic = torch.ones_like(candidate) * 2
        support = torch.ones(1, 4, 1)
        first, _ = ledger.consume(candidate, semantic, support, layer_id=14)
        second, _ = ledger.consume(candidate, semantic, support, layer_id=15)
        applied = masked_energy(first, support) + masked_energy(second, support)
        reference = 0.01 * 2 * masked_energy(semantic, support)
        self.assertTrue(bool((applied <= reference + 1e-6).all().item()))

    def test_cfg_cap_inside_and_outside(self) -> None:
        delta = torch.ones(2, 3, 4, 4)
        reference = torch.ones_like(delta) * 2
        support = torch.zeros(1, 3, 4, 4)
        support[:, :, :, :2] = 1
        capped, record = cap_prediction_delta(
            delta,
            reference,
            support,
            inside_ratio=0.1,
            outside_ratio=0.01,
        )
        self.assertLess(torch.linalg.vector_norm(capped), torch.linalg.vector_norm(delta))
        self.assertLess(record["cfg_outside_coefficient"], record["cfg_inside_coefficient"])

    def test_writer_uses_positive_minus_negative(self) -> None:
        root = Path("/home/liuzhirui/Project/physGen/code/v1/demo/P02/P02-V2-planimg.json")
        raw = json.loads(root.read_text(encoding="utf-8"))
        compiled = compile_trace_plan_v3(TracePlan.from_dict(raw), raw, mode="i2v")
        contexts = {}
        for name in compiled.context_texts():
            value = 2.0 if name.endswith("positive") else 1.0
            contexts[name] = torch.full((1, 2, 4), value)
        config = TraceWriterConfigV3(
            dynamic_support_mode="off",
            token_cap_ratio=10,
            layer_cap_ratio=10,
            group_cap_ratio=10,
            temporal_derivative_ratio=0,
            lambda0=1,
        )
        runtime = build_routing_runtime_v3(
            compiled,
            contexts,
            height=1,
            width=1,
            seq_len=25,
            config=config,
        )
        query = torch.zeros(1, 25, 4)
        semantic = torch.ones_like(query)
        result = build_trace_residual_v3(
            MeanContextAttention(),
            query,
            semantic,
            runtime,
            context_lens=None,
            scale=1,
            step_index=0,
            layer_id=14,
        )
        setup_core = runtime.stages[0].core_indices[0].item()
        self.assertTrue(torch.allclose(result.candidate[0, setup_core], torch.ones(4)))


if __name__ == "__main__":
    unittest.main()
