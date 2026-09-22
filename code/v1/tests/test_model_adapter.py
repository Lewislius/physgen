from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from ace_router.model_adapter import (
    AceModelProxy,
    ace_block_forward,
    cap_residual_by_semantic_norm,
)


class _ZeroSelfAttention(nn.Module):
    def forward(self, x, seq_lens, grid_sizes, freqs):
        return torch.zeros_like(x)


class _ContextMeanCrossAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, x, context, context_lens):
        self.calls += 1
        return torch.ones_like(x) * context.mean()


class _CountingZeroFFN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return torch.zeros_like(x)


class _FakeBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.modulation = nn.Parameter(torch.zeros(1, 6, dim))
        self.norm1 = nn.Identity()
        self.norm2 = nn.Identity()
        self.norm3 = nn.Identity()
        self.self_attn = _ZeroSelfAttention()
        self.cross_attn = _ContextMeanCrossAttention()
        self.ffn = _CountingZeroFFN()


class _FastPathBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model_type = "ti2v"
        self.num_layers = 30
        self.text_len = 512
        self.blocks = nn.ModuleList([nn.Identity() for _ in range(30)])
        self.marker = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.calls = 0

    def forward(self, x, t, context, seq_len, y=None):
        self.calls += 1
        return [x[0] + 7]


class ModelAdapterTests(unittest.TestCase):
    def test_block_formula_and_call_counts(self) -> None:
        dim = 4
        block = _FakeBlock(dim)
        x = torch.zeros(1, 2, dim)
        e = torch.zeros(1, 2, 6, dim, dtype=torch.float32)
        output = ace_block_forward(
            block,
            x,
            e=e,
            seq_lens=torch.tensor([2]),
            grid_sizes=torch.tensor([[1, 1, 2]]),
            freqs=torch.zeros(1),
            semantic_context=torch.ones(1, 3, dim),
            positive_context=torch.full((1, 3, dim), 3.0),
            counterfactual_context=torch.full((1, 3, dim), 2.0),
            residual_mode="positive_minus_counterfactual",
            context_lens=None,
            layer_id=14,
            layer_gate=1.0,
            step_gate=1.0,
            lambda0=0.5,
            step_index=0,
            total_steps=50,
            diagnostic_sink=None,
            active_layer_ids=(14,),
        )
        self.assertTrue(torch.allclose(output, torch.full_like(output, 1.5)))
        self.assertEqual(block.cross_attn.calls, 3)
        self.assertEqual(block.ffn.calls, 1)

    def test_positive_only_uses_semantic_as_residual_reference(self) -> None:
        dim = 4
        block = _FakeBlock(dim)
        output = ace_block_forward(
            block,
            torch.zeros(1, 2, dim),
            e=torch.zeros(1, 2, 6, dim, dtype=torch.float32),
            seq_lens=torch.tensor([2]),
            grid_sizes=torch.tensor([[1, 1, 2]]),
            freqs=torch.zeros(1),
            semantic_context=torch.ones(1, 3, dim),
            positive_context=torch.full((1, 3, dim), 3.0),
            counterfactual_context=None,
            residual_mode="positive_only",
            context_lens=None,
            layer_id=14,
            layer_gate=1.0,
            step_gate=1.0,
            lambda0=0.5,
            step_index=0,
            total_steps=50,
            diagnostic_sink=None,
            active_layer_ids=(14,),
        )
        self.assertTrue(torch.allclose(output, torch.full_like(output, 2.0)))
        self.assertEqual(block.cross_attn.calls, 2)
        self.assertEqual(block.ffn.calls, 1)

    def test_relative_cap_limits_block_residual_without_amplifying(self) -> None:
        dim = 4
        block = _FakeBlock(dim)
        output = ace_block_forward(
            block,
            torch.zeros(1, 2, dim),
            e=torch.zeros(1, 2, 6, dim, dtype=torch.float32),
            seq_lens=torch.tensor([2]),
            grid_sizes=torch.tensor([[1, 1, 2]]),
            freqs=torch.zeros(1),
            semantic_context=torch.ones(1, 3, dim),
            positive_context=torch.full((1, 3, dim), 3.0),
            counterfactual_context=torch.full((1, 3, dim), 2.0),
            residual_mode="positive_minus_counterfactual",
            context_lens=None,
            layer_id=14,
            layer_gate=1.0,
            step_gate=1.0,
            lambda0=0.5,
            step_index=0,
            total_steps=50,
            diagnostic_sink=None,
            active_layer_ids=(14,),
            residual_cap_ratio=0.1,
        )
        self.assertTrue(torch.allclose(output, torch.full_like(output, 1.1)))

        weak = torch.full((2, 3, 4), 0.01)
        semantic = torch.ones_like(weak)
        applied, coefficient = cap_residual_by_semantic_norm(
            weak, semantic, 0.1
        )
        self.assertTrue(torch.equal(applied, weak))
        self.assertTrue(torch.equal(coefficient, torch.ones(2, 1, 1)))

    def test_relative_cap_is_per_sample(self) -> None:
        semantic = torch.ones(2, 2, 2)
        candidate = torch.stack(
            [torch.ones(2, 2), torch.full((2, 2), 0.01)]
        )
        applied, coefficient = cap_residual_by_semantic_norm(
            candidate, semantic, 0.1
        )
        self.assertAlmostEqual(float(coefficient[0].item()), 0.1, places=6)
        self.assertAlmostEqual(float(coefficient[1].item()), 1.0, places=6)
        ratios = torch.linalg.vector_norm(applied, dim=(1, 2)) / torch.linalg.vector_norm(
            semantic, dim=(1, 2)
        )
        self.assertLessEqual(float(ratios.max().item()), 0.1 + 1e-6)

    def test_proxy_lambda_zero_calls_original_forward(self) -> None:
        base = _FastPathBase()
        proxy = AceModelProxy(base)
        x = torch.zeros(1)
        output = proxy(
            [x],
            t=torch.zeros(1),
            context=[torch.zeros(1, 1)],
            seq_len=1,
            lambda0=0.0,
            layer_gates=[1.0] * 30,
            step_gate=1.0,
        )
        self.assertEqual(base.calls, 1)
        self.assertTrue(torch.equal(output[0], x + 7))

    def test_proxy_rejects_missing_active_contexts(self) -> None:
        proxy = AceModelProxy(_FastPathBase())
        with self.assertRaises(ValueError):
            proxy(
                [torch.zeros(1)],
                t=torch.zeros(1),
                context=[torch.zeros(1, 1)],
                seq_len=1,
                lambda0=0.5,
                layer_gates=[1.0] * 30,
                step_gate=1.0,
            )

    def test_proxy_positive_only_does_not_require_counterfactual(self) -> None:
        proxy = AceModelProxy(_FastPathBase())
        marker = [torch.ones(1)]
        with patch(
            "ace_router.model_adapter.ace_model_forward", return_value=marker
        ) as mocked_forward:
            output = proxy(
                [torch.zeros(1)],
                t=torch.zeros(1),
                context=[torch.zeros(1, 1)],
                seq_len=1,
                causal_pos_context=[torch.ones(1, 1)],
                residual_mode="positive_only",
                lambda0=0.5,
                layer_gates=[1.0] * 30,
                step_gate=1.0,
            )
        self.assertIs(output, marker)
        self.assertEqual(mocked_forward.call_args.kwargs["residual_mode"], "positive_only")
        self.assertIsNone(mocked_forward.call_args.kwargs["causal_neg_context"])

    def test_proxy_validates_gates_even_on_fast_path(self) -> None:
        proxy = AceModelProxy(_FastPathBase())
        common = {
            "x": [torch.zeros(1)],
            "t": torch.zeros(1),
            "context": [torch.zeros(1, 1)],
            "seq_len": 1,
            "lambda0": 0.0,
        }
        with self.assertRaises(ValueError):
            proxy(layer_gates=[0.0] * 29, **common)
        with self.assertRaises(ValueError):
            proxy(layer_gates=[0.0] * 29 + [float("nan")], **common)
        with self.assertRaises(ValueError):
            proxy(layer_gates=[0.0] * 29 + [-0.1], **common)

    def test_proxy_zero_step_and_zero_layers_use_original_forward(self) -> None:
        base = _FastPathBase()
        proxy = AceModelProxy(base)
        common = {
            "x": [torch.zeros(1)],
            "t": torch.zeros(1),
            "context": [torch.zeros(1, 1)],
            "seq_len": 1,
            "lambda0": 0.5,
        }
        proxy(layer_gates=[1.0] * 30, step_gate=0.0, **common)
        proxy(layer_gates=[0.0] * 30, step_gate=1.0, **common)
        self.assertEqual(base.calls, 2)


if __name__ == "__main__":
    unittest.main()
