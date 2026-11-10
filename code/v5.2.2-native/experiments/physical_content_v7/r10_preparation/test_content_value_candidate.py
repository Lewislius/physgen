"""Bounded CPU implementation checks; no checkpoints, models or videos loaded."""
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
R9 = ROOT / 'r1_content_reader_r9/snapshots/20261031T150803Z'
sys.path.insert(0, str(R9))
import bootstrap
import torch
from torch import nn
from torch.nn import functional as F
from physical_content.model import Enhancement as R9Enhancement, Fusion
from physical_content.evidence import resample_words
from content_value_candidate import ContentValueEnhancement, conditioning_choice

torch.set_num_threads(2)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim, self.num_heads, self.head_dim = 16, 2, 8
        for name in ['q', 'k', 'v', 'o']:
            setattr(self, name, nn.Linear(16, 16, bias=False))
        self.norm_q, self.norm_k = nn.RMSNorm(16), nn.RMSNorm(16)

    def forward(self, x, context):
        return x


class CandidateTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        self.cfg = dict(model=dict(qwen_width=32, width=16, q_heads=4,
            q_tokens=8, rounds=2, grounded_read_heads=2),
            train=dict(gradient_checkpointing=True))
        self.interface = dict(mean=torch.randn(32), scale=torch.rand(32)+.5,
            weight=torch.randn(16, 32)*.05, bias=torch.randn(16)*.01,
            target_scale=torch.tensor(.1))
        self.evidence = dict(words=torch.randn(6, 32), word_groups=torch.arange(6)//3,
            request_words=torch.randn(5, 32))

    def test_same_initial_C_and_RNG_preserve_existing_start(self):
        torch.manual_seed(42)
        original = R9Enhancement(self.cfg, self.interface)
        rng = torch.get_rng_state().clone()
        torch.manual_seed(42)
        new = ContentValueEnhancement(self.cfg, self.interface)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for name, parameter in original.named_parameters():
            torch.testing.assert_close(parameter, dict(new.named_parameters())[name], rtol=0, atol=0)
        torch.testing.assert_close(original(self.evidence)[0], new(self.evidence)[0], rtol=0, atol=0)
        self.assertEqual(new.shared_offset.numel(), self.cfg['model']['width'])

    def test_common_attention_queries_still_read_input_values(self):
        model = ContentValueEnhancement(self.cfg, self.interface)
        with torch.no_grad():
            model.block.cross.q.weight.zero_()
            model.block.cross.v.weight.copy_(torch.eye(32))
        self.assertIsNone(model.block.cross.v.bias)
        state = torch.ones(1, 8, 32)
        memory, words = torch.randn(1, 11, 32), torch.randn(1, 6, 32)
        groups = torch.arange(8)//4
        before = model.read_values(state, memory, words, groups, self.evidence['word_groups'])
        shift = torch.linspace(-.3, .4, 32).reshape(1, 1, 32)
        after = model.read_values(state, memory+shift, words+shift, groups, self.evidence['word_groups'])
        torch.testing.assert_close(after-before, shift.expand_as(before), rtol=1e-5, atol=1e-6)

    def test_anchor_control_disables_shared_and_input_corrections(self):
        model = ContentValueEnhancement(self.cfg, self.interface)
        with torch.no_grad():
            model.project.weight.normal_(std=.02)
            model.shared_offset.fill_(.3)
        words = (self.evidence['words']-model.anchor_mean)/model.anchor_scale
        anchor, _, _ = resample_words(words, self.evidence['word_groups'], 8)
        expected = F.linear(anchor, model.anchor_weight, model.anchor_bias)
        torch.testing.assert_close(model(self.evidence, anchor_only=True)[0], expected, rtol=0, atol=0)
        image_only = dict(self.evidence, request_words=torch.empty(0, 32))
        self.assertTrue(torch.isfinite(model(image_only)[0]).all())

    def test_real_fusion_gradient_reaches_input_selection(self):
        model = ContentValueEnhancement(self.cfg, self.interface)
        attention = Attention().requires_grad_(False)
        fusion = Fusion(attention)
        optimizer = torch.optim.SGD([*model.parameters(), *fusion.parameters()], lr=.2)
        x, context, target = torch.randn(1, 4, 16), torch.randn(1, 5, 16), torch.randn(1, 2, 16)
        norms = []
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            c, groups = model(self.evidence)
            y, _ = fusion(x, context, c, torch.tensor([[0., 1.]]), (2, 1, 2), attention, groups)
            F.mse_loss(y[:, 2:], target).backward()
            names = dict(output=fusion.output.weight, project=model.project.weight,
                offset=model.shared_offset, seed=model.seed, query=model.block.cross.q.weight)
            norms.append({name:float(p.grad.norm()) for name,p in names.items()})
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            optimizer.step()
        self.assertGreater(norms[0]['output'], 0.)
        for name in ['project', 'offset']:
            self.assertGreater(norms[1][name], 0.)
        for name in ['seed', 'query']:
            self.assertGreater(norms[2][name], 0.)

    def test_condition_exposure_preserves_original_draws(self):
        counts = dict(both_present=0, joint_null=0, qwen_present_native_null=0)
        for cursor in range(1600):
            old = torch.rand(2, generator=torch.Generator().manual_seed(42*1000003+cursor))
            chosen = conditioning_choice(42, cursor, joint_probability=.1, native_only_probability=.2)
            self.assertEqual(chosen['qwen_request_dropped'], bool(old[0]<.1))
            self.assertEqual(chosen['sigma'], float(.001+.998*old[1]))
            self.assertFalse(chosen['qwen_request_dropped'] and not chosen['native_text_dropped'])
            counts[chosen['mode']] += 1
        self.assertEqual(counts['joint_null'], 154)
        self.assertGreater(counts['qwen_present_native_null'], 0)
        self.assertGreater(counts['both_present'], counts['qwen_present_native_null'])


if __name__ == '__main__':
    unittest.main()
