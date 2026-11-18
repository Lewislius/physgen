"""B400 joint-read equivalence and R16Gt T2V time/write contracts."""
import math
import sys
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bootstrap  # noqa: F401

from physical_content.legacy_joint_read import read_physical
from physical_content.model import Fusion


class RMS(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * self.weight


class FakeAttention(nn.Module):
    def __init__(self, width=8, heads=2):
        super().__init__()
        self.dim, self.num_heads, self.head_dim = width, heads, width // heads
        self.q, self.k, self.v, self.o = (nn.Linear(width, width) for _ in range(4))
        self.norm_q, self.norm_k = RMS(width), RMS(width)

    def forward(self, x, context=None):
        src = x if context is None else context
        split = lambda value: value.reshape(value.shape[0], -1,
                                            self.num_heads, self.head_dim).transpose(1, 2)
        q, k, v = split(self.norm_q(self.q(x))), split(self.norm_k(self.k(src))), split(self.v(src))
        return self.o(F.scaled_dot_product_attention(q, k, v)
                      .transpose(1, 2).flatten(2))


class Recorder(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width
        self.last = None

    def forward(self, value):
        self.last = value
        return torch.zeros(value.shape[0], value.shape[1], self.width)


def reference_joint(q, k, v, request_keys, reference, prior, relation_count):
    scale = math.sqrt(q.shape[-1])
    binding = ((request_keys.float() @ k.float().transpose(-1, -2)) / scale
               + prior[:-1]).softmax(-1)
    bound = binding.to(v) @ v
    bound_prior = torch.full((request_keys.shape[-2],),
                             math.log(relation_count) - math.log(request_keys.shape[-2]))
    keys = torch.cat((k, request_keys, reference), -2)
    values = torch.cat((v, bound, torch.zeros_like(reference)), -2)
    physical_prior = torch.cat((prior[:-1], bound_prior), -1) - math.log(2.)
    logits = q.float() @ keys.float().transpose(-1, -2) / scale
    logits = logits + torch.cat((physical_prior, prior[-1:]), -1)
    return logits.softmax(-1).to(values) @ values


def identity_rope(q, grid, freqs):
    return q


class Contracts(unittest.TestCase):
    def test_joint_formula_matches_explicit_reference(self):
        torch.manual_seed(7)
        batch, heads, dim = 1, 2, 4
        q = torch.randn(batch, heads, 8, dim)
        k = torch.randn(batch, heads, 6, dim)
        v = torch.randn(batch, heads, 6, dim)
        request = torch.randn(batch, heads, 3, dim)
        reference = torch.randn(batch, heads, 1, dim)
        prior = torch.tensor([-.4, -.2, -.1, -.3, -.5, -.6, 0.])
        got = read_physical(q, k, v, request, reference, prior, 'joint',
                            relation_count=4)
        expected = reference_joint(q, k, v, request, reference, prior, 4)
        self.assertTrue(torch.allclose(got, expected, atol=1e-6, rtol=1e-5))

    def test_t2v_uses_every_time_point(self):
        width = 8
        fusion = Fusion(FakeAttention(width), FakeAttention(width),
                        response_mode='triple_global_pure_local')
        fusion.time = Recorder(width)
        x = torch.randn(1, 8, width)
        video_state = torch.randn(1, 8, width)
        context = torch.randn(1, 5, width)
        content = dict(routing=torch.randn(1, 6, width),
                       execution=torch.randn(1, 6, width))
        groups = torch.tensor([0, 0, 1, 1, 2, 2])
        times = torch.tensor([[0., 0.5]])
        geometry = (2, 2, 2)
        fusion(x, video_state, content, times, geometry,
               FakeAttention(width), groups, torch.tensor([[2, 2, 2]]), None,
               identity_rope, context, 3, nn.Linear(width, width, bias=False))
        self.assertEqual(tuple(fusion.time.last.shape), (1, 2, 256))

    def test_zero_init_writes_nothing_and_t2v_writes_every_position(self):
        width = 8
        attention = FakeAttention(width)
        fusion = Fusion(attention, FakeAttention(width),
                        response_mode='triple_global_pure_local')
        x = torch.randn(1, 8, width)
        video_state = torch.randn(1, 8, width)
        context = torch.randn(1, 5, width)
        content = dict(routing=torch.randn(1, 6, width),
                       execution=torch.randn(1, 6, width))
        groups = torch.tensor([0, 0, 1, 1, 2, 2])
        times = torch.tensor([[0., 0.5]])
        geometry = (2, 2, 2)
        correction = nn.Linear(width, width, bias=False)
        zero = fusion(x, video_state, content, times, geometry, attention, groups,
                      torch.tensor([[2, 2, 2]]), None, identity_rope, context, 3,
                      correction)
        self.assertEqual(float(zero.detach().abs().max()), 0.)
        with torch.no_grad():
            fusion.output.weight.normal_(std=.1)
            fusion.condition_query.weight.normal_(std=.1)
            fusion.time.weight.normal_(std=.1)
            for name in ('direct_output', 'request_output', 'joint_output',
                         'direct_state_value', 'request_state_value',
                         'joint_state_value'):
                getattr(fusion, name).weight.normal_(std=.1)
        full = fusion(x, video_state, content, times, geometry, attention, groups,
                      torch.tensor([[2, 2, 2]]), None, identity_rope, context, 3,
                      correction)
        per_position = full.detach().abs().amax(dim=-1)[0]
        self.assertEqual(tuple(per_position.shape), (8,))
        self.assertTrue(bool((per_position > 1e-7).all()), per_position)
        local_only = fusion(x, video_state, content, times, geometry, attention,
                            groups, torch.tensor([[2, 2, 2]]), None, identity_rope,
                            context, 3, correction, global_enabled=False)
        self.assertGreater(float(local_only.detach().abs().max()), 0.)


if __name__ == '__main__':
    unittest.main()
