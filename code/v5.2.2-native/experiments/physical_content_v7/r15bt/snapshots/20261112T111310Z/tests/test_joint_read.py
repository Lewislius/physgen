"""B400 joint-read equivalence and T2V time/write-mask contracts."""
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

from physical_content.model import Fusion, read_physical


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


class JointRead(unittest.TestCase):
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

    def test_t2v_uses_all_time_points(self):
        width = 8
        attention = FakeAttention(width)
        fusion = Fusion(attention)
        fusion.time = Recorder(width)
        x = torch.randn(1, 8, width)
        context = torch.randn(1, 5, width)
        content = dict(routing=torch.randn(1, 6, width),
                       execution=torch.randn(1, 6, width))
        groups = torch.tensor([0, 0, 1, 1, 2, 2])
        times = torch.tensor([[0., 0.5]])
        geometry = (2, 2, 2)  # two latent frames, 4 spatial tokens
        fusion(x, context, content, times, geometry, attention, groups,
               context_length=3, known_frames=0)
        self.assertEqual(tuple(fusion.time.last.shape), (1, 2, 256))
        fusion(x, context, content, times, geometry, attention, groups,
               context_length=3, known_frames=4)
        self.assertEqual(tuple(fusion.time.last.shape), (1, 1, 256))

    def test_zero_init_writes_nothing_and_t2v_writes_every_position(self):
        width = 8
        attention = FakeAttention(width)
        fusion = Fusion(attention)
        x = torch.randn(1, 8, width)
        context = torch.randn(1, 5, width)
        content = dict(routing=torch.randn(1, 6, width),
                       execution=torch.randn(1, 6, width))
        groups = torch.tensor([0, 0, 1, 1, 2, 2])
        times = torch.tensor([[0., 0.5]])
        geometry = (2, 2, 2)
        zero, _ = fusion(x, context, content, times, geometry, attention, groups,
                         context_length=3, known_frames=0)
        text = self.native_text(attention, x, context)
        self.assertTrue(torch.equal(zero, text))
        with torch.no_grad():
            fusion.output.weight.copy_(torch.eye(width))
            fusion.state_value.weight.zero_()
            fusion.time.weight.normal_()
        t2v, _ = fusion(x, context, content, times, geometry, attention, groups,
                        context_length=3, known_frames=0)
        i2v, _ = fusion(x, context, content, times, geometry, attention, groups,
                        context_length=3, known_frames=4)
        self.assertFalse(torch.equal(t2v[:, :4], text[:, :4]))
        self.assertTrue(torch.equal(i2v[:, :4], text[:, :4]))

    @staticmethod
    def native_text(attention, x, context):
        batch, length, width = x.shape
        heads, dim = attention.num_heads, attention.head_dim
        split = lambda value: value.reshape(batch, -1, heads, dim)
        qt = split(attention.norm_q(attention.q(x)))
        kt = split(attention.norm_k(attention.k(context)))
        vt = split(attention.v(context))
        kernel = attention.forward.__func__.__globals__.get('flash_attention')
        if kernel is None:
            value = F.scaled_dot_product_attention(
                qt.transpose(1, 2), kt.transpose(1, 2).to(qt),
                vt.transpose(1, 2).to(qt)).transpose(1, 2)
        else:
            value = kernel(qt, kt, vt, k_lens=None)
        return attention.o(value.flatten(2))


if __name__ == '__main__':
    unittest.main()
