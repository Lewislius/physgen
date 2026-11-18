"""Tiny fake-Wan forward/backward for both mechanisms and task boundaries."""
import sys
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bootstrap  # noqa: F401

from physical_content.common import load_config
from physical_content.model import PhysicalModel

W, FF, H = 8, 16, 2
D = W // H


class RMS(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(W))

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * self.weight


class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim, self.num_heads, self.head_dim = W, H, D
        self.q, self.k, self.v, self.o = (nn.Linear(W, W) for _ in range(4))
        self.norm_q, self.norm_k = RMS(), RMS()

    def split(self, x):
        return x.reshape(x.shape[0], -1, H, D).transpose(1, 2)

    def forward(self, x, seq_lens=None, grid_sizes=None, freqs=None,
                context=None, context_lens=None):
        # Native cross-attention takes context as its second positional arg.
        if torch.is_tensor(seq_lens) and seq_lens.ndim == 3:
            context = seq_lens
        src = x if context is None else context
        q = self.split(self.norm_q(self.q(x)))
        k = self.split(self.norm_k(self.k(src)))
        v = self.split(self.v(src))
        y = F.scaled_dot_product_attention(q, k, v)
        return self.o(y.transpose(1, 2).flatten(2))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn, self.cross_attn = Attn(), Attn()
        self.norm1 = nn.LayerNorm(W, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(W, elementwise_affine=False)
        self.norm3 = nn.Identity()
        self.ffn = nn.Sequential(nn.Linear(W, FF), nn.GELU(approximate='tanh'),
                                 nn.Linear(FF, W))
        self.modulation = nn.Parameter(torch.randn(1, 6, W) / W ** .5)


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(W, 48 * 4)

    def forward(self, x, e):
        return self.linear(x)


class FakeWan(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim, self.text_len, self.freq_dim = W, 8, 256
        self.freqs = None
        self.blocks = nn.ModuleList([Block() for _ in range(30)])
        self.patch_embedding = nn.Conv3d(48, W, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.head = Head()
        self.time_embedding = nn.Linear(256, 16)
        self.time_projection = nn.Linear(16, 6 * W)
        self.text_embedding = nn.Sequential(
            nn.Linear(4096, W), nn.GELU(approximate='tanh'), nn.Linear(W, W))

    def unpatchify(self, x, grid):
        t, h, w = (int(v) for v in grid[0])
        return x.reshape(1, t, h, w, 48, 1, 2, 2).permute(
            0, 4, 1, 5, 2, 6, 3, 7).reshape(1, 48, t, h * 2, w * 2)


def evidence():
    return dict(
        words=torch.randn(12, 4096),
        word_groups=torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]),
        word_positions=torch.arange(12), word_strings=['a'] * 12,
        request_words=torch.randn(4, 4096), request_word_strings=['x'] * 4,
        relations=['a', 'b', 'c', 'd'], caption='x y', sample='s',
        scope='first_image_and_request_only', native_words=torch.randn(12, 3072))


class TinyModelTests(unittest.TestCase):
    def run_plan(self, plan, task):
        cfg = load_config(ROOT / 'configs' / f'{plan}.yaml')
        cfg['model']['width'] = W
        cfg['model']['ffn_width'] = FF
        cfg['model']['injection_layers'] = [0]
        cfg['model']['q_tokens'] = 8
        cfg['model']['q_heads'] = 4
        cfg['model']['grounded_heads'] = 2
        cfg['train']['gradient_checkpointing'] = False
        interface = dict(mean=torch.zeros(4096), scale=torch.ones(4096),
                         weight=torch.randn(W, 4096) * .01, bias=torch.zeros(W),
                         target_scale=torch.tensor(1.0))
        model = PhysicalModel(cfg, device='cpu', wan=FakeWan().requires_grad_(False),
                              interface=interface)
        model.task = task
        text = torch.randn(1, 10, 4096)
        content, groups = model.prepare_content(evidence(), text)
        noisy = torch.randn(1, 48, 2, 4, 4)
        first = torch.randn(1, 48, 1, 4, 4) if task == 'i2v' else None
        prediction, _ = model.velocity(
            noisy, first, text, torch.tensor(0.5), content, groups,
            torch.tensor([[0., 0.5]]))
        self.assertEqual(tuple(prediction.shape), (1, 48, 2, 4, 4))
        self.assertTrue(torch.isfinite(prediction).all())
        prediction[:, :, 0].square().mean().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()
                            if parameter.requires_grad))

    def test_r18f_both_tasks(self):
        self.run_plan('r18f', 'i2v')
        self.run_plan('r18f', 't2v')

    def test_r18h_both_tasks(self):
        self.run_plan('r18h', 'i2v')
        self.run_plan('r18h', 't2v')


if __name__ == '__main__':
    unittest.main()
