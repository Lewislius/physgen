import math

import torch
from torch import nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, width, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        return (x.float()*torch.rsqrt(x.float().square().mean(-1, keepdim=True)+self.eps)*self.weight.float()).to(x.dtype)


class Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads, self.head_dim = heads, width//heads
        self.q, self.k, self.v, self.o = (nn.Linear(width, width) for _ in range(4))
        self.norm_q, self.norm_k = RMSNorm(width), RMSNorm(width)

    def forward(self, x, context=None, valid=None):
        context = x if context is None else context
        shape = lambda t: t.reshape(t.shape[0], t.shape[1], self.heads, self.head_dim).transpose(1, 2)
        q, k, v = shape(self.norm_q(self.q(x))), shape(self.norm_k(self.k(context))), shape(self.v(context))
        mask = None if valid is None else valid[:, None, None, :].bool()
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o(out.transpose(1, 2).flatten(2))

def xy_grid(height, width, device):
    y, x = torch.meshgrid((torch.arange(height, device=device)+.5)/height,
                          (torch.arange(width, device=device)+.5)/width, indexing="ij")
    return torch.stack((x, y), -1).reshape(-1, 2)


def fourier(coords, width):
    """Deterministic full-width position/time/source encoding; no learned bottleneck."""
    bands = math.ceil(width/(coords.shape[-1]*2))
    frequencies = torch.exp(torch.linspace(0, math.log(128), bands, device=coords.device))
    phase = coords.float().unsqueeze(-1)*frequencies*math.pi
    return torch.cat((phase.sin(), phase.cos()), -1).flatten(-2)[..., :width]


def dense_init(module, source):
    """Copy full pretrained tensors without sharing storage with frozen Wan."""
    module.load_state_dict({k: v.detach().float().clone() for k, v in source.state_dict().items()}, strict=True)
