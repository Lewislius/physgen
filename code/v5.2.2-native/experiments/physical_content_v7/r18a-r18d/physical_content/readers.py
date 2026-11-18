"""Common full-width physical reader and the A/B independent semantic reader."""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from .lora import FrozenBaseLoRALinear


class PhysicalReader(nn.Module):
    """Read Cr/Ce at every video token, including the known first-frame tokens.

    Q is the current block's official frozen cross-attention Q and norm_q.  K/V
    are fresh full-width dense trainable maps initialized from official Wk/Wv.
    There is no null reference slot and no legacy state-value branch.  The
    message is mapped to video residual coordinates by the official Wo weight
    without its bias; the per-expert zero-initialized writer is a separate map.
    """

    def __init__(self, attention, chunk=1024):
        super().__init__()
        width = attention.dim
        self.width = width
        self.heads = attention.num_heads
        self.head_dim = attention.head_dim
        self.chunk = int(chunk)
        if self.heads * self.head_dim != width:
            raise ValueError('Physical reader attention shape differs')
        self.query_norm = nn.LayerNorm(width, eps=1e-6, elementwise_affine=False)
        self.query_base = attention.q.requires_grad_(False)
        self.query_norm_base = attention.norm_q.requires_grad_(False)
        self.key = nn.Linear(width, width, bias=False, device=attention.k.weight.device)
        self.value = nn.Linear(width, width, bias=False, device=attention.v.weight.device)
        with torch.no_grad():
            self.key.weight.copy_(attention.k.weight)
            self.value.weight.copy_(attention.v.weight)
        self.key_norm = copy.deepcopy(attention.norm_k).float().requires_grad_(True)
        self.output_base = attention.o.requires_grad_(False)
        if self.output_base.bias is not None:
            # Only the weight is used; the bias is deliberately excluded.
            self.output_bias_present = True
        else:
            self.output_bias_present = False

    def forward(self, hidden, route, execute, groups):
        if route.ndim != 3 or execute.ndim != 3 or route.shape != execute.shape:
            raise ValueError('Physical reader needs matching [B,S,D] Cr/Ce')
        if groups.ndim != 1 or groups.numel() != route.shape[1]:
            raise ValueError('Physical reader needs one group label per C slot')
        batch, length, width = hidden.shape
        if batch != 1:
            raise ValueError('R18 readers use one video per GPU')
        split = lambda value: value.reshape(batch, -1, self.heads, self.head_dim).transpose(1, 2)
        query_input = self.query_norm(hidden).to(self.query_base.weight.dtype)
        q = split(self.query_norm_base(self.query_base(query_input)))
        k = split(self.key_norm(self.key(route.float()))).to(q.dtype)
        v = split(self.value(execute.float())).to(q.dtype)

        counts = torch.bincount(groups.detach().cpu(), minlength=int(groups.max()) + 1).float()
        group_index = groups.to(q.device)
        prior = -counts.to(q.device)[group_index].log().to(q)[None, None, None, :]
        pieces = []
        for start in range(0, length, self.chunk):
            q_chunk = q[:, :, start:start + self.chunk].float()
            logits = q_chunk @ k.float().transpose(-1, -2) / math.sqrt(self.head_dim)
            logits = logits + prior
            weights = logits.softmax(-1)
            pieces.append(weights.to(v.dtype) @ v)
        message = torch.cat(pieces, 2).transpose(1, 2).flatten(2)
        return F.linear(message.to(self.output_base.weight.dtype), self.output_base.weight, None)


class SemanticReader(nn.Module):
    """A/B independent full-width semantic read of the frozen native text context."""

    def __init__(self, attention, rank=32, alpha=32):
        super().__init__()
        self.width = attention.dim
        self.heads = attention.num_heads
        self.head_dim = attention.head_dim
        self.query_norm = nn.LayerNorm(self.width, eps=1e-6, elementwise_affine=False)
        self.q = FrozenBaseLoRALinear(attention.q, rank, alpha)
        self.k = FrozenBaseLoRALinear(attention.k, rank, alpha)
        self.v = FrozenBaseLoRALinear(attention.v, rank, alpha)
        self.o = FrozenBaseLoRALinear(attention.o, rank, alpha)
        self.norm_q = attention.norm_q.requires_grad_(False)
        self.norm_k = attention.norm_k.requires_grad_(False)

    def forward(self, hidden, context, text_mask):
        batch = hidden.shape[0]
        split = lambda value: value.reshape(batch, -1, self.heads, self.head_dim).transpose(1, 2)
        q = split(self.norm_q(self.q(self.query_norm(hidden))))
        k = split(self.norm_k(self.k(context)))
        v = split(self.v(context))
        allowed = text_mask.to(device=hidden.device, dtype=torch.bool)[:, None, None, :]
        value = F.scaled_dot_product_attention(q, k.to(q.dtype), v.to(q.dtype), attn_mask=allowed)
        return self.o(value.transpose(1, 2).flatten(2))
