"""In-block NoChunks-v2 mechanisms.

Both mechanisms leave every native Wan operator in place.  Mechanism 1 adds a
direct physical read and a request-associated read whose location comes from the
same native text association used by the block.  Mechanism 2 adds one joint
physical read and a rank-64 adapter on the real post-GELU native FFN activation.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .content import norm0
from .readers import DirectPhysicalReader, JointPhysicalReader, semantic_binding


def _scatter_future(value, length, future_start):
    if future_start is None:
        return value
    return torch.cat((value.new_zeros(value.shape[0], future_start, value.shape[-1]),
                      value), 1)


def _future_inputs(hidden, times, future_start):
    if future_start is None:
        return hidden, times, None
    return hidden[:, future_start:], times[:, future_start:], future_start


class Mechanism1(nn.Module):
    """Native text-association-located dual physical write."""

    def __init__(self, base_bank, layer, cfg):
        super().__init__()
        self.layer = layer
        width = cfg['model']['width']
        attention = base_bank.attention(layer, 'cross_attn')
        self.physical = DirectPhysicalReader(attention)
        self.state = nn.Linear(width, width, bias=False)
        self.direct_writer = nn.Linear(width, width, bias=False)
        self.semantic_writer = nn.Linear(width, width, bias=False)
        for module in (self.state, self.direct_writer, self.semantic_writer):
            nn.init.zeros_(module.weight)

    def forward(self, hidden, x_norm3, qt, text_keys, text_values, text_mask,
                route, execute, groups, times, spatial, future_start):
        future_x, future_times, start = _future_inputs(x_norm3, times, future_start)
        direct = self.physical(x_norm3, route, execute, groups, times, spatial,
                               future_start=future_start)
        zs = semantic_binding(
            qt[:, :, start:] if start is not None else qt,
            text_keys, text_values, text_mask, route, execute,
            self.physical.key_norm, self.physical.key, self.physical.value, groups)
        state = self.state(norm0(future_x))
        direct = self.direct_writer(direct + state * direct)
        semantic = self.semantic_writer(zs + state * zs)
        return _scatter_future(direct + semantic, hidden.shape[1], future_start)


class ConditionedFFN(nn.Module):
    """Mechanism 2's rank-64 adapter on the real native post-GELU activation."""

    def __init__(self, ffn_width, width, rank=64):
        super().__init__()
        self.a = nn.Linear(ffn_width, rank, bias=False)
        self.r = nn.Linear(width, rank, bias=False)
        self.b = nn.Linear(rank, width, bias=False)
        nn.init.zeros_(self.b.weight)

    def forward(self, activated, z_full, mask):
        value = self.b(self.a(activated) * self.r(norm0(z_full)))
        return value * mask


class Mechanism2(nn.Module):
    """Joint direct/request physical read plus conditioned native FFN adapter."""

    def __init__(self, base_bank, layer, cfg):
        super().__init__()
        self.layer = layer
        width = cfg['model']['width']
        ffn_width = cfg['model']['ffn_width']
        rank = int(cfg['model']['conditioned_ffn_rank'])
        if rank != 64:
            raise ValueError('Mechanism 2 conditioned adapter is rank 64')
        attention = base_bank.attention(layer, 'cross_attn')
        self.physical = JointPhysicalReader(attention)
        self.state = nn.Linear(width, width, bias=False)
        self.writer = nn.Linear(width, width, bias=False)
        nn.init.zeros_(self.state.weight)
        nn.init.zeros_(self.writer.weight)
        self.conditioned = ConditionedFFN(ffn_width, width, rank)

    def forward(self, hidden, x_norm3, text_keys, text_values, text_mask,
                route, execute, groups, times, spatial, future_start):
        future_x, _, start = _future_inputs(x_norm3, times, future_start)
        raw = self.physical(x_norm3, route, execute, groups, text_keys, text_values,
                            text_mask, times, spatial, future_start=future_start)
        state = self.state(norm0(future_x))
        direct = self.writer(raw + state * raw)
        z_full = _scatter_future(raw, hidden.shape[1], future_start)
        direct = _scatter_future(direct, hidden.shape[1], future_start)
        return direct, z_full
