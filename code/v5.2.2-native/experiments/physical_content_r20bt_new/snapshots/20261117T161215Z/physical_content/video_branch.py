"""Depth-matched, per-site physical block for R20BT-new.

Each block is an independent full-grid physical executor initialised from the
official Wan block at its own injection site.  There is no shared physical
video stream: the block receives the actual native block input at its site.

Time conditioning is the native Wan ``block.modulation + native_mod`` only.
There is no shared ``NoiseConditioning``.  The attention shift/scale/gate from
that native modulation drive both the physical self-attention and the C reader
query/gate.  Presentation time enters only through the reader's per-block
``VideoTimeEmbedding`` inside ``DirectPhysicalReader``.
"""
import copy

import torch
from torch import nn
from torch.nn import functional as F

from .r18e_reference.readers import DirectPhysicalReader


class PhysicalConditionRead(nn.Module):
    """Read shared full-width C into the current per-site physical state.

    The reader is private to this physical block and is initialised from the
    same-site ``cross_attn`` passed by ``PhysicalVideoBlock``.  The attention
    shift/scale/gate come from ``block.modulation + native_mod``; no separate
    shared noise projection exists.
    """

    def __init__(self, attention):
        super().__init__()
        width = attention.dim
        self.direct = DirectPhysicalReader(copy.deepcopy(attention))
        # _PhysicalBase freezes the supplied query; this physical copy is
        # explicitly trainable, as in R20B/R20F.
        self.direct.query_base.float().requires_grad_(True)
        self.direct.query_norm_base.float().requires_grad_(True)
        self.direct_output = nn.Linear(width, width, bias=False)
        nn.init.zeros_(self.direct_output.weight)

    def forward(self, current, route, execute, groups, times,
                shift, scale, gate, spatial, known):
        # ``shift``/``scale``/``gate`` are per-token [1, N, width], taken from
        # the same native modulation used by the physical self-attention.
        query = F.layer_norm(current.float(), (current.shape[-1],), eps=1e-6)
        query = query * (1 + scale) + shift
        read = self.direct(query, route, execute, groups, times, spatial,
                           future_start=known or None)
        delta = self.direct_output(read) * gate
        if known:
            delta = torch.cat((torch.zeros_like(delta[:, :known]), delta[:, known:]), 1)
        return delta


class PhysicalVideoBlock(nn.Module):
    """One independent physical executor, initialised from its own site's block.

    The forward returns only the hint written after the matching native block.
    The block's internal state is private and is never fed to another site.
    The original caption cross-attention and its norm are frozen; the physical
    self-attention, FFN, modulation, reader and output map are trainable.
    """

    def __init__(self, native_block):
        super().__init__()
        self.native = copy.deepcopy(native_block).float().requires_grad_(True)
        self.native.cross_attn.requires_grad_(False)
        self.native.norm3.requires_grad_(False)
        self.conditioning = PhysicalConditionRead(self.native.cross_attn)
        self.output = nn.Linear(native_block.dim, native_block.dim, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, state, native_mod, lengths, grid, freqs, context,
                route, execute, groups, times, spatial, known, mode):
        block = self.native
        parts = (block.modulation.unsqueeze(0) + native_mod.float()).unbind(2)
        value = block.self_attn(
            block.norm1(state).float() * (1 + parts[1]) + parts[0],
            lengths, grid, freqs)
        state = state.float() + value.float() * parts[2]
        text_input = block.norm3(state)
        state = state + block.cross_attn(text_input, context, None)
        if mode != 'physical_condition_off':
            # The C reader uses the same per-block attention modulation as the
            # physical self-attention; there is no shared noise projection.
            state = state + self.conditioning(
                state, route, execute, groups, times,
                parts[0], parts[1], parts[2], spatial, known)
        value = block.ffn(
            block.norm2(state).float() * (1 + parts[4]) + parts[3])
        state = state + value.float() * parts[5]
        hint = self.output(state)
        if known:
            hint = torch.cat((torch.zeros_like(hint[:, :known]), hint[:, known:]), 1)
        return hint
