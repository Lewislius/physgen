"""I2V physical interaction response, using current video states as values.

C changes Q/K of an independent full-grid video attention. Both attention
arms share the SAME parameters and video V; only their difference is written.
An official-initialized FFN processes that video response, also as a shared
conditional difference. No independent unconditioned video prediction is added.
"""
import copy

import torch
from torch import nn
from torch.nn import functional as F

from .video_branch import PhysicalConditionRead


def native_block_with_text_input(block, x, native_mod, lengths, grid, freqs, context):
    """Unchanged native arithmetic, also returning its actual caption query input."""
    parts = (block.modulation.unsqueeze(0)+native_mod.float()).unbind(2)
    value = block.self_attn(block.norm1(x).float()*(1+parts[1])+parts[0],
                            lengths, grid, freqs)
    x = x.float()+value.float()*parts[2]
    text_input = block.norm3(x)
    x = x+block.cross_attn(text_input, context, None)
    value = block.ffn(block.norm2(x).float()*(1+parts[4])+parts[3])
    x = x.float()+value.float()*parts[5]
    return x, text_input


class PhysicalInteractionBlock(nn.Module):
    def __init__(self, native_block, associated):
        super().__init__()
        self.native = copy.deepcopy(native_block).float().requires_grad_(True)
        self.native.cross_attn.requires_grad_(False)
        self.native.norm3.requires_grad_(False)
        self.native.self_attn.o.bias.requires_grad_(False)
        # The last FFN bias cancels exactly in the shared conditional response.
        self.native.ffn[2].bias.requires_grad_(False)
        self.conditioning = PhysicalConditionRead(self.native.cross_attn, associated)

    def forward(self, state, text_input, native_mod, lengths, grid, freqs,
                context, text_mask, route, execute, groups, times,
                physical_mod, spatial, known, mode):
        if mode == 'physical_condition_off':
            return torch.zeros_like(state)
        # Import after the configured official Wan namespace has been loaded.
        from wan.modules.model import flash_attention, rope_apply
        block = self.native
        parts = (block.modulation.unsqueeze(0)+native_mod.float()).unbind(2)
        read = self.conditioning(state, text_input, block.cross_attn, context,
                                 text_mask, route, execute, groups, times,
                                 physical_mod, spatial, known, mode)
        h = block.norm1(state).float()*(1+parts[1])+parts[0]
        attention = block.self_attn
        batch, length = h.shape[:2]
        shape = (batch, length, attention.num_heads, attention.head_dim)
        q0 = attention.norm_q(attention.q(h)).view(shape)
        k0 = attention.norm_k(attention.k(h)).view(shape)
        qc = attention.norm_q(attention.q(h+read)).view(shape)
        kc = attention.norm_k(attention.k(h+read)).view(shape)
        # Physical text never enters V or the FFN as an additive content value.
        values = attention.v(h).view(shape)
        def attend(q, k):
            return flash_attention(q=rope_apply(q, grid, freqs),
                                   k=rope_apply(k, grid, freqs), v=values,
                                   k_lens=lengths, window_size=attention.window_size)
        difference = (attend(qc, kc).float()-attend(q0, k0).float()).flatten(2)
        # Output bias cancels between the two arms; adding it once would be wrong.
        delta = F.linear(difference, attention.o.weight, None).float()*parts[2]
        if known:
            delta = torch.cat((torch.zeros_like(delta[:, :known]), delta[:, known:]), 1)
        if mode != 'physical_ffn_off':
            base = block.norm2(state).float()*(1+parts[4])+parts[3]
            changed = block.norm2(state+delta).float()*(1+parts[4])+parts[3]
            delta = delta+(block.ffn(changed).float()-block.ffn(base).float())*parts[5]
        if known:
            delta = torch.cat((torch.zeros_like(delta[:, :known]), delta[:, known:]), 1)
        return delta
