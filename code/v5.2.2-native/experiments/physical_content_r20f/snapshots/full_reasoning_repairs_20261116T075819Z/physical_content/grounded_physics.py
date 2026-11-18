"""Whole-video C/video interaction and depth-matched physical execution.

Working architecture under redesign, without boxes or box-derived supervision.
Video values, rotary positions and caption context use the Wan representation.
"""
import copy
import math

import torch
from torch import nn


def unit_rms(value):
    value = value.float()
    return value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)


class GroundedCondition(nn.Module):
    """Private full-width C/video binding; no geometric labels or mask gate.

One affinity supports video-to-C evidence pooling and C-to-video reading.
"""
    def __init__(self, native_attention):
        super().__init__()
        self.width = native_attention.dim
        self.heads = native_attention.num_heads
        self.head_dim = self.width // self.heads
        self.query = copy.deepcopy(native_attention.q).float().requires_grad_(True)
        # Native caption V has a bias. Copying it into a physical-only reader
        # would introduce a content-independent write even for zero C.
        self.key = nn.Linear(self.width, self.width, bias=False)
        self.value = nn.Linear(self.width, self.width, bias=False)
        with torch.no_grad():
            self.key.weight.copy_(native_attention.k.weight)
            self.value.weight.copy_(native_attention.v.weight)
        self.query_norm = copy.deepcopy(native_attention.norm_q).float().requires_grad_(True)
        self.key_norm = copy.deepcopy(native_attention.norm_k).float().requires_grad_(True)
        self.video_to_content = nn.Linear(self.width, self.width, bias=False)
        nn.init.zeros_(self.video_to_content.weight)

    def split(self, value):
        return value.reshape(value.shape[0], value.shape[1], self.heads,
                             self.head_dim).transpose(1, 2)

    def forward(self, native_state, route, execute, *, source_present=True):
        query = self.split(self.query_norm(self.query(native_state))).float()
        key = self.split(self.key_norm(self.key(route))).float()
        score = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        # Each C token reads current denoising-state evidence, not a future frame.
        video = self.split(unit_rms(native_state))
        # The condition-specific observation excludes the scene-wide mean.
        # Otherwise an uninformative C reads the same global appearance vector
        # and recreates an unconditional appearance hint through this path.
        back_read = score.transpose(-2, -1).softmax(-1) @ (video-video.mean(2,keepdim=True))
        back_read = back_read.transpose(1, 2).flatten(2)
        grounded = execute + self.video_to_content(back_read)
        values = self.split(self.value(grounded))
        read = score.softmax(-1).to(values.dtype) @ values
        # Preserve the actual read magnitude. Per-head unit normalization would
        # amplify even a weak/near-zero conditional read into a full-strength one.
        if not source_present:
            read = read * 0
        read = read.transpose(1, 2).flatten(2)
        return read


class GroundedPhysicalBlock(nn.Module):
    """Two evaluations of ONE pretrained physical block, with shared parameters.

Only the C-dependent response is written after the corresponding native block.
The reference evaluation is the same block and the same video/text input with
no C modulation; it is not a zero-valued attention token or another learned
network. This excludes a condition-independent appearance hint by construction.
Physical SA Q/K/V/O, caption attention and FFN are independently trainable.
The original backbone block and its text Q/K/V/O are untouched.
"""
    def __init__(self, native_block):
        super().__init__()
        self.native = copy.deepcopy(native_block).float().requires_grad_(True)
        # The final affine bias cancels exactly in the shared response.
        self.native.ffn[2].bias.requires_grad_(False)
        self.conditioning = GroundedCondition(native_block.cross_attn)
        width = native_block.dim
        self.query_update = nn.Linear(width, width, bias=False)
        self.key_update = nn.Linear(width, width, bias=False)
        nn.init.zeros_(self.query_update.weight)
        nn.init.zeros_(self.key_update.weight)
        self.output = nn.Linear(width, width, bias=False)
        nn.init.eye_(self.output.weight)

    def _execute(self, state, native_mod, lengths, grid, freqs, context, read):
        from wan.modules.model import flash_attention, rope_apply
        block = self.native
        shift, scale, gate, ff_shift, ff_scale, ff_gate = (
            block.modulation.unsqueeze(0) + native_mod.float()).unbind(2)
        normalized = block.norm1(state).float() * (1 + scale) + shift
        attention = block.self_attn
        query, key = attention.q(normalized), attention.k(normalized)
        if read is not None:
            query = query + self.query_update(read)
            key = key + self.key_update(read)
        batch, length, _ = state.shape
        heads, dim = attention.num_heads, attention.head_dim
        query = attention.norm_q(query).reshape(batch, length, heads, dim)
        key = attention.norm_k(key).reshape(batch, length, heads, dim)
        # Physical information changes interactions; values remain video features.
        value = attention.v(normalized).reshape(batch, length, heads, dim)
        attended = flash_attention(q=rope_apply(query, grid, freqs),
                                  k=rope_apply(key, grid, freqs), v=value,
                                  k_lens=lengths, window_size=attention.window_size)
        state = state.float() + attention.o(attended.flatten(2)).float() * gate
        state = state + block.cross_attn(block.norm3(state), context, None)
        state = state + block.ffn(block.norm2(state).float() * (1 + ff_scale)
                                  + ff_shift).float() * ff_gate
        return state

    def forward(self, state, native_output, native_mod, lengths, grid, freqs,
                context, route, execute, *, mode='normal', source_present=True):
        read = self.conditioning(self.native.norm3(native_output), route, execute,
                                 source_present=source_present)
        reference = self._execute(state, native_mod, lengths, grid, freqs, context, None)
        conditioned = self._execute(state, native_mod, lengths, grid, freqs, context, read)
        hint = self.output(conditioned - reference)
        return hint
