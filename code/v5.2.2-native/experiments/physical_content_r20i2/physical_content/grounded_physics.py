"""R20IT2 dual-branch physical execution.

The R20FT 256-slot C branch is the visual/global anchor.  A separate PWCA
module reads complete native reasoning words and updates only the physical
attention Q/K, never the state.  Word K/V come from the Wan native text
cross-attention weights of the corresponding depth, not random projections.
"""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F


def unit_rms(value):
    value = value.float()
    return value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)


def native_block_with_read_state(block, state, native_mod, lengths, grid, freqs, context):
    """Original Wan arithmetic, exposing its actual text-attention query input."""
    shift, scale, gate, ff_shift, ff_scale, ff_gate = (
        block.modulation.unsqueeze(0) + native_mod.float()).unbind(2)
    attended = block.self_attn(block.norm1(state).float() * (1 + scale) + shift,
                               lengths, grid, freqs)
    state = state.float() + attended.float() * gate
    read_state = block.norm3(state)
    state = state + block.cross_attn(read_state, context, None)
    state = state + block.ffn(block.norm2(state).float() * (1 + ff_scale)
                              + ff_shift).float() * ff_gate
    return state, read_state


class GroundedCondition(nn.Module):
    """Private full-width source-C read at the actual current video state."""
    def __init__(self, native_attention):
        super().__init__()
        self.width = native_attention.dim
        self.heads = native_attention.num_heads
        self.head_dim = self.width // self.heads
        self.query = copy.deepcopy(native_attention.q).float().requires_grad_(True)
        self.key = nn.Linear(self.width, self.width, bias=False)
        self.value = nn.Linear(self.width, self.width, bias=False)
        with torch.no_grad():
            self.key.weight.copy_(native_attention.k.weight)
            self.value.weight.copy_(native_attention.v.weight)
        self.query_norm = copy.deepcopy(native_attention.norm_q).float().requires_grad_(True)
        self.key_norm = copy.deepcopy(native_attention.norm_k).float().requires_grad_(True)
        # Exact R20FT C branch: zero-initialised video-to-C binding.  This is
        # part of the C execution value, not a state residual on the video.
        self.video_to_content = nn.Linear(self.width, self.width, bias=False)
        nn.init.zeros_(self.video_to_content.weight)

    def split(self, value):
        return value.reshape(value.shape[0], value.shape[1], self.heads,
                             self.head_dim).transpose(1, 2)

    def forward(self, native_state, route, execute, *, source_present=True,
                video_binding=True):
        query = self.split(self.query_norm(self.query(native_state))).float()
        key = self.split(self.key_norm(self.key(route))).float()
        score = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        if video_binding:
            video = self.split(unit_rms(native_state))
            back_read = (score.transpose(-1, -2).softmax(-1)
                         @ (video - video.mean(2, keepdim=True)))
            back_read = back_read.transpose(1, 2).flatten(2)
            grounded = execute + self.video_to_content(back_read)
        else:
            grounded = execute
        values = self.split(self.value(grounded))
        read = score.softmax(-1).to(values.dtype) @ values
        if not source_present:
            read = read * 0
        return read.transpose(1, 2).flatten(2)


class GroundedPhysicalBlock(nn.Module):
    """R20FT C branch plus native-word PWCA Q/K injection.

    The word K/V are per-depth copies of the Wan native text cross-attention
    K/V and remain frozen.  Word attention updates only the physical SA Q/K
    through zero-initialised projections; it never adds a state vector and
    never uses a random word projection, q-memory, or a feature gain.
    """
    def __init__(self, native_block):
        super().__init__()
        self.native = copy.deepcopy(native_block).float().requires_grad_(True)
        self.native.ffn[2].bias.requires_grad_(False)
        self.conditioning = GroundedCondition(native_block.cross_attn)
        width = native_block.dim
        self.query_update = nn.Linear(width, width, bias=False)
        self.key_update = nn.Linear(width, width, bias=False)
        nn.init.zeros_(self.query_update.weight)
        nn.init.zeros_(self.key_update.weight)
        self.output = nn.Linear(width, width, bias=False)
        nn.init.eye_(self.output.weight)

        # Per-depth native text K/V, frozen.  No random projection/source.
        native_text = native_block.cross_attn
        self.word_query = copy.deepcopy(native_text.q).float().requires_grad_(True)
        self.word_query_norm = copy.deepcopy(native_text.norm_q).float().requires_grad_(True)
        self.word_key = copy.deepcopy(native_text.k).float().requires_grad_(False)
        self.word_key_norm = copy.deepcopy(native_text.norm_k).float().requires_grad_(False)
        self.word_value = copy.deepcopy(native_text.v).float().requires_grad_(False)
        self.word_query_update = nn.Linear(width, width, bias=False)
        self.word_key_update = nn.Linear(width, width, bias=False)
        nn.init.zeros_(self.word_query_update.weight)
        nn.init.zeros_(self.word_key_update.weight)
        self.word_time = nn.Embedding(128, width)
        nn.init.normal_(self.word_time.weight, std=0.002)
        self.word_local = nn.Linear(width, width, bias=False)
        nn.init.zeros_(self.word_local.weight)
        self.word_heads = native_text.num_heads
        self.word_head_dim = width // self.word_heads
        self.word_memory_statistics = None

    def _word_attention(self, native_read_state, grid, native_words,
                        source_present):
        state = native_read_state.float()
        native_words = native_words.float()
        if not source_present or native_words.numel() == 0:
            return state.new_zeros(state.shape)
        frames, height, width = (int(value) for value in grid[0].tolist())
        spatial = height * width
        if frames * spatial != state.shape[1]:
            raise ValueError('PWCA grid does not match the video token sequence')
        frame_index = torch.arange(frames, device=state.device).repeat_interleave(spatial)
        query = self.word_query(state) + self.word_time(frame_index)[None]
        if frames > 1:
            cube = state.reshape(1, frames, spatial, state.shape[-1])
            previous = torch.cat((cube[:, :1], cube[:, :-1]), 1)
            following = torch.cat((cube[:, 1:], cube[:, -1:]), 1)
            local = cube - 0.5 * (previous + following)
            query = query + self.word_local(local.reshape(1, frames * spatial, -1))
        query = self.word_query_norm(query)
        key = self.word_key_norm(self.word_key(native_words))
        value = self.word_value(native_words)
        heads, dim = self.word_heads, self.word_head_dim
        query = query.reshape(1, -1, heads, dim).transpose(1, 2)
        key = key.reshape(1, -1, heads, dim).transpose(1, 2)
        value = value.reshape(1, -1, heads, dim).transpose(1, 2)
        score = query @ key.transpose(-2, -1) / math.sqrt(dim)
        attention = score.softmax(-1)
        read = attention @ value
        read = read.transpose(1, 2).flatten(2)
        if self.training:
            with torch.no_grad():
                detached = attention.detach().float()
                self.word_memory_statistics = dict(
                    attention_entropy=float(
                        (-(detached.clamp_min(1e-9).log() * detached).sum(-1)).mean()
                        / math.log(detached.shape[-1])),
                    read_token_variation=float(
                        (read.detach() - read.detach().mean(1, keepdim=True)).square().sum()
                        / read.detach().square().sum().clamp_min(1e-20)),
                )
        return read

    def _execute(self, state, native_mod, lengths, grid, freqs, context,
                 read_c, read_word):
        block = self.native
        shift, scale, gate, ff_shift, ff_scale, ff_gate = (
            block.modulation.unsqueeze(0) + native_mod.float()).unbind(2)
        normalized = block.norm1(state).float() * (1 + scale) + shift
        attention = block.self_attn
        query, key = attention.q(normalized), attention.k(normalized)
        if read_c is not None:
            query = query + self.query_update(read_c)
            key = key + self.key_update(read_c)
        if read_word is not None:
            query = query + self.word_query_update(read_word)
            key = key + self.word_key_update(read_word)
        batch, length, _ = state.shape
        heads, dim = attention.num_heads, attention.head_dim
        query = attention.norm_q(query).reshape(batch, length, heads, dim)
        key = attention.norm_k(key).reshape(batch, length, heads, dim)
        value = attention.v(normalized).reshape(batch, length, heads, dim)
        from wan.modules.model import flash_attention, rope_apply
        attended = flash_attention(q=rope_apply(query, grid, freqs),
                                   k=rope_apply(key, grid, freqs), v=value,
                                   k_lens=lengths, window_size=attention.window_size)
        state = state.float() + attention.o(attended.flatten(2)).float() * gate
        state = state + block.cross_attn(block.norm3(state), context, None)
        ffn_input = block.norm2(state).float() * (1 + ff_scale) + ff_shift
        state = state + block.ffn(ffn_input).float() * ff_gate
        return state

    def forward(self, state, native_read_state, native_mod, lengths, grid, freqs,
                context, route, execute, word_memory, *,
                mode='normal', source_present=True):
        if mode not in ('normal', 'off'):
            raise ValueError(mode)
        state = state.float()
        if mode == 'off':
            return state.new_zeros(state.shape)
        read_c = self.conditioning(native_read_state, route, execute,
                                   source_present=source_present)
        read_word = self._word_attention(native_read_state, grid, word_memory,
                                         source_present)
        reference = self._execute(state, native_mod, lengths, grid, freqs,
                                  context, None, None)
        conditioned = self._execute(state, native_mod, lengths, grid, freqs,
                                    context, read_c, read_word)
        return self.output(conditioned - reference)
