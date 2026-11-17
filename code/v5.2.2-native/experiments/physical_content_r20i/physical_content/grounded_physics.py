"""R20IT dual-branch physical execution.

The 256-slot source-valued C branch is retained as a global physical prior.
A separate per-token word cross-attention (PWCA) reads the complete reasoning
words and supplies the token-level physical path.  The native physical block
FFN is intact; there is no C-conditioned full-width feature gain.
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

    def split(self, value):
        return value.reshape(value.shape[0], value.shape[1], self.heads,
                             self.head_dim).transpose(1, 2)

    def forward(self, native_state, route, execute, *, source_present=True):
        query = self.split(self.query_norm(self.query(native_state))).float()
        key = self.split(self.key_norm(self.key(route))).float()
        score = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        values = self.split(self.value(execute))
        read = score.softmax(-1).to(values.dtype) @ values
        if not source_present:
            read = read * 0
        return read.transpose(1, 2).flatten(2)


def _projection(rows, columns, *, seed=42):
    """A deterministic full-rank source projection, kept fixed after init.

    Random full-rank projections preserve word-level distances much better
    than a jointly trained 256-slot reader in the measured probe.  They are
    buffers, not trainable parameters, so training cannot collapse them.
    """
    generator = torch.Generator(device='cpu').manual_seed(seed)
    return torch.randn(rows, columns, generator=generator) / math.sqrt(rows)


class GroundedPhysicalBlock(nn.Module):
    """Source C global branch plus complete-word token-level PWCA.

    The C branch supplies a global physical prior.  PWCA supplies token-level
    physical content and is zero-initialised, so the initial model follows the
    native/C path exactly.  The native physical block FFN remains untouched and
    is not multiplied by a C-conditioned full-width gain.
    """
    def __init__(self, native_block, *, qwen_width=4096, word_heads=24,
                 frozen_seed=42):
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

        # Complete-word PWCA: K/V are frozen source projections; only Q,
        # temperature and the zero-initialised output projection train.
        candidate = min(int(word_heads), max(1, width // 32))
        while candidate > 1 and width % candidate:
            candidate -= 1
        self.word_heads = candidate
        self.word_head_dim = width // self.word_heads
        self.word_query = copy.deepcopy(native_block.cross_attn.q).float().requires_grad_(True)
        self.register_buffer('word_key', _projection(width, qwen_width, seed=frozen_seed),
                             persistent=True)
        self.register_buffer('word_value', _projection(width, width, seed=frozen_seed + 1),
                             persistent=True)
        self.word_key.requires_grad_(False)
        self.word_value.requires_grad_(False)
        self.word_out = nn.Linear(width, width, bias=False)
        nn.init.zeros_(self.word_out.weight)
        self.word_time = nn.Embedding(128, width)
        nn.init.normal_(self.word_time.weight, std=0.02)
        self.word_local = nn.Linear(width, width, bias=False)
        nn.init.zeros_(self.word_local.weight)
        self.word_logit_scale = nn.Parameter(torch.tensor(2.0))
        self.word_memory_statistics = None

    def _word_attention(self, native_read_state, grid, word_memory, source_present):
        state = native_read_state.float()
        qwen_words, native_words = word_memory
        qwen_words = qwen_words.float()
        native_words = native_words.float()
        if (not source_present or qwen_words.numel() == 0
                or native_words.numel() == 0):
            return state.new_zeros(state.shape)
        frames, height, width = (int(value) for value in grid[0].tolist())
        spatial = height * width
        if frames * spatial != state.shape[1]:
            raise ValueError('PWCA grid does not match the video token sequence')
        frame_index = torch.arange(frames, device=state.device).repeat_interleave(spatial)
        if frame_index.numel() != state.shape[1]:
            raise ValueError('PWCA frame index does not match video tokens')
        query = self.word_query(state) + self.word_time(frame_index)[None]
        if frames > 1:
            cube = state.reshape(1, frames, spatial, state.shape[-1])
            previous = torch.cat((cube[:, :1], cube[:, :-1]), 1)
            following = torch.cat((cube[:, 1:], cube[:, -1:]), 1)
            local = cube - 0.5 * (previous + following)
            query = query + self.word_local(local.reshape(1, frames * spatial, -1))
        if qwen_words.shape[0] != native_words.shape[0]:
            raise ValueError('Qwen and native word counts differ in PWCA')
        key = F.linear(qwen_words, self.word_key)
        value = F.linear(native_words, self.word_value)
        heads, head_dim = self.word_heads, self.word_head_dim
        query = query.reshape(1, -1, heads, head_dim).transpose(1, 2)
        key = key.reshape(1, -1, heads, head_dim).transpose(1, 2)
        value = value.reshape(1, -1, heads, head_dim).transpose(1, 2)
        score = query @ key.transpose(-2, -1) / math.sqrt(head_dim)
        score = score * self.word_logit_scale.float()
        attention = score.softmax(-1)
        read = attention @ value
        read = read.transpose(1, 2).flatten(2)
        output = self.word_out(read)
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
                    output_relative_rms=float(
                        output.detach().square().mean().sqrt()
                        / state.square().mean().sqrt().clamp_min(1e-8)),
                )
        return output

    def _execute(self, state, native_mod, lengths, grid, freqs, context, read):
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
        read = self.conditioning(native_read_state, route, execute,
                                 source_present=source_present)
        word = self._word_attention(native_read_state, grid, word_memory,
                                    source_present)
        conditioned_state = state + word
        reference = self._execute(state, native_mod, lengths, grid, freqs, context, None)
        conditioned = self._execute(conditioned_state, native_mod, lengths, grid,
                                    freqs, context, read)
        return self.output(conditioned - reference)
