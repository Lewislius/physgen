"""Physical conditions change video relations and full-width feature execution.

C supplies a condition; the current post-text video state supplies the features
being transformed. There is no raw C vector added to the video residual. The
independent full-width executor starts from official FFN weights, never from a
project checkpoint; the original video/text operators stay intact.
"""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from .readers import JointPhysicalReader


class PhysicalReader(JointPhysicalReader):
    """Joint direct/request C read with independent, full-width physical Q/K/V."""

    def __init__(self, attention):
        super().__init__(attention)
        # The inherited native references stay frozen. Independent copies train.
        self.query_projection = copy.deepcopy(attention.q).float().requires_grad_(True)
        self.query_normalization = copy.deepcopy(attention.norm_q).float().requires_grad_(True)

    def query(self, hidden, frame_time, spatial):
        projected = self.query_projection(hidden.to(self.query_projection.weight.dtype))
        projected = projected + frame_time.to(projected.dtype).repeat_interleave(
            spatial, 1)[:, :hidden.shape[1]]
        return self.split(self.query_normalization(projected)).float()

    def forward(self, hidden, route, execute, groups, text_keys, text_values,
                text_mask, times, spatial, future_start=None, mode='normal'):
        frame_time = self.time(times)
        if future_start is not None:
            hidden = hidden[:, future_start:]
            frame_time = frame_time[:, future_start // spatial:]
        q = (super().query(hidden, frame_time, spatial) if mode == 'native_query'
             else self.query(hidden, frame_time, spatial))
        k, v = self.physical_kv(route, execute)
        relation_count = int(groups.max()) + 1
        counts = torch.bincount(groups, minlength=relation_count).to(q)
        prior = -counts[groups].log()
        valid = text_mask[0].bool()
        request_keys = text_keys[:, :, valid].to(q)
        if request_keys.shape[2] == 0:
            raise ValueError('Native text includes at least its EOS token')
        binding = (request_keys @ k.transpose(-1, -2) / math.sqrt(self.head_dim)
                   + prior).softmax(-1)
        bound_values = binding.to(v) @ v
        bound_prior = prior.new_full((request_keys.shape[2],),
                                     math.log(relation_count / request_keys.shape[2]))
        if mode == 'direct_only':
            keys, values, priors = [k], [v], [prior]
        elif mode == 'request_only':
            keys, values, priors = [request_keys], [bound_values], [bound_prior]
        else:
            keys, values = [k, request_keys], [v, bound_values]
            priors = [prior - math.log(2.), bound_prior - math.log(2.)]
        reference = self.split(self.reference_key).to(k).expand(k.shape[0], -1, -1, -1)
        keys = torch.cat([*keys, reference], 2)
        values = torch.cat([*values, torch.zeros_like(v[:, :, :1])], 2)
        priors = torch.cat([*priors, prior.new_zeros(1)])
        pieces = []
        for start in range(0, q.shape[2], self.chunk):
            weights = (q[:, :, start:start + self.chunk] @ keys.transpose(-1, -2)
                       / math.sqrt(self.head_dim) + priors).softmax(-1)
            pieces.append(weights.to(values) @ values)
        field = self.merge(torch.cat(pieces, 2))
        if mode == 'spatial_read_shift':
            field = field.unflatten(1, (-1, spatial)).roll(
                shifts=max(1, spatial // 2), dims=2).flatten(1, 2)
        return field


class VideoRelations(nn.Module):
    """C changes Q/K relations; both attention arms use the SAME video values.

    For the actual native self-attention input h, return
        O[(A(Q(h)+Pq(C), K(h)+Pk(C)) - A(Q(h), K(h))) V(h)].
    C is read using a provisional native text update. This relation correction
    is applied before the final native text read, so that the original text
    attention processes the physically updated video state. Q/K normalization,
    real-video RoPE and diffusion modulation are retained.
    The official-initialized Q/K/V/O are independent trainable copies. Only
    Pq/Pk start at zero. There is no C addition to V or the video residual.
    Global video attention includes the known first frame as a value source;
    the I2V write mask prevents changes to its state through this operator.
    """

    def __init__(self, base_bank, layer, width):
        super().__init__()
        source = base_bank.attention(layer, 'self_attn')
        self.heads, self.head_dim = source.num_heads, source.head_dim
        self.window_size = getattr(source, 'window_size', (-1, -1))
        for name in ('q', 'k', 'norm_q', 'norm_k'):
            setattr(self, name, copy.deepcopy(getattr(source, name)).float().requires_grad_(True))
        # V and O biases cancel between the two attention arms.
        for name in ('v', 'o'):
            original = getattr(source, name)
            copied = nn.Linear(width, width, bias=False,
                               device=original.weight.device, dtype=torch.float32)
            with torch.no_grad():
                copied.weight.copy_(original.weight)
            setattr(self, name, copied)
        self.condition_q = nn.Linear(width, width, bias=False,
                                     device=source.q.weight.device)
        self.condition_k = nn.Linear(width, width, bias=False,
                                     device=source.k.weight.device)
        nn.init.zeros_(self.condition_q.weight)
        nn.init.zeros_(self.condition_k.weight)
        self.norm = base_bank.block_norm(layer, 'norm1')

    def _split(self, value):
        return value.unflatten(-1, (self.heads, self.head_dim))

    def _attend(self, q, k, v, lengths):
        if q.is_cuda:
            from wan.modules.attention import flash_attention
            return flash_attention(q=q, k=k, v=v, k_lens=lengths,
                                   window_size=self.window_size)
        return F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)

    def forward(self, hidden, content_read, write_mask, parts, lengths, grid, freqs):
        state = self.norm(hidden).float() * (1 + parts[1]) + parts[0]
        q, k = self.q(state), self.k(state)
        cq = self.condition_q(content_read) * write_mask
        ck = self.condition_k(content_read) * write_mask
        q0, k0 = self._split(self.norm_q(q)), self._split(self.norm_k(k))
        q1, k1 = self._split(self.norm_q(q + cq)), self._split(self.norm_k(k + ck))
        if freqs is not None:
            from wan.modules.model import rope_apply
            q0, k0, q1, k1 = (rope_apply(value, grid, freqs)
                               for value in (q0, k0, q1, k1))
        values = self._split(self.v(state))
        original = self._attend(q0, k0, values, lengths)
        conditioned = self._attend(q1, k1, values, lengths)
        update = self.o((conditioned.float() - original.float()).flatten(2))
        return update.float() * parts[2] * write_mask


class StateExecutor(nn.Module):
    """C -> full-width FFN-input shift -> native video-feature transformation.

For u = the block's time-modulated, post-text FFN input:
    delta = W2 [GELU(W1(u + O(C_read)) + b1) - GELU(W1 u + b1)].
    W1/W2 are independent trainable copies of the official video FFN. The block
    then applies its existing diffusion-time output modulation. O starts
at zero, so the fresh module is exactly neutral without suppressing its gradient.
"""

    def __init__(self, base_bank, layer, cfg):
        super().__init__()
        attention = base_bank.attention(layer, 'cross_attn')
        self.physical = PhysicalReader(attention)
        width = int(cfg['model']['width'])
        self.relations = VideoRelations(base_bank, layer, width)
        self.condition = nn.Linear(width, width, bias=False,
                                   device=attention.q.weight.device)
        nn.init.zeros_(self.condition.weight)
        self.ffn_input = base_bank.linear(layer, 'ffn.0')
        self.ffn_output = base_bank.linear(layer, 'ffn.2')
        self.executor_input = copy.deepcopy(self.ffn_input).float().requires_grad_(True)
        self.executor_output = nn.Linear(
            self.ffn_output.in_features, self.ffn_output.out_features, bias=False,
            device=self.ffn_output.weight.device, dtype=torch.float32)
        with torch.no_grad():
            self.executor_output.weight.copy_(self.ffn_output.weight)

    def read(self, post_text_norm, text_keys, text_values, text_mask,
             route, execute, groups, times, spatial, future_start, mode='normal'):
        value = self.physical(
            post_text_norm, route, execute, groups, text_keys, text_values,
            text_mask, times, spatial, future_start=future_start, mode=mode)
        if future_start is not None:
            value = torch.cat((value.new_zeros(value.shape[0], future_start,
                                               value.shape[-1]), value), 1)
        return value

    def forward(self, video_input, content_read, write_mask, *, official_executor=False):
        input_map = self.ffn_input if official_executor else self.executor_input
        output_map = self.ffn_output if official_executor else self.executor_output
        shift = self.condition(content_read) * write_mask
        base = F.gelu(input_map(video_input), approximate='tanh')
        conditioned = F.gelu(input_map(video_input + shift), approximate='tanh')
        # The output bias cancels in the difference; no unused trainable bias.
        delta = F.linear(conditioned - base, output_map.weight)
        return delta * write_mask
