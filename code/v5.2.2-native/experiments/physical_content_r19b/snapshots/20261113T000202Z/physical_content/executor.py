"""A physical write refined by a complete pretrained video-block response.

The physical branch has independent trainable video self-attention and FFN.
It reads the original request through the unchanged native text operators.
No expert router, future-state predictor, or additional supervision is used.
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
        return self.merge(torch.cat(pieces, 2))


class TransformerExecutor(nn.Module):
    """Return B_phys(h + O(C_read), text) - B_phys(h, text).

This includes the direct C write, its video-token interaction, the response of
the frozen text read, and the full-width FFN response. B_phys starts from the
corresponding official video block. The difference is evaluated term by term;
the final FFN output bias cancels and is not a trainable parameter.
"""

    def __init__(self, base_bank, layer, cfg):
        super().__init__()
        attention = base_bank.attention(layer, 'cross_attn')
        self.physical = PhysicalReader(attention)
        width = int(cfg['model']['width'])
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
        official = base_bank.attention(layer, 'self_attn')
        self.video_attention = copy.deepcopy(official).float().requires_grad_(True)
        # Preserve the ORIGINAL Linear references when main-video LoRA later
        # replaces the parent attention's child modules. No extra base weights.
        self.official_attention = copy.copy(official)
        self.official_attention._modules = official._modules.copy()
        self.text_attention = attention
        self.norm1 = base_bank.block_norm(layer, 'norm1')
        self.norm2 = base_bank.block_norm(layer, 'norm2')
        self.norm3 = base_bank.block_norm(layer, 'norm3')

    def read(self, post_text_norm, text_keys, text_values, text_mask,
             route, execute, groups, times, spatial, future_start, mode='normal'):
        value = self.physical(
            post_text_norm, route, execute, groups, text_keys, text_values,
            text_mask, times, spatial, future_start=future_start, mode=mode)
        if future_start is not None:
            value = torch.cat((value.new_zeros(value.shape[0], future_start,
                                               value.shape[-1]), value), 1)
        return value

    def _interact(self, hidden, parts, lengths, grid, freqs, context, attention):
        video_input = self.norm1(hidden).float() * (1 + parts[1]) + parts[0]
        interaction = attention(video_input, lengths, grid, freqs).float()
        state = hidden.float() + interaction * parts[2]
        semantic = self.text_attention(self.norm3(state), context, None).float()
        state = state + semantic
        ffn_input = self.norm2(state).float() * (1 + parts[4]) + parts[3]
        return interaction, semantic, ffn_input

    def forward(self, hidden, content_read, write_mask, parts, lengths, grid,
                freqs, context, *, mode='normal'):
        official_executor = mode == 'official_executor'
        attention = self.official_attention if official_executor else self.video_attention
        input_map = self.ffn_input if official_executor else self.executor_input
        output_map = self.ffn_output if official_executor else self.executor_output
        shift = self.condition(content_read) * write_mask
        if mode == 'direct_write_only':
            return shift
        base_sa, base_text, base_input = self._interact(
            hidden, parts, lengths, grid, freqs, context, attention)
        cond_sa, cond_text, cond_input = self._interact(
            hidden + shift, parts, lengths, grid, freqs, context, attention)
        base = F.gelu(input_map(base_input), approximate='tanh')
        conditioned = F.gelu(input_map(cond_input), approximate='tanh')
        ffn_delta = F.linear(conditioned - base, output_map.weight).float()
        delta = (shift.float() + (cond_sa - base_sa) * parts[2]
                 + (cond_text - base_text) + ffn_delta * parts[5])
        return delta * write_mask
