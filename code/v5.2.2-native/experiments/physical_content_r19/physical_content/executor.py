"""Full-width physical conditioning executed in the pretrained video FFN.

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
        return self.merge(torch.cat(pieces, 2))


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
