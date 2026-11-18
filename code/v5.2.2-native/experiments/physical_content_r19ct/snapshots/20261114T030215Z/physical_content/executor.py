"""Full-width physical conditioning inside the actual native FFN execution.

Native text and C read the same state. Physical read magnitude is preserved;
the ordinary native FFN output, including its LoRA, executes the feature change.
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


class NativeFeatureExecutor(nn.Module):
    """C modulates actual native FFN features, using the native output map.

    f' = f + f * O_physical(r). No independent video V/O or direct C
    residual is introduced. r keeps its read magnitude, so absent/weak
    physical evidence is not normalized back to a full-strength write.
    """

    def __init__(self, base_bank, layer, cfg):
        super().__init__()
        attention = base_bank.attention(layer, 'cross_attn')
        self.physical = PhysicalReader(attention)
        self.condition = nn.Linear(cfg['model']['width'], cfg['model']['ffn_width'],
                                   bias=False, device=attention.q.weight.device)
        nn.init.zeros_(self.condition.weight)

    def read(self, video_state_norm, text_keys, text_values, text_mask,
             route, execute, groups, times, spatial, future_start, mode='normal'):
        return self.physical(video_state_norm, route, execute, groups,
                             text_keys, text_values, text_mask, times, spatial,
                             future_start=future_start, mode=mode)

    def forward(self, activated, content_read, write_mask):
        gain = self.condition(content_read)
        return activated * gain * write_mask
