"""Full-width physical readers for NoChunks-v2 mechanisms 1 and 2."""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from physgen_native.generator import sinusoidal_embedding_1d


def make_text_mask(context, text_mask):
    if text_mask is None:
        text_mask = torch.ones(context.shape[:2], dtype=torch.bool, device=context.device)
    return text_mask.to(device=context.device, dtype=torch.bool)


def time_features(times):
    """Presentation time and adjacent-frame interval, never diffusion sigma."""
    if times.ndim == 1:
        times = times[None]
    delta = torch.cat((times[:, :1] * 0, times[:, 1:] - times[:, :-1]), 1)
    a = sinusoidal_embedding_1d(128, times.flatten())
    b = sinusoidal_embedding_1d(128, delta.flatten())
    return torch.cat((a, b), -1).reshape(*times.shape, 256).float()


class VideoTimeEmbedding(nn.Module):
    """Trainable E_video_time; zero-initialized so the start is native-only."""

    def __init__(self, width):
        super().__init__()
        self.proj = nn.Linear(256, width, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self, times):
        return self.proj(time_features(times).to(self.proj.weight.dtype))


class _PhysicalBase(nn.Module):
    def __init__(self, attention, chunk=1024):
        super().__init__()
        width = attention.dim
        self.width = width
        self.heads = attention.num_heads
        self.head_dim = attention.head_dim
        self.chunk = int(chunk)
        if self.heads * self.head_dim != width:
            raise ValueError('Physical reader attention shape differs')
        self.query_base = attention.q.requires_grad_(False)
        self.query_norm_base = attention.norm_q.requires_grad_(False)
        self.key = nn.Linear(width, width, bias=False, device=attention.k.weight.device)
        self.value = nn.Linear(width, width, bias=False, device=attention.v.weight.device)
        with torch.no_grad():
            self.key.weight.copy_(attention.k.weight)
            self.value.weight.copy_(attention.v.weight)
        self.key_norm = copy.deepcopy(attention.norm_k).float().requires_grad_(True)
        self.time = VideoTimeEmbedding(width)
        # B400+I uses a learned zero-initialized reference key; its value is
        # fixed to zero and its prior is zero.
        self.reference_key = nn.Parameter(
            torch.zeros(1, 1, width, device=attention.q.weight.device))

    def split(self, value):
        return value.reshape(value.shape[0], -1, self.heads, self.head_dim).transpose(1, 2)

    def merge(self, value):
        return value.transpose(1, 2).flatten(2)

    def physical_kv(self, route, execute):
        k = self.split(self.key_norm(self.key(route.float()))).float()
        v = self.split(self.value(execute.float())).float()
        return k, v

    def query(self, hidden, frame_time, spatial):
        q_input = self.query_base(hidden.to(self.query_base.weight.dtype))
        q_input = q_input + frame_time.to(q_input.dtype).repeat_interleave(
            spatial, 1)[:, :hidden.shape[1]]
        return self.split(self.query_norm_base(q_input)).float()


class DirectPhysicalReader(_PhysicalBase):
    """Mechanism 1: direct read with a fixed zero reference key and group prior."""

    def forward(self, hidden, route, execute, groups, times, spatial,
                future_start=None):
        frame_time = self.time(times)
        if future_start is not None:
            hidden = hidden[:, future_start:]
            frame_time = frame_time[:, future_start // spatial:]
        q = self.query(hidden, frame_time, spatial)
        k, v = self.physical_kv(route, execute)
        counts = torch.bincount(groups.detach().cpu(), minlength=int(groups.max()) + 1).float()
        prior = -counts.to(q.device)[groups.to(q.device)].log().to(q.dtype)
        reference = self.split(self.reference_key).to(k).expand(k.shape[0], -1, -1, -1)
        k = torch.cat((k, reference), 2)
        v = torch.cat((v, torch.zeros_like(v[:, :, :1])), 2)
        prior = torch.cat((prior, prior.new_zeros(1)))
        pieces = []
        scale = math.sqrt(self.head_dim)
        for start in range(0, q.shape[2], self.chunk):
            logits = q[:, :, start:start + self.chunk] @ k.transpose(-1, -2) / scale
            weights = (logits + prior).softmax(-1)
            pieces.append(weights.to(v.dtype) @ v)
        return self.merge(torch.cat(pieces, 2))


class JointPhysicalReader(_PhysicalBase):
    """Mechanism 2: direct physical slots plus request-bound physical values."""

    def forward(self, hidden, route, execute, groups, text_keys, text_values,
                text_mask, times, spatial, future_start=None):
        frame_time = self.time(times)
        if future_start is not None:
            hidden = hidden[:, future_start:]
            frame_time = frame_time[:, future_start // spatial:]
        q = self.query(hidden, frame_time, spatial)
        k, v = self.physical_kv(route, execute)
        counts = torch.bincount(groups.detach().cpu(), minlength=int(groups.max()) + 1).float()
        direct_prior = -counts.to(q.device)[groups.to(q.device)].log().to(q.dtype)
        valid = text_mask[0].bool()
        request_keys = text_keys[:, :, valid]
        request_values = text_values[:, :, valid]
        relation_count = int(groups.max()) + 1
        valid_count = max(1, int(valid.sum()))
        binding = ((request_keys.float() @ k.float().transpose(-1, -2))
                   / math.sqrt(self.head_dim) + direct_prior)
        bound_values = binding.softmax(-1).to(v.dtype) @ v
        bound_prior = q.new_full((valid_count,),
                                 math.log(relation_count) - math.log(valid_count))
        reference = self.split(self.reference_key).to(k).expand(k.shape[0], -1, -1, -1)
        keys = torch.cat((k, request_keys, reference), 2)
        values = torch.cat((v, bound_values, torch.zeros_like(v[:, :, :1])), 2)
        prior = torch.cat((direct_prior - math.log(2.), bound_prior - math.log(2.),
                           direct_prior.new_zeros(1)))
        pieces = []
        scale = math.sqrt(self.head_dim)
        for start in range(0, q.shape[2], self.chunk):
            logits = q[:, :, start:start + self.chunk] @ keys.transpose(-1, -2) / scale
            weights = (logits + prior).softmax(-1)
            pieces.append(weights.to(values.dtype) @ values)
        return self.merge(torch.cat(pieces, 2))


def semantic_binding(qt, text_keys, text_values, text_mask, route, execute,
                     key_norm, key, value, groups, chunk=1024):
    """Mechanism 1 request binding and native-text association.

    ``qt`` is [B,H,N,dh] for the queried video positions.  Returns
    ``zs = softmax(Qt Kt_full^T + mask) @ PadZero(Binding Vc)``.
    """
    batch, heads, length, head_dim = qt.shape
    full_keys = text_keys
    full_values = text_values
    token_mask = text_mask.to(device=qt.device, dtype=torch.bool)[:, None, None, :]
    binding_mask = text_mask.to(device=qt.device, dtype=torch.bool)[:, None, :, None]
    # Binding is computed for every text token, then invalid request slots are
    # zeroed.  This is the documented Kt_valid path without a second tokenizer.
    kc = key_norm(key(route.float())).reshape(batch, -1, heads, head_dim).transpose(1, 2).float()
    vc = value(execute.float()).reshape(batch, -1, heads, head_dim).transpose(1, 2).float()
    counts = torch.bincount(groups.detach().cpu(), minlength=int(groups.max()) + 1).float()
    prior = -counts.to(qt.device)[groups.to(qt.device)].log().to(qt.dtype)
    binding = (full_keys.float() @ kc.transpose(-1, -2)
               / math.sqrt(head_dim) + prior)
    binding = binding.softmax(-1) * binding_mask.to(binding.dtype)
    request_values = binding.to(vc.dtype) @ vc
    pieces = []
    for start in range(0, length, chunk):
        logits = (qt[:, :, start:start + chunk].float()
                  @ full_keys.float().transpose(-1, -2) / math.sqrt(head_dim))
        logits = logits.masked_fill(~token_mask, torch.finfo(logits.dtype).min)
        weights = logits.softmax(-1)
        pieces.append(weights.to(request_values.dtype) @ request_values)
    return torch.cat(pieces, 2).transpose(1, 2).flatten(2)
