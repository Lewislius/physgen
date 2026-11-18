"""Whole-video physical binding and a pretrained physical execution block.

The same binding weights read current video evidence into C and return C to
video positions. Their spatial marginals are the supervised localization.
Physical conditions select/compose video attention heads and video FFN
features; C is not projected directly into a visible image residual.
"""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F


class GroundedPhysics(nn.Module):
    def __init__(self, native_block):
        super().__init__()
        text = native_block.cross_attn
        self.width, self.heads = text.dim, text.num_heads
        self.head_dim = self.width // self.heads
        # Independent physical Q/K/V, initialized in the same native language
        # addressing space as the input-anchored routing/execution C.
        self.video_query = copy.deepcopy(text.q).float().requires_grad_(True)
        self.query_norm = copy.deepcopy(text.norm_q).float().requires_grad_(True)
        self.condition_key = copy.deepcopy(text.k).float().requires_grad_(True)
        self.key_norm = copy.deepcopy(text.norm_k).float().requires_grad_(True)
        self.condition_value = copy.deepcopy(text.v).float().requires_grad_(True)
        self.video_value = copy.deepcopy(native_block.self_attn.v).float().requires_grad_(True)
        self.grounded_output = nn.Linear(self.width, self.width, bias=False)
        nn.init.zeros_(self.grounded_output.weight)

        self.norm1 = copy.deepcopy(native_block.norm1).float().requires_grad_(True)
        self.norm2 = copy.deepcopy(native_block.norm2).float().requires_grad_(True)
        self.attention = copy.deepcopy(native_block.self_attn).float().requires_grad_(True)
        if self.attention.o.bias is not None:
            self.attention.o.bias.requires_grad_(False)
        self.ffn = copy.deepcopy(native_block.ffn).float().requires_grad_(True)
        self.modulation = nn.Parameter(native_block.modulation.detach().float().clone())
        self.head_selection = nn.Linear(self.width, self.heads, bias=False)
        self.feature_selection = nn.Linear(self.width, self.width, bias=False)
        nn.init.zeros_(self.head_selection.weight)
        nn.init.zeros_(self.feature_selection.weight)

    def split(self, x):
        return x.unflatten(-1, (self.heads, self.head_dim)).transpose(1, 2)

    def join(self, x):
        return x.transpose(1, 2).flatten(-2)

    def bind(self, state, route, execute, *, grounding=True):
        local = F.layer_norm(state.float(), (self.width,), eps=1e-6)
        q = self.split(self.query_norm(self.video_query(local)))
        k = self.split(self.key_norm(self.condition_key(route)))
        # C is global language memory: no fabricated spatial/temporal RoPE
        # coordinate is assigned to its learned query slots.
        logits = torch.matmul(k.float(), q.float().transpose(-1, -2)) / math.sqrt(self.head_dim)
        binding = logits.softmax(-1)
        values = self.split(self.condition_value(execute)).float()
        if grounding:
            observed = torch.matmul(binding, self.split(self.video_value(local)).float())
            values = values + self.split(self.grounded_output(self.join(observed))).float()
        # Normalize each C value BEFORE scattering, preserving the local
        # binding mass. Uniform binding has unit mean write weight N/Q.
        values = self.split(F.layer_norm(self.join(values), (self.width,), eps=1e-6))
        read = torch.matmul(binding.transpose(-1, -2), values.float())
        read = self.join(read) * (state.shape[1] / route.shape[1])
        distribution = binding.mean((1, 2))
        return read, distribution

    def forward(self, state, native_mod, lengths, grid, freqs, route, execute,
                mode='normal'):
        from wan.modules.model import flash_attention, rope_apply
        if mode not in ('normal', 'grounding_off'):
            raise ValueError(mode)
        read, distribution = self.bind(state, route, execute, grounding=mode != 'grounding_off')
        shift_a, scale_a, time_a, shift_f, scale_f, time_f = (
            self.modulation.unsqueeze(0) + native_mod.float()).unbind(2)
        h = self.norm1(state).float() * (1 + scale_a) + shift_a
        attn = self.attention
        shape = (*h.shape[:2], self.heads, self.head_dim)
        q, k = attn.norm_q(attn.q(h)).view(shape), attn.norm_k(attn.k(h)).view(shape)
        v = attn.v(h).view(shape)
        # Actual native full-video 3D coordinates; no chunks/local time resets.
        heads = flash_attention(q=rope_apply(q, grid, freqs), k=rope_apply(k, grid, freqs),
            v=v, k_lens=lengths, window_size=attn.window_size)
        selection = self.head_selection(read).tanh().to(heads.dtype)
        # Each head is a full-video interaction operator; local C controls
        # its contribution. The original text-conditioned backbone is intact.
        update_a = F.linear((heads * selection[..., None]).flatten(2), attn.o.weight, None).float()
        state = state.float() + update_a * time_a
        h = self.norm2(state).float() * (1 + scale_f) + shift_f
        feature_gate = self.feature_selection(read).tanh()
        update_f = self.ffn(h).float() * feature_gate * time_f
        state = state + update_f
        stats = torch.stack((selection.detach().float().abs().mean(),
            feature_gate.detach().float().abs().mean(),
            distribution.detach().float().amax()))
        return state, distribution, stats
