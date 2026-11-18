"""A/B/C/D execution modules, each using frozen official Wan bases."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .content import norm0
from .lora import FrozenBaseLoRALinear
from .readers import PhysicalReader, SemanticReader


def split_heads(value, heads, head_dim):
    batch = value.shape[0]
    return value.reshape(batch, -1, heads, head_dim)


def apply_rope(value, grid, freqs):
    if freqs is None:
        return value
    from wan.modules.model import rope_apply
    return rope_apply(value, grid, freqs)


def video_attention(q, k, v, lengths, grid, freqs):
    """Efficient native-grid attention; CPU SDPA exists only for contract tests."""
    q = apply_rope(q, grid, freqs)
    k = apply_rope(k, grid, freqs)
    if q.is_cuda:
        from wan.modules.attention import flash_attention
        return flash_attention(q=q, k=k, v=v, k_lens=lengths)
    return F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)


class Writer(nn.Module):
    """Independent full-width zero-initialized future-only final writer."""

    def __init__(self, width):
        super().__init__()
        self.width = width
        self.linear = nn.Linear(width, width, bias=False)
        nn.init.zeros_(self.linear.weight)

    def forward(self, payload, known):
        if payload.ndim != 3 or payload.shape[-1] != self.width:
            raise ValueError('Writer payload must be [B,N,3072]')
        value = self.linear(payload[:, known:].float())
        return torch.cat((payload.new_zeros(payload.shape[0], known, self.width), value), 1)


class ExpertAttention(nn.Module):
    def __init__(self, base_attention, rank, alpha, *, value_from_payload,
                 use_value_bias, use_output_bias):
        super().__init__()
        self.width = base_attention.dim
        self.heads = base_attention.num_heads
        self.head_dim = base_attention.head_dim
        self.q = FrozenBaseLoRALinear(base_attention.q, rank, alpha)
        self.k = FrozenBaseLoRALinear(base_attention.k, rank, alpha)
        self.v = FrozenBaseLoRALinear(base_attention.v, rank, alpha)
        self.o = FrozenBaseLoRALinear(base_attention.o, rank, alpha)
        self.norm_q = base_attention.norm_q.requires_grad_(False)
        self.norm_k = base_attention.norm_k.requires_grad_(False)
        self.value_from_payload = value_from_payload
        self.use_value_bias = use_value_bias
        self.use_output_bias = use_output_bias

    def forward(self, qk_input, payload, *, q_extra=None, k_extra=None,
                lengths=None, grid=None, freqs=None):
        q = self.q(qk_input)
        k = self.k(qk_input)
        if q_extra is not None:
            q = q + q_extra
        if k_extra is not None:
            k = k + k_extra
        value_input = payload if self.value_from_payload else qk_input
        q = split_heads(self.norm_q(q), self.heads, self.head_dim)
        k = split_heads(self.norm_k(k), self.heads, self.head_dim)
        v = split_heads(self.v(value_input, use_bias=self.use_value_bias),
                        self.heads, self.head_dim)
        value = video_attention(q, k.to(q.dtype), v.to(q.dtype), lengths, grid, freqs)
        return self.o(value.flatten(2), use_bias=self.use_output_bias)


class ExpertFFN(nn.Module):
    def __init__(self, base_ffn, rank, alpha, *, use_bias):
        super().__init__()
        self.linear1 = FrozenBaseLoRALinear(base_ffn[0], rank, alpha)
        self.linear2 = FrozenBaseLoRALinear(base_ffn[2], rank, alpha)
        self.use_bias = use_bias

    def forward(self, state):
        hidden = F.gelu(self.linear1(state, use_bias=self.use_bias), approximate='tanh')
        return self.linear2(hidden, use_bias=self.use_bias)


class JointPhysicsExecutor(nn.Module):
    """A: semantic and physical content enter one complete video SA/FFN expert."""

    def __init__(self, base_bank, layer, cfg):
        super().__init__()
        rank = cfg['model']['expert_lora_rank']
        alpha = cfg['model']['expert_lora_alpha']
        width = cfg['model']['width']
        self.layer = layer
        self.semantic = SemanticReader(
            base_bank.attention(layer, 'cross_attn'),
            cfg['model']['semantic_reader_lora_rank'],
            cfg['model']['semantic_reader_lora_rank'])
        self.physical = PhysicalReader(base_bank.attention(layer, 'cross_attn'))
        self.attention = ExpertAttention(
            base_bank.attention(layer, 'self_attn'), rank, alpha,
            value_from_payload=False, use_value_bias=True, use_output_bias=True)
        self.ffn = ExpertFFN(base_bank.ffn(layer), rank, alpha, use_bias=True)
        self.norm1 = base_bank.block_norm(layer, 'norm1').requires_grad_(False)
        self.norm2 = base_bank.block_norm(layer, 'norm2').requires_grad_(False)
        self.writer = Writer(width)

    def forward(self, hidden, context, text_mask, route, execute, groups, parts,
                known, lengths, grid, freqs):
        semantic = self.semantic(hidden, context, text_mask)
        physical = self.physical(hidden + semantic, route, execute, groups)
        joint = hidden + semantic + physical
        xqk = self.norm1(joint) * (1 + parts[1]) + parts[0]
        attention = parts[2] * self.attention(
            xqk, joint, lengths=lengths, grid=grid, freqs=freqs)
        ua = joint + attention
        z = self.norm2(ua) * (1 + parts[4]) + parts[3]
        ffn = parts[5] * self.ffn(z)
        return self.writer(semantic + physical + attention + ffn, known)


class InBlockInjector(nn.Module):
    """B: one joint semantic/physical write between native text and native FFN."""

    def __init__(self, base_bank, layer, cfg):
        super().__init__()
        self.layer = layer
        self.semantic = SemanticReader(
            base_bank.attention(layer, 'cross_attn'),
            cfg['model']['semantic_reader_lora_rank'],
            cfg['model']['semantic_reader_lora_rank'])
        self.physical = PhysicalReader(base_bank.attention(layer, 'cross_attn'))
        self.writer = Writer(cfg['model']['width'])

    def forward(self, hidden, context, text_mask, route, execute, groups, known):
        semantic = self.semantic(hidden, context, text_mask)
        physical = self.physical(hidden + semantic, route, execute, groups)
        return self.writer(semantic + physical, known)


class PhysicalExecutor(nn.Module):
    """C: propagate physical values; no direct semantic video-value write."""

    def __init__(self, base_bank, layer, cfg):
        super().__init__()
        rank = cfg['model']['expert_lora_rank']
        alpha = cfg['model']['expert_lora_alpha']
        width = cfg['model']['width']
        self.layer = layer
        self.physical = PhysicalReader(base_bank.attention(layer, 'cross_attn'))
        self.attention = ExpertAttention(
            base_bank.attention(layer, 'self_attn'), rank, alpha,
            value_from_payload=True, use_value_bias=False, use_output_bias=False)
        self.ffn = ExpertFFN(base_bank.ffn(layer), rank, alpha, use_bias=False)
        self.writer = Writer(width)

    def forward(self, hidden, route, execute, groups, parts, known, lengths, grid, freqs):
        physical = self.physical(hidden, route, execute, groups)
        xqk = norm0(hidden + physical) * (1 + parts[1]) + parts[0]
        attention = parts[2] * self.attention(
            xqk, physical, lengths=lengths, grid=grid, freqs=freqs)
        p1 = physical + attention
        z = norm0(p1) * (1 + parts[4])
        p2 = p1 + parts[5] * self.ffn(z)
        return self.writer(p2, known)


class GuidedPhysicsExecutor(nn.Module):
    """D: native text_update guides how physical values propagate/transform."""

    def __init__(self, base_bank, layer, cfg):
        super().__init__()
        rank = cfg['model']['expert_lora_rank']
        alpha = cfg['model']['expert_lora_alpha']
        width = cfg['model']['width']
        self.layer = layer
        self.physical = PhysicalReader(base_bank.attention(layer, 'cross_attn'))
        self.attention = ExpertAttention(
            base_bank.attention(layer, 'self_attn'), rank, alpha,
            value_from_payload=True, use_value_bias=False, use_output_bias=False)
        self.ffn = ExpertFFN(base_bank.ffn(layer), rank, alpha, use_bias=False)
        self.uq = nn.Linear(width, width, bias=False)
        self.uk = nn.Linear(width, width, bias=False)
        self.g = nn.Linear(width, width, bias=False)
        for module in (self.uq, self.uk, self.g):
            nn.init.zeros_(module.weight)
        self.writer = Writer(width)

    def forward(self, hidden, text_update, route, execute, groups, parts,
                known, lengths, grid, freqs):
        physical = self.physical(hidden, route, execute, groups)
        xqk = norm0(hidden + physical) * (1 + parts[1]) + parts[0]
        guide = norm0(text_update)
        q_extra = self.uq(guide)
        k_extra = self.uk(guide)
        attention = parts[2] * self.attention(
            xqk, physical, q_extra=q_extra, k_extra=k_extra,
            lengths=lengths, grid=grid, freqs=freqs)
        p1 = physical + attention
        z = norm0(p1) * (1 + parts[4]) * (1 + self.g(guide))
        p2 = p1 + parts[5] * self.ffn(z)
        return self.writer(p2, known)
