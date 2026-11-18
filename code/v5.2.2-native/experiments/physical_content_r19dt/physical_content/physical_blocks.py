"""Source-backed content and visual/physical interaction for the DT candidate.

Its contracts address two observed CT shortcuts: unrestricted C-only content
rewriting, and a state expert that can emit its message without video input.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class QueryRead(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(width, width, bias=False)

    def forward(self, query, memory):
        if memory.shape[1] == 0:
            return query
        q, m = self.query_norm(query), self.memory_norm(memory)
        split = lambda x: x.unflatten(-1, (self.heads, -1)).transpose(1, 2)
        read = F.scaled_dot_product_attention(
            split(self.query(q)), split(self.key(m)), split(self.value(m)))
        return query + self.output(read.transpose(1, 2).flatten(2))


class SourceContent(nn.Module):
    """Queries select native values; they cannot add an arbitrary content vector.

Reasoning and request inputs are both frozen Qwen-language word features.
The native physical values are aligned with those reasoning-word positions.
Routing values come from the original native caption, with its padding masked.
All matrices of returned source weights are row-stochastic. No trainable value
or additive output projection is hidden inside content construction; the later
independent physical reader/executor performs the trainable visual mapping.
"""
    def __init__(self, query_width=4096, content_width=3072, heads=32,
                 slots=256, rounds=2):
        super().__init__()
        self.seed = nn.Parameter(torch.randn(1, slots, query_width) * .02)
        self.reasoning_read = QueryRead(query_width, heads)
        self.request_read = QueryRead(query_width, heads)
        self.exchange = QueryRead(query_width, heads)
        self.norm = nn.LayerNorm(query_width)
        self.ffn = nn.Sequential(nn.Linear(query_width, 2 * query_width, bias=False),
                                 nn.GELU(),
                                 nn.Linear(2 * query_width, query_width, bias=False))
        self.source_query = nn.Linear(query_width, content_width, bias=False)
        self.caption_query = nn.Linear(query_width, content_width, bias=False)
        nn.init.zeros_(self.source_query.weight)
        nn.init.zeros_(self.caption_query.weight)
        self.rounds = rounds

    @staticmethod
    def bind(query, source, valid):
        scores = (F.normalize(query.float(), dim=-1)
                  @ F.normalize(source.float(), dim=-1).transpose(-1, -2))
        scores = scores * math.sqrt(query.shape[-1])
        if not valid.any(-1).all():
            raise ValueError('Every query must have a real source, including EOS for null inputs')
        weights = scores.masked_fill(~valid, -torch.inf).softmax(-1)
        return weights.to(source) @ source, weights

    def forward(self, query_anchor, native_anchor, reasoning, request,
                native_words, slot_groups, word_groups, native_caption,
                caption_valid, *, learned_queries=True):
        if reasoning.shape[:2] != native_words.shape[:2]:
            raise ValueError('Reasoning and native word positions must match')
        query = query_anchor + self.seed if learned_queries else query_anchor
        for _ in range(self.rounds if learned_queries else 0):
            query = self.reasoning_read(query, reasoning)
            query = self.request_read(query, request)
            query = self.exchange(query, query)
            query = query + self.ffn(self.norm(query))
        allowed = slot_groups[:, None] == word_groups[None, :]
        execute, source_weights = self.bind(
            native_anchor + (self.source_query(query) if learned_queries else 0),
            native_words, allowed[None])
        route, caption_weights = self.bind(
            execute + (self.caption_query(query) if learned_queries else 0), native_caption,
            caption_valid[:, None, :])
        return dict(route=route, execute=execute, queries=query,
                    source_weights=source_weights, caption_weights=caption_weights)


class ZeroPreservingAttention(nn.Module):
    """Bias-free values and outputs preserve zero interaction input exactly."""
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        for name in ('q', 'k', 'v', 'o'):
            setattr(self, name, nn.Linear(width, width, bias=False))
        self.norm_q = nn.RMSNorm(width, eps=1e-6)
        self.norm_k = nn.RMSNorm(width, eps=1e-6)

    @torch.no_grad()
    def initialize_from_official(self, attention):
        for name in ('q', 'k', 'v', 'o', 'norm_q', 'norm_k'):
            getattr(self, name).weight.copy_(getattr(attention, name).weight)

    @staticmethod
    def temporal_rope(value, times):
        width = value.shape[-1]
        temporal_width = width - 4 * (width // 6)
        frequency = 10000. ** (-torch.arange(0, temporal_width, 2,
                                           device=value.device).float() / temporal_width)
        phase = times[:, None, :, None].float() * frequency
        a, b = value[..., :temporal_width].float().unflatten(-1, (-1, 2)).unbind(-1)
        rotated = torch.stack((a * phase.cos() - b * phase.sin(),
                               a * phase.sin() + b * phase.cos()), -1).flatten(-2)
        return torch.cat((rotated.to(value), value[..., temporal_width:]), -1)

    def forward(self, value, times=None):
        split = lambda x: x.unflatten(-1, (self.heads, -1)).transpose(1, 2)
        q, k, v = split(self.norm_q(self.q(value))), split(self.norm_k(self.k(value))), split(self.v(value))
        if times is not None:
            q, k = self.temporal_rope(q, times), self.temporal_rope(k, times)
        return self.o(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).flatten(2))


class InteractionExpert(nn.Module):
    """Process actual video/content interactions, not a C-only synthesized state.

Input is [batch,time,slots,width]. Both input streams are necessary for a
nonzero output. Temporal positions describe the video, never Q-slot indices.
Native denoising scales/gates can be supplied per frame; they only modulate
existing interaction features and cannot create a constant value from zero.
"""
    def __init__(self, width=3072, heads=24, hidden=14336):
        super().__init__()
        self.video_norm = nn.LayerNorm(width, elementwise_affine=False)
        self.condition = nn.Linear(width, width, bias=False)
        self.relation = ZeroPreservingAttention(width, heads)
        self.temporal = ZeroPreservingAttention(width, heads)
        self.relation_norm = nn.LayerNorm(width, elementwise_affine=False)
        self.temporal_norm = nn.LayerNorm(width, elementwise_affine=False)
        self.ffn_norm = nn.LayerNorm(width, elementwise_affine=False)
        self.ffn = nn.Sequential(nn.Linear(width, hidden, bias=False),
                                 nn.GELU(approximate='tanh'),
                                 nn.Linear(hidden, width, bias=False))

    @torch.no_grad()
    def initialize_from_official(self, block):
        self.condition.weight.copy_(block.cross_attn.v.weight)
        self.relation.initialize_from_official(block.self_attn)
        self.temporal.initialize_from_official(block.self_attn)
        for index in (0, 2):
            self.ffn[index].weight.copy_(block.ffn[index].weight)

    def forward(self, grounded, execute, times, native_parts=None,
                *, temporal=True, relations=True):
        batch, frames, slots, width = grounded.shape
        state = self.video_norm(grounded) * F.silu(self.condition(execute))[:, None]
        if native_parts is None:
            scale = state.new_zeros(batch, frames, width)
            gate = torch.ones_like(scale)
            ffn_scale, ffn_gate = scale, gate
        else:
            scale, gate, ffn_scale, ffn_gate = (native_parts[i] for i in (1, 2, 4, 5))
        if relations:
            addressed = self.relation_norm(state) * (1 + scale[:, :, None])
            delta = self.relation(addressed.reshape(batch * frames, slots, width)).reshape_as(state)
            state = state + delta * gate[:, :, None]
        if temporal:
            addressed = self.temporal_norm(state) * (1 + scale[:, :, None])
            sequence = addressed.transpose(1, 2).reshape(batch * slots, frames, width)
            position = times[:, None].expand(batch, slots, frames).reshape(batch * slots, frames)
            delta = self.temporal(sequence, position)
            delta = delta.reshape(batch, slots, frames, width).transpose(1, 2)
            state = state + delta * gate[:, :, None]
        value = self.ffn_norm(state) * (1 + ffn_scale[:, :, None])
        return state + self.ffn(value) * ffn_gate[:, :, None]


class NativeFeatureGate(nn.Module):
    """Reweight actual native post-GELU features; no additive appearance field.

The ordinary native FFN output remains the actual output operator. At zero
interaction, the channel multiplier is exactly one. The bounded sigmoid is a
feature-selection activation, not a runtime video-quality detector or clamp.
"""
    def __init__(self, width=3072, ffn_width=14336):
        super().__init__()
        self.output = nn.Linear(width, ffn_width, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, activated, interaction):
        return activated * (2 * torch.sigmoid(self.output(interaction)))


class SourceGroundedRead(nn.Module):
    """Direct physics and request-associated addresses share one video-value read.

Each slot's direct/associated keys refer to the same execution content. Their
joint attention can therefore be evaluated with logaddexp, retaining the exact
joint softmax mass rather than averaging two independently normalized outputs.
The zero-value choice and relation-size prior are retained at write time.
"""
    def __init__(self, width=3072, heads=24, value_projection=None):
        super().__init__()
        self.heads, self.head_dim = heads, width // heads
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.norm_q = nn.RMSNorm(width, eps=1e-6)
        self.norm_k = nn.RMSNorm(width, eps=1e-6)
        self.value = value_projection or nn.Linear(width, width, bias=False)
        if value_projection is None:
            nn.init.eye_(self.value.weight)

    @torch.no_grad()
    def initialize_from_official(self, attention):
        for target, source in ((self.query, attention.q), (self.key, attention.k)):
            target.weight.copy_(source.weight)
            if source.bias is None:
                target.bias.zero_()
            else:
                target.bias.copy_(source.bias)
        self.norm_q.weight.copy_(attention.norm_q.weight)
        self.norm_k.weight.copy_(attention.norm_k.weight)

    def split(self, value):
        return value.unflatten(-1, (self.heads, self.head_dim)).transpose(-3, -2)

    def forward(self, video, query_state, content, groups, times, expert,
                native_parts=None, *, temporal=True, relations=True,
                video_feedback=True, return_details=False):
        q = self.split(self.norm_q(self.query(query_state))).float()
        physical_keys = self.split(self.norm_k(self.key(content['execute'])))[:, None].float()
        request_keys = self.split(self.norm_k(self.key(content['route'])))[:, None].float()
        direct = (q @ physical_keys.transpose(-1, -2)) / math.sqrt(self.head_dim)
        associated = (q @ request_keys.transpose(-1, -2)) / math.sqrt(self.head_dim)
        logits = torch.logaddexp(direct, associated) - math.log(2.)
        values = self.split(self.value(video))
        read_weights = logits.transpose(-1, -2).softmax(-1)
        grounded = (read_weights.to(values) @ values).transpose(-3, -2).flatten(-2)
        if not video_feedback:
            grounded = torch.zeros_like(grounded)
        interaction = expert(grounded, content['execute'], times, native_parts,
                             temporal=temporal, relations=relations)
        counts = torch.bincount(groups, minlength=int(groups.max()) + 1).to(logits)
        prior = -counts[groups].log()
        weights = F.pad(logits + prior, (0, 1), value=0.).softmax(-1)[..., :-1]
        message = (weights.to(interaction) @ self.split(interaction)).transpose(-3, -2).flatten(-2)
        if not return_details:
            return message
        return message, dict(direct_logits=direct, associated_logits=associated,
                             write_weights=weights, grounded=grounded,
                             interaction=interaction, prior=prior)
