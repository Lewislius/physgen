"""Unsubmitted whole-path prototype; not an approved training architecture.

Two input experts form request-grounded physical content. During denoising,
physical slots read the current video, interact across relations and time,
then address the native execution through the same video/content affinities.
No clean future video, teacher prediction, or external object tracks are inputs.
"""
import math
import copy

import torch
from torch import nn
from torch.nn import functional as F


class NativeStyleAttention(nn.Module):
    """Full native-width Q/K/V/O with Wan's projection-side RMS norms."""
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(width, width)
        self.k = nn.Linear(width, width)
        self.v = nn.Linear(width, width)
        self.out_proj = nn.Linear(width, width)
        self.norm_q = nn.RMSNorm(width, eps=1e-6)
        self.norm_k = nn.RMSNorm(width, eps=1e-6)

    def forward(self, query, key, value, need_weights=False, key_padding_mask=None,
                positions=None):
        split = lambda x: x.unflatten(-1, (self.heads, -1)).transpose(1, 2)
        q, k = split(self.norm_q(self.q(query))), split(self.norm_k(self.k(key)))
        v = split(self.v(value))
        if positions is not None:
            q, k = temporal_rope(q, positions), temporal_rope(k, positions)
        mask = None if key_padding_mask is None else ~key_padding_mask[:, None, None, :]
        y = F.scaled_dot_product_attention(q.to(v.dtype), k.to(v.dtype), v, attn_mask=mask)
        return self.out_proj(y.transpose(1, 2).flatten(2)), None

    @torch.no_grad()
    def initialize_from_official(self, attention):
        for dest, source in [(self.q, attention.q), (self.k, attention.k),
                             (self.v, attention.v), (self.out_proj, attention.o)]:
            dest.weight.copy_(source.weight)
            dest.bias.copy_(source.bias if source.bias is not None else torch.zeros_like(dest.bias))
        self.norm_q.weight.copy_(attention.norm_q.weight)
        self.norm_k.weight.copy_(attention.norm_k.weight)


class ResidualAttention(nn.Module):
    def __init__(self, width, heads, neutral=False, native_style=False):
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.attention = (NativeStyleAttention(width, heads) if native_style else
                          nn.MultiheadAttention(width, heads, batch_first=True))
        if neutral:
            nn.init.zeros_(self.attention.out_proj.weight)
            nn.init.zeros_(self.attention.out_proj.bias)

    def forward(self, query, memory, valid=None):
        value, _ = self.attention(
            self.query_norm(query), self.memory_norm(memory),
            self.memory_norm(memory), key_padding_mask=None if valid is None else ~valid,
            need_weights=False)
        return query + value


class ResidualFFN(nn.Module):
    def __init__(self, width, neutral=False, hidden=None):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        hidden = hidden or 2 * width
        self.layers = nn.Sequential(nn.Linear(width, hidden), nn.GELU(approximate='tanh'),
                                    nn.Linear(hidden, width))
        if neutral:
            nn.init.zeros_(self.layers[-1].weight)
            nn.init.zeros_(self.layers[-1].bias)

    def forward(self, value):
        return value + self.layers(self.norm(value))


class SemanticPhysicalEncoder(nn.Module):
    """Separate trainable request and reasoning experts, not a MoE router.

    Native caption values provide the route's initial semantic coordinates.
    The execution values retain the physical reasoning: semantic coordination
    must not discard useful physical information. These hypotheses are processed
    with the actual video by the state expert, never written as a raw C residual.
    This is not itself a factual-verification mechanism.
    """
    def __init__(self, qwen_width=4096, width=3072, heads=32, slots=256, rounds=2):
        super().__init__()
        self.seed = nn.Parameter(torch.randn(1, slots, qwen_width) * .02)
        self.semantic_attention = ResidualAttention(qwen_width, heads, neutral=True)
        self.semantic_ffn = ResidualFFN(qwen_width, neutral=True)
        self.physical_read = ResidualAttention(qwen_width, heads)
        self.request_read = ResidualAttention(qwen_width, heads)
        self.physical_exchange = ResidualAttention(qwen_width, heads)
        self.physical_ffn = ResidualFFN(qwen_width)
        self.address_query = nn.Linear(width, width, bias=False)
        self.address_key = nn.Linear(width, width, bias=False)
        self.address_refinement = nn.Linear(qwen_width, width, bias=False)
        self.semantic_refinement = nn.Linear(qwen_width, width, bias=False)
        nn.init.eye_(self.address_query.weight)
        nn.init.eye_(self.address_key.weight)
        nn.init.zeros_(self.address_refinement.weight)
        nn.init.zeros_(self.semantic_refinement.weight)
        self.route_delta = nn.Linear(qwen_width, width, bias=False)
        self.execute_delta = nn.Linear(qwen_width, width, bias=False)
        nn.init.zeros_(self.route_delta.weight)
        nn.init.zeros_(self.execute_delta.weight)
        self.rounds = rounds

    def forward(self, query_anchor, reasoning, caption, native_caption, caption_valid,
                native_physics,
                *, semantic_expert=True):
        if not caption_valid.any(-1).all():
            raise ValueError('Caption memory includes at least its EOS token')
        if caption.shape[:2] != native_caption.shape[:2]:
            raise ValueError('Native and T5 caption token positions must match')
        semantic = caption
        if semantic_expert:
            semantic = self.semantic_attention(semantic, semantic, caption_valid)
            semantic = self.semantic_ffn(semantic)
        physical = query_anchor + self.seed
        for _ in range(self.rounds):
            physical = self.physical_read(physical, reasoning)
            if semantic_expert:
                physical = self.request_read(physical, semantic, caption_valid)
            physical = self.physical_exchange(physical, physical)
            physical = self.physical_ffn(physical)
        # Initialize addressing with the frozen T5/Wan semantic coordinates,
        # rather than random Qwen-to-T5 dot products. Reasoning is a query;
        # the values come from what the request actually specifies.
        q = F.normalize(self.address_query(native_physics +
                        self.address_refinement(physical)), dim=-1)
        k = F.normalize(self.address_key(native_caption +
                        self.semantic_refinement(semantic - caption)), dim=-1)
        weights = (q @ k.transpose(-1, -2) * math.sqrt(q.shape[-1])).masked_fill(
            ~caption_valid[:, None], -torch.inf).softmax(-1)
        semantic_anchor = weights.to(native_caption) @ native_caption
        if not semantic_expert:
            semantic_anchor = native_physics
        return dict(route=semantic_anchor + self.route_delta(physical),
                    execute=native_physics + self.execute_delta(physical),
                    caption_binding=weights, physical_queries=physical)


def temporal_rope(value, times):
    """Rotate Wan's temporal head subspace; no fictitious Q-slot coordinates."""
    width = value.shape[-1]
    temporal_width = width - 4 * (width // 6)
    frequency = 10000. ** (-torch.arange(0, temporal_width, 2, device=value.device,
                                       dtype=torch.float32) / temporal_width)
    phase = times[:, None, :, None].float() * frequency
    pairs = value[..., :temporal_width].float().unflatten(-1, (-1, 2))
    a, b = pairs.unbind(-1)
    rotated = torch.stack((a * phase.cos() - b * phase.sin(),
                           a * phase.sin() + b * phase.cos()), -1).flatten(-2)
    return torch.cat((rotated.to(value.dtype), value[..., temporal_width:]), -1)


class PhysicalStateExpert(nn.Module):
    """Full-width relation and temporal interaction on video-grounded C.

    Axial attention avoids a quadratic attention matrix over all frame/slot
    combinations. It is not a separate future-state predictor or rollout.
    """
    def __init__(self, width=3072, heads=24, ffn_width=14336):
        super().__init__()
        self.video_norm = nn.LayerNorm(width)
        self.condition_attention = NativeStyleAttention(width, heads)
        self.relation_norm = nn.LayerNorm(width, elementwise_affine=False)
        self.temporal_norm = nn.LayerNorm(width, elementwise_affine=False)
        self.relation_attention = NativeStyleAttention(width, heads)
        self.temporal_attention = NativeStyleAttention(width, heads)
        self.ffn = ResidualFFN(width, hidden=ffn_width)
        # Current Wan denoising-time embedding, not video presentation time.
        # No hardcoded timestep shuts off the physical branch.
        self.noise_modulation = nn.Linear(width, 9 * width)
        nn.init.zeros_(self.noise_modulation.weight)
        nn.init.zeros_(self.noise_modulation.bias)

    @torch.no_grad()
    def initialize_from_official_block(self, block):
        self.condition_attention.initialize_from_official(block.cross_attn)
        self.relation_attention.initialize_from_official(block.self_attn)
        self.temporal_attention.initialize_from_official(block.self_attn)
        self.relation_norm = copy.deepcopy(block.norm1).float().requires_grad_(True)
        self.temporal_norm = copy.deepcopy(block.norm1).float().requires_grad_(True)
        self.video_norm = copy.deepcopy(block.norm3).float().requires_grad_(True)
        self.ffn.norm = copy.deepcopy(block.norm2).float().requires_grad_(True)
        for index in (0, 2):
            self.ffn.layers[index].load_state_dict(block.ffn[index].state_dict())
        self.ffn.layers[1] = copy.deepcopy(block.ffn[1])

    def forward(self, grounded, route, execute, times, noise_embedding, native_parts=None,
                *, temporal=True):
        batch, frames, slots, width = grounded.shape
        if noise_embedding.shape != (batch, frames, width):
            raise ValueError('Use the current native denoising embedding for each frame')
        rs, rb, rg, ts, tb, tg, fs, fb, fg = self.noise_modulation(
            F.silu(noise_embedding)).chunk(9, -1)
        if native_parts is None:
            # Module-only tests; production always supplies the actual native
            # layer's timestep-dependent scale, shift and residual gates.
            zero = torch.zeros_like(noise_embedding)
            one = torch.ones_like(noise_embedding)
            native_parts = (zero, zero, one, zero, zero, one)
        rb, rs, rg = rb + native_parts[0], rs + native_parts[1], rg + native_parts[2]
        tb, ts, tg = tb + native_parts[0], ts + native_parts[1], tg + native_parts[2]
        fb, fs, fg = fb + native_parts[3], fs + native_parts[4], fg + native_parts[5]
        # Cross attention learns the map from content coordinates to actual
        # video-state coordinates. Equal width alone does not justify adding
        # normalized language and video embeddings coordinate by coordinate.
        sequence = grounded.reshape(batch, frames * slots, width)
        conditioned, _ = self.condition_attention(self.video_norm(sequence), route, execute)
        state = (sequence + conditioned).reshape(batch, frames, slots, width)
        sequence = state.reshape(batch * frames, slots, width)
        addressed = (self.relation_norm(state) * (1 + rs[:, :, None]) + rb[:, :, None]
                     ).reshape(batch * frames, slots, width)
        delta, _ = self.relation_attention(addressed, addressed, addressed)
        state = (sequence.reshape_as(state) + delta.reshape_as(state) * rg[:, :, None])
        if temporal:
            sequence = state.transpose(1, 2).reshape(batch * slots, frames, width)
            position = times[:, None].expand(batch, slots, frames).reshape(batch * slots, frames)
            scale = ts[:, None].expand(batch, slots, frames, width).reshape_as(sequence)
            bias = tb[:, None].expand(batch, slots, frames, width).reshape_as(sequence)
            addressed = self.temporal_norm(sequence) * (1 + scale) + bias
            delta, _ = self.temporal_attention(addressed, addressed, addressed, positions=position)
            gate = tg[:, None].expand(batch, slots, frames, width).reshape_as(sequence)
            state = (sequence + delta * gate).reshape(batch, slots, frames, width).transpose(1, 2)
        activated = self.ffn.norm(state) * (1 + fs[:, :, None]) + fb[:, :, None]
        return state + self.ffn.layers(activated) * fg[:, :, None]


class GroundedPhysicalFusion(nn.Module):
    """Per-site video read -> per-site physical state expert -> native execution.

    Each injection site owns its own Q/K/V/O read projections and its own
    ``PhysicalStateExpert``; nothing is shared across sites.  The read/write
    compatibility field uses separate softmax axes.  The write has a zero-value
    option and retains its physical attention mass; no final message LayerNorm.
    """
    def __init__(self, width=3072, ffn_width=14336, heads=24,
                 video_value_init='official_cross_attn_v',
                 read_output_init='official_cross_attn_o'):
        super().__init__()
        if width % heads:
            raise ValueError('Physical heads must cover the full width')
        if video_value_init not in ('official_cross_attn_v', 'identity'):
            raise ValueError(f'Unknown video_value_init: {video_value_init}')
        if read_output_init not in ('official_cross_attn_o', 'identity'):
            raise ValueError(f'Unknown read_output_init: {read_output_init}')
        # V and O are a value->residual pair.  Keep them coherent: official V
        # with official O, or identity V with identity O.  A mixed pair would
        # rotate the read into a space that the same-depth state expert's video
        # query does not expect.
        if (video_value_init, read_output_init) not in {
                ('official_cross_attn_v', 'official_cross_attn_o'),
                ('identity', 'identity')}:
            raise ValueError(
                'video_value_init and read_output_init must be a coherent pair '
                '(official V + official O, or identity V + identity O)')
        self.width, self.heads, self.head_dim = width, heads, width // heads
        self.video_value_init = video_value_init
        self.read_output_init = read_output_init
        self.video_query = nn.Linear(width, width)
        self.content_key = nn.Linear(width, width)
        self.video_value = nn.Linear(width, width)
        # Per-site read output.  This is the same-layer cross-attention O prior,
        # not a shared projection; it is initialized in initialize_from_official_block.
        self.read_output = nn.Linear(width, width)
        self.query_norm = nn.RMSNorm(width, eps=1e-6)
        self.key_norm = nn.RMSNorm(width, eps=1e-6)
        self.feature_output = nn.Linear(width, ffn_width, bias=False)
        # Neutral fallback; official initialization overwrites Q/K/V/O below.
        nn.init.eye_(self.video_value.weight)
        nn.init.zeros_(self.video_value.bias)
        nn.init.eye_(self.read_output.weight)
        nn.init.zeros_(self.read_output.bias)
        nn.init.zeros_(self.feature_output.weight)

    @torch.no_grad()
    def initialize_from_official_block(self, block):
        """Copy same-depth official Q/K/V/O, without aliasing the backbone.

        This module is intentionally per-injection-site.  Q/K/V/O and their
        native RMS norms all come from the *same* Wan block.  The write map
        (feature_output) stays zero-initialized so the physical branch starts as
        an exact no-op on the native path.
        """
        for target, source in [(self.video_query, block.cross_attn.q),
                               (self.content_key, block.cross_attn.k),
                               (self.video_value, block.cross_attn.v),
                               (self.read_output, block.cross_attn.o)]:
            target.weight.copy_(source.weight)
            if source.bias is None:
                target.bias.zero_()
            else:
                target.bias.copy_(source.bias)
        if self.video_value_init == 'identity':
            nn.init.eye_(self.video_value.weight)
            nn.init.zeros_(self.video_value.bias)
        if self.read_output_init == 'identity':
            nn.init.eye_(self.read_output.weight)
            nn.init.zeros_(self.read_output.bias)
        self.query_norm.weight.copy_(block.cross_attn.norm_q.weight)
        self.key_norm.weight.copy_(block.cross_attn.norm_k.weight)
        nn.init.zeros_(self.feature_output.weight)

    def heads_view(self, value):
        return value.unflatten(-1, (self.heads, self.head_dim)).transpose(-3, -2)

    def forward(self, video, content, times, noise_embedding, groups, expert, *, temporal=True,
                video_feedback=True, return_details=False, query_state=None, native_parts=None):
        # video: [B,T,spatial,D]; content tensors: [B,Q,D]
        query_state = video if query_state is None else query_state
        q = self.heads_view(self.query_norm(self.video_query(query_state))).float()
        k = self.heads_view(self.key_norm(self.content_key(content['route'])))[:, None].float()
        logits = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        value = self.heads_view(self.video_value(video))
        read_weights = logits.transpose(-1, -2).softmax(-1)
        grounded = (read_weights.to(value) @ value).transpose(-3, -2).flatten(-2)
        # Same-layer O prior.  Apply it before the no-feedback zeroing so the
        # no_video_feedback intervention removes the entire video read, including
        # the O bias, rather than leaving a constant offset.
        grounded = self.read_output(grounded)
        if not video_feedback:
            grounded = torch.zeros_like(grounded)
        state = expert(grounded, content['route'], content['execute'], times,
                       noise_embedding, native_parts, temporal=temporal)
        counts = torch.bincount(groups, minlength=int(groups.max()) + 1).to(logits)
        prior = -counts[groups].log()
        write_logits = logits + prior
        write_weights = F.pad(write_logits, (0, 1), value=0.).softmax(-1)[..., :-1]
        state_values = self.heads_view(state)
        read = (write_weights.to(state_values) @ state_values).transpose(-3, -2).flatten(-2)
        shift = self.feature_output(read)
        if not return_details:
            return shift
        return shift, dict(state=state, grounded=grounded, read=read,
                           read_weights=read_weights, write_weights=write_weights)
