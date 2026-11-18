"""Unsubmitted R20 candidate components; not evidence of video-quality gains.

Content uses the measured R18E normalized query update with exact native input
anchors. Physical execution has its own learned reader and diffusion-time
modulation, and returns a residual to add AFTER an intact native block.
No temporal coordinate is invented for the global physical-condition slots.
"""
import copy

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .r18e_reference.content import ContentBlock, Enhancement, norm0
from .r18e_reference.evidence import resample_words
from .r18e_reference.readers import DirectPhysicalReader, semantic_binding


class NativeAnchoredContent(nn.Module):
    """Qwen queries enhance source-grounded, full-width routing/execution C.

This retains R18E's normalized per-round query residual, rather than CT's
unrestricted stacked content FFNs. Source anchors are the frozen native text
encodings of INPUT reasoning, never future-video features or target outputs.
"""
    def __init__(self, query_width=4096, width=3072, heads=32,
                 grounded_heads=16, slots=256, rounds=2, checkpointing=False,
                 source_scale=1.0):
        super().__init__()
        self.seed = nn.Parameter(torch.randn(1, slots, query_width)*.02)
        self.block = ContentBlock(query_width, heads, grounded_heads)
        self.route_update = nn.Linear(query_width, width, bias=False)
        self.execute_update = nn.Linear(query_width, width, bias=False)
        nn.init.zeros_(self.route_update.weight)
        nn.init.zeros_(self.execute_update.weight)
        self.rounds = rounds
        self.checkpointing = checkpointing
        self.register_buffer('source_scale', torch.as_tensor(source_scale).detach().float().clone())

    def forward(self, words, request, native_words, word_groups):
        if words.ndim != 2 or request.ndim != 2 or native_words.ndim != 2:
            raise ValueError('One sample with unpadded input word sequences is required')
        if words.shape[0] != native_words.shape[0]:
            raise ValueError('Native and Qwen reasoning words must be aligned')
        anchor, groups, _ = resample_words(words, word_groups, self.seed.shape[1])
        native, native_groups, _ = resample_words(native_words, word_groups, self.seed.shape[1])
        if not torch.equal(groups, native_groups):
            raise ValueError('Physical relation groups differ across sources')
        memory = torch.cat((request, words), 0)[None]
        mask = torch.ones(memory.shape[:2], dtype=torch.bool, device=words.device)
        state = anchor + self.seed
        for _ in range(self.rounds):
            args = (state, anchor, memory, mask, groups, words[None], word_groups)
            state = (checkpoint(self.block, *args, use_reentrant=False)
                     if self.checkpointing and torch.is_grad_enabled()
                     else self.block(*args))
        # Preserve a separate address and execution value as in R18E. Both
        # retain their actual input anchor; no learned shared content offset.
        delta = state-anchor
        read_values = Enhancement.read_values(
            self, state, memory, words[None], groups, word_groups)
        return dict(route=native+self.route_update(read_values),
                    execute=native+self.execute_update(delta),
                    native=native, groups=groups, query_state=state)

    def routing_fidelity(self, content):
        source = content['native'].detach().float()
        # A fixed TRAINING-source scale also handles genuinely empty/zero
        # input encodings. Per-sample division would explode for those inputs.
        return (content['route'].float()-source).square().mean()/self.source_scale.square().clamp_min(1e-8)


class DiffusionModulation(nn.Module):
    """Shared noise-level conditioning, separate from video presentation time.

    Four full-width vectors are direct shift/scale/gate and the associated
    write gate. The associated query retains the actual native text address;
    it is not shifted by a separately learned query modulation.
"""
    def __init__(self, width=3072):
        super().__init__()
        self.project = nn.Sequential(nn.SiLU(), nn.Linear(width, 4*width))
        nn.init.zeros_(self.project[1].weight)
        nn.init.zeros_(self.project[1].bias)
        with torch.no_grad():
            self.project[1].bias.view(4, width)[2:].fill_(1)

    def forward(self, noise_embedding):
        return self.project(noise_embedding).unflatten(-1, (4, noise_embedding.shape[-1]))


class IndependentPhysicalResidual(nn.Module):
    """Direct/request-associated full-width physics alongside the native block.

Native text weights are never trained or replaced. Its frozen query operator is
used only as a semantic address for the associated physical read. The direct
physical query is an independent trainable copy. Both physical writers have
their own zero-initialized output maps and a video-state interaction.
"""
    def __init__(self, native_attention):
        super().__init__()
        width = native_attention.dim
        # _PhysicalBase historically shares and freezes its supplied Q.
        # Give it a private copy first, then explicitly train that copy.
        self.direct = DirectPhysicalReader(copy.deepcopy(native_attention))
        self.direct.query_base.float().requires_grad_(True)
        self.direct.query_norm_base.float().requires_grad_(True)
        self.semantic_query = copy.deepcopy(native_attention.q).requires_grad_(False)
        self.semantic_query_norm = copy.deepcopy(native_attention.norm_q).requires_grad_(False)
        self.state = nn.Linear(width, width, bias=False)
        self.direct_output = nn.Linear(width, width, bias=False)
        self.associated_output = nn.Linear(width, width, bias=False)
        for module in (self.state, self.direct_output, self.associated_output):
            nn.init.zeros_(module.weight)
        self.modulation = nn.Parameter(torch.zeros(1, 1, 4, width))

    def forward(self, native_hidden, normalized_hidden, route, execute, groups,
                text_keys, text_values, text_mask, presentation_times,
                noise_modulation, spatial, *, known_tokens=0,
                direct=True, associated=True, diffusion_modulation=True,
                semantic_hidden=None):
        batch, length, width = native_hidden.shape
        frames = presentation_times.shape[1]
        if length != frames*spatial:
            raise ValueError('Video/time grid mismatch')
        if known_tokens not in (0, spatial):
            raise ValueError('Only T2V or a known first latent frame is supported')
        if normalized_hidden.shape != native_hidden.shape:
            raise ValueError('Native hidden shapes differ')
        if noise_modulation.shape != (batch, frames, 4, width):
            raise ValueError('Noise modulation must be frame-aligned')
        mod = noise_modulation + self.modulation if diffusion_modulation else torch.zeros_like(noise_modulation)
        shift_d, scale_d, gate_d, gate_s = mod.unbind(-2)
        if not diffusion_modulation:
            gate_d, gate_s = torch.ones_like(gate_d), torch.ones_like(gate_s)
        expand = lambda value: value.repeat_interleave(spatial, dim=1)
        start = known_tokens or None
        current = native_hidden[:, known_tokens:]
        state = self.state(norm0(current))
        delta = torch.zeros_like(current)
        if direct:
            query = normalized_hidden*(1+expand(scale_d))+expand(shift_d)
            read = self.direct(query, route, execute, groups, presentation_times,
                               spatial, future_start=start)
            delta = delta+self.direct_output(read+state*read)*expand(gate_d)[:, known_tokens:]
        if associated:
            if semantic_hidden is None or semantic_hidden.shape != native_hidden.shape:
                raise ValueError('Associated read needs the actual native text-attention input')
            query = semantic_hidden.to(self.semantic_query.weight.dtype)
            query = self.direct.split(self.semantic_query_norm(self.semantic_query(query))).float()
            read = semantic_binding(
                query[:, :, known_tokens:], text_keys, text_values, text_mask,
                route, execute, self.direct.key_norm, self.direct.key,
                self.direct.value, groups)
            delta = delta+self.associated_output(read+state*read)*expand(gate_s)[:, known_tokens:]
        if known_tokens:
            delta = torch.cat((torch.zeros_like(native_hidden[:, :known_tokens]), delta), dim=1)
        return delta
