"""R20 source-preserving physical components; task status is in each config.

Content uses the measured R18E normalized query update with exact native input
anchors. Physical execution has its own learned reader and diffusion-time
modulation, and returns a residual to add AFTER an intact native block.
No temporal coordinate is invented for the global physical-condition slots.
Implementation checks are not evidence of video-quality gains.
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
    def __init__(self, query_width=4096, width=1536, heads=32,
                 grounded_heads=16, slots=256, rounds=2, checkpointing=False,
                 source_scale=1.0):
        if width != 1536:
            raise ValueError('R20FT-Wan2.1 content C must be the full 1536 width')
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
