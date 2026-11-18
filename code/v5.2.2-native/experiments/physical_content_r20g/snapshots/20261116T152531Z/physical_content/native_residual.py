"""GT: queries select full-width execution values from the input reasoning.

FT's projection of Q-anchor became a nearly common execution direction across
the fixed16 inputs. Here Q selects source details. A full-width trainable
projection still enhances Qwen physical information, but its input is retrieved
source content rather than a normalized query residual. Physical execution
remains freely trainable.
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

    def source_details(self, state, words, native_words, groups, word_groups):
        """Same reasoning-word indices for Qwen keys and native Wan values.

Relation heads preserve the slot's physical statement; contextual heads can
select complementary physical statements. The query state already includes
the original request. No request token is accidentally paired with a
reasoning value, and no timestamp is fabricated for a global Q slot.
"""
        attention = self.block.cross
        split = lambda x: x.unflatten(-1, (attention.heads, attention.head_dim)).transpose(1, 2)
        query = split(attention.q(self.block.query_norm(state))).float()
        key = split(attention.k(self.block.memory_norm(words[None]))).float()
        score = query @ key.transpose(-2, -1) / attention.head_dim ** .5
        local = self.block.grounded_heads
        allowed = groups[:, None] == word_groups[None, :]
        grounded = score[:, :local].masked_fill(~allowed[None, None], float('-inf')).softmax(-1)
        contextual = score[:, local:].softmax(-1)
        # Each slot selects complete native-width source vectors. Center the
        # retrieval against the same heads' uniform source read: learning can
        # emphasize actual source details without adding another shared mean.
        # The caller retains the original per-slot native anchor in full.
        weights = torch.cat((grounded, contextual), 1).mean(1)
        relation_weights = allowed.to(weights.dtype) / allowed.sum(-1, keepdim=True)
        baseline = (relation_weights[None] * (local / attention.heads)
                    + (1 - local / attention.heads) / words.shape[0])
        detail_weights = weights - baseline
        return (detail_weights.to(native_words.dtype) @ native_words[None],
                detail_weights.to(words.dtype) @ words[None])

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
        read_values = Enhancement.read_values(
            self, state, memory, words[None], groups, word_groups)
        native_detail, qwen_detail = self.source_details(
            state, words, native_words, groups, word_groups)
        return dict(route=native+self.route_update(read_values),
                    execute=native+native_detail+self.execute_update(qwen_detail),
                    native=native, groups=groups, query_state=state)

    def routing_fidelity(self, content):
        source = content['native'].detach().float()
        # A fixed TRAINING-source scale also handles genuinely empty/zero
        # input encodings. Per-sample division would explode for those inputs.
        return (content['route'].float()-source).square().mean()/self.source_scale.square().clamp_min(1e-8)
