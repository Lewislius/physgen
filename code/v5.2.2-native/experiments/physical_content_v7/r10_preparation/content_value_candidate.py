"""Unsubmitted R10 candidate: Q selects input values rather than writing its state.

This is an implementation candidate, not a chosen training recipe or a claim of
physical benefit. The caller must load the frozen R9 bootstrap before importing.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from physical_content.model import Enhancement as R9Enhancement
from physical_content.evidence import resample_words


class ContentValueEnhancement(R9Enhancement):
    def __init__(self, cfg, interface=None):
        super().__init__(cfg, interface)
        # A shared correction remains expressible without requiring every Q to
        # encode it. This starts at zero and does not consume additional RNG.
        self.shared_offset = nn.Parameter(torch.zeros(cfg['model']['width']))

    def read_values(self, state, memory, words, groups, word_groups):
        """Use refined Q for weights; values project the fitted input words.

        There is no seed/FFN residual added after this read and no output RMSNorm
        that fixes its magnitude. The existing full-width project stays learned.
        The bias-free shared value map sees all input channels for every head;
        values bypass memory LayerNorm and its learned affine offset. This does
        not prevent learned attention or projection from ignoring data.
        """
        attention = self.block.cross
        split = lambda x: x.unflatten(-1, (attention.heads, attention.head_dim)).transpose(1, 2)
        q = split(attention.q(self.block.query_norm(state)))
        keys = split(attention.k(self.block.memory_norm(memory)))
        word_keys = split(attention.k(self.block.memory_norm(words)))
        values = split(attention.v(memory))
        word_values = split(attention.v(words))
        local = self.block.grounded_heads
        allowed = groups[:, None] == word_groups[None, :]
        grounded = F.scaled_dot_product_attention(
            q[:, :local], word_keys[:, :local], word_values[:, :local],
            attn_mask=allowed[None, None])
        contextual = F.scaled_dot_product_attention(
            q[:, local:], keys[:, local:], values[:, local:])
        return torch.cat((grounded, contextual), 1).transpose(1, 2).flatten(2)

    def forward(self, evidence, *, anchor_only=False):
        words = evidence['words'].detach().float()
        words = (words - self.anchor_mean) / self.anchor_scale
        anchor, groups, _ = resample_words(words, evidence['word_groups'], self.seed.shape[1])
        native = F.linear(anchor, self.anchor_weight, self.anchor_bias)
        # Integration must use this explicit control: zeroing project.weight
        # alone would leave shared_offset and would mislabel the ablation.
        if anchor_only:
            return native, groups
        words = words[None] if words.ndim == 2 else words
        request = evidence['request_words'].detach().float()
        request = (request - self.anchor_mean) / self.anchor_scale
        request = request[None] if request.ndim == 2 else request
        memory = torch.cat((request, words), 1)
        mask = torch.ones(memory.shape[:2], dtype=torch.bool, device=memory.device)
        state = anchor + self.seed
        for _ in range(self.rounds):
            arguments = (state, anchor, memory, mask, groups, words, evidence['word_groups'])
            state = (checkpoint(self.block, *arguments, use_reentrant=False)
                     if self.checkpointing and torch.is_grad_enabled() else self.block(*arguments))
        values = self.read_values(state, memory, words, groups, evidence['word_groups'])
        return native + self.shared_offset + self.project(values), groups


def conditioning_choice(seed, cursor, *, joint_probability, native_only_probability):
    """Candidate training exposure; no active trainer imports this function.

    Preserve the original joint-dropout and sigma draws. The second mask uses a
    separate cursor-seeded generator; it cannot change sample/noise selection.
    native_only_probability is conditional on retaining the Qwen request.
    The first-image condition remains present in every case.
    """
    if not (0 <= joint_probability <= 1 and 0 <= native_only_probability <= 1):
        raise ValueError('Condition probabilities must be in [0, 1]')
    original = torch.rand(2, generator=torch.Generator().manual_seed(seed * 1000003 + cursor))
    joint_drop = bool(original[0] < joint_probability)
    extra = torch.rand((), generator=torch.Generator().manual_seed(seed * 3000017 + cursor))
    native_only = not joint_drop and bool(extra < native_only_probability)
    return dict(qwen_request_dropped=joint_drop,
                native_text_dropped=joint_drop or native_only,
                sigma=float(.001 + .998 * original[1]),
                mode='joint_null' if joint_drop else
                     'qwen_present_native_null' if native_only else 'both_present')
