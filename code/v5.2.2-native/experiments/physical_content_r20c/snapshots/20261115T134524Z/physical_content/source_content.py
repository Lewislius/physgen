"""Query enhancement with source-valued, full-width physical content.

Qwen-space queries learn which input physical words to combine. The values
are their aligned frozen native-Wan encodings. Independent physical K/V maps
then address and execute the SAME content, rather than unrelated learned
3072-dimensional offsets. This representation property is not a quality claim.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .r18e_reference.content import ContentBlock
from .r18e_reference.evidence import resample_words


class SourceValuedContent(nn.Module):
    def __init__(self, query_width=4096, width=3072, heads=32,
                 grounded_heads=16, slots=256, rounds=2, checkpointing=False,
                 source_scale=1.0):
        super().__init__()
        self.seed = nn.Parameter(torch.randn(1, slots, query_width)*.02)
        self.block = ContentBlock(query_width, heads, grounded_heads)
        self.source_query = nn.Linear(query_width, query_width, bias=False)
        self.source_key = nn.Linear(query_width, query_width, bias=False)
        nn.init.eye_(self.source_query.weight)
        nn.init.eye_(self.source_key.weight)
        self.heads, self.grounded_heads = heads, grounded_heads
        self.rounds, self.checkpointing = rounds, checkpointing
        self.register_buffer('source_scale', torch.as_tensor(source_scale).detach().float().clone())

    def forward(self, words, request, native_words, word_groups):
        if words.ndim != 2 or request.ndim != 2 or native_words.ndim != 2:
            raise ValueError('Unpadded single-sample input word sequences required')
        if words.shape[0] != native_words.shape[0]:
            raise ValueError('Qwen and native physical words must be aligned')
        anchor, groups, _ = resample_words(words, word_groups, self.seed.shape[1])
        native, native_groups, _ = resample_words(native_words, word_groups, self.seed.shape[1])
        if not torch.equal(groups, native_groups):
            raise ValueError('Physical source groups differ')
        memory = torch.cat((request, words), 0)[None]
        mask = torch.ones(memory.shape[:2], dtype=torch.bool, device=words.device)
        state = anchor+self.seed
        for _ in range(self.rounds):
            args = (state, anchor, memory, mask, groups, words[None], word_groups)
            state = (checkpoint(self.block, *args, use_reentrant=False)
                     if self.checkpointing and torch.is_grad_enabled() else self.block(*args))
        split = lambda x: x.unflatten(-1, (self.heads, words.shape[-1]//self.heads)).transpose(1, 2)
        q = split(self.source_query(F.layer_norm(state.float(), (state.shape[-1],))))
        k = split(self.source_key(F.layer_norm(words[None].float(), (words.shape[-1],))))
        logits = q.float()@k.float().transpose(-1, -2)/(q.shape[-1]**.5)
        local = groups[:, None] == word_groups[None, :]
        allowed = torch.ones_like(logits, dtype=torch.bool)
        allowed[:, :self.grounded_heads] = local[None, None]
        # Every head chooses whole native word values, not arbitrary channel
        # slices. Their mean remains a normalized source-word mixture.
        weights = logits.masked_fill(~allowed, float('-inf')).softmax(-1).mean(1)
        content = weights@native_words[None].float()
        return dict(route=content, execute=content, native=native, groups=groups,
                    query_state=state)

    def routing_fidelity(self, content):
        # Source-valued construction replaces the former free-offset fidelity
        # penalty; formal C uses only its existing real-video objective.
        return content['route'].new_zeros(())
