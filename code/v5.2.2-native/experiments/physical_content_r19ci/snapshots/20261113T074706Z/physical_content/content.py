"""Qwen physical enhancement grounded in native encodings of input reasoning."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from physgen_native.message_conditioning import MessageAttention
from .evidence import resample_words


def norm0(x, eps=1e-6):
    return F.layer_norm(x.float(), (x.shape[-1],), eps=eps).to(x.dtype)


class ContentBlock(nn.Module):
    """R10A read/exchange structure, with all weights fresh in this run."""

    def __init__(self, width, heads, grounded_heads):
        super().__init__()
        if not 0 < grounded_heads < heads:
            raise ValueError('Retain both relation-grounded and full-context reads')
        self.grounded_heads = grounded_heads
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.cross = MessageAttention(width, heads)
        self.exchange_norm = nn.LayerNorm(width)
        self.exchange = MessageAttention(width, heads)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, 2 * width, bias=False), nn.GELU(),
            nn.Linear(2 * width, width, bias=False))
        self.update_norm = nn.RMSNorm(width, eps=1e-6)

    def read_heads(self, query, memory, memory_mask, words, word_groups, groups):
        attention = self.cross
        split = lambda x: x.unflatten(-1, (attention.heads, attention.head_dim)).transpose(1, 2)
        q = split(attention.q(query))
        local = self.grounded_heads
        k = split(attention.k(memory))
        v = split(attention.v(memory))
        wk = split(attention.k(words))
        wv = split(attention.v(words))
        allowed = groups[:, None] == word_groups[None, :]
        grounded = F.scaled_dot_product_attention(
            q[:, :local], wk[:, :local], wv[:, :local], attn_mask=allowed[None, None])
        contextual = F.scaled_dot_product_attention(
            q[:, local:], k[:, local:], v[:, local:], attn_mask=memory_mask[:, None, None, :])
        return torch.cat((grounded, contextual), 1)

    def forward(self, state, anchor, memory, memory_mask, groups, words, word_groups):
        read = self.read_heads(
            self.query_norm(state), self.memory_norm(memory), memory_mask,
            self.memory_norm(words), word_groups, groups)
        state = state + self.cross.o(read.transpose(1, 2).flatten(2))
        normalized = self.exchange_norm(state)
        summaries = torch.stack(
            [normalized[:, groups == r].mean(1) for r in range(int(groups.max()) + 1)], 1)
        exchange_memory = torch.cat((normalized, summaries), 1)
        local = groups[:, None] == groups[None, :]
        allowed = torch.cat(
            (local, torch.ones(len(groups), len(summaries[0]), device=state.device, dtype=torch.bool)), 1)
        attention = self.exchange
        split = lambda x: x.unflatten(-1, (attention.heads, attention.head_dim)).transpose(1, 2)
        values = F.scaled_dot_product_attention(
            split(attention.q(normalized)), split(attention.k(exchange_memory)),
            split(attention.v(exchange_memory)), attn_mask=allowed[None, None])
        state = state + attention.o(values.transpose(1, 2).flatten(2))
        state = state + self.ffn(self.ffn_norm(state))
        return anchor + self.update_norm(state - anchor)


class Enhancement(nn.Module):
    """Native physical words with full-width learned Qwen refinements."""

    def __init__(self, cfg, interface):
        super().__init__()
        model = cfg['model']
        for name in ('mean', 'scale', 'weight', 'bias', 'target_scale'):
            if name not in interface:
                raise KeyError(f'Fresh interface is missing {name}')
            self.register_buffer(f'anchor_{name}', interface[name].float())
        self.seed = nn.Parameter(torch.randn(1, model['q_tokens'], model['qwen_width']) * .02)
        self.block = ContentBlock(
            model['qwen_width'], model['q_heads'], model['grounded_heads'])
        self.project = nn.Linear(model['qwen_width'], model['width'], bias=False)
        nn.init.zeros_(self.project.weight)
        self.rounds = model['content_rounds']
        self.checkpointing = cfg['train']['gradient_checkpointing']
        self.shared_offset = nn.Parameter(torch.zeros(model['width']))
        self.execution_project = nn.Linear(model['qwen_width'], model['width'], bias=False)
        nn.init.zeros_(self.execution_project.weight)

    def read_values(self, state, memory, words, groups, word_groups):
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

    def forward(self, evidence, *, anchor_only=False, routing_only=False,
                anchor_source='native'):
        words = evidence['words'].detach().float()
        words = (words - self.anchor_mean) / self.anchor_scale
        anchor, groups, _ = resample_words(words, evidence['word_groups'], self.seed.shape[1])
        if anchor_source == 'native':
            # This is the same INPUT reasoning encoded by frozen T5/Wan, not
            # information derived from the target video or evaluator.
            native_words = evidence['native_condition'].detach().float()
            if native_words.shape[-2] != words.shape[-2]:
                raise ValueError('Native/Qwen physical words have different spans')
            native, native_groups, _ = resample_words(
                native_words, evidence['word_groups'], self.seed.shape[1])
            if not torch.equal(native_groups, groups):
                raise ValueError('Native/Qwen physical relation groups differ')
        elif anchor_source == 'estimated':
            native = F.linear(anchor, self.anchor_weight, self.anchor_bias)
        else:
            raise ValueError(anchor_source)
        if anchor_only:
            return dict(routing=native, execution=native), groups
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
        routing = native + self.shared_offset + self.project(values)
        execution = native if routing_only else native + self.execution_project(state - anchor)
        return dict(routing=routing, execution=execution), groups

    def content_loss(self, content, evidence):
        raise RuntimeError('R19CI uses native physics as input; no route-regression loss')
