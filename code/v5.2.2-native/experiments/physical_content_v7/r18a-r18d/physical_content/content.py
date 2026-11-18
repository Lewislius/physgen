"""Fresh 256-slot content enhancement and C's shared semantic-memory enhancer."""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from physgen_native.message_conditioning import MessageAttention
from .lora import FrozenBaseLoRALinear
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
    """Fresh Qwen-to-C interface; raw Cr/Ce remain available for route loss."""

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

    def forward(self, evidence, *, anchor_only=False, routing_only=False):
        words = evidence['words'].detach().float()
        words = (words - self.anchor_mean) / self.anchor_scale
        anchor, groups, _ = resample_words(words, evidence['word_groups'], self.seed.shape[1])
        native = F.linear(anchor, self.anchor_weight, self.anchor_bias)
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
        if 'native_words' not in evidence:
            raise KeyError('Route loss requires the training-side native word target')
        routing = content['routing']
        target, _, _ = resample_words(
            evidence['native_words'].detach().float(), evidence['word_groups'], routing.shape[1])
        return ((routing.float() - target).square().mean()
                / self.anchor_target_scale.square().clamp_min(1e-8))


class MemoryCrossAttention(nn.Module):
    def __init__(self, source_attention, rank, alpha):
        super().__init__()
        self.q = FrozenBaseLoRALinear(source_attention.q, rank, alpha)
        self.k = FrozenBaseLoRALinear(source_attention.k, rank, alpha)
        self.v = FrozenBaseLoRALinear(source_attention.v, rank, alpha)
        self.o = FrozenBaseLoRALinear(source_attention.o, rank, alpha)
        self.norm_q = copy.deepcopy(source_attention.norm_q).float().requires_grad_(False)
        self.norm_k = copy.deepcopy(source_attention.norm_k).float().requires_grad_(False)
        self.heads = source_attention.num_heads
        self.head_dim = source_attention.head_dim
        self.eps = 1e-6

    def forward(self, state, context, text_mask):
        batch = state.shape[0]
        split = lambda x: x.reshape(batch, -1, self.heads, self.head_dim).transpose(1, 2)
        q = split(self.norm_q(self.q(norm0(state, self.eps))))
        k = split(self.norm_k(self.k(context)))
        v = split(self.v(context))
        value = F.scaled_dot_product_attention(
            q, k.to(q.dtype), v.to(q.dtype), attn_mask=text_mask[:, None, None, :])
        return self.o(value.transpose(1, 2).flatten(2))


class MemorySelfAttention(nn.Module):
    def __init__(self, source_attention, rank, alpha):
        super().__init__()
        self.q = FrozenBaseLoRALinear(source_attention.q, rank, alpha)
        self.k = FrozenBaseLoRALinear(source_attention.k, rank, alpha)
        self.v = FrozenBaseLoRALinear(source_attention.v, rank, alpha)
        self.o = FrozenBaseLoRALinear(source_attention.o, rank, alpha)
        self.norm_q = copy.deepcopy(source_attention.norm_q).float().requires_grad_(False)
        self.norm_k = copy.deepcopy(source_attention.norm_k).float().requires_grad_(False)
        self.heads = source_attention.num_heads
        self.head_dim = source_attention.head_dim
        self.eps = 1e-6

    def forward(self, state):
        batch = state.shape[0]
        split = lambda x: x.reshape(batch, -1, self.heads, self.head_dim).transpose(1, 2)
        q = split(self.norm_q(self.q(norm0(state, self.eps))))
        k = split(self.norm_k(self.k(norm0(state, self.eps))))
        v = split(self.v(norm0(state, self.eps)))
        value = F.scaled_dot_product_attention(q, k.to(q.dtype), v.to(q.dtype))
        return self.o(value.transpose(1, 2).flatten(2))


class MemoryFFN(nn.Module):
    def __init__(self, source_ffn, rank, alpha):
        super().__init__()
        self.linear1 = FrozenBaseLoRALinear(source_ffn[0], rank, alpha)
        self.linear2 = FrozenBaseLoRALinear(source_ffn[2], rank, alpha)
        self.eps = 1e-6

    def forward(self, state):
        hidden = F.gelu(self.linear1(norm0(state, self.eps)), approximate='tanh')
        return self.linear2(hidden)


class SemanticPhysicsMemoryEnhancer(nn.Module):
    """C's two shared rounds over the 256 physical positions."""

    def __init__(self, source_block, cfg):
        super().__init__()
        memory = cfg['model']['memory_enhancer']
        rank, alpha = memory['lora_rank'], memory['lora_rank']
        self.cross = MemoryCrossAttention(source_block.cross_attn, rank, alpha)
        self.self_attention = MemorySelfAttention(source_block.self_attn, rank, alpha)
        self.ffn = MemoryFFN(source_block.ffn, rank, alpha)
        self.rounds = memory['rounds_shared']
        self.width = cfg['model']['width']
        self.route_projection = nn.Linear(self.width, self.width, bias=False)
        if memory['route_projection_init'] != 'identity':
            raise ValueError('C route projection must be identity-initialized')
        nn.init.eye_(self.route_projection.weight)
        self.eps = 1e-6

    def forward(self, execution, context, text_mask):
        state = execution
        for _ in range(self.rounds):
            state = state + self.cross(norm0(state, self.eps), context, text_mask)
            state = state + self.self_attention(norm0(state, self.eps))
            state = state + self.ffn(norm0(state, self.eps))
        return state

    def enhance(self, raw, context, text_mask):
        enhanced_execution = self.forward(raw['execution'], context, text_mask)
        enhanced_routing = raw['routing'] + self.route_projection(
            enhanced_execution.float() - raw['execution'].float())
        return dict(
            routing=enhanced_routing,
            execution=enhanced_execution,
            raw=dict(routing=raw['routing'], execution=raw['execution']),
        )
