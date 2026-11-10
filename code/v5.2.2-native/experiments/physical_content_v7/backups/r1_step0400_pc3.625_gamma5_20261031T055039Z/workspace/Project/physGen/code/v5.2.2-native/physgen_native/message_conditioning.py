"""Frozen-Qwen content messages and a fixed 128-query condition workspace.

Both routes produce a single Wan-width condition, never concat(content, R).
The Qwen boundary is detached; P, the shared update block and alignment learn.
v25-v28 preserve Qwen width inside G and apply P only at the final Wan interface.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .attention import RMSNorm
from .context_alignment import FullWidthAlignment


MESSAGE_MODES = ('h_content_message', 'q_seed128_message')


class MessageAttention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        if width % heads:
            raise ValueError('Message width must divide into complete attention heads')
        self.heads, self.head_dim = heads, width // heads
        self.q, self.k, self.v, self.o = (nn.Linear(width, width, bias=False) for _ in range(4))

    def forward(self, query, memory, valid):
        def split(value):
            return value.unflatten(-1, (self.heads, self.head_dim)).transpose(1, 2)
        q, k, v = split(self.q(query)), split(self.k(memory)), split(self.v(memory))
        value = F.scaled_dot_product_attention(q, k, v, attn_mask=valid[:, None, None, :])
        return self.o(value.transpose(1, 2).flatten(2))


class MessageBlock(nn.Module):
    def __init__(self, width, heads, multiplier, *, exchange):
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.cross = MessageAttention(width, heads)
        self.exchange = exchange
        if exchange:
            self.exchange_norm = nn.LayerNorm(width)
            self.self_attention = MessageAttention(width, heads)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, width * multiplier, bias=False), nn.GELU(),
                                 nn.Linear(width * multiplier, width, bias=False))

    def forward(self, state, memory, memory_mask, state_mask):
        state = state + self.cross(self.query_norm(state), self.memory_norm(memory), memory_mask)
        if self.exchange:
            normalized = self.exchange_norm(state)
            state = state + self.self_attention(normalized, normalized, state_mask)
        state = state + self.ffn(self.ffn_norm(state))
        return state.masked_fill(~state_mask[..., None], 0)


class MessageCondition(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg['knowledge']
        self.mode = self.cfg['reasoning_mode']
        if self.mode not in MESSAGE_MODES:
            raise ValueError('Unknown message workspace')
        options = cfg['message']
        wan_width = cfg['model']['width']
        width = options.get('workspace_width', wan_width)
        self.output_projection = options.get('projection', 'input') == 'output'
        self.norm = RMSNorm(cfg['model']['qwen_width'])
        self.project = nn.Linear(cfg['model']['qwen_width'], wan_width, bias=False)
        self.block = MessageBlock(width, options['heads'], options['ffn_multiplier'],
                                  exchange=self.mode == 'q_seed128_message')
        if self.mode == 'q_seed128_message':
            self.query_seed = nn.Parameter(torch.empty(1, 128, width))
            nn.init.normal_(self.query_seed, std=options['init_std'])
        self.alignment = FullWidthAlignment(wan_width, cfg['alignment']['eps'])
        self.checkpointing = self.cfg['gradient_checkpointing']

    def _exit(self, requested):
        if requested is None:
            if self.training and torch.is_grad_enabled() and any(p.requires_grad for p in self.parameters()):
                exits = self.cfg['train_reasoning_rounds']
                requested = exits[0] if len(exits) == 1 else exits[int(torch.randint(len(exits), ()).item())]
            else:
                requested = self.cfg['inference_reasoning_round']
        if type(requested) is not int or not 0 <= requested <= self.cfg['latent_reasoning_steps']:
            raise ValueError('Message exit is outside the configured round range')
        return requested

    def forward(self, evidence, *, reasoning_round=None, all_rounds=False):
        chosen = self._exit(reasoning_round)
        memory = self.norm(evidence['memory'].detach())
        if not self.output_projection: memory = self.project(memory)
        memory_mask = evidence['memory_mask'].to(device=memory.device, dtype=torch.bool)
        if memory_mask.shape != memory.shape[:2] or not bool(memory_mask.any(-1).all()):
            raise ValueError('Message evidence requires nonempty valid memory per sample')
        positions = evidence['content_positions'].to(device=memory.device, dtype=torch.long)
        if positions.ndim == 1:
            positions = positions[None].expand(memory.shape[0], -1)
        content_mask = evidence.get('content_mask', torch.ones_like(positions, dtype=torch.bool))
        content_mask = content_mask.to(device=memory.device, dtype=torch.bool)
        if positions.shape != content_mask.shape or positions.shape[0] != memory.shape[0]:
            raise ValueError('Content positions and mask must have shape [B,T]')
        if not bool(content_mask.any(-1).all()):
            raise ValueError('Message content cannot be empty')
        # Padded positions are irrelevant and need not contain a valid index.
        positions = positions.masked_fill(~content_mask, 0)
        content = memory.gather(1, positions[..., None].expand(-1, -1, memory.shape[-1]))
        content = content.masked_fill(~content_mask[..., None], 0)
        query_mode = self.mode == 'q_seed128_message'
        state = self.query_seed.expand(memory.shape[0], -1, -1).to(memory.dtype) if query_mode else content
        mask = torch.ones(state.shape[:2], device=state.device, dtype=torch.bool) if query_mode else content_mask
        limit = self.cfg['latent_reasoning_steps'] if all_rounds else chosen
        states = [] if query_mode else [state]
        # Content U0 is the mapped content; query R0 is the first grounded read.
        for _ in range(limit + int(query_mode)):
            if self.checkpointing and torch.is_grad_enabled():
                state = checkpoint(self.block, state, memory, memory_mask, mask, use_reentrant=False)
            else:
                state = self.block(state, memory, memory_mask, mask)
            states.append(state)
        self.last_round, self.last_mask = chosen, mask
        def align(value):
            if self.output_projection: value = self.project(value)
            return self.alignment(value).masked_fill(~mask[..., None], 0)
        if all_rounds:
            return dict(content=content, relation_states=tuple(states), readouts=tuple(states),
                        conditions=tuple(align(s) for s in states), attention_mask=mask,
                        reasoning=evidence.get('reasoning', ''), reasoning_round=chosen)
        return align(states[chosen])
