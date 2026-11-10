"""Full-depth Qwen recurrence over an immutable, differentiable prefix.

No second transformer, pooled evidence, detached KV, or mutable decode cache.
The prefix is encoded once by the multimodal model (including DeepStack). Each
readout traverses the SAME native text layers from their input embedding space.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import math
import torch
from torch import nn
from .lora import role_checkpoint


@contextmanager
def capture_prefix_projections(language):
    """Capture native K-normalization/V outputs without changing their graph.

Hooks are removed before backward. Non-reentrant checkpoint recomputation only
reproduces native tensor operations; it cannot overwrite retained prefix KV.
RoPE is applied to captured K once after the multimodal forward.
"""
    keys, values, handles = {}, {}, []
    for index, layer in enumerate(language.layers):
        def key_hook(module, args, output, index=index): keys[index] = output
        def value_hook(module, args, output, index=index): values[index] = output
        handles.append(layer.self_attn.k_norm.register_forward_hook(key_hook))
        handles.append(layer.self_attn.v_proj.register_forward_hook(value_hook))
    try:
        yield keys, values
    finally:
        for handle in handles: handle.remove()


@dataclass(frozen=True)
class PrefixMemory:
    kv: tuple
    valid: torch.Tensor
    positions: torch.Tensor

    @classmethod
    def from_projections(cls, language, captured, hidden, positions, valid):
        from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb
        keys, values = captured
        if set(keys) != set(range(len(language.layers))) or set(values) != set(keys):
            raise RuntimeError('Every native Qwen layer must contribute prefix KV')
        cos, sin = language.rotary_emb(hidden, positions[-3:])
        pairs = []
        for index, layer in enumerate(language.layers):
            # k_norm output is [B,L,KV_heads,head_dim], before rotary embedding.
            key = keys[index].transpose(1, 2)
            key = apply_rotary_pos_emb(key, key, cos, sin)[1]
            value = values[index].reshape(*hidden.shape[:2], -1, layer.self_attn.head_dim).transpose(1, 2)
            pairs.append((key, value))
        return cls(tuple(pairs), valid.bool(), positions)


class _ReadOnlyLayerCache:
    """The native attention API requests update; return a concatenation only."""
    def __init__(self, key, value, index):
        self.key, self.value, self.index = key, value, index

    def update(self, key, value, layer_idx, cache_kwargs=None):
        if layer_idx != self.index: raise ValueError('Prefix KV belongs to another layer')
        return torch.cat((self.key, key), -2), torch.cat((self.value, value), -2)


def read_prefix(language, embeddings, memory, *, checkpointing=False):
    """One full native Qwen pass over working tokens, with fixed prefix KV."""
    batch, count, _ = embeddings.shape
    if len(memory.kv) != len(language.layers): raise ValueError('Prefix memory must cover every Qwen layer')
    if memory.valid.shape[0] != batch or not bool(memory.valid.any(-1).all()):
        raise ValueError('Each working sequence needs a nonempty valid prefix')
    length = memory.valid.shape[1]
    # Text continuation after the last VALID multimodal coordinate, identical
    # across rounds. Physical cache indices and mRoPE coordinates are distinct.
    coords = memory.positions[-3:].masked_fill(~memory.valid[None], -1)
    start = coords.amax(dim=(0, 2)) + 1
    positions = (start[:, None] + torch.arange(count, device=embeddings.device)[None])[None].expand(3, -1, -1)
    rotary = language.rotary_emb(embeddings, positions)
    local = torch.ones(count, count, device=embeddings.device, dtype=torch.bool).tril()
    allowed = torch.cat((memory.valid[:, None].expand(-1, count, -1), local[None].expand(batch, -1, -1)), -1)
    mask = embeddings.new_zeros(batch, 1, count, length+count).masked_fill(~allowed[:, None], float('-inf'))
    cache_positions = torch.arange(length, length+count, device=embeddings.device)
    state = embeddings
    for index, (layer, (key, value)) in enumerate(zip(language.layers, memory.kv)):
        def run(hidden, prefix_key, prefix_value, layer=layer, index=index):
            # Bypass GradientCheckpointingLayer.__call__: that wrapper clears
            # past_key_values. Our functional checkpoint takes KV explicitly.
            return layer.forward(hidden, attention_mask=mask, position_ids=positions[0],
                past_key_values=_ReadOnlyLayerCache(prefix_key, prefix_value, index),
                cache_position=cache_positions, position_embeddings=rotary, use_cache=False)
        state = role_checkpoint(run, state, key, value) if checkpointing and torch.is_grad_enabled() else run(state, key, value)
    return language.norm(state)


class RecurrentWorkspace(nn.Module):
    def __init__(self, width, tokens=32, init_std=.02):
        super().__init__()
        self.query_seed = nn.Parameter(torch.empty(1, tokens, width))
        self.feedback_norm = nn.RMSNorm(width)
        # Full-width, full-rank-parameterized bridge: no 64-dimensional layer.
        self.feedback = nn.Linear(width, width, bias=False)
        nn.init.normal_(self.query_seed, std=init_std)
        nn.init.normal_(self.feedback.weight, std=init_std/math.sqrt(width))

    def input_from(self, previous, batch, dtype):
        query = self.query_seed.expand(batch, -1, -1)
        if previous is not None:
            value = self.feedback_norm(previous.to(self.feedback_norm.weight.dtype))
            query = query + self.feedback(value.to(self.feedback.weight.dtype)).to(query.dtype)
        return query.to(dtype)

    def forward(self, language, memory, dtype, *, steps, checkpointing=False):
        if not isinstance(steps, int) or steps < 0: raise ValueError('Nonnegative additional update count required')
        states, state = [], None
        for _ in range(steps+1):
            embeddings = self.input_from(state, memory.valid.shape[0], dtype)
            state = read_prefix(language, embeddings, memory, checkpointing=checkpointing)
            states.append(state)
        return tuple(states)


def pack_conditions(tokens):
    """Pad independently encoded [1,N,D] conditions; True means valid token.

    Pass BOTH returned values to generator(..., condition_mask=mask). Padding
    alone is not neutral under attention, even if its numerical value is zero.
    """
    if not tokens or any(t.ndim != 3 or t.shape[0] != 1 or t.shape[1] == 0 for t in tokens):
        raise ValueError('Expected nonempty independently encoded [1,N,D] conditions')
    width = tokens[0].shape[-1]
    if any(t.shape[-1] != width or t.device != tokens[0].device or t.dtype != tokens[0].dtype for t in tokens):
        raise ValueError('Condition widths, devices and dtypes must match')
    lengths = torch.tensor([t.shape[1] for t in tokens], device=tokens[0].device)
    padded = torch.nn.utils.rnn.pad_sequence([t[0] for t in tokens], batch_first=True)
    mask = torch.arange(padded.shape[1], device=padded.device)[None] < lengths[:, None]
    return padded, mask
