"""Physical conditions control native interactions and local feature formation.

One native attention and one native FFN remain. The condition steers video Q/K
and scales actual native FFN preactivations; it supplies no additive image
feature or independently learned output basis. The local execution read sees
the state after the original caption attention.
"""
import copy

import torch
from torch import nn
from torch.nn import functional as F

from .r18e_reference.readers import DirectPhysicalReader


class NativePhysicalExecution(nn.Module):
    def __init__(self, native_caption_attention, ffn_width):
        super().__init__()
        width = native_caption_attention.dim
        self.reader = DirectPhysicalReader(copy.deepcopy(native_caption_attention))
        self.reader.query_base.float().requires_grad_(True)
        self.reader.query_norm_base.float().requires_grad_(True)
        # Only the Q/K update maps start at zero. The independent physical O
        # starts from the native caption O, avoiding a doubly-zero gradient path.
        self.read_output = copy.deepcopy(native_caption_attention.o).float().requires_grad_(True)
        if self.read_output.bias is not None:
            self.read_output.bias.requires_grad_(False)
        self.query_update = nn.Linear(width, width, bias=False)
        self.key_update = nn.Linear(width, width, bias=False)
        nn.init.zeros_(self.query_update.weight)
        nn.init.zeros_(self.key_update.weight)
        self.feature_gain = nn.Linear(width, ffn_width, bias=False)
        nn.init.zeros_(self.feature_gain.weight)
        self.layer_modulation = nn.Parameter(torch.zeros(1, 1, 3, width))

    def read_state(self, state, route, execute, groups, times, physical_mod,
                   spatial, mode):
        mod = physical_mod+self.layer_modulation
        shift, scale, gate = mod.unbind(-2)
        if mode == 'noise_modulation_off':
            shift, scale, gate = torch.zeros_like(shift), torch.zeros_like(scale), torch.ones_like(gate)
        expand = lambda v: v.repeat_interleave(spatial, dim=1)
        query = F.layer_norm(state.float(), (state.shape[-1],), eps=1e-6)
        query = query*(1+expand(scale))+expand(shift)
        read = self.reader(query, route, execute, groups, times, spatial)
        return F.linear(read, self.read_output.weight, None)*expand(gate)

    def condition_features(self, preactivated, read, local):
        # The full native FFN width is conditioned. Zero initialization gives
        # exactly the original activation; no C-only bias can create features
        # where the native preactivation is zero. The positive gain preserves
        # feature sign while allowing suppression and amplification.
        # A common physical value must still interact with this position's
        # actual native features, rather than becoming one global style gain.
        local = F.layer_norm(local.float(), (local.shape[-1],), eps=1e-6)
        gain = self.feature_gain(read*local).tanh().to(preactivated.dtype)
        return preactivated*(1+gain), gain

    def forward(self, block, state, native_mod, lengths, grid, freqs, context,
                route, execute, groups, times, physical_mod, spatial, mode):
        from wan.modules.model import flash_attention, rope_apply
        parts = (block.modulation.unsqueeze(0)+native_mod.float()).unbind(2)
        h = block.norm1(state).float()*(1+parts[1])+parts[0]
        attention = block.self_attn
        q, k = attention.q(h), attention.k(h)
        update_rms = state.new_zeros(())
        feature_mean, feature_max = state.new_zeros(()), state.new_zeros(())
        if mode != 'physical_condition_off':
            read = self.read_state(state, route, execute, groups, times,
                                   physical_mod, spatial, mode)
            # Couple physical information to the local native video features.
            # A shared physical read must not become an independent common Q/K
            # offset that acquires a spatial pattern solely through RoPE.
            dq, dk = self.query_update(q*read), self.key_update(k*read)
            if self.training and torch.is_grad_enabled():
                update_rms = (dq.detach().float().square().mean()+dk.detach().float().square().mean()).sqrt()
                update_rms = update_rms/(q.detach().float().square().mean()+k.detach().float().square().mean()).sqrt().clamp_min(1e-8)
            q, k = q+dq, k+dk
        shape = (*h.shape[:2], attention.num_heads, attention.head_dim)
        # Normalize AFTER physical conditioning: routing changes directions,
        # without adding uncontrolled attention-logit scale to the native block.
        q, k = attention.norm_q(q).view(shape), attention.norm_k(k).view(shape)
        values = attention.v(h).view(shape)
        attended = flash_attention(q=rope_apply(q, grid, freqs),
            k=rope_apply(k, grid, freqs), v=values,
            k_lens=lengths, window_size=attention.window_size)
        state = state.float()+attention.o(attended.flatten(2)).float()*parts[2]
        # The original caption update is applied once, before local execution.
        state = state+block.cross_attn(block.norm3(state), context, None)
        local = block.norm2(state).float()*(1+parts[4])+parts[3]
        preactivated = block.ffn[0](local)
        if mode not in ('physical_condition_off', 'physical_feature_off'):
            read = self.read_state(state, route, execute, groups, times,
                                   physical_mod, spatial, mode)
            preactivated, gain = self.condition_features(preactivated, read, local)
            if self.training and torch.is_grad_enabled():
                feature_mean = gain.detach().float().abs().mean()
                feature_max = gain.detach().float().abs().max()
        value = block.ffn[2](block.ffn[1](preactivated))
        state = state.float()+value.float()*parts[5]
        return state, torch.stack((update_rms, feature_mean, feature_max))
