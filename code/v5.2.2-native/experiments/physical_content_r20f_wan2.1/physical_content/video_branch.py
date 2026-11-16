"""Full-grid physical video processing; native Wan geometry/time coordinates."""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from .r18e_reference.readers import DirectPhysicalReader, semantic_binding


class NoiseConditioning(nn.Module):
    def __init__(self, width, associated):
        super().__init__()
        self.vectors = 4 if associated else 3
        self.project = nn.Sequential(nn.SiLU(), nn.Linear(width, self.vectors*width))
        nn.init.zeros_(self.project[1].weight)
        nn.init.zeros_(self.project[1].bias)
        with torch.no_grad():
            self.project[1].bias.view(self.vectors, width)[2:].fill_(1)

    def forward(self, embedding):
        return self.project(embedding).unflatten(-1, (self.vectors, embedding.shape[-1]))


class PhysicalConditionRead(nn.Module):
    """C conditions the physical VIDEO stream, rather than directly writing Wan."""
    def __init__(self, attention, associated):
        super().__init__()
        width = attention.dim
        self.associated = associated
        self.direct = DirectPhysicalReader(copy.deepcopy(attention))
        self.direct.query_base.float().requires_grad_(True)
        self.direct.query_norm_base.float().requires_grad_(True)
        self.direct_output = nn.Linear(width, width, bias=False)
        nn.init.zeros_(self.direct_output.weight)
        if associated:
            self.semantic_query = copy.deepcopy(attention.q).float().requires_grad_(False)
            self.semantic_norm = copy.deepcopy(attention.norm_q).float().requires_grad_(False)
            self.associated_output = nn.Linear(width, width, bias=False)
            nn.init.zeros_(self.associated_output.weight)
        self.layer_modulation = nn.Parameter(torch.zeros(1, 1, 4 if associated else 3, width))

    def forward(self, current, text_input, caption_attention, context, mask,
                route, execute, groups, times, modulation, spatial, known, mode):
        mod = modulation+self.layer_modulation
        shift, scale, gate = mod[:, :, 0], mod[:, :, 1], mod[:, :, 2]
        if mode == 'noise_modulation_off':
            shift, scale, gate = torch.zeros_like(shift), torch.zeros_like(scale), torch.ones_like(gate)
        expand = lambda v: v.repeat_interleave(spatial, dim=1)
        query = F.layer_norm(current.float(), (current.shape[-1],), eps=1e-6)
        query = query*(1+expand(scale))+expand(shift)
        read = self.direct(query, route, execute, groups, times, spatial, future_start=known or None)
        delta = self.direct_output(read)*expand(gate)[:, known:]
        if self.associated and mode != 'direct_only':
            query = self.direct.split(self.semantic_norm(self.semantic_query(text_input))).float()
            keys = self.direct.split(caption_attention.norm_k(caption_attention.k(context))).float()
            read = semantic_binding(query[:, :, known:], keys, None, mask, route, execute,
                                    self.direct.key_norm, self.direct.key, self.direct.value, groups)
            gate = mod[:, :, 3] if mode != 'noise_modulation_off' else torch.ones_like(mod[:, :, 3])
            delta = delta+self.associated_output(read)*expand(gate)[:, known:]
        if known:
            delta = torch.cat((torch.zeros_like(current[:, :known]), delta), 1)
        return delta


class PhysicalVideoBlock(nn.Module):
    """Full video state persists across blocks; original caption attention is frozen.

    C is processed by this FFN and subsequent 3-D attention. Separate output
    hints start at zero. The stream is not compressed to physical-relation slots.
    """
    def __init__(self, native_block, associated):
        super().__init__()
        self.native = copy.deepcopy(native_block).float().requires_grad_(True)
        self.native.cross_attn.requires_grad_(False)
        self.native.norm3.requires_grad_(False)
        self.conditioning = PhysicalConditionRead(self.native.cross_attn, associated)
        self.output = nn.Linear(native_block.dim, native_block.dim, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, state, native_mod, lengths, grid, freqs, context, text_mask,
                route, execute, groups, times, physical_mod, spatial, known, mode):
        block = self.native
        parts = (block.modulation.unsqueeze(0)+native_mod.float()).unbind(2)
        value = block.self_attn(block.norm1(state).float()*(1+parts[1])+parts[0], lengths, grid, freqs)
        state = state.float()+value.float()*parts[2]
        text_input = block.norm3(state)
        state = state+block.cross_attn(text_input, context, None)
        if mode != 'physical_condition_off':
            state = state+self.conditioning(state, text_input, block.cross_attn,
                                             context, text_mask, route, execute,
                                             groups, times, physical_mod, spatial, known, mode)
        value = block.ffn(block.norm2(state).float()*(1+parts[4])+parts[3])
        state = state+value.float()*parts[5]
        hint = self.output(state)
        if known:
            hint = torch.cat((torch.zeros_like(hint[:, :known]), hint[:, known:]), 1)
        return state, hint
