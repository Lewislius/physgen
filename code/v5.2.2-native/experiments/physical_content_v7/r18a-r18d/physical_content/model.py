"""R18A-D velocity model.

Every variant traverses the native Wan blocks exactly once.  Injection layers
use an explicit native SA -> text-CA -> new content -> optional injection ->
native FFN order.  A/C/D add their expert after the complete native block; D
uses the already computed text update from that same layer and never reruns the
native text attention.
"""
import copy
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from physgen_native.context_alignment import native_text_context
from physgen_native.generator import load_wan, sinusoidal_embedding_1d

from .base_bank import OfficialBaseBank
from .content import Enhancement, SemanticPhysicsMemoryEnhancer
from .experts import (GuidedPhysicsExecutor, InBlockInjector, JointPhysicsExecutor,
                      PhysicalExecutor)
from .lora import assert_native_text_frozen, install_video_lora
from .readers import PhysicalReader, SemanticReader
from .wan_loader import ensure_wan_namespace


def make_text_mask(text, text_len, device):
    if text.ndim == 2:
        text = text[None]
    length = min(int(text.shape[1]), int(text_len))
    mask = torch.zeros((text.shape[0], text_len), dtype=torch.bool, device=device)
    mask[:, :length] = True
    return text, mask


class PhysicalModel(nn.Module):
    def __init__(self, cfg, device='cuda', wan=None, interface=None):
        super().__init__()
        self.cfg = cfg
        self.variant = cfg['variant']
        if wan is None:
            ensure_wan_namespace(cfg)
            self.wan = load_wan(cfg, device)
        else:
            self.wan = wan.requires_grad_(False)
        self.base_bank = OfficialBaseBank(self.wan)
        if interface is None:
            interface = torch.load(cfg['paths']['interface'], map_location='cpu', weights_only=True)
        seed = int(cfg['initialization']['new_adapter_seed'])
        devices = [torch.device(device).index or 0] if str(device).startswith('cuda') else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            if devices:
                torch.cuda.manual_seed_all(seed)
            self.enhancement = Enhancement(cfg, interface).to(device)
            if self.variant == 'A':
                self.experts = nn.ModuleDict({
                    str(site): JointPhysicsExecutor(self.base_bank, site, cfg).to(device)
                    for site in cfg['model']['sites_zero_based']})
            elif self.variant == 'B':
                self.injectors = nn.ModuleDict({
                    str(site): InBlockInjector(self.base_bank, site, cfg).to(device)
                    for site in cfg['model']['sites_zero_based']})
            elif self.variant == 'C':
                self.memory = SemanticPhysicsMemoryEnhancer(
                    self.wan.blocks[cfg['model']['memory_enhancer']['source_layer_zero_based']],
                    cfg).to(device)
                self.executors = nn.ModuleDict({
                    str(site): PhysicalExecutor(self.base_bank, site, cfg).to(device)
                    for site in cfg['model']['sites_zero_based']})
            elif self.variant == 'D':
                self.executors = nn.ModuleDict({
                    str(site): GuidedPhysicsExecutor(self.base_bank, site, cfg).to(device)
                    for site in cfg['model']['sites_zero_based']})
            else:
                raise ValueError(self.variant)
            self.video_lora_targets = install_video_lora(self.wan, cfg)
        assert_native_text_frozen(self.wan)
        self.learned_names = tuple(
            name for name, parameter in self.named_parameters() if parameter.requires_grad)
        self.checkpointing = cfg['train']['gradient_checkpointing']
        self.base_fingerprints = self.base_bank.hashes(full=False)

    # ------------------------------------------------------------------ content
    def _context(self, text):
        text, text_mask = make_text_mask(text, self.wan.text_len, text.device)
        parameter = next(self.wan.text_embedding.parameters())
        with torch.no_grad():
            context = native_text_context(self.wan, text.to(parameter.dtype))
        return context, text_mask

    def prepare_content(self, evidence, text):
        """Raw C plus the optional C-only semantic-memory transformation.

        The returned context/mask are immutable inputs for the same condition
        arm and may be reused across denoising steps.  No memory is detached in
        training; when this method is called under autograd its graph is kept.
        """
        raw, groups = self.enhancement(evidence)
        context, text_mask = self._context(text)
        if self.variant == 'C':
            content = self.memory.enhance(raw, context, text_mask)
        else:
            content = dict(routing=raw['routing'], execution=raw['execution'], raw=raw)
        content['context'] = context
        content['text_mask'] = text_mask
        return content, groups

    # --------------------------------------------------------------- native parts
    @staticmethod
    def native_parts(block, modulation):
        return tuple(piece.squeeze(2) for piece in
                     (block.modulation.unsqueeze(0).float() + modulation).chunk(6, dim=2))

    @staticmethod
    def native_self(block, hidden, parts, lengths, grid, freqs):
        y = block.self_attn(
            block.norm1(hidden).float() * (1 + parts[1]) + parts[0],
            lengths, grid, freqs)
        return hidden.float() + y.float() * parts[2]

    @staticmethod
    def native_text(block, hidden, context):
        return block.cross_attn(block.norm3(hidden), context, None)

    @staticmethod
    def native_ffn(block, hidden, parts):
        y = block.ffn(block.norm2(hidden).float() * (1 + parts[4]) + parts[3])
        return hidden.float() + y.float() * parts[5]

    # ------------------------------------------------------------------ velocity
    def velocity(self, noisy, first, text, sigma, c, groups=None, times=None,
                 target=None, valid=None, **legacy):
        wan = self.wan
        if noisy.shape[0] != 1:
            raise ValueError('One video per GPU; use gradient accumulation')
        if groups is None:
            raise ValueError('Every R18 reader requires relation group labels')
        if wan.freqs is not None and wan.freqs.device != noisy.device:
            wan.freqs = wan.freqs.to(noisy.device)
        x = wan.patch_embedding(torch.cat((first, noisy[:, :, 1:]), 2))
        geometry = tuple(x.shape[2:])
        x = x.flatten(2).transpose(1, 2)
        length, known = x.shape[1], geometry[1] * geometry[2]
        position = torch.cat((sigma.new_zeros(known), sigma.expand(length - known) * 1000))
        with torch.autocast(noisy.device.type, enabled=False):
            e = wan.time_embedding(sinusoidal_embedding_1d(wan.freq_dim, position).float()[None])
            modulation = wan.time_projection(e).unflatten(-1, (6, wan.dim))
        if 'context' in c:
            context, text_mask = c['context'], c['text_mask']
        else:
            context, text_mask = self._context(text)
        route, execute = c['routing'], c['execution']
        grid = torch.tensor([geometry], dtype=torch.long)
        lengths = torch.tensor([length], dtype=torch.long)
        aux = x.new_zeros((), dtype=torch.float32)
        sites = {str(site) for site in self.cfg['model']['sites_zero_based']}

        for index, block in enumerate(wan.blocks):
            site = str(index)

            def run(hidden, route_, execute_, layer=block, site_=site):
                if site_ not in sites:
                    return layer(hidden, modulation, lengths, grid, wan.freqs, context, None)
                parts = self.native_parts(layer, modulation)
                if self.variant == 'A':
                    native = layer(hidden, modulation, lengths, grid, wan.freqs, context, None)
                    delta = self.experts[site_](
                        native.float(), context, text_mask, route_, execute_, groups,
                        parts, known, lengths, grid, wan.freqs)
                    return native.float() + delta.float()
                if self.variant == 'B':
                    h_sa = self.native_self(layer, hidden, parts, lengths, grid, wan.freqs)
                    text_update = self.native_text(layer, h_sa, context)
                    h_t = h_sa + text_update.float()
                    delta = self.injectors[site_](
                        h_t, context, text_mask, route_, execute_, groups, known)
                    h_mix = h_t + delta.float()
                    return self.native_ffn(layer, h_mix, parts)
                if self.variant == 'C':
                    native = layer(hidden, modulation, lengths, grid, wan.freqs, context, None)
                    delta = self.executors[site_](
                        native.float(), route_, execute_, groups, parts, known,
                        lengths, grid, wan.freqs)
                    return native.float() + delta.float()
                if self.variant == 'D':
                    h_sa = self.native_self(layer, hidden, parts, lengths, grid, wan.freqs)
                    text_update = self.native_text(layer, h_sa, context)
                    h_t = h_sa + text_update.float()
                    native = self.native_ffn(layer, h_t, parts)
                    delta = self.executors[site_](
                        native.float(), text_update, route_, execute_, groups,
                        parts, known, lengths, grid, wan.freqs)
                    return native.float() + delta.float()
                raise ValueError(self.variant)

            if self.checkpointing and torch.is_grad_enabled():
                x = checkpoint(run, x, route, execute, use_reentrant=False)
            else:
                x = run(x, route, execute)

        prediction = wan.unpatchify(wan.head(x, e), grid)[0][None].float()
        return prediction, aux

    # ---------------------------------------------------------------- parameter roles
    def _role(self, name):
        if name.startswith('wan.'):
            if 'cross_attn' in name:
                raise ValueError(f'Native text adapter appeared: {name}')
            return 'wan_video_lora'
        if name.startswith('enhancement.'):
            return 'enhancement'
        if name.startswith('memory.'):
            return 'content_memory'
        if '.writer.' in name:
            return 'writer'
        if '.semantic.' in name or name.startswith('semantic_readers.'):
            return 'semantic_reader_lora'
        if '.physical.' in name or name.startswith('physical_readers.'):
            return 'physics_read'
        if '.uq.' in name or '.uk.' in name or '.g.' in name:
            return 'semantic_guide'
        if '.attention.' in name or '.ffn.' in name:
            return 'expert_lora'
        raise ValueError(f'Unassigned trainable parameter: {name}')

    def optim_groups(self):
        named = dict(self.named_parameters())
        grouped = {}
        for name in self.learned_names:
            role = self._role(name)
            decay = named[name].ndim >= 2
            key = (role, decay)
            grouped.setdefault(key, []).append(named[name])
        groups = []
        for role in ('wan_video_lora', 'enhancement', 'physics_read', 'writer',
                     'expert_lora', 'semantic_reader_lora', 'content_memory',
                     'semantic_guide'):
            for decay in (True, False):
                params = grouped.get((role, decay), [])
                if params:
                    groups.append(dict(
                        name=f'{role}:{"matrix" if decay else "no_decay"}',
                        role=role,
                        params=params,
                        lr=self.cfg['train']['peak_lrs'][role],
                        weight_decay=(0.01 if decay else 0.0),
                        grad_clip=self.cfg['train']['grad_clip_per_group'],
                    ))
        assigned = sum(len(group['params']) for group in groups)
        if assigned != len(self.learned_names):
            raise AssertionError('Some trainable parameters were not assigned exactly once')
        return groups

    def set_phase(self, step):
        self.wan.eval()
        return 'joint'

    def constant_state(self):
        return {name: value.detach().cpu() for name, value in self.named_buffers()
                if name.startswith('enhancement.')}

    def trainable_count(self):
        named = dict(self.named_parameters())
        return {name: named[name].numel() for name in self.learned_names}

    def assert_base_unchanged(self, before, full=False):
        after = self.base_bank.hashes(full=full)
        if after != before:
            changed = [name for name in before if before[name] != after.get(name)]
            raise AssertionError(f'Official frozen base changed: {changed[:8]}')
        return True
