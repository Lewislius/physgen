"""R18E-H NoChunks-v2 velocity model.

The native Wan block is traversed once.  At a mechanism site we explicitly run
native self-attention, native text attention, the selected physical mechanism,
and (for mechanism 2) the native FFN with its conditioned rank-64 residual.
Every other block calls the official block unchanged.
"""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from physgen_native.context_alignment import native_text_context
from physgen_native.generator import load_wan, sinusoidal_embedding_1d

from .base_bank import OfficialBaseBank
from .content import Enhancement
from .lora import assert_native_text_frozen, install_video_lora
from .mechanisms import Mechanism1, Mechanism2
from .wan_loader import ensure_wan_namespace


def make_text_mask(text, text_len, device):
    if text.ndim == 2:
        text = text[None]
    length = min(int(text.shape[1]), int(text_len))
    mask = torch.zeros((text.shape[0], text_len), dtype=torch.bool, device=device)
    mask[:, :length] = True
    return text, mask


def _attention(q, k, v):
    """Native text attention in [B,H,N,dh] layout for both kernels."""
    if q.is_cuda:
        from wan.modules.attention import flash_attention
        return flash_attention(q=q.transpose(1, 2), k=k.transpose(1, 2),
                               v=v.transpose(1, 2), k_lens=None).transpose(1, 2)
    return F.scaled_dot_product_attention(q, k, v)


class PhysicalModel(nn.Module):
    def __init__(self, cfg, device='cuda', wan=None, interface=None):
        super().__init__()
        self.cfg = cfg
        self.plan = cfg['plan']
        self.task = cfg['task']
        self.mechanism = int(cfg['mechanism'])
        if wan is None:
            ensure_wan_namespace(cfg)
            self.wan = load_wan(cfg, device)
        else:
            self.wan = wan.requires_grad_(False)
        self.base_bank = OfficialBaseBank(self.wan)
        if interface is None:
            interface = torch.load(cfg['paths']['interface'], map_location='cpu',
                                   weights_only=True)
        seed = int(cfg['initialization']['new_adapter_seed'])
        devices = [torch.device(device).index or 0] if str(device).startswith('cuda') else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            if devices:
                torch.cuda.manual_seed_all(seed)
            self.enhancement = Enhancement(cfg, interface).to(device)
            module = Mechanism1 if self.mechanism == 1 else Mechanism2
            self.mechanisms = nn.ModuleDict({
                str(site): module(self.base_bank, site, cfg).to(device)
                for site in cfg['model']['injection_layers']})
            self.video_lora_targets = install_video_lora(self.wan, cfg)
        assert_native_text_frozen(self.wan)
        self.learned_names = tuple(
            name for name, parameter in self.named_parameters() if parameter.requires_grad)
        self.checkpointing = cfg['train']['gradient_checkpointing']
        self.base_fingerprints = self.base_bank.hashes(full=False)

    def _context(self, text):
        text, text_mask = make_text_mask(text, self.wan.text_len, text.device)
        parameter = next(self.wan.text_embedding.parameters())
        with torch.no_grad():
            context = native_text_context(self.wan, text.to(parameter.dtype))
        return context, text_mask

    def prepare_content(self, evidence, text):
        raw, groups = self.enhancement(evidence)
        context, text_mask = self._context(text)
        content = dict(
            routing=raw['routing'], execution=raw['execution'], raw=raw,
            context=context, text_mask=text_mask)
        return content, groups

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
    def text_parts(block, x_norm3, context):
        attention = block.cross_attn
        batch = x_norm3.shape[0]
        split = lambda value: value.reshape(batch, -1, attention.num_heads,
                                            attention.head_dim).transpose(1, 2)
        q = split(attention.norm_q(attention.q(x_norm3))).float()
        k = split(attention.norm_k(attention.k(context))).float()
        v = split(attention.v(context)).float()
        update = attention.o(_attention(q, k, v).transpose(1, 2).flatten(2)).float()
        return q, k, v, update

    @staticmethod
    def native_ffn(block, hidden, parts):
        y = block.ffn(block.norm2(hidden).float() * (1 + parts[4]) + parts[3])
        return hidden.float() + y.float() * parts[5]

    def _geometry(self, noisy, first):
        wan = self.wan
        if self.task == 't2v':
            if first is not None:
                raise ValueError('T2V must not receive an external first frame')
            x = wan.patch_embedding(noisy)
            known = 0
        else:
            if first is None:
                raise ValueError('I2V requires its known first latent')
            x = wan.patch_embedding(torch.cat((first, noisy[:, :, 1:]), 2))
            known = None
        geometry = tuple(x.shape[2:])
        x = x.flatten(2).transpose(1, 2)
        if known is None:
            known = geometry[1] * geometry[2]
        return x, geometry, known

    def velocity(self, noisy, first, text, sigma, c, groups=None, times=None,
                 target=None, valid=None, physical_mode='normal', **legacy):
        if physical_mode not in ('normal', 'off', 'ffn_off'):
            raise ValueError(f'Unknown physical intervention: {physical_mode}')
        if noisy.shape[0] != 1:
            raise ValueError('One video per GPU; use gradient accumulation')
        if groups is None:
            raise ValueError('Every NoChunks reader requires relation group labels')
        wan = self.wan
        if wan.freqs is not None and wan.freqs.device != noisy.device:
            wan.freqs = wan.freqs.to(noisy.device)
        x, geometry, known = self._geometry(noisy, first)
        length = x.shape[1]
        if self.task == 'i2v':
            position = torch.cat((sigma.new_zeros(known),
                                  sigma.expand(length - known) * 1000))
        else:
            position = sigma.expand(length) * 1000
        with torch.autocast(noisy.device.type, enabled=False):
            e = wan.time_embedding(sinusoidal_embedding_1d(wan.freq_dim, position).float()[None])
            modulation = wan.time_projection(e).unflatten(-1, (6, wan.dim))
        context, text_mask = c['context'], c['text_mask']
        route, execute = c['routing'], c['execution']
        grid = torch.tensor([geometry], dtype=torch.long)
        lengths = torch.tensor([length], dtype=torch.long)
        sites = {str(site) for site in self.cfg['model']['injection_layers']}
        if physical_mode == 'off':
            sites = set()
        spatial = geometry[1] * geometry[2]
        future_start = known if self.task == 'i2v' else None
        write_mask = (torch.cat((x.new_zeros(1, known, 1), x.new_ones(1, length - known, 1)), 1)
                      if self.task == 'i2v' else x.new_ones(1, length, 1))

        def run_block(hidden, route_, execute_, block, site):
            parts = self.native_parts(block, modulation)
            h_sa = self.native_self(block, hidden, parts, lengths, grid, wan.freqs)
            x_norm3 = block.norm3(h_sa)
            if site in sites:
                qt, text_keys, text_values, text_update = self.text_parts(
                    block, x_norm3, context)
                module = self.mechanisms[site]
                if self.mechanism == 1:
                    delta = module(
                        h_sa, x_norm3, qt, text_keys, text_values, text_mask,
                        route_, execute_, groups, times, spatial, future_start)
                    h_mix = h_sa + text_update + delta
                    z_full = None
                else:
                    direct, z_full = module(
                        h_sa, x_norm3, text_keys, text_values, text_mask,
                        route_, execute_, groups, times, spatial, future_start)
                    h_mix = h_sa + text_update + direct
            else:
                text_update = self.native_text(block, h_sa, context)
                h_mix = h_sa + text_update
                z_full = None

            u = block.norm2(h_mix).float() * (1 + parts[4]) + parts[3]
            if self.mechanism == 2 and site in sites:
                f = F.gelu(block.ffn[0](u), approximate='tanh')
                y = block.ffn[2](f)
                delta_ff = (torch.zeros_like(y) if physical_mode == 'ffn_off' else
                            self.mechanisms[site].conditioned(f, z_full, write_mask))
                return h_mix + (y + delta_ff) * parts[5]
            y = block.ffn(u)
            return h_mix + y * parts[5]

        for index, block in enumerate(wan.blocks):
            site = str(index)
            if self.checkpointing and torch.is_grad_enabled():
                x = checkpoint(run_block, x, route, execute, block, site,
                               use_reentrant=False)
            else:
                x = run_block(x, route, execute, block, site)

        prediction = wan.unpatchify(wan.head(x, e), grid)[0][None].float()
        return prediction, x.new_zeros((), dtype=torch.float32)

    def _role(self, name):
        if name.startswith('wan.'):
            if 'cross_attn' in name:
                raise ValueError(f'Native text adapter appeared: {name}')
            return 'wan_video_lora'
        if name.startswith('enhancement.'):
            return 'enhancement'
        if '.conditioned.' in name:
            return 'mechanism_adapter'
        if '.physical.' in name:
            return 'physics_read'
        if any(token in name for token in ('.state.', '.direct_writer.',
                                           '.semantic_writer.', '.writer.')):
            return 'physics_write'
        raise ValueError(f'Unassigned trainable parameter: {name}')

    def optim_groups(self):
        named = dict(self.named_parameters())
        grouped = {}
        for name in self.learned_names:
            role = self._role(name)
            decay = named[name].ndim >= 2
            grouped.setdefault((role, decay), []).append(named[name])
        groups = []
        for role in ('wan_video_lora', 'enhancement', 'physics_read',
                     'physics_write', 'mechanism_adapter'):
            for decay in (True, False):
                params = grouped.get((role, decay), [])
                if params:
                    groups.append(dict(
                        name=f'{role}:{"matrix" if decay else "no_decay"}',
                        role=role, params=params,
                        lr=self.cfg['train']['peak_lrs'][role],
                        weight_decay=(0.01 if decay else 0.0),
                        grad_clip=1.0))
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
