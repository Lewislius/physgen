"""R19 video-state physical execution, with an unchanged native text path."""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from physgen_native.context_alignment import native_text_context
from physgen_native.generator import load_wan, sinusoidal_embedding_1d

from .base_bank import OfficialBaseBank
from .physical_blocks import SourceContent, InteractionExpert, SourceGroundedRead, NativeFeatureGate
from .evidence import resample_words
from .lora import assert_native_text_frozen, install_video_lora
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
            model = cfg['model']
            self.enhancement = SourceContent(
                model['qwen_width'], model['width'], model['q_heads'],
                model['q_tokens'], model['content_rounds']).to(device)
            self.register_buffer('condition_mean', interface['mean'].float().to(device))
            self.register_buffer('condition_scale', interface['scale'].float().to(device))
            self.physical_value = nn.Linear(model['width'], model['width'], bias=False).to(device)
            nn.init.eye_(self.physical_value.weight)
            self.state_expert = InteractionExpert(
                model['width'], model['native_heads'], model['ffn_width']).to(device)
            # Initialize from official pretrained operators BEFORE video LoRA.
            # The shared expert owns copies; no frozen backbone tensor is aliased.
            self.state_expert.initialize_from_official(
                self.wan.blocks[model['state_initialization_layer']])
            self.mechanisms = nn.ModuleDict({
                str(site): SourceGroundedRead(
                    model['width'], model['native_heads'],
                    value_projection=self.physical_value).to(device)
                for site in cfg['model']['injection_layers']})
            for site, module in self.mechanisms.items():
                module.initialize_from_official(self.wan.blocks[int(site)].cross_attn)
            self.feature_gates = nn.ModuleDict({
                str(site): NativeFeatureGate(model['width'], model['ffn_width']).to(device)
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

    def prepare_content(self, evidence, text, *, learned_queries=True):
        words = evidence['words'].detach().float()
        words = (words - self.condition_mean) / self.condition_scale
        anchor, groups, _ = resample_words(words, evidence['word_groups'],
                                         self.cfg['model']['q_tokens'])
        physics, _, _ = resample_words(evidence['native_condition'].detach().float(),
                                       evidence['word_groups'], self.cfg['model']['q_tokens'])
        request = (evidence['request_words'].detach().float() - self.condition_mean) / self.condition_scale
        native_words = evidence['native_condition'].detach().float()
        context, text_mask = self._context(text)
        text = text[None] if text.ndim == 2 else text
        count = min(text.shape[1], self.wan.text_len)
        raw = self.enhancement(
            anchor, physics, words[None] if words.ndim == 2 else words,
            request[None] if request.ndim == 2 else request,
            native_words[None] if native_words.ndim == 2 else native_words,
            groups, evidence['word_groups'], context[:, :count].float(),
            text_mask[:, :count], learned_queries=learned_queries)
        raw.update(routing=raw['route'], execution=raw['execute'])
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
        if physical_mode not in ('normal', 'off', 'no_video_feedback', 'no_temporal', 'no_relations'):
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
        write_mask = (torch.cat((x.new_zeros(1, known, 1), x.new_ones(1, length - known, 1)), 1)
                      if self.task == 'i2v' else x.new_ones(1, length, 1))
        if times is None:
            raise ValueError('Physical temporal reasoning requires presentation timestamps')
        times = times.reshape(1, geometry[0])

        def run_block(hidden, route_, execute_, block, site):
            parts = self.native_parts(block, modulation)
            h_sa = self.native_self(block, hidden, parts, lengths, grid, wan.freqs)
            h_text = h_sa + self.native_text(block, h_sa, context)
            if site in sites:
                module = self.mechanisms[site]
                content_read = module(
                    h_text.reshape(1, geometry[0], spatial, wan.dim),
                    block.norm3(h_text).reshape(1, geometry[0], spatial, wan.dim),
                    dict(route=route_, execute=execute_), groups, times, self.state_expert,
                    temporal=physical_mode != 'no_temporal',
                    relations=physical_mode != 'no_relations',
                    video_feedback=physical_mode != 'no_video_feedback',
                    native_parts=tuple(p.reshape(1, geometry[0], spatial, wan.dim)[:, :, 0]
                                       for p in parts))
                content_read = content_read.flatten(1, 2) * write_mask
            else:
                content_read = None
            u = block.norm2(h_text).float() * (1 + parts[4]) + parts[3]
            preactivated = block.ffn[0](u)
            activated = block.ffn[1](preactivated)
            if content_read is not None:
                activated = self.feature_gates[site](activated, content_read)
            # Ordinary and physical features use the SAME native output map
            # including its ordinary video LoRA, rather than a second basis.
            y = block.ffn[2](activated)
            return h_text + y * parts[5]

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
        if name.startswith(('state_expert.', 'physical_value.')):
            return 'physics_state'
        if name.startswith('mechanisms.'):
            return 'physics_read'
        if name.startswith('feature_gates.'):
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
        for role in ('wan_video_lora', 'enhancement', 'physics_state', 'physics_read', 'physics_write'):
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
        interface_steps = int(self.cfg['train'].get('interface_updates', 0))
        joint = step >= interface_steps
        for name, parameter in self.wan.named_parameters():
            if 'wan.' + name in self.learned_names:
                parameter.requires_grad_(joint)
        return 'joint' if joint else 'physical_interface'

    def constant_state(self):
        return {name: value.detach().cpu() for name, value in self.named_buffers()
                if name.startswith('enhancement.') or name in ('condition_mean', 'condition_scale')}

    def trainable_count(self):
        named = dict(self.named_parameters())
        return {name: named[name].numel() for name in self.learned_names}

    def assert_base_unchanged(self, before, full=False):
        after = self.base_bank.hashes(full=full)
        if after != before:
            changed = [name for name in before if before[name] != after.get(name)]
            raise AssertionError(f'Official frozen base changed: {changed[:8]}')
        return True
