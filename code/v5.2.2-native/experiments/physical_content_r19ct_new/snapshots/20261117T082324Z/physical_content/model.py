"""R19CTN per-site physical state execution, with an unchanged native text path."""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from physgen_native.context_alignment import native_text_context
from physgen_native.generator import load_wan, sinusoidal_embedding_1d

from .base_bank import OfficialBaseBank
from .grounded_experts import SemanticPhysicalEncoder, PhysicalStateExpert, GroundedPhysicalFusion
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
            self.enhancement = SemanticPhysicalEncoder(
                model['qwen_width'], model['width'], model['q_heads'],
                model['q_tokens'], model['content_rounds']).to(device)
            self.register_buffer('condition_mean', interface['mean'].float().to(device))
            self.register_buffer('condition_scale', interface['scale'].float().to(device))
            sites = [int(site) for site in cfg['model']['injection_layers']]
            if len(set(sites)) != len(sites):
                raise ValueError('Injection layers must be unique')
            if sites != sorted(sites):
                raise ValueError('Injection layers must be in ascending order')
            if any(site < 0 or site >= int(model['native_layers']) for site in sites):
                raise ValueError('Injection layer outside the native Wan depth')
            declared = cfg['model'].get('physical_initialization_blocks')
            if declared is not None and [int(value) for value in declared] != sites:
                raise ValueError(
                    'physical_initialization_blocks must equal injection_layers for per-site initialization')
            video_value_init = str(model.get('video_value_init', 'official_cross_attn_v'))
            read_output_init = str(model.get('read_output_init', 'official_cross_attn_o'))
            # One independent physical state expert per injection site.  Every
            # expert is initialized from the *same-numbered* official Wan block;
            # there is no shared expert and no cross-layer fallback.
            self.state_experts = nn.ModuleDict({
                str(site): PhysicalStateExpert(
                    model['width'], model['native_heads'], model['ffn_width']).to(device)
                for site in sites})
            self.sites = tuple(sites)
            self.mechanisms = nn.ModuleDict({
                str(site): GroundedPhysicalFusion(
                    model['width'], model['ffn_width'], model['native_heads'],
                    video_value_init=video_value_init,
                    read_output_init=read_output_init).to(device)
                for site in sites})
            if set(self.state_experts) != set(self.mechanisms):
                raise AssertionError('Per-site experts and read modules must cover the same sites')
            # Initialize from official pretrained operators BEFORE video LoRA.
            # Each site owns independent Q/K/V/O, read output and state expert
            # copies from its own depth; no frozen backbone tensor is aliased.
            for site in sites:
                block = self.wan.blocks[site]
                self.mechanisms[str(site)].initialize_from_official_block(block)
                self.state_experts[str(site)].initialize_from_official_block(block)
            self.video_lora_targets = install_video_lora(self.wan, cfg)
        assert_native_text_frozen(self.wan)
        self.learned_names = tuple(
            name for name, parameter in self.named_parameters() if parameter.requires_grad)
        self.checkpointing = cfg['train']['gradient_checkpointing']
        self.base_fingerprints = self.base_bank.hashes(full=False)
        self.initialization_report = None
        if bool(cfg['train'].get('verify_initialization', True)):
            self.initialization_report = self.audit_initialization()

    @staticmethod
    def _same_tensor(actual, expected):
        # The official Wan weights are BF16 while the fresh physical modules are
        # FP32.  copy_ preserves the exact BF16 values in FP32, so compare in a
        # common float32 view rather than requiring equal dtypes.
        left = actual.detach().float()
        right = expected.detach().float()
        return left.shape == right.shape and torch.equal(left, right)

    @classmethod
    def _same_module(cls, actual, expected):
        left, right = actual.state_dict(), expected.state_dict()
        return (set(left) == set(right)
                and all(cls._same_tensor(left[key], right[key]) for key in left))

    @staticmethod
    def _bias_pair(actual, expected):
        target = expected.bias if expected.bias is not None else None
        if target is None:
            target = torch.zeros_like(actual.bias)
        return actual.bias, target

    @torch.no_grad()
    def audit_initialization(self):
        """Fail fast if any per-site physical parameter came from the wrong depth.

        This runs immediately after official initialization and before any
        optimizer step.  Every copied parameter is checked against the
        same-numbered Wan block; per-site parameter objects must be disjoint.
        """
        sites = list(self.sites)
        if set(self.mechanisms) != {str(site) for site in sites}:
            raise AssertionError('Mechanism sites differ from configured injection layers')
        if set(self.state_experts) != {str(site) for site in sites}:
            raise AssertionError('State expert sites differ from configured injection layers')
        per_site_ids, per_site_counts = {}, {}
        value_modes = {module.video_value_init for module in self.mechanisms.values()}
        read_modes = {module.read_output_init for module in self.mechanisms.values()}
        if len(value_modes) != 1 or len(read_modes) != 1:
            raise AssertionError('All per-site mechanisms must use one coherent V/O initialization')
        value_init, read_init = value_modes.pop(), read_modes.pop()
        if (value_init, read_init) not in {
                ('official_cross_attn_v', 'official_cross_attn_o'),
                ('identity', 'identity')}:
            raise AssertionError('video_value_init and read_output_init are not a coherent pair')
        for site in sites:
            mech = self.mechanisms[str(site)]
            expert = self.state_experts[str(site)]
            block = self.wan.blocks[site]
            cross = block.cross_attn
            self_q = self.base_bank.linear(site, 'self_attn.q')
            self_k = self.base_bank.linear(site, 'self_attn.k')
            self_v = self.base_bank.linear(site, 'self_attn.v')
            self_o = self.base_bank.linear(site, 'self_attn.o')
            ffn0 = self.base_bank.linear(site, 'ffn.0')
            ffn2 = self.base_bank.linear(site, 'ffn.2')
            checks = {
                f'mechanisms.{site}.video_query': (mech.video_query.weight, cross.q.weight),
                f'mechanisms.{site}.content_key': (mech.content_key.weight, cross.k.weight),
                f'mechanisms.{site}.query_norm': (mech.query_norm.weight, cross.norm_q.weight),
                f'mechanisms.{site}.key_norm': (mech.key_norm.weight, cross.norm_k.weight),
                f'state_experts.{site}.condition.q': (expert.condition_attention.q.weight, cross.q.weight),
                f'state_experts.{site}.condition.k': (expert.condition_attention.k.weight, cross.k.weight),
                f'state_experts.{site}.condition.v': (expert.condition_attention.v.weight, cross.v.weight),
                f'state_experts.{site}.condition.o': (expert.condition_attention.out_proj.weight, cross.o.weight),
                f'state_experts.{site}.condition.norm_q': (expert.condition_attention.norm_q.weight, cross.norm_q.weight),
                f'state_experts.{site}.condition.norm_k': (expert.condition_attention.norm_k.weight, cross.norm_k.weight),
                f'state_experts.{site}.relation.q': (expert.relation_attention.q.weight, self_q.weight),
                f'state_experts.{site}.relation.k': (expert.relation_attention.k.weight, self_k.weight),
                f'state_experts.{site}.relation.v': (expert.relation_attention.v.weight, self_v.weight),
                f'state_experts.{site}.relation.o': (expert.relation_attention.out_proj.weight, self_o.weight),
                f'state_experts.{site}.temporal.q': (expert.temporal_attention.q.weight, self_q.weight),
                f'state_experts.{site}.temporal.k': (expert.temporal_attention.k.weight, self_k.weight),
                f'state_experts.{site}.temporal.v': (expert.temporal_attention.v.weight, self_v.weight),
                f'state_experts.{site}.temporal.o': (expert.temporal_attention.out_proj.weight, self_o.weight),
                f'state_experts.{site}.ffn.0': (expert.ffn.layers[0].weight, ffn0.weight),
                f'state_experts.{site}.ffn.2': (expert.ffn.layers[2].weight, ffn2.weight),
            }
            checks.update({
                f'mechanisms.{site}.video_query.bias': self._bias_pair(mech.video_query, cross.q),
                f'mechanisms.{site}.content_key.bias': self._bias_pair(mech.content_key, cross.k),
                f'state_experts.{site}.condition.q.bias': self._bias_pair(
                    expert.condition_attention.q, cross.q),
                f'state_experts.{site}.condition.k.bias': self._bias_pair(
                    expert.condition_attention.k, cross.k),
                f'state_experts.{site}.condition.v.bias': self._bias_pair(
                    expert.condition_attention.v, cross.v),
                f'state_experts.{site}.condition.o.bias': self._bias_pair(
                    expert.condition_attention.out_proj, cross.o),
                f'state_experts.{site}.relation.q.bias': self._bias_pair(
                    expert.relation_attention.q, self_q),
                f'state_experts.{site}.relation.k.bias': self._bias_pair(
                    expert.relation_attention.k, self_k),
                f'state_experts.{site}.relation.v.bias': self._bias_pair(
                    expert.relation_attention.v, self_v),
                f'state_experts.{site}.relation.o.bias': self._bias_pair(
                    expert.relation_attention.out_proj, self_o),
                f'state_experts.{site}.temporal.q.bias': self._bias_pair(
                    expert.temporal_attention.q, self_q),
                f'state_experts.{site}.temporal.k.bias': self._bias_pair(
                    expert.temporal_attention.k, self_k),
                f'state_experts.{site}.temporal.v.bias': self._bias_pair(
                    expert.temporal_attention.v, self_v),
                f'state_experts.{site}.temporal.o.bias': self._bias_pair(
                    expert.temporal_attention.out_proj, self_o),
                f'state_experts.{site}.ffn.0.bias': self._bias_pair(
                    expert.ffn.layers[0], ffn0),
                f'state_experts.{site}.ffn.2.bias': self._bias_pair(
                    expert.ffn.layers[2], ffn2),
            })
            if value_init == 'official_cross_attn_v':
                checks[f'mechanisms.{site}.video_value'] = (mech.video_value.weight, cross.v.weight)
                checks[f'mechanisms.{site}.video_value.bias'] = self._bias_pair(
                    mech.video_value, cross.v)
            else:
                identity = torch.eye(mech.width, device=mech.video_value.weight.device,
                                     dtype=mech.video_value.weight.dtype)
                checks[f'mechanisms.{site}.video_value'] = (mech.video_value.weight, identity)
                if not torch.equal(mech.video_value.bias,
                                   torch.zeros_like(mech.video_value.bias)):
                    raise AssertionError(f'mechanisms.{site}.video_value identity bias is non-zero')
            if read_init == 'official_cross_attn_o':
                checks[f'mechanisms.{site}.read_output'] = (mech.read_output.weight, cross.o.weight)
                checks[f'mechanisms.{site}.read_output.bias'] = self._bias_pair(
                    mech.read_output, cross.o)
            else:
                identity = torch.eye(mech.width, device=mech.read_output.weight.device,
                                     dtype=mech.read_output.weight.dtype)
                checks[f'mechanisms.{site}.read_output'] = (mech.read_output.weight, identity)
                if not torch.equal(mech.read_output.bias,
                                   torch.zeros_like(mech.read_output.bias)):
                    raise AssertionError(f'mechanisms.{site}.read_output identity bias is non-zero')
            for name, (actual, expected) in checks.items():
                if not self._same_tensor(actual, expected):
                    raise AssertionError(f'Wrong-depth physical initialization: {name}')
            module_checks = {
                f'state_experts.{site}.relation_norm': (expert.relation_norm,
                                                        self.base_bank.block_norm(site, 'norm1')),
                f'state_experts.{site}.temporal_norm': (expert.temporal_norm,
                                                        self.base_bank.block_norm(site, 'norm1')),
                f'state_experts.{site}.video_norm': (expert.video_norm,
                                                     self.base_bank.block_norm(site, 'norm3')),
                f'state_experts.{site}.ffn_norm': (expert.ffn.norm,
                                                   self.base_bank.block_norm(site, 'norm2')),
                f'state_experts.{site}.relation_q_norm': (
                    expert.relation_attention.norm_q,
                    self.base_bank.norm(site, 'self_attn', 'norm_q')),
                f'state_experts.{site}.temporal_k_norm': (
                    expert.temporal_attention.norm_k,
                    self.base_bank.norm(site, 'self_attn', 'norm_k')),
            }
            for name, (actual, expected) in module_checks.items():
                if not self._same_module(actual, expected):
                    raise AssertionError(f'Wrong-depth physical normalization: {name}')
            if float(mech.feature_output.weight.detach().abs().max()) != 0.0:
                raise AssertionError(f'mechanisms.{site}.feature_output must start at zero')
            if float(expert.noise_modulation.weight.detach().abs().max()) != 0.0:
                raise AssertionError(f'state_experts.{site}.noise_modulation must start at zero')
            ids = ({id(p) for p in mech.parameters()} |
                   {id(p) for p in expert.parameters()})
            per_site_ids[site] = ids
            per_site_counts[site] = sum(p.numel() for p in
                                        list(mech.parameters()) + list(expert.parameters()))
        for index, left in enumerate(sites):
            for right in sites[index + 1:]:
                if per_site_ids[left] & per_site_ids[right]:
                    raise AssertionError(
                        f'Physical parameters are shared between sites {left} and {right}')
        return dict(
            sites=sites, video_value_init=value_init,
            read_output_init=read_init,
            per_site_physical_parameters=per_site_counts,
            physical_parameters=sum(per_site_counts.values()),
            same_depth_qkvo=True, independent_per_site=True,
            zero_initialized_write=True)

    def _context(self, text):
        text, text_mask = make_text_mask(text, self.wan.text_len, text.device)
        parameter = next(self.wan.text_embedding.parameters())
        with torch.no_grad():
            context = native_text_context(self.wan, text.to(parameter.dtype))
        return context, text_mask

    def prepare_content(self, evidence, text, *, anchor_source='native', semantic_expert=True):
        if anchor_source != 'native':
            raise ValueError('This design uses request-grounded native coordinates')
        words = evidence['words'].detach().float()
        words = (words - self.condition_mean) / self.condition_scale
        anchor, groups, _ = resample_words(words, evidence['word_groups'],
                                         self.cfg['model']['q_tokens'])
        physics, _, _ = resample_words(evidence['native_condition'].detach().float(),
                                       evidence['word_groups'], self.cfg['model']['q_tokens'])
        context, text_mask = self._context(text)
        text = text[None] if text.ndim == 2 else text
        count = min(text.shape[1], self.wan.text_len)
        raw = self.enhancement(
            anchor, words[None] if words.ndim == 2 else words,
            text[:, :count].float(), context[:, :count].float(), text_mask[:, :count],
            physics, semantic_expert=semantic_expert)
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
        if physical_mode not in ('normal', 'off', 'no_video_feedback', 'no_temporal'):
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
        noise_embedding = e.reshape(1, geometry[0], spatial, wan.dim).mean(2)

        def run_block(hidden, route_, execute_, block, site):
            parts = self.native_parts(block, modulation)
            h_sa = self.native_self(block, hidden, parts, lengths, grid, wan.freqs)
            h_text = h_sa + self.native_text(block, h_sa, context)
            if site in sites:
                module = self.mechanisms[site]
                content_read = module(
                    h_text.reshape(1, geometry[0], spatial, wan.dim),
                    dict(route=route_, execute=execute_), times, noise_embedding,
                    groups, self.state_experts[site],
                    temporal=physical_mode != 'no_temporal',
                    video_feedback=physical_mode != 'no_video_feedback',
                    query_state=block.norm3(h_text).reshape(1, geometry[0], spatial, wan.dim),
                    native_parts=tuple(p.reshape(1, geometry[0], spatial, wan.dim)[:, :, 0]
                                       for p in parts))
                content_read = content_read.flatten(1, 2) * write_mask
            else:
                content_read = None
            u = block.norm2(h_text).float() * (1 + parts[4]) + parts[3]
            preactivated = block.ffn[0](u)
            if content_read is not None:
                preactivated = preactivated + content_read.to(preactivated.dtype)
            activated = block.ffn[1](preactivated)
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
        if name.startswith('state_experts.'):
            return 'physics_state'
        if name.startswith('mechanisms.') and '.feature_output.' not in name:
            return 'physics_read'
        if '.feature_output.' in name:
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
