"""R20BT-new: depth-matched per-site physical blocks beside intact Wan.

Changes from R20BT:
- each physical block copies the official Wan block at its own injection site;
- the shared serial 5-block physical stream is removed; every site runs its
  own independent block on the actual native block input at that site;
- the shared NoiseConditioning is removed; the native
  ``block.modulation + native_mod`` is the only diffusion-time modulation and
  also drives the C reader's query/gate;
- shared C, text context, grid, freqs and lengths remain unchanged;
- presentation-time ``times`` remains, through the per-block
  VideoTimeEmbedding inside DirectPhysicalReader.

This candidate is T2V-only.  No I2V path is implemented or silently exercised.
"""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from physgen_native.context_alignment import native_text_context
from physgen_native.generator import sinusoidal_embedding_1d

from .native_residual import NativeAnchoredContent
from .video_branch import PhysicalVideoBlock


class NativePhysicalVelocity(nn.Module):
    def __init__(self, wan, *, task, interface, query_width=4096, q_heads=32,
                 grounded_heads=16, slots=256, rounds=2,
                 injection_layers=(2, 8, 14, 20, 26), checkpointing=True):
        super().__init__()
        if task != 't2v':
            raise ValueError('R20BT-new is a T2V-only candidate')
        self.task = task
        self.wan = wan.requires_grad_(False).eval()
        self.checkpointing = checkpointing
        self.injection_layers = tuple(int(site) for site in injection_layers)
        if len(set(self.injection_layers)) != len(self.injection_layers):
            raise ValueError('Physical injection sites must be distinct')
        if any(site < 0 or site >= len(self.wan.blocks) for site in self.injection_layers):
            raise ValueError('Physical injection site is outside the native model')
        self.register_buffer('condition_mean', interface['mean'].float())
        self.register_buffer('condition_scale', interface['scale'].float())
        self.enhancement = NativeAnchoredContent(
            query_width, wan.dim, q_heads, grounded_heads, slots, rounds,
            checkpointing=checkpointing, source_scale=interface.get('native_scale', 1.))
        self.physical_blocks = nn.ModuleList([
            PhysicalVideoBlock(wan.blocks[site]) for site in self.injection_layers])
        self._validate_depth_mapping()
        self.learned_names = tuple(n for n, p in self.named_parameters() if p.requires_grad)
        self.write_statistics = {}

    @torch.no_grad()
    def _validate_depth_mapping(self):
        """Fail loudly if a physical block or reader is not the site-matched copy.

        This is deliberately a startup check, not a comment.  It is cheap and
        catches the historical BT error (copy blocks 0..4, write at 2/8/14/20/26).
        """
        for site, physical in zip(self.injection_layers, self.physical_blocks):
            source = self.wan.blocks[site]
            for field in ('self_attn.q.weight', 'self_attn.k.weight',
                          'self_attn.v.weight', 'self_attn.o.weight',
                          'ffn.0.weight', 'ffn.2.weight'):
                expected = dict(source.named_parameters())[field].detach().float()
                actual = dict(physical.native.named_parameters())[field].detach().float()
                if not torch.equal(expected, actual):
                    raise ValueError(
                        f'Physical block at site {site} is not a copy of the '
                        f'same-depth native block: {field}')
            expected_mod = source.modulation.detach().float()
            actual_mod = physical.native.modulation.detach().float()
            if not torch.equal(expected_mod, actual_mod):
                raise ValueError(
                    f'Physical block at site {site} has a different native modulation')
            for source_field, reader_field in (
                    ('q.weight', 'query_base.weight'),
                    ('k.weight', 'key.weight'),
                    ('v.weight', 'value.weight')):
                expected = dict(source.cross_attn.named_parameters())[source_field].detach().float()
                actual = dict(physical.conditioning.direct.named_parameters())[reader_field].detach().float()
                if not torch.equal(expected, actual):
                    raise ValueError(
                        f'Physical reader at site {site} is not initialised from '
                        f'the same-site cross-attention: {source_field}')
        return True

    def prepare_content(self, evidence, text, *, anchor_only=False):
        words = (evidence['words'].detach().float() - self.condition_mean) / self.condition_scale
        request = (evidence['request_words'].detach().float() - self.condition_mean) / self.condition_scale
        raw = self.enhancement(
            words, request, evidence['native_condition'].detach().float(),
            evidence['word_groups'])
        if anchor_only:
            raw['route'] = raw['execute'] = raw['native']
        text = text[None] if text.ndim == 2 else text
        if text.shape[1] > self.wan.text_len:
            raise ValueError('Input text would be truncated')
        with torch.no_grad():
            context = native_text_context(
                self.wan, text.to(next(self.wan.text_embedding.parameters()).dtype))
        raw.update(routing=raw['route'], execution=raw['execute'])
        # One C is produced per sample and shared by every physical site,
        # matching R20FT.  No per-site C tensors are created here.
        return dict(routing=raw['route'], execution=raw['execute'], raw=raw,
                    context=context), raw['groups']

    def velocity(self, noisy, first, text, sigma, content, groups, times, *,
                 physical_mode='normal'):
        if physical_mode not in ('normal', 'off', 'physical_condition_off'):
            raise ValueError(physical_mode)
        if first is not None:
            raise ValueError('R20BT-new is T2V-only; no external first image')
        if noisy.shape[0] != 1:
            raise ValueError('Use one video and gradient accumulation')
        latent = noisy
        wan = self.wan
        if wan.freqs.device != noisy.device:
            wan.freqs = wan.freqs.to(noisy.device)
        x = wan.patch_embedding(latent)
        geometry = tuple(x.shape[2:])
        frames, height, width = geometry
        spatial = height * width
        x = x.flatten(2).transpose(1, 2)
        known = 0
        position = sigma.expand(x.shape[1]) * 1000
        with torch.autocast(noisy.device.type, enabled=False):
            e = wan.time_embedding(
                sinusoidal_embedding_1d(wan.freq_dim, position).float()[None])
            native_mod = wan.time_projection(e).unflatten(-1, (6, wan.dim))
        grid = torch.tensor([geometry], dtype=torch.long)
        lengths = torch.tensor([x.shape[1]], dtype=torch.long)
        if times.numel() != frames:
            raise ValueError('Presentation times must match the video frame count')
        times = times.reshape(1, frames)
        if not torch.isfinite(times).all():
            raise ValueError('Presentation times must be finite')
        if frames > 1 and torch.any(times[:, 1:] < times[:, :-1]):
            raise ValueError('Presentation times must be non-decreasing')
        context = content['context']
        sites = dict(zip(self.injection_layers, self.physical_blocks))
        self.write_statistics = {}
        for index, block in enumerate(wan.blocks):
            block_input = x
            args = (x, native_mod, lengths, grid, wan.freqs, context, None)
            x = (checkpoint(block, *args, use_reentrant=False)
                 if self.checkpointing and torch.is_grad_enabled() else block(*args))
            if index in sites and physical_mode != 'off':
                physical = sites[index]
                physical_args = (
                    block_input, native_mod, lengths, grid, wan.freqs, context,
                    content['routing'], content['execution'], groups, times,
                    spatial, known, physical_mode)
                hint = (checkpoint(physical, *physical_args, use_reentrant=False)
                        if self.checkpointing and torch.is_grad_enabled()
                        else physical(*physical_args))
                if self.training and torch.is_grad_enabled():
                    self.write_statistics[str(index)] = (
                        hint.detach().float().square().mean().sqrt()
                        / x.detach().float().square().mean().sqrt().clamp_min(1e-8))
                x = x.float() + hint.float()
        prediction = wan.unpatchify(wan.head(x, e), grid)[0][None].float()
        return prediction, prediction.new_zeros(())
