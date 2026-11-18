"""Unsubmitted source-preserving R20 integration prototype.

Every original Wan block runs intact, once. The added physical operator reads
the completed block state and adds a separate diffusion-conditioned residual.
This provides an executable integration candidate, not a final R20 design or
evidence of good generated videos. It never reuses trained R-series weights.
"""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from physgen_native.context_alignment import native_text_context
from physgen_native.generator import sinusoidal_embedding_1d

from .native_residual import NativeAnchoredContent, DiffusionModulation, IndependentPhysicalResidual


class NativePhysicalVelocity(nn.Module):
    def __init__(self, wan, *, task, interface, query_width=4096,
                 q_heads=32, grounded_heads=16, slots=256, rounds=2,
                 injection_layers=(2, 5, 8, 11, 14, 17, 20, 23, 26, 29),
                 checkpointing=True):
        super().__init__()
        if task not in ('t2v', 'i2v'):
            raise ValueError(task)
        self.task = task
        self.wan = wan.requires_grad_(False).eval()
        self.checkpointing = checkpointing
        self.register_buffer('condition_mean', interface['mean'].float())
        self.register_buffer('condition_scale', interface['scale'].float())
        self.enhancement = NativeAnchoredContent(
            query_width, wan.dim, q_heads, grounded_heads, slots, rounds,
            checkpointing=checkpointing, source_scale=interface.get('native_scale', 1.0))
        self.noise_modulation = DiffusionModulation(wan.dim)
        self.physics = nn.ModuleDict({
            str(site): IndependentPhysicalResidual(wan.blocks[site].cross_attn)
            for site in injection_layers})
        self.learned_names = tuple(n for n, p in self.named_parameters() if p.requires_grad)
        self.write_statistics = {}

    def prepare_content(self, evidence, text, *, anchor_only=False):
        words = (evidence['words'].detach().float()-self.condition_mean)/self.condition_scale
        request = (evidence['request_words'].detach().float()-self.condition_mean)/self.condition_scale
        raw = self.enhancement(words, request, evidence['native_condition'].detach().float(),
                               evidence['word_groups'])
        if anchor_only:
            raw['route'] = raw['native']
            raw['execute'] = raw['native']
        text = text[None] if text.ndim == 2 else text
        if text.shape[1] > self.wan.text_len:
            raise ValueError('Input text would be truncated')
        with torch.no_grad():
            context = native_text_context(self.wan, text.to(next(self.wan.text_embedding.parameters()).dtype))
        mask = torch.arange(self.wan.text_len, device=text.device)[None] < text.shape[1]
        raw.update(routing=raw['route'], execution=raw['execute'])
        return dict(routing=raw['route'], execution=raw['execute'], raw=raw,
                    context=context, text_mask=mask), raw['groups']

    def velocity(self, noisy, first, text, sigma, content, groups, times,
                 *, physical_mode='normal'):
        if physical_mode not in ('normal', 'off', 'direct_only', 'associated_only', 'noise_modulation_off'):
            raise ValueError(physical_mode)
        if self.task == 't2v' and first is not None:
            raise ValueError('T2V cannot receive an external first frame')
        if self.task == 'i2v' and first is None:
            raise ValueError('I2V requires its known first latent')
        if noisy.shape[0] != 1:
            raise ValueError('One video per device; accumulate gradients across samples')
        latent = noisy if self.task == 't2v' else torch.cat((first, noisy[:, :, 1:]), 2)
        wan = self.wan
        if wan.freqs.device != noisy.device:
            wan.freqs = wan.freqs.to(noisy.device)
        x = wan.patch_embedding(latent)
        geometry = tuple(x.shape[2:])
        frames, height, width = geometry
        spatial = height*width
        x = x.flatten(2).transpose(1, 2)
        known = spatial if self.task == 'i2v' else 0
        length = x.shape[1]
        position = sigma.expand(length)*1000
        if known:
            position = torch.cat((position.new_zeros(known), position[known:]))
        with torch.autocast(noisy.device.type, enabled=False):
            e = wan.time_embedding(sinusoidal_embedding_1d(wan.freq_dim, position).float()[None])
            native_mod = wan.time_projection(e).unflatten(-1, (6, wan.dim))
            # Every spatial token in a frame has the same diffusion timestep.
            frame_e = e.reshape(1, frames, spatial, wan.dim)[:, :, 0]
            physical_mod = self.noise_modulation(frame_e)
        grid = torch.tensor([geometry], dtype=torch.long)
        lengths = torch.tensor([length], dtype=torch.long)
        times = times.reshape(1, frames)
        context, text_mask = content['context'], content['text_mask']
        self.write_statistics = {}

        def run_block(hidden, route, execute, physical_mod_, block, site):
            if physical_mode == 'off' or site not in self.physics:
                return block(hidden, native_mod, lengths, grid, wan.freqs, context, None)
            # Capture the input of the REAL native text read without replacing
            # its forward or recomputing text attention on modified states.
            captured = {}
            def capture(_module, args):
                captured['text_input'] = args[0]
            handle = block.cross_attn.register_forward_pre_hook(capture)
            try:
                native = block(hidden, native_mod, lengths, grid, wan.freqs, context, None)
            finally:
                handle.remove()
            module = self.physics[site]
            attention = block.cross_attn
            split = lambda v: v.reshape(1, -1, attention.num_heads, attention.head_dim).transpose(1, 2)
            # Frozen caption keys/values retain their original meaning.
            text_keys = split(attention.norm_k(attention.k(context))).float()
            # The physical branch uses its own norm; the original block's
            # cross-attention input is never replaced or re-evaluated.
            normalized = nn.functional.layer_norm(native.float(), (wan.dim,), eps=1e-6)
            residual = module(native, normalized, route, execute, groups,
                              text_keys, None, text_mask, times,
                              physical_mod_, spatial, known_tokens=known,
                              direct=physical_mode != 'associated_only',
                              associated=physical_mode != 'direct_only',
                              diffusion_modulation=physical_mode != 'noise_modulation_off',
                              semantic_hidden=captured['text_input'])
            if self.training and torch.is_grad_enabled():
                # Reuse this actual forward. This is logging, not a loss, gate
                # or additional diagnostic generation pass.
                self.write_statistics[site] = (
                    residual.detach()[:, known:].float().square().mean().sqrt()
                    / native.detach()[:, known:].float().square().mean().sqrt().clamp_min(1e-8))
            return native.float()+residual.float()

        for index, block in enumerate(wan.blocks):
            args = (x, content['routing'], content['execution'], physical_mod, block, str(index))
            x = (checkpoint(run_block, *args, use_reentrant=False)
                 if self.checkpointing and torch.is_grad_enabled()
                 else run_block(*args))
        prediction = wan.unpatchify(wan.head(x, e), grid)[0][None].float()
        return prediction, prediction.new_zeros(())

    def trainable_count(self):
        return {n: p.numel() for n, p in self.named_parameters() if p.requires_grad}
