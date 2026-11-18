"""R20FT distributed, depth-matched physical execution beside intact Wan."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from physgen_native.context_alignment import native_text_context
from physgen_native.generator import sinusoidal_embedding_1d
from .native_residual import NativeAnchoredContent
from .grounded_physics import GroundedPhysicalBlock


class NativePhysicalVelocity(nn.Module):
    def __init__(self, wan, *, task, interface, query_width=4096, q_heads=32,
                 grounded_heads=16, slots=256, rounds=2,
                 injection_layers=(0, 7, 14, 21, 28), checkpointing=True):
        super().__init__()
        if task not in ('t2v', 'i2v'):
            raise ValueError(task)
        self.task, self.wan, self.checkpointing = task, wan.requires_grad_(False).eval(), checkpointing
        self.injection_layers = tuple(injection_layers)
        self.register_buffer('condition_mean', interface['mean'].float())
        self.register_buffer('condition_scale', interface['scale'].float())
        self.enhancement = NativeAnchoredContent(
            query_width, wan.dim, q_heads, grounded_heads, slots, rounds,
            checkpointing=checkpointing, source_scale=interface.get('native_scale', 1.))
        if task != 't2v':
            raise ValueError('FT is the authorized text-to-video candidate')
        if len(set(self.injection_layers)) != len(self.injection_layers):
            raise ValueError('Physical block locations must be distinct')
        self.physical_blocks = nn.ModuleList([
            GroundedPhysicalBlock(wan.blocks[site]) for site in self.injection_layers])
        self.learned_names = tuple(n for n,p in self.named_parameters() if p.requires_grad)
        self.write_statistics = {}

    def prepare_content(self, evidence, text, *, anchor_only=False):
        words = (evidence['words'].detach().float()-self.condition_mean)/self.condition_scale
        request = (evidence['request_words'].detach().float()-self.condition_mean)/self.condition_scale
        raw = self.enhancement(words, request, evidence['native_condition'].detach().float(), evidence['word_groups'])
        present = any(str(value).strip() for value in evidence.get('relations', ['present']))
        if not present:
            raw['route'], raw['execute'] = raw['route']*0, raw['execute']*0
        if anchor_only:
            raw['route'] = raw['execute'] = raw['native']
        text = text[None] if text.ndim == 2 else text
        if text.shape[1] > self.wan.text_len:
            raise ValueError('Input text would be truncated')
        with torch.no_grad():
            context = native_text_context(self.wan, text.to(next(self.wan.text_embedding.parameters()).dtype))
        mask = torch.arange(self.wan.text_len, device=text.device)[None] < text.shape[1]
        raw.update(routing=raw['route'], execution=raw['execute'])
        return dict(routing=raw['route'], execution=raw['execute'], raw=raw,
                    context=context, text_mask=mask, physical_present=present), raw['groups']

    def velocity(self, noisy, first, text, sigma, content, groups, times, *,
                 physical_mode='normal'):
        if physical_mode not in ('normal','off','physical_condition_off'):
            raise ValueError(physical_mode)
        if (self.task == 't2v' and first is not None) or (self.task == 'i2v' and first is None):
            raise ValueError('External first-image boundary differs from task')
        if noisy.shape[0] != 1:
            raise ValueError('Use one video and gradient accumulation')
        latent = noisy if self.task == 't2v' else torch.cat((first, noisy[:,:,1:]), 2)
        wan = self.wan
        if wan.freqs.device != noisy.device:
            wan.freqs = wan.freqs.to(noisy.device)
        x = wan.patch_embedding(latent)
        geometry = tuple(x.shape[2:]); frames,height,width = geometry; spatial = height*width
        x = x.flatten(2).transpose(1,2)
        known = spatial if self.task == 'i2v' else 0
        position = sigma.expand(x.shape[1])*1000
        if known:
            position = torch.cat((position.new_zeros(known),position[known:]))
        with torch.autocast(noisy.device.type, enabled=False):
            e = wan.time_embedding(sinusoidal_embedding_1d(wan.freq_dim, position).float()[None])
            native_mod = wan.time_projection(e).unflatten(-1,(6,wan.dim))
        grid = torch.tensor([geometry],dtype=torch.long)
        lengths = torch.tensor([x.shape[1]],dtype=torch.long)
        times = times.reshape(1,frames)
        context = content['context']
        sites = dict(zip(self.injection_layers, self.physical_blocks))
        self.write_statistics = {}
        for index,block in enumerate(wan.blocks):
            block_input = x
            args = (x,native_mod,lengths,grid,wan.freqs,context,None)
            x = (checkpoint(block,*args,use_reentrant=False)
                 if self.checkpointing and torch.is_grad_enabled() else block(*args))
            if index in sites and physical_mode not in ('off','physical_condition_off'):
                # Actual native features at this depth, not a shallow physical
                # stream initialized from layers 0..4 at all five locations.
                physical = sites[index]
                args = (block_input,x,native_mod,lengths,grid,wan.freqs,context,
                        content['routing'],content['execution'])
                present = content.get('physical_present', True)
                hint = (checkpoint(physical,*args,mode=physical_mode,source_present=present,use_reentrant=False)
                    if self.checkpointing and torch.is_grad_enabled()
                    else physical(*args,mode=physical_mode,source_present=present))
                if self.training and torch.is_grad_enabled():
                    self.write_statistics[str(index)] = (hint.detach()[:,known:].float().square().mean().sqrt()
                        /x.detach()[:,known:].float().square().mean().sqrt().clamp_min(1e-8))
                x = x.float()+hint.float()
        prediction = wan.unpatchify(wan.head(x,e),grid)[0][None].float()
        return prediction,prediction.new_zeros(())
