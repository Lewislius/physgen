"""R20FT candidate: whole-video grounded physical execution after native DiT."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from physgen_native.context_alignment import native_text_context
from physgen_native.generator import sinusoidal_embedding_1d
from .native_residual import NativeAnchoredContent
from .grounded_physics import GroundedPhysics


class NativePhysicalVelocity(nn.Module):
    def __init__(self, wan, *, task, interface, query_width=4096, q_heads=32,
                 grounded_heads=16, slots=256, rounds=2,
                 injection_layers=(29,), checkpointing=True):
        super().__init__()
        if task != 't2v':
            raise ValueError(task)
        self.task, self.wan, self.checkpointing = task, wan.requires_grad_(False).eval(), checkpointing
        self.injection_layers = tuple(injection_layers)
        self.register_buffer('condition_mean', interface['mean'].float())
        self.register_buffer('condition_scale', interface['scale'].float())
        self.enhancement = NativeAnchoredContent(
            query_width, wan.dim, q_heads, grounded_heads, slots, rounds,
            checkpointing=checkpointing, source_scale=interface.get('native_scale', 1.))
        if self.injection_layers != (len(wan.blocks)-1,):
            raise ValueError('The grounded executor follows the final native block')
        self.physical_blocks = nn.ModuleList([GroundedPhysics(wan.blocks[-1])])
        self.local_distribution = None
        self.learned_names = tuple(n for n,p in self.named_parameters() if p.requires_grad)
        self.write_statistics = {}
        self.feature_statistics = {}

    def prepare_content(self, evidence, text, *, anchor_only=False):
        words = (evidence['words'].detach().float()-self.condition_mean)/self.condition_scale
        request = (evidence['request_words'].detach().float()-self.condition_mean)/self.condition_scale
        raw = self.enhancement(words, request, evidence['native_condition'].detach().float(), evidence['word_groups'])
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
                    context=context, text_mask=mask), raw['groups']

    def velocity(self, noisy, first, text, sigma, content, groups, times, *, physical_mode='normal'):
        if physical_mode not in ('normal','off','grounding_off'):
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
        context,mask = content['context'],content['text_mask']
        self.write_statistics = {}
        self.feature_statistics = {}
        self.local_distribution = None
        for block in wan.blocks:
            args = (x,native_mod,lengths,grid,wan.freqs,context,None)
            x = (checkpoint(block,*args,use_reentrant=False)
                 if self.checkpointing and torch.is_grad_enabled() else block(*args))
        if physical_mode != 'off':
            args = (x,native_mod,lengths,grid,wan.freqs,
                    content['routing'],content['execution'],physical_mode)
            physical = self.physical_blocks[0]
            x,distribution,statistics = (checkpoint(physical,*args,use_reentrant=False)
                if self.checkpointing and torch.is_grad_enabled() else physical(*args))
            self.local_distribution = distribution.reshape(1,frames,height,width)
            self.write_statistics['tail'] = statistics[0]
            self.feature_statistics['tail'] = statistics[1:]
        prediction = wan.unpatchify(wan.head(x,e),grid)[0][None].float()
        return prediction,prediction.new_zeros(())
