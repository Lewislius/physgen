"""R20FT distributed, depth-matched physical execution beside intact Wan."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from physgen_native.context_alignment import native_text_context
from physgen_native.generator import sinusoidal_embedding_1d
from .native_residual import NativeAnchoredContent
from .grounded_physics import GroundedPhysicalBlock, native_block_with_read_state


class NativePhysicalVelocity(nn.Module):
    def __init__(self, wan, *, task, interface, query_width=4096, q_heads=32,
                 grounded_heads=16, slots=256, rounds=2,
                 injection_layers=(0, 7, 14, 21, 28), checkpointing=True):
        super().__init__()
        if task not in ('t2v', 'i2v'):
            raise ValueError(task)
        if wan.dim != 1536:
            raise ValueError(f'R20FT-Wan2.1 requires width 1536, got {wan.dim}')
        self.task, self.wan, self.checkpointing = task, wan.requires_grad_(False).eval(), checkpointing
        self.injection_layers = tuple(injection_layers)
        if len(wan.blocks) != 30:
            raise ValueError(f'Wan2.1-T2V-1.3B must have 30 blocks, got {len(wan.blocks)}')
        if max(self.injection_layers) >= len(wan.blocks):
            raise ValueError('Physical injection depth exceeds the Wan2.1 backbone')
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
        native = evidence['native_condition'].detach().float()
        if native.ndim != 2 or native.shape[1] != self.wan.dim:
            raise ValueError(
                f'Native physical words must be [N, {self.wan.dim}], got {tuple(native.shape)}')
        if words.shape[0] != native.shape[0]:
            raise ValueError('Qwen and native Wan2.1 word counts differ')
        raw = self.enhancement(words, request, native, evidence['word_groups'])
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
        if physical_mode not in ('normal','off','video_binding_off'):
            raise ValueError(physical_mode)
        if (self.task == 't2v' and first is not None) or (self.task == 'i2v' and first is None):
            raise ValueError('External first-image boundary differs from task')
        if noisy.shape[0] != 1:
            raise ValueError('Use one video and gradient accumulation')
        if noisy.shape[1] != self.wan.in_dim:
            raise ValueError(
                f'Wan2.1 latent must have {self.wan.in_dim} channels, got {noisy.shape[1]}')
        latent = noisy if self.task == 't2v' else torch.cat((first, noisy[:,:,1:]), 2)
        wan = self.wan
        if wan.freqs.device != noisy.device:
            wan.freqs = wan.freqs.to(noisy.device)
        x = wan.patch_embedding(latent.to(wan.patch_embedding.weight.dtype))
        geometry = tuple(x.shape[2:]); frames,height,width = geometry; spatial = height*width
        x = x.flatten(2).transpose(1,2)
        known = spatial if self.task == 'i2v' else 0
        # Wan2.1's WanAttentionBlock/Head take one diffusion-time vector per
        # batch ([B,6,C]/[B,C]); Wan2.2's implementation used per-token
        # modulation.  R20FT-Wan2.1 is T2V-only and every video token shares
        # the same sigma, so the native Wan2.1 shape is the correct one.
        sigma = torch.as_tensor(sigma, device=noisy.device, dtype=torch.float32)
        position = sigma.reshape(1) * 1000
        with torch.autocast(noisy.device.type, enabled=False):
            e = wan.time_embedding(
                sinusoidal_embedding_1d(wan.freq_dim, position).float())
            native_mod = wan.time_projection(e).unflatten(-1, (6, wan.dim))
        grid = torch.tensor([geometry],dtype=torch.long)
        lengths = torch.tensor([x.shape[1]],dtype=torch.long)
        times = times.reshape(1,frames)
        if times.shape[1] != frames:
            raise ValueError(f'Presentation time grid must have {frames} values')
        context = content['context']
        sites = dict(zip(self.injection_layers, self.physical_blocks))
        self.write_statistics = {}
        for index,block in enumerate(wan.blocks):
            block_input = x
            if index in sites:
                args = (block,x,native_mod,lengths,grid,wan.freqs,context)
                x,read_state = (checkpoint(native_block_with_read_state,*args,use_reentrant=False)
                    if self.checkpointing and torch.is_grad_enabled()
                    else native_block_with_read_state(*args))
            else:
                args = (x,native_mod,lengths,grid,wan.freqs,context,None)
                x = (checkpoint(block,*args,use_reentrant=False)
                     if self.checkpointing and torch.is_grad_enabled() else block(*args))
            if index in sites and physical_mode != 'off':
                # Actual native features at this depth, not a shallow physical
                # stream initialized from layers 0..4 at all five locations.
                physical = sites[index]
                args = (block_input,read_state,native_mod,lengths,grid,wan.freqs,context,
                        content['routing'],content['execution'])
                present = content.get('physical_present', True)
                hint = (checkpoint(physical,*args,mode=physical_mode,source_present=present,use_reentrant=False)
                    if self.checkpointing and torch.is_grad_enabled()
                    else physical(*args,mode=physical_mode,source_present=present))
                if self.training and torch.is_grad_enabled():
                    self.write_statistics[str(index)] = (hint.detach()[:,known:].float().square().mean().sqrt()
                        /x.detach()[:,known:].float().square().mean().sqrt().clamp_min(1e-8))
                x = x.float()+hint.float()
        head_dtype = wan.head.head.weight.dtype
        prediction = wan.unpatchify(wan.head(x.to(head_dtype),e),grid)[0][None].float()
        return prediction,prediction.new_zeros(())
