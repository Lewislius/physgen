"""Frozen Wan VAE with functional temporal caches and differentiable frame selection."""
from pathlib import Path
import importlib.util
import sys
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class VideoDecoder(nn.Module):
    def __init__(self, cfg, device='cuda'):
        super().__init__()
        # This native module is self-contained. Loading it directly avoids
        # importing Wan's unrelated speech/music pipelines in observer jobs.
        name = 'physgen_native._wan_vae2_2'
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name,Path(cfg['paths']['wan_code'])/'wan/modules/vae2_2.py')
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        module = sys.modules[name]
        Wan2_2_VAE, count_conv3d, unpatchify = module.Wan2_2_VAE, module.count_conv3d, module.unpatchify
        wrapper = Wan2_2_VAE(vae_pth=str(Path(cfg['paths']['wan_checkpoint'])/'Wan2.2_VAE.pth'),
                            dtype=torch.bfloat16, device=device)
        self.vae = wrapper.model.requires_grad_(False).eval()
        self.register_buffer('mean', wrapper.scale[0].float())
        self.register_buffer('inverse_std', wrapper.scale[1].float())
        self.cache_count = count_conv3d(self.vae.decoder)
        self.unpatchify = unpatchify
        self.checkpointing = cfg['train']['gradient_checkpointing']

    def train(self, mode=True):
        super().train(mode); self.vae.eval(); return self

    def forward(self, latent, frame_indices=None, resize=None):
        """Decode in true temporal order; never reset the VAE at arbitrary windows.

        Checkpoint functions own their cache lists. Recomputing a chunk cannot
        overwrite the mutable caches of another forward or supervision branch.
        """
        if latent.shape[0] != 1: raise ValueError('The recipe uses microbatch one')
        total = 1+4*(latent.shape[2]-1)
        selected = list(range(total)) if frame_indices is None else list(frame_indices)
        if not selected or selected != sorted(set(selected)) or selected[0] < 0 or selected[-1] >= total:
            raise ValueError('Frame indices must be sorted unique valid RGB positions')
        z = latent.float()/self.inverse_std.view(1, -1, 1, 1, 1)+self.mean.view(1, -1, 1, 1, 1)
        z = self.vae.conv2(z.to(next(self.vae.parameters()).dtype))
        cache = tuple([None]*self.cache_count)
        outputs = []
        last_latent = (selected[-1]+3)//4
        recompute = self.checkpointing and torch.is_grad_enabled() and latent.requires_grad
        # Two checkpoint levels retain only segment-boundary caches. Saving the
        # entire decoder cache for each of 21 chunks exceeds a resident 96-GiB run.
        # These are computation segments, NOT temporal resets or truncated BPTT.
        for begin in range(0, last_latent+1, 4):
            end = min(last_latent+1, begin+4)
            def decode_segment(values, *previous, base=begin):
                segment_cache, pieces = previous, []
                for j in range(values.shape[2]):
                    absolute = base+j
                    first = absolute == 0
                    offset = 0 if first else 1+4*(absolute-1)
                    length = 1 if first else 4
                    local = [index-offset for index in selected if offset <= index < offset+length]
                    def decode_chunk(value, *old, is_first=first, positions=tuple(local)):
                        current = list(old)
                        result = self.vae.decoder(value, feat_cache=current, feat_idx=[0], first_chunk=is_first)
                        result = self.unpatchify(result, patch_size=2).float()
                        result = result[:, :, list(positions)].permute(0, 2, 1, 3, 4).flatten(0, 1)
                        if resize is not None and len(positions):
                            result = F.interpolate(result, size=tuple(resize), mode='bicubic', align_corners=False, antialias=True)
                        return result.clamp(-1, 1), tuple(current)
                    if recompute:
                        frames, segment_cache = checkpoint(decode_chunk, values[:, :, j:j+1], *segment_cache, use_reentrant=False)
                    else:
                        frames, segment_cache = decode_chunk(values[:, :, j:j+1], *segment_cache)
                    if local: pieces.append(frames)
                return torch.cat(pieces) if pieces else frames, segment_cache
            if recompute:
                # Pass each cache tensor as a top-level argument. A tuple argument
                # is held strongly by PyTorch's input saver, bypassing parent
                # checkpoint hooks and retaining every intermediate chunk cache.
                frames, cache = checkpoint(decode_segment, z[:, :, begin:end], *cache, use_reentrant=False)
            else:
                frames, cache = decode_segment(z[:, :, begin:end], *cache)
            if len(frames): outputs.append(frames)
        return torch.cat(outputs)
