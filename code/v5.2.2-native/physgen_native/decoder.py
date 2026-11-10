"""Frozen Wan VAE with functional temporal caches and differentiable frame selection."""
from pathlib import Path
import importlib.util
import sys
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .geometry import uses_native_geometry, require_native_size


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
        self.preserve_aspect = uses_native_geometry(cfg)
        self.native_size = tuple(cfg['geometry']['rgb_size']) if self.preserve_aspect else None

    def train(self, mode=True):
        super().train(mode); self.vae.eval(); return self

    def _checkpointed_chunk(self, value, cache, first):
        """Recompute native residual blocks, not an entire high-resolution chunk.

        A temporal chunk checkpoint still rebuilds ALL decoder activations
        together during backward. Bound that inner peak with functional native
        blocks. Each block owns only its own cache entries; returning them keeps
        the exact derivative to earlier video times without shared mutations.
        """
        decoder = self.vae.decoder
        conv_type = type(decoder.conv1)
        position = 0

        def cached_convolution(layer, x):
            nonlocal position
            old = cache[position]
            def run(current, previous):
                following = current[:, :, -2:].clone()
                if following.shape[2] < 2 and previous is not None:
                    following = torch.cat((previous[:, :, -1:].to(current.device), following), 2)
                return layer(current, previous), following
            x, following = checkpoint(run, x, old, use_reentrant=False)
            cache[position] = following
            position += 1
            return x

        def cached_block(layer, x):
            nonlocal position
            if hasattr(layer, 'residual'):
                count = sum(isinstance(part, conv_type) for part in layer.residual)
            else:
                count = int(hasattr(layer, 'time_conv'))
            old = cache[position:position+count]
            def run(current, *previous):
                local, index = list(previous), [0]
                output = layer(current, feat_cache=local, feat_idx=index)
                if index[0] != count:
                    raise RuntimeError('Native decoder cache consumption changed')
                return (output, *local)
            output, *following = checkpoint(run, x, *old, use_reentrant=False)
            cache[position:position+count] = following
            position += count
            return output

        x = cached_convolution(decoder.conv1, value)
        for layer in decoder.middle:
            x = cached_block(layer, x) if hasattr(layer, 'residual') else checkpoint(layer, x, use_reentrant=False)
        for stage in decoder.upsamples:
            main = x.clone()
            for layer in stage.upsamples:
                main = cached_block(layer, main)
            x = main+stage.avg_shortcut(x, first) if stage.avg_shortcut is not None else main
        for layer in decoder.head:
            x = cached_convolution(layer, x) if isinstance(layer, conv_type) else checkpoint(layer, x, use_reentrant=False)
        return x

    def forward(self, latent, frame_indices=None, resize=None):
        """Decode in true temporal order; never reset the VAE at arbitrary windows.

        Checkpoint functions own their cache lists. Recomputing a chunk cannot
        overwrite the mutable caches of another forward or supervision branch.
        """
        if latent.shape[0] != 1: raise ValueError('The recipe uses microbatch one')
        if getattr(self, 'preserve_aspect', False):
            require_native_size([16*x for x in latent.shape[-2:]], self.native_size)
            if resize is not None: require_native_size(resize, self.native_size)
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
                        if recompute and getattr(self, 'block_checkpointing', True):
                            result = self._checkpointed_chunk(value, current, is_first)
                        else:
                            result = self.vae.decoder(value, feat_cache=current, feat_idx=[0], first_chunk=is_first)
                        result = self.unpatchify(result, patch_size=2).float()
                        result = result[:, :, list(positions)].permute(0, 2, 1, 3, 4).flatten(0, 1)
                        if getattr(self, 'preserve_aspect', False):
                            require_native_size(result.shape[-2:], self.native_size)
                        elif resize is not None and len(positions):
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
