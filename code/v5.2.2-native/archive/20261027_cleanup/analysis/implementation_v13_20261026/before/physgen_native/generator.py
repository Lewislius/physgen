"""Native Wan spatiotemporal traversal with one direct knowledge input."""
from pathlib import Path
import json
import sys
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .conditioning import NativeInterface
from .lora import install_wan_lora
from .native_attention import NativeAttentionInterface, NativeAttentionContext


def sinusoidal_embedding_1d(dim, position):
    # Native Wan formula, kept local so CPU tests need not initialize wan's CUDA-only T5 imports.
    half = dim//2
    position = position.to(torch.float64)
    angle = torch.outer(position, torch.pow(10000, -torch.arange(half).to(position)/half))
    return torch.cat((angle.cos(), angle.sin()), -1)


def load_wan(cfg, device='cpu'):
    sys.path.insert(0, cfg['paths']['wan_code'])
    from wan.modules.model import WanModel
    from safetensors import safe_open
    root = Path(cfg['paths']['wan_checkpoint'])
    wan = WanModel.from_pretrained(str(root), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, local_files_only=True)
    mapping = json.loads((root/'diffusion_pytorch_model.safetensors.index.json').read_text())['weight_map']
    prefixes = ['time_embedding.', 'time_projection.', 'head.']
    for i in range(len(wan.blocks)):
        prefixes += [f'blocks.{i}.{s}' for s in ('modulation', 'norm3.', 'self_attn.norm_q.', 'self_attn.norm_k.',
                                              'cross_attn.norm_q.', 'cross_attn.norm_k.')]
    for name, parameter in wan.named_parameters():
        if any(name.startswith(p) for p in prefixes):
            with safe_open(str(root/mapping[name]), framework='pt', device='cpu') as reader:
                parameter.data = reader.get_tensor(name).float().clone()
    return wan.eval().requires_grad_(False).to(device)


class ConditionedWan(nn.Module):
    def __init__(self, wan, cfg):
        super().__init__()
        self.wan = wan
        self.native_C = cfg['model'].get('injection')=='native_cross_attention'
        interface = NativeAttentionInterface if self.native_C else NativeInterface
        self.interface = interface(cfg, len(wan.blocks))
        self.checkpointing = cfg['train']['gradient_checkpointing']
        install_wan_lora(wan, cfg['wan_lora'])
        if self.native_C:self.interface.attach(wan.blocks)

    def forward(self, noisy, observed, text, sigma, tokens, times, strength=1., *, auxiliary=False):
        wan = self.wan
        if wan.freqs.device != noisy.device: wan.freqs = wan.freqs.to(noisy.device)
        x = wan.patch_embedding(torch.cat((observed, noisy[:, :, 1:]), 2))
        geometry = tuple(x.shape[2:])
        x = x.flatten(2).transpose(1, 2)
        length, known = x.shape[1], geometry[1]*geometry[2]
        positions = torch.cat((sigma.new_zeros(known), sigma.expand(length-known)*1000))
        with torch.autocast(noisy.device.type, enabled=False):
            e = wan.time_embedding(sinusoidal_embedding_1d(wan.freq_dim, positions).float()[None])
            modulation = wan.time_projection(e).unflatten(-1, (6, wan.dim))
        text = text[:, :wan.text_len]
        context = wan.text_embedding(F.pad(text, (0, 0, 0, wan.text_len-text.shape[1])))
        grid = torch.tensor([geometry], dtype=torch.long)
        lengths = torch.tensor([length], dtype=torch.long)
        kv = None if self.native_C else self.interface.prepare(tokens)
        intensities = []
        for index, block in enumerate(wan.blocks):
            if self.native_C:
                def run_native(hidden, c, layer=block, site=index):
                    packet = NativeAttentionContext(context,c,self.interface,geometry,strength,site) if site in self.interface.layers else None
                    hidden=layer(hidden,modulation,lengths,grid,wan.freqs,packet if packet is not None else context,None)
                    intensity=packet.intensity if packet is not None else None
                    return hidden,intensity if intensity is not None else hidden.new_empty(0)
                if self.checkpointing and torch.is_grad_enabled():x,intensity=checkpoint(run_native,x,tokens,use_reentrant=False)
                else:x,intensity=run_native(x,tokens)
                if auxiliary and intensity.numel():intensities.append(intensity)
                continue
            def run(hidden, keys, values, layer=block, site=index):
                hidden = layer(hidden, modulation, lengths, grid, wan.freqs, context, None)
                hidden, intensity = self.interface(hidden, site, (keys, values), sigma, times, geometry, strength)
                # Empty tensors keep the checkpoint signature uniform.
                return hidden, intensity if intensity is not None else hidden.new_empty(0)
            if self.checkpointing and torch.is_grad_enabled():
                x, intensity = checkpoint(run, x, *kv, use_reentrant=False)
            else:
                x, intensity = run(x, *kv)
            if auxiliary and intensity.numel(): intensities.append(intensity)
        velocity = wan.unpatchify(wan.head(x, e), grid)[0][None].float()
        if not auxiliary: return velocity
        return velocity, torch.stack(intensities).mean(0)
