import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .model import bounded_residual, noise_window
from .runtime import autocast


def fp32_islands(model):
    # Native norms multiply BF16 activations by FP32 scales; BF16 scales silently
    # introduce an extra rounding before RoPE. Modulation additions also stay FP32.
    modules = ["time_embedding", "time_projection", "head"]
    parameters = []
    for i in range(len(model.blocks)):
        modules.extend(f"blocks.{i}.{name}" for name in (
            "norm3", "self_attn.norm_q", "self_attn.norm_k", "cross_attn.norm_q", "cross_attn.norm_k"))
        parameters.append(f"blocks.{i}.modulation")
    return modules, parameters


def load_wan(path, device):
    from safetensors import safe_open
    from wan.modules.model import WanModel
    model = WanModel.from_pretrained(path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    if ((len(model.blocks), model.dim, model.in_dim, model.out_dim, model.text_len) != (30, 3072, 48, 48, 512)
            or tuple(model.patch_size) != (1, 2, 2)):
        raise ValueError("Expected original Wan2.2-TI2V-5B")
    index = json.loads((Path(path) / "diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]
    # Restore native FP32 islands from ORIGINAL weights, not from BF16 rounded tensors.
    modules, parameters = fp32_islands(model)
    for name in modules:
        module = model.get_submodule(name).float()
        values = {}
        for key, filename in index.items():
            if key.startswith(name + "."):
                with safe_open(str(Path(path) / filename), framework="pt", device="cpu") as reader:
                    values[key[len(name) + 1:]] = reader.get_tensor(key).float()
        module.load_state_dict(values, strict=True)
    for name in parameters:
        with safe_open(str(Path(path) / index[name]), framework="pt", device="cpu") as reader:
            model.get_parameter(name).data = reader.get_tensor(name).float().clone()
    model.eval().requires_grad_(False).to(device)
    model.freqs = model.freqs.to(device)
    return model


def fix_first(x, first):
    return x.float() if first is None else torch.cat((first.float(), x[:, :, 1:].float()), dim=2)


class ProcessWan(nn.Module):
    def __init__(self, wan, adapter, recompute=True):
        super().__init__()
        self.wan, self.adapter, self.recompute = wan, adapter, recompute

    def inputs(self, x, text, sigma, first):
        from wan.modules.model import sinusoidal_embedding_1d
        wan = self.wan
        embedded = wan.patch_embedding(fix_first(x, first))
        grid = torch.tensor([embedded.shape[2:]], dtype=torch.long)
        h = embedded.flatten(2).transpose(1, 2)
        length = h.shape[1]
        known = int(grid[0, 1] * grid[0, 2]) if first is not None else 0
        t = torch.cat((sigma.new_zeros(known), sigma.expand(length - known) * 1000))[None]
        with torch.autocast(x.device.type, enabled=False):
            e = wan.time_embedding(sinusoidal_embedding_1d(wan.freq_dim, t.flatten()).reshape(1, length, -1).float())
            e0 = wan.time_projection(e).unflatten(2, (6, wan.dim))
        if text.shape[1] > wan.text_len:
            raise ValueError("T5 context exceeds native length")
        padding = text.new_zeros(1, wan.text_len - text.shape[1], text.shape[2])
        context = wan.text_embedding(torch.cat((text, padding), dim=1))
        args = dict(e=e0, seq_lens=torch.tensor([length], dtype=torch.long), grid_sizes=grid,
                    freqs=wan.freqs, context=context, context_lens=None)
        return h, args, e, grid

    def block(self, block, h, args):
        if self.recompute and torch.is_grad_enabled() and h.requires_grad:
            # Each recomputation creates its own autocast cache; no no_grad cache is reused.
            def call(value):
                with autocast(value.device):
                    return block(value, **args)
            return checkpoint(call, h, use_reentrant=False)
        return block(h, **args)

    def head(self, h, e, grid):
        return self.wan.unpatchify(self.wan.head(h, e), grid)[0][None].float()

    @torch.no_grad()
    def observe(self, x, first, text, sigma):
        with autocast(x.device):
            h, args, e, grid = self.inputs(x, text, sigma, first)
            observations = []
            for number, block in enumerate(self.wan.blocks[:15], 1):
                h = block(h, **args)
                if number in (5, 15):
                    observations.append(h.detach())
        return tuple(observations)

    @torch.no_grad()
    def negative(self, x, first, text, sigma):
        with autocast(x.device):
            h, args, e, grid = self.inputs(x, text, sigma, first)
            for block in self.wan.blocks:
                h = block(h, **args)
            return self.head(h, e, grid)

    def forward(self, x, first, text, sigma, duration, geo, p0, negative=None, ramp=1.):
        # A frozen common prefix followed by independent positive base/plus suffixes.
        with torch.no_grad(), autocast(x.device):
            h, args, e, grid = self.inputs(x, text, sigma, first)
            for block in self.wan.blocks[:5]:
                h = block(h, **args)
            h5 = h.detach()
            h15base = None
            for number, block in enumerate(self.wan.blocks[5:], 6):
                h = block(h, **args)
                if number == 15:
                    h15base = h.detach()
            base = self.head(h, e, grid).detach()
        with autocast(x.device):
            a = self.adapter
            p5, mp5 = a.correctors["5"].core(p0, h5, text, sigma, duration, 0, geo, a.s_z, return_metrics=True)
            h, m5 = a.correctors["5"].writer(h5, p5, h5, sigma, ramp, first is not None, 5, geo)
            for block in self.wan.blocks[5:15]:
                h = self.block(block, h, args)
            h15 = h
            p15, mp15 = a.correctors["15"].core(p5, h15, text, sigma, duration, 1, geo, a.s_z, return_metrics=True)
            h, m15 = a.correctors["15"].writer(h15, p15, h15base, sigma, ramp, first is not None, 15, geo)
            for block in self.wan.blocks[15:]:
                h = self.block(block, h, args)
            plus = self.head(h, e, grid)
        shape = base.shape
        def tokens(v):
            return v.permute(0, 2, 3, 4, 1).reshape(1, shape[2], -1, shape[1])
        delta, mv = bounded_residual(tokens(plus - base), tokens(base), tokens(base), geo.v_weight,
                                     (.03 + .17 * noise_window(sigma)) * ramp,
                                     first is not None, a.s_v)
        delta = delta.reshape(1, shape[2], shape[3], shape[4], shape[1]).permute(0, 4, 1, 2, 3)
        cond = base + delta
        guided = None
        if negative is not None:
            bn = self.negative(x, first, negative, sigma)
            guided = (bn + 5 * (base - bn)).detach() + delta
        return dict(cond=cond, guided=guided, base=base, p0=p0.detach(), p5=p5.detach(), p15=p15.detach(),
                    observations=(h5.detach(), h15.detach()),
                    metrics=dict(state5=mp5, state15=mp15, writer5=m5, writer15=m15, velocity=mv))
