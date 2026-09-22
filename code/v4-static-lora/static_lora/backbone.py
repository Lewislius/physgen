"""Use the actual Wan forward, adding only LoRA and activation recomputation."""
import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .native_wan import fix_first, load_wan
from .lora import inject_lora
from .runtime import autocast


def enable_checkpointing(wan):
    def wrap(original):
        def forward(x, *args, **kwargs):
            if not torch.is_grad_enabled():
                return original(x, *args, **kwargs)
            def call(value):
                with autocast(value.device):
                    return original(value, *args, **kwargs)
            # Non-reentrant checkpointing also trains LoRA in the first block,
            # whose input comes from frozen embeddings and requires no gradient.
            return checkpoint(call, x, use_reentrant=False)
        return forward
    for block in wan.blocks:
        if getattr(block, "_static_lora_checkpointed", False):
            raise ValueError("Block checkpointing already installed")
        block.forward = wrap(block.forward)
        block._static_lora_checkpointed = True


def model_inputs(wan, x, first, text, sigma):
    if x.ndim != 5 or x.shape[0] != 1 or text.ndim != 3 or text.shape[0] != 1:
        raise ValueError("The baseline runs microbatch=1 with accumulation=8")
    if text.shape[1] > wan.text_len:
        raise ValueError("Text exceeds native Wan context length")
    shape, patch = x.shape[2:], wan.patch_size
    if any(size % width for size, width in zip(shape, patch)):
        raise ValueError("Latent shape must be divisible by Wan patch size")
    grid = [size // width for size, width in zip(shape, patch)]
    length = math.prod(grid)
    if first is not None and (patch[0] != 1 or first.shape != (1, x.shape[1], 1, *x.shape[-2:])):
        raise ValueError("Invalid native I2V first-slice condition")
    known = grid[1] * grid[2] if first is not None else 0
    sigma = torch.as_tensor(sigma, device=x.device, dtype=torch.float32).reshape(())
    t = torch.cat((sigma.new_zeros(known), sigma.expand(length - known) * 1000))[None]
    return dict(x=[fix_first(x, first)[0]], t=t, context=[text[0]], seq_len=length)


class WanLoRA(nn.Module):
    def __init__(self, wan, spec, recompute=True):
        super().__init__()
        self.wan = wan
        self.targets = inject_lora(wan, spec)
        if recompute:
            enable_checkpointing(wan)

    def forward(self, x, first, text, sigma):
        with autocast(x.device):
            return self.wan(**model_inputs(self.wan, x, first, text, sigma))[0][None].float()

    def guided(self, x, first, text, negative, sigma, guidance=5.):
        # Native CFG: both conditional and unconditional predictions include LoRA.
        cond = self(x, first, text, sigma)
        uncond = self(x, first, negative, sigma)
        return uncond + guidance * (cond - uncond)


def load_model(cfg, device, training=True):
    wan = load_wan(cfg["paths"]["wan_checkpoint"], device)
    model = WanLoRA(wan, cfg["lora"], recompute=training and cfg["train"]["gradient_checkpointing"])
    # Original Wan is deterministic in eval mode. LoRA dropout is fixed to zero;
    # gradients and block recomputation remain enabled during training.
    model.eval()
    return model if training else model.requires_grad_(False)
