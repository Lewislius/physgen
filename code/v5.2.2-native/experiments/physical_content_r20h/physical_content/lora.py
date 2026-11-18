"""Independent LoRA adapters over frozen official Wan linear operators.

Only the six declared video targets receive a main LoRA.  Native text
cross-attention is never adapted.  Expert adapters live in their own modules and
never alias a main adapter.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

EXPECTED_TARGETS = {
    'self_attn.q', 'self_attn.k', 'self_attn.v', 'self_attn.o', 'ffn.0', 'ffn.2',
}


class FrozenBaseLoRALinear(nn.Module):
    """`W0(x) + (alpha/r) B(Ax)`, with an optional frozen W0 bias on a path.

    ``use_bias=False`` is used by C/D physical-value V/O and FFN operators.  It
    removes the *official base bias* for that call without modifying the base.
    """

    def __init__(self, base, rank, alpha):
        super().__init__()
        if rank <= 0 or alpha <= 0:
            raise ValueError('LoRA rank and alpha must be positive')
        if not isinstance(base, nn.Linear):
            raise TypeError('FrozenBaseLoRALinear requires an official nn.Linear')
        self.base = base.requires_grad_(False)
        self.rank = int(rank)
        self.scale = float(alpha) / float(rank)
        device = base.weight.device
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32, device=device))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32, device=device))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x, *, use_bias=True):
        # Keep the official base in the model's normal autocast path.  Only the
        # small adapter parameters are FP32; the incremental result is cast back.
        base_x = x.to(self.base.weight.dtype)
        base = F.linear(base_x, self.base.weight,
                        self.base.bias if use_bias and self.base.bias is not None else None)
        delta = F.linear(F.linear(x.to(self.lora_a.dtype), self.lora_a), self.lora_b)
        return base + (delta * self.scale).to(base.dtype)

    def extra_repr(self):
        return f'in={self.base.in_features}, out={self.base.out_features}, rank={self.rank}, scale={self.scale}'


def _ffn_targets(block):
    if not isinstance(block.ffn, nn.Sequential) or len(block.ffn) != 3:
        raise TypeError('Official Wan FFN must be Linear-GELU-Linear')
    if not isinstance(block.ffn[0], nn.Linear) or not isinstance(block.ffn[2], nn.Linear):
        raise TypeError('Official Wan FFN layers 0 and 2 must be Linear')
    return (block.ffn[0], block.ffn[2])


def install_video_lora(wan, cfg):
    """Install exactly one rank32 main adapter on each of the 180 allowlisted layers."""
    base_cfg = cfg['base']
    rank = int(base_cfg['video_lora_rank'])
    alpha = int(base_cfg['video_lora_alpha'])
    if rank != 32 or alpha != 32:
        raise ValueError('NoChunks-v2 main LoRA must be rank32/alpha32')
    installed = []
    for index, block in enumerate(wan.blocks):
        for name in ('q', 'k', 'v', 'o'):
            base = getattr(block.self_attn, name)
            if isinstance(base, FrozenBaseLoRALinear):
                raise ValueError(f'Adapter already installed at block {index} self_attn.{name}')
            setattr(block.self_attn, name, FrozenBaseLoRALinear(base, rank, alpha))
            installed.append(f'blocks.{index}.self_attn.{name}')
        first, second = _ffn_targets(block)
        if isinstance(first, FrozenBaseLoRALinear) or isinstance(second, FrozenBaseLoRALinear):
            raise ValueError(f'FFN adapter already installed at block {index}')
        block.ffn[0] = FrozenBaseLoRALinear(first, rank, alpha)
        block.ffn[2] = FrozenBaseLoRALinear(second, rank, alpha)
        installed += [f'blocks.{index}.ffn.0', f'blocks.{index}.ffn.2']
    if len(installed) != 180 or len(set(installed)) != 180:
        raise AssertionError(f'Expected 180 unique video LoRA targets, got {len(installed)}')
    if any('cross_attn' in name for name in installed):
        raise AssertionError('Native text cross-attention must never receive a main LoRA')
    return installed


def video_lora_parameters(wan):
    return [(name, parameter) for name, parameter in wan.named_parameters()
            if parameter.requires_grad and (name.endswith('.lora_a') or name.endswith('.lora_b'))
            and '.cross_attn.' not in name]


def assert_native_text_frozen(wan):
    for name, parameter in wan.named_parameters():
        if '.cross_attn.' in name and parameter.requires_grad:
            raise ValueError(f'Native text parameter is trainable: {name}')
    return True
