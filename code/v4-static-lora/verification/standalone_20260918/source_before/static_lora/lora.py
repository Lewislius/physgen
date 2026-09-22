"""Ordinary W x + (alpha/r) B A x; frozen original weight and bias, no extra gates."""
import math

import torch
from torch import nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        if not isinstance(base, nn.Linear) or rank <= 0:
            raise ValueError("LoRA requires a Linear layer and positive rank")
        self.base = base.requires_grad_(False)
        self.in_features, self.out_features = base.in_features, base.out_features
        self.rank, self.alpha = rank, float(alpha)
        self.scale = self.alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale


def inject_lora(wan, spec):
    if spec["dropout"] != 0:
        raise ValueError("This baseline fixes LoRA dropout to zero")
    wan.requires_grad_(False)
    names = []
    for index, block in enumerate(wan.blocks):
        for target in spec["targets"]:
            parent_name, name = target.rsplit(".", 1)
            parent = block.get_submodule(parent_name)
            layer = getattr(parent, name)
            if not isinstance(layer, nn.Linear):
                raise ValueError(f"Expected unmodified Linear: blocks.{index}.{target}")
            setattr(parent, name, LoRALinear(layer, spec["rank"], spec["alpha"]))
            names.append(f"blocks.{index}.{target}")
    if len(names) != len(wan.blocks) * 8:
        raise ValueError("Expected q/k/v/o in self- and cross-attention of every block")
    assert_lora_only(wan)
    return names


def lora_named_parameters(model):
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            prefix = name + "." if name else ""
            yield prefix + "lora_A", module.lora_A
            yield prefix + "lora_B", module.lora_B


def assert_lora_only(model):
    expected = dict(lora_named_parameters(model))
    actual = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if not expected or actual.keys() != expected.keys() or any(p.dtype != torch.float32 for p in actual.values()):
        raise ValueError("Only FP32 LoRA A/B matrices may be trainable")
    return sum(p.numel() for p in actual.values())


def lora_state_dict(model):
    return {name: p.detach().cpu().clone() for name, p in lora_named_parameters(model)}


@torch.no_grad()
def load_lora_state_dict(model, values):
    expected = dict(lora_named_parameters(model))
    if set(expected) != set(values):
        raise ValueError("Checkpoint LoRA target names differ; full-model/adapter weights are not accepted")
    for name, param in expected.items():
        value = values[name]
        if value.shape != param.shape or value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
            raise ValueError(f"Invalid LoRA weight: {name}")
    for name, param in expected.items():
        param.copy_(values[name])
