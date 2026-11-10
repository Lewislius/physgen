"""Frozen bases with explicit, checkpoint-safe knowledge/teacher adapter roles."""
from contextlib import contextmanager
from contextvars import ContextVar
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

_knowledge = ContextVar('physgen_knowledge_adapter', default=False)


@contextmanager
def knowledge_role(enabled):
    token = _knowledge.set(bool(enabled))
    try: yield
    finally: _knowledge.reset(token)


def role_checkpoint(function, *args, **kwargs):
    # Capture the role NOW, not the ambient role during backward recomputation.
    enabled = _knowledge.get()
    def run(*inputs):
        with knowledge_role(enabled):
            return function(*inputs, **kwargs)
    return checkpoint(run, *args, use_reentrant=False)


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha, *, knowledge=False):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.knowledge = knowledge
        self.scale = alpha/rank
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32, device=base.weight.device))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x):
        result = self.base(x)
        if not self.knowledge or _knowledge.get():
            delta = F.linear(F.linear(x.to(self.lora_a.dtype), self.lora_a), self.lora_b)
            result = result + (delta*self.scale).to(result.dtype)
        return result


def install_wan_lora(wan, cfg):
    for block in wan.blocks:
        for attention in (block.self_attn, block.cross_attn):
            for name in ('q', 'k', 'v', 'o'):
                setattr(attention, name, LoRALinear(getattr(attention, name), cfg['rank'], cfg['alpha']))


def install_qwen_lora(model, cfg):
    for layer in model.model.language_model.layers[-cfg['lora_last_layers']:]:
        for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
            attn = layer.self_attn
            setattr(attn, name, LoRALinear(getattr(attn, name), cfg['lora_rank'], cfg['lora_alpha'], knowledge=True))
    if cfg['gradient_checkpointing']:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        for module in model.modules():
            if hasattr(module, '_gradient_checkpointing_func'):
                module._gradient_checkpointing_func = role_checkpoint
