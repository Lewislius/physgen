"""Frozen bases with explicit, checkpoint-safe knowledge/teacher adapter roles."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

_knowledge = ContextVar('physgen_knowledge_adapter', default=False)
_verifier = ContextVar('physgen_verifier_adapter', default=False)
_checkpoint_blocks = ContextVar('physgen_checkpoint_eval_blocks', default=False)


@contextmanager
def checkpoint_eval_blocks(enabled):
    token = _checkpoint_blocks.set(bool(enabled))
    try: yield
    finally: _checkpoint_blocks.reset(token)


def install_eval_checkpointing(model):
    """Bound verifier memory without enabling dropout or changing state keys.

    Transformers' native block checkpoints require training=True. Our shared
    frozen Qwen must stay in eval mode, including during RGB differentiation.
    Whole-read checkpointing alone still retains every layer on recomputation.
    """
    for block in (*model.model.visual.blocks, *model.model.language_model.layers):
        if getattr(block, '_physgen_eval_checkpoint', False):
            continue
        def wrap(forward):
            @wraps(forward)
            def run(*args, **kwargs):
                if _checkpoint_blocks.get() and torch.is_grad_enabled():
                    return role_checkpoint(forward, *args, **kwargs)
                return forward(*args, **kwargs)
            return run
        block.forward = wrap(block.forward)
        block._physgen_eval_checkpoint = True


@contextmanager
def knowledge_role(enabled):
    token = _knowledge.set(bool(enabled))
    try: yield
    finally: _knowledge.reset(token)


@contextmanager
def verifier_role(enabled):
    token = _verifier.set(bool(enabled))
    try: yield
    finally: _verifier.reset(token)


def role_checkpoint(function, *args, **kwargs):
    # Capture the role NOW, not the ambient role during backward recomputation.
    enabled, judging = _knowledge.get(), _verifier.get()
    def run(*inputs):
        with knowledge_role(enabled), verifier_role(judging):
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


class VerifierLoRALinear(nn.Module):
    """Separate adapter ownership; never changes generation-side Qwen features."""
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.scale = alpha / rank
        self.verifier_lora_a = nn.Parameter(torch.empty(rank, base.in_features,
            dtype=torch.float32, device=base.weight.device))
        self.verifier_lora_b = nn.Parameter(torch.zeros(base.out_features, rank,
            dtype=torch.float32, device=base.weight.device))
        nn.init.kaiming_uniform_(self.verifier_lora_a, a=math.sqrt(5))

    def forward(self, x):
        result = self.base(x)
        if _verifier.get():
            delta = F.linear(F.linear(x.to(self.verifier_lora_a.dtype), self.verifier_lora_a), self.verifier_lora_b)
            result = result + (delta * self.scale).to(result.dtype)
        return result


def install_verifier_lora(model, cfg):
    layers = model.model.language_model.layers
    if not 1 <= cfg['last_layers'] <= len(layers):
        raise ValueError('Verifier LoRA depth exceeds the native Qwen language model')
    for layer in layers[-cfg['last_layers']:]:
        for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
            base = getattr(layer.self_attn, name)
            if not isinstance(base, nn.Linear):
                raise ValueError('Verifier LoRA requires an unadapted frozen Qwen base')
            setattr(layer.self_attn, name, VerifierLoRALinear(base, cfg['rank'], cfg['alpha']))


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
