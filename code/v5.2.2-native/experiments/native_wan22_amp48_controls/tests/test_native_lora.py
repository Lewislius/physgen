"""Small-Wan test of the native video-LoRA training/inference operators."""
import copy
from pathlib import Path
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bootstrap
from native_control.lora import (install_video_lora, video_lora_parameters)
from native_control.wan_loader import ensure_wan_namespace
from train_lora import _velocity


def main():
    torch.set_num_threads(2)
    ensure_wan_namespace({'paths': {'wan_code': '/home/liuzhirui/model/Wan2.2'}})
    import importlib
    native = importlib.import_module('wan.modules.model')
    from torch.nn import functional as F
    native.flash_attention = lambda q, k, v, **kw: F.scaled_dot_product_attention(
        q.transpose(1, 2).float(), k.transpose(1, 2).float(),
        v.transpose(1, 2).float()).transpose(1, 2)
    base = native.WanModel(model_type='t2v', dim=128, ffn_dim=256, num_heads=4,
                           num_layers=30, text_len=12, text_dim=64, freq_dim=16,
                           in_dim=4, out_dim=4, patch_size=(1, 2, 2)).eval()
    nn.init.normal_(base.head.head.weight, std=.04)
    cfg = {'base': {'video_lora_rank': 32, 'video_lora_alpha': 32}}
    official_copy = copy.deepcopy(base)
    base.requires_grad_(False)
    targets = install_video_lora(base, cfg)
    assert len(targets) == 180 and len(set(targets)) == 180
    parameters = [p for _, p in video_lora_parameters(base)]
    assert len(parameters) == 360
    assert all(p.requires_grad for p in parameters)
    assert all(not p.requires_grad for name, p in base.named_parameters()
               if not (name.endswith('.lora_a') or name.endswith('.lora_b')))

    noisy = torch.randn(1, 4, 3, 4, 10)
    text = torch.randn(1, 7, 64)
    sigma = torch.tensor(.5)
    prediction = _velocity(base, noisy, text, sigma, checkpointing=True)
    assert prediction.shape == noisy.shape
    loss = prediction.square().mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in parameters)
    assert all(p.grad is None for name, p in base.named_parameters()
               if not (name.endswith('.lora_a') or name.endswith('.lora_b')))

    optimizer = torch.optim.AdamW(parameters, lr=1e-3)
    optimizer.step()
    state = {name: value.detach().clone()
             for name, value in video_lora_parameters(base)}
    with torch.no_grad():
        expected = _velocity(base, noisy, text, sigma, checkpointing=False)

    # Build a clean copy from the official pre-LoRA weights and reload the
    # trained LoRA state; inference must match bit-for-bit.
    clean = official_copy
    clean.requires_grad_(False)
    install_video_lora(clean, cfg)
    named = dict(clean.named_parameters())
    with torch.no_grad():
        for name, value in state.items():
            named[name].copy_(value)
        restored = _velocity(clean, noisy, text, sigma, checkpointing=False)
    torch.testing.assert_close(restored, expected, atol=0, rtol=0)

    print('PASS native video-LoRA: 180 targets, frozen native text/base, '
          'checkpointed forward, LoRA-only gradients, state reload, exact inference.')


if __name__ == '__main__':
    main()
