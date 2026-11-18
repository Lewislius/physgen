"""Small real-Wan checks of stages, generation gradients and resumable weights."""
import copy
import importlib
import json
from pathlib import Path
import sys
import tempfile
import warnings

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bootstrap
from physical_content.common import load_config
from physical_content.model import PhysicalModel
from physical_content.wan_loader import ensure_wan_namespace
from physical_content.objectives import backward_micro
from physical_content.schedule import set_lrs
from physical_content.checkpointing import save_checkpoint, restore_checkpoint


def cpu_attention(q, k, v, **kwargs):
    return F.scaled_dot_product_attention(q.transpose(1, 2).float(), k.transpose(1, 2).float(),
                                         v.transpose(1, 2).float()).transpose(1, 2)


def main():
    torch.set_num_threads(2)
    torch.manual_seed(73)
    warnings.filterwarnings('ignore', message='User provided device_type of.*')
    ensure_wan_namespace({'paths': {'wan_code': '/home/liuzhirui/model/Wan2.2'}})
    native = importlib.import_module('wan.modules.model')
    native.flash_attention = cpu_attention
    base = native.WanModel(model_type='t2v', dim=128, ffn_dim=256, num_heads=4,
                           num_layers=30, text_len=12, text_dim=64, freq_dim=16,
                           in_dim=4, out_dim=4, patch_size=(1, 2, 2)).eval()
    nn.init.normal_(base.head.head.weight, std=.04)
    interface = dict(mean=torch.zeros(32), scale=torch.ones(32), native_scale=1.3)
    evidence = dict(words=torch.randn(6, 32), request_words=torch.randn(5, 32),
                    native_condition=torch.randn(6, 128)*.1,
                    word_groups=torch.tensor([0, 0, 0, 1, 1, 1]))
    records = []
    for plan in ('AT', 'AI'):
        cfg = load_config(ROOT/'configs'/f'r20{plan.lower()}.yaml')
        cfg['model'].update(width=128, ffn_width=256, native_heads=4, qwen_width=32,
                            q_tokens=8, q_heads=4, grounded_heads=2)
        cfg['train']['peak_lrs'] = {k: 1e-3 for k in cfg['train']['peak_lrs']}
        model = PhysicalModel(cfg, device='cpu', wan=copy.deepcopy(base), interface=interface)
        blank = {'native': torch.zeros(1, 8, 128), 'route': torch.full((1, 8, 128), 1e-4)}
        assert 0 < model.enhancement.routing_fidelity(blank) < 1e-5
        optimizer = torch.optim.AdamW(model.optim_groups(), lr=1e-3)
        sample = dict(latent=torch.randn(1, 4, 3, 4, 4), valid=torch.ones(4, 4),
                      times=torch.arange(3)[None]/6, text=torch.randn(1, 7, 64))
        if cfg['task'] == 'i2v':
            sample['first'] = sample['latent'][:, :, :1].clone()
        sigma, noise = torch.tensor(.7), torch.randn_like(sample['latent'])
        before = model.base_bank.hashes(full=True)
        assert model.set_phase(0) == 'physical_generation'
        assert not any(p.requires_grad for p in model.wan.parameters())
        set_lrs(optimizer, cfg, 1)
        assert all(g['lr'] == 0 for g in optimizer.param_groups if g['role'] == 'wan_video_lora')
        norms = {}
        for step in (1, 32, 128, 129, 200):
            phase = model.set_phase(step-1)
            set_lrs(optimizer, cfg, step)
            optimizer.zero_grad(set_to_none=True)
            value = backward_micro(model, evidence, sample, sample['text'], sigma, noise, 0., 1)
            if step <= 128:
                assert all(p.grad is None for p in model.wan.parameters())
            else:
                assert phase == 'joint'
                assert model.wan.blocks[0].self_attn.q.lora_b.grad is not None
            for name, p in model.named_parameters():
                if p.grad is not None:
                    norms[name] = max(norms.get(name, 0), float(p.grad.norm()))
            optimizer.step()
        for name in ('enhancement.seed', 'enhancement.route_update.weight',
                     'enhancement.execute_update.weight', 'physics.2.direct.query_base.weight',
                     'physics.2.direct_output.weight', 'physics.2.associated_output.weight',
                     'noise_modulation.project.1.weight', 'wan.blocks.0.self_attn.q.lora_b'):
            assert norms.get(name, 0) > 0, name
        assert all(not p.requires_grad for n, p in model.wan.named_parameters() if '.cross_attn.' in n)
        model.assert_base_unchanged(before, full=True)
        with tempfile.TemporaryDirectory(prefix='r20_small_resume_') as temp:
            checkpoint = save_checkpoint(model, optimizer, cfg, temp, 200, 800)
            restored = PhysicalModel(cfg, device='cpu', wan=copy.deepcopy(base), interface=interface)
            restored_optimizer = torch.optim.AdamW(restored.optim_groups(), lr=1e-3)
            update, cursor, _ = restore_checkpoint(restored, restored_optimizer, checkpoint, cfg)
            assert (update, cursor) == (200, 800)
            with torch.no_grad():
                c, groups = model.prepare_content(evidence, sample['text'])
                c2, groups2 = restored.prepare_content(evidence, sample['text'])
                x = (1-sigma)*sample['latent']+sigma*noise
                y, _ = model.velocity(x, sample.get('first'), sample['text'], sigma, c, groups, sample['times'])
                z, _ = restored.velocity(x, sample.get('first'), sample['text'], sigma, c2, groups2, sample['times'])
                torch.testing.assert_close(y, z, atol=0, rtol=0)
        records.append(dict(task=cfg['task'], stages='128 physical generation + 272 joint',
                            generation_gradient_paths=len(norms), resume_equal=True,
                            frozen_native_unchanged=True, final_write_ratio=value['write_relative_rms_mean']))
    report = dict(passed=True, scope='Small real Wan CPU; not generation-quality evidence', records=records)
    (ROOT/'reports/training_path_tests.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
