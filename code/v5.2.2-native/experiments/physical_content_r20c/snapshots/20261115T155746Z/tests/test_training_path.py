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
from physical_content.interaction_branch import native_block_with_text_input


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
    for plan in ('CT','CI'):
        cfg = load_config(ROOT/'configs'/f'r20{plan.lower()}.yaml')
        cfg['model'].update(width=128, ffn_width=256, native_heads=4, qwen_width=32,
                            q_tokens=8, q_heads=4, grounded_heads=2)
        cfg['train']['peak_lrs'] = {k: 1e-3 for k in cfg['train']['peak_lrs']}
        model = PhysicalModel(cfg, device='cpu', wan=copy.deepcopy(base), interface=interface)
        blank = {'native': torch.zeros(1, 8, 128), 'route': torch.full((1, 8, 128), 1e-4)}
        assert model.enhancement.routing_fidelity(blank) == 0
        optimizer = torch.optim.AdamW(model.optim_groups(), lr=1e-3)
        sample = dict(latent=torch.randn(1, 4, 3, 4, 4), valid=torch.ones(4, 4),
                      times=torch.arange(3)[None]/6, text=torch.randn(1, 7, 64))
        if cfg['task'] == 'i2v':
            sample['first'] = sample['latent'][:, :, :1].clone()
        sigma, noise = torch.tensor(.7), torch.randn_like(sample['latent'])
        before = model.base_bank.hashes(full=True)
        # Zero output hints must preserve the actual native main-stream result.
        with torch.no_grad():
            c, groups = model.prepare_content(evidence, sample['text'])
            # A learned query may select different source words, but cannot
            # invent a free output value or decouple addressing from content.
            torch.testing.assert_close(c['routing'], c['execution'], atol=0, rtol=0)
            low = evidence['native_condition'].min(0).values
            high = evidence['native_condition'].max(0).values
            assert (c['execution'] >= low-1e-6).all()
            assert (c['execution'] <= high+1e-6).all()
            uniform_source = dict(evidence, native_condition=evidence['native_condition'][:1].expand(6,-1))
            constant, _ = model.prepare_content(uniform_source, sample['text'])
            torch.testing.assert_close(constant['execution'], uniform_source['native_condition'][:1][None].expand(1,8,-1), atol=1e-6, rtol=1e-6)
            noisy = (1-sigma)*sample['latent']+sigma*noise
            native_output, _ = model.velocity(noisy, sample.get('first'), sample['text'],
                                              sigma, c, groups, sample['times'], physical_mode='off')
            initial_output, _ = model.velocity(noisy, sample.get('first'), sample['text'],
                                               sigma, c, groups, sample['times'])
            torch.testing.assert_close(initial_output,native_output,atol=0,rtol=0)
            block = model.physical_blocks[0]
            h = torch.randn(1,12,128)
            mod = torch.randn(1,12,6,128)*.05
            grid = torch.tensor([[3,2,2]])
            lengths = torch.tensor([12])
            pmod = model.noise_modulation(torch.randn(1,3,128))
            if plan=='CI':
                state,text_input = native_block_with_text_input(model.wan.blocks[2],h,mod,lengths,
                                                               grid,model.wan.freqs,c['context'])
                expected = model.wan.blocks[2](h,mod,lengths,grid,model.wan.freqs,c['context'],None)
                torch.testing.assert_close(state,expected,atol=0,rtol=0)
                hint = block(h,text_input,mod,lengths,grid,model.wan.freqs,c['context'],c['text_mask'],
                             c['routing'],c['execution'],groups,sample['times'],pmod,4,4,'physical_condition_off')
            else:
                state, hint = block(h,mod,lengths,grid,model.wan.freqs,c['context'],c['text_mask'],
                                    c['routing'],c['execution'],groups,sample['times'],pmod,4,0,'physical_condition_off')
                expected = block.native(h,mod,lengths,grid,model.wan.freqs,c['context'],None)
                torch.testing.assert_close(state,expected,atol=1e-6,rtol=1e-6)
            assert hint.count_nonzero()==0
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
        required = ['enhancement.seed', 'enhancement.source_query.weight',
                    'enhancement.source_key.weight',
                    'physical_blocks.0.conditioning.direct.query_base.weight',
                    'physical_blocks.0.conditioning.direct_output.weight',
                    'physical_blocks.0.native.self_attn.q.weight',
                    'physical_blocks.4.native.ffn.0.weight',
                    'physical_blocks.4.output.weight',
                    'noise_modulation.project.1.weight', 'wan.blocks.0.self_attn.q.lora_b']
        if plan=='CI':
            required[required.index('physical_blocks.4.output.weight')] = 'physical_blocks.4.native.self_attn.o.weight'
        if cfg['model']['physical_read']=='direct_and_caption_associated':
            required.append('physical_blocks.0.conditioning.associated_output.weight')
        for name in required:
            assert norms.get(name, 0) > 0, name
        assert all(not p.requires_grad for n, p in model.wan.named_parameters() if '.cross_attn.' in n)
        assert all(not p.requires_grad for n,p in model.physical_blocks.named_parameters() if '.native.cross_attn.' in '.'+n)
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
                bypass, _ = model.velocity(x,sample.get('first'),sample['text'],sigma,c,groups,
                                           sample['times'],physical_mode='physical_condition_off')
                assert not torch.allclose(y,bypass,atol=1e-8,rtol=1e-8)
                last = model.physical_blocks[-1]
                with torch.no_grad():
                    if plan=='CI':
                        hint = last(h,h,mod,lengths,grid,model.wan.freqs,c['context'],c['text_mask'],
                                    c['routing'],c['execution'],groups,sample['times'],
                                    model.noise_modulation(torch.randn(1,3,128)),4,4,'normal')
                        off,_ = model.velocity(x,sample['first'],sample['text'],sigma,c,groups,
                                              sample['times'],physical_mode='off')
                        torch.testing.assert_close(bypass,off,atol=0,rtol=0)
                        # A constant video value cannot become a global C-value
                        # broadcast, even with a nonzero trained condition read.
                        constant_h=h[:,:1].expand_as(h)
                        constant_mod=mod[:,:1].expand_as(mod)
                        constant_hint=last(constant_h,constant_h,constant_mod,lengths,grid,model.wan.freqs,
                            c['context'],c['text_mask'],c['routing'],c['execution'],groups,sample['times'],
                            model.noise_modulation(torch.randn(1,3,128)),4,0,'normal')
                        assert constant_hint.abs().max()<5e-6
                    else:
                        hidden,hint = last(h,mod,lengths,grid,model.wan.freqs,c['context'],c['text_mask'],
                                           c['routing'],c['execution'],groups,sample['times'],
                                           model.noise_modulation(torch.randn(1,3,128)),4,0,'normal')
                    assert hint[:,4:].abs().sum()>0
                    if cfg['task']=='i2v':
                        assert hint[:,:4].count_nonzero()==0
        records.append(dict(task=cfg['task'], stages='128 physical generation + 272 joint',
                            generation_gradient_paths=len(norms), resume_equal=True,
                            frozen_native_unchanged=True, final_write_ratio=value['write_relative_rms_mean']))
    report = dict(passed=True, scope='Small real Wan CPU; not generation-quality evidence', records=records)
    (ROOT/'reports/training_path_tests.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
