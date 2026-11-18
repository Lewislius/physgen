"""Small actual-Wan checks for R20BT-new.

Checks, in order:
- every physical block is the same-depth copy at its site, and every reader is
  initialised from the same-site cross-attention;
- physical blocks are independent (no tied parameters, no shared state);
- each physical block receives the actual native block input at its site;
- C, text context, grid, freqs and lengths are shared across sites;
- times enters the per-block reader and receives generation gradients;
- native equivalence at initialisation, stage ownership, gradients, frozen
  native text and checkpoint resume.

This is implementation evidence, not generation-quality evidence.
"""
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
from physical_content.common import load_config, validate_config
from physical_content.model import PhysicalModel
from physical_content.wan_loader import ensure_wan_namespace
from physical_content.objectives import backward_micro
from physical_content.schedule import set_lrs
from physical_content.checkpointing import save_checkpoint, restore_checkpoint


def cpu_attention(q, k, v, **kwargs):
    return F.scaled_dot_product_attention(
        q.transpose(1, 2).float(), k.transpose(1, 2).float(),
        v.transpose(1, 2).float()).transpose(1, 2)


def main():
    torch.set_num_threads(2)
    torch.manual_seed(73)
    warnings.filterwarnings('ignore', message='User provided device_type of.*')
    ensure_wan_namespace({'paths': {'wan_code': '/home/liuzhirui/model/Wan2.2'}})
    native = importlib.import_module('wan.modules.model')
    native.flash_attention = cpu_attention
    base = native.WanModel(
        model_type='t2v', dim=128, ffn_dim=256, num_heads=4, num_layers=30,
        text_len=12, text_dim=64, freq_dim=16, in_dim=4, out_dim=4,
        patch_size=(1, 2, 2)).eval()
    nn.init.normal_(base.head.head.weight, std=.04)
    interface = dict(mean=torch.zeros(32), scale=torch.ones(32), native_scale=1.3)
    evidence = dict(
        words=torch.randn(6, 32), request_words=torch.randn(5, 32),
        native_condition=torch.randn(6, 128) * .1,
        word_groups=torch.tensor([0, 0, 0, 1, 1, 1]))
    cfg = load_config(ROOT / 'configs' / 'r20bt_new.yaml')
    bad = copy.deepcopy(cfg)
    bad['model']['physical_initialization_blocks'] = [0, 1, 2, 3, 4]
    try:
        validate_config(bad)
    except ValueError:
        pass
    else:
        raise AssertionError('Mismatched physical initialization was accepted')
    cfg['model'].update(width=128, ffn_width=256, native_heads=4, qwen_width=32,
                        q_tokens=8, q_heads=4, grounded_heads=2)
    cfg['train']['peak_lrs'] = {key: 1e-3 for key in cfg['train']['peak_lrs']}
    model = PhysicalModel(cfg, device='cpu', wan=copy.deepcopy(base), interface=interface)
    base_before = model.base_bank.hashes(full=True)

    # 1. Depth copy, reader copy, independence, no NoiseConditioning.
    assert tuple(model.injection_layers) == (2, 8, 14, 20, 26)
    assert cfg['model']['physical_initialization_blocks'] == cfg['model']['injection_layers']
    assert not hasattr(model, 'noise_modulation')
    assert 'physics_state' not in cfg['train']['peak_lrs']
    for k, site in enumerate(model.injection_layers):
        physical = model.physical_blocks[k]
        bank = model.base_bank
        for field in ('self_attn.q', 'self_attn.k', 'self_attn.v', 'self_attn.o',
                      'ffn.0', 'ffn.2'):
            assert torch.equal(
                bank.linear(site, field).weight.float(),
                dict(physical.native.named_parameters())[field + '.weight'].float()), (site, field)
        assert torch.equal(
            bank.modulation(site).float(), physical.native.modulation.float())
        assert torch.equal(
            bank.linear(site, 'cross_attn.q').weight.float(),
            physical.conditioning.direct.query_base.weight.float())
        assert torch.equal(
            bank.linear(site, 'cross_attn.k').weight.float(),
            physical.conditioning.direct.key.weight.float())
        assert torch.equal(
            bank.linear(site, 'cross_attn.v').weight.float(),
            physical.conditioning.direct.value.weight.float())
    parameter_ids = [id(p) for block in model.physical_blocks for p in block.parameters()]
    assert len(parameter_ids) == len(set(parameter_ids)), 'physical blocks share parameters'

    sample = dict(latent=torch.randn(1, 4, 3, 4, 4), valid=torch.ones(4, 4),
                  times=torch.arange(3)[None] / 6, text=torch.randn(1, 7, 64))
    sigma, noise = torch.tensor(.7), torch.randn_like(sample['latent'])
    with torch.no_grad():
        content, groups = model.prepare_content(evidence, sample['text'])
        noisy = (1 - sigma) * sample['latent'] + sigma * noise
        native_output, _ = model.velocity(
            noisy, None, sample['text'], sigma, content, groups,
            sample['times'], physical_mode='off')
        initial_output, _ = model.velocity(
            noisy, None, sample['text'], sigma, content, groups, sample['times'])
        torch.testing.assert_close(initial_output, native_output, atol=0, rtol=0)
        cond_off, _ = model.velocity(
            noisy, None, sample['text'], sigma, content, groups,
            sample['times'], physical_mode='physical_condition_off')
        torch.testing.assert_close(cond_off, native_output, atol=0, rtol=0)

    # 2. Per-site native input and shared C/context/grid/freqs/lengths.
    physical_inputs, native_inputs = {}, {}
    routes, executes, contexts, grids, freqs, lengths = {}, {}, {}, {}, {}, {}
    handles = []
    for k, site in enumerate(model.injection_layers):
        def physical_hook(site=site):
            def hook(module, args):
                physical_inputs[site] = args[0].detach().clone()
                lengths[site] = args[2].detach().clone()
                grids[site] = args[3].detach().clone()
                freqs[site] = args[4].detach().clone()
                contexts[site] = args[5].detach().clone()
                routes[site] = args[6].detach().clone()
                executes[site] = args[7].detach().clone()
            return hook
        def native_hook(site=site):
            def hook(module, args):
                native_inputs[site] = args[0].detach().clone()
            return hook
        handles.append(model.physical_blocks[k].register_forward_pre_hook(physical_hook()))
        handles.append(model.wan.blocks[site].register_forward_pre_hook(native_hook()))
    with torch.no_grad():
        model.velocity(noisy, None, sample['text'], sigma, content, groups,
                       sample['times'], physical_mode='normal')
    for handle in handles:
        handle.remove()
    for site in model.injection_layers:
        assert torch.equal(physical_inputs[site], native_inputs[site]), site
    first = model.injection_layers[0]
    for site in model.injection_layers[1:]:
        assert torch.equal(routes[site], routes[first]), 'C route is not shared'
        assert torch.equal(executes[site], executes[first]), 'C execute is not shared'
        assert torch.equal(contexts[site], contexts[first]), 'context is not shared'
        assert torch.equal(grids[site], grids[first]), 'grid is not shared'
        assert torch.equal(freqs[site], freqs[first]), 'freqs are not shared'
        assert torch.equal(lengths[site], lengths[first]), 'lengths are not shared'
    # times must be consumed by the per-block reader.
    time_calls = []
    time_handle = model.physical_blocks[0].conditioning.direct.time.register_forward_hook(
        lambda module, args, output: time_calls.append(int(output.shape[-1])))
    with torch.no_grad():
        model.velocity(noisy, None, sample['text'], sigma, content, groups,
                       sample['times'], physical_mode='normal')
    time_handle.remove()
    assert time_calls and all(value == cfg['model']['width'] for value in time_calls)

    # 3. Stages, gradients, frozen native text, resume.
    optimizer = torch.optim.AdamW(model.optim_groups(), lr=1e-3)
    assert model.set_phase(0) == 'physical_generation'
    assert not any(p.requires_grad for p in model.wan.parameters())
    set_lrs(optimizer, cfg, 1)
    assert all(g['lr'] == 0 for g in optimizer.param_groups
               if g['role'] == 'wan_video_lora')
    norms = {}
    final = None
    for step in (1, 2, 8, 32, 128, 129, 200):
        phase = model.set_phase(step - 1)
        set_lrs(optimizer, cfg, step)
        optimizer.zero_grad(set_to_none=True)
        final = backward_micro(model, evidence, sample, sample['text'],
                               sigma, noise, 0., 1)
        if step <= 128:
            assert all(p.grad is None for p in model.wan.parameters())
        else:
            assert phase == 'joint'
            assert model.wan.blocks[0].self_attn.q.lora_b.grad is not None
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                norms[name] = max(norms.get(name, 0.), float(parameter.grad.norm()))
        optimizer.step()
    required = [
        'enhancement.seed', 'enhancement.route_update.weight',
        'enhancement.execute_update.weight',
        'physical_blocks.0.conditioning.direct.query_base.weight',
        'physical_blocks.0.conditioning.direct.key.weight',
        'physical_blocks.0.conditioning.direct.value.weight',
        'physical_blocks.0.conditioning.direct.time.proj.weight',
        'physical_blocks.0.conditioning.direct_output.weight',
        'physical_blocks.0.native.self_attn.q.weight',
        'physical_blocks.0.native.modulation',
        'physical_blocks.4.native.ffn.0.weight',
        'physical_blocks.4.output.weight',
        'wan.blocks.0.self_attn.q.lora_b']
    for name in required:
        assert norms.get(name, 0.) > 0., name
    assert all(not p.requires_grad for n, p in model.wan.named_parameters()
               if '.cross_attn.' in n)
    assert all(not p.requires_grad for n, p in model.physical_blocks.named_parameters()
               if '.native.cross_attn.' in '.' + n)
    model.assert_base_unchanged(base_before, full=True)

    # After training, the per-block times embedding must affect the generated
    # velocity, not merely be called.
    with torch.no_grad():
        c, g = model.prepare_content(evidence, sample['text'])
        y_times, _ = model.velocity(
            noisy, None, sample['text'], sigma, c, g, sample['times'],
            physical_mode='normal')
        y_times_shift, _ = model.velocity(
            noisy, None, sample['text'], sigma, c, g, sample['times'] + 0.05,
            physical_mode='normal')
        assert not torch.allclose(y_times, y_times_shift, atol=1e-8, rtol=1e-8)

    with tempfile.TemporaryDirectory(prefix='r20bt_new_small_resume_') as temp:
        checkpoint = save_checkpoint(model, optimizer, cfg, temp, 200, 800)
        restored = PhysicalModel(
            cfg, device='cpu', wan=copy.deepcopy(base), interface=interface)
        restored_optimizer = torch.optim.AdamW(restored.optim_groups(), lr=1e-3)
        update, cursor, _ = restore_checkpoint(
            restored, restored_optimizer, checkpoint, cfg)
        assert (update, cursor) == (200, 800)
        with torch.no_grad():
            c1, g1 = model.prepare_content(evidence, sample['text'])
            c2, g2 = restored.prepare_content(evidence, sample['text'])
            y, _ = model.velocity(noisy, None, sample['text'], sigma, c1, g1,
                                  sample['times'], physical_mode='normal')
            z, _ = restored.velocity(noisy, None, sample['text'], sigma, c2, g2,
                                     sample['times'], physical_mode='normal')
            torch.testing.assert_close(y, z, atol=0, rtol=0)
            bypass, _ = model.velocity(
                noisy, None, sample['text'], sigma, c1, g1,
                sample['times'], physical_mode='physical_condition_off')
            assert not torch.allclose(y, bypass, atol=1e-8, rtol=1e-8)
            off, _ = model.velocity(
                noisy, None, sample['text'], sigma, c1, g1,
                sample['times'], physical_mode='off')
            assert not torch.allclose(y, off, atol=1e-8, rtol=1e-8)

    report = dict(
        passed=True,
        scope='Small real Wan CPU; implementation evidence only, not quality evidence',
        injection_layers=list(model.injection_layers),
        physical_initialization_blocks=cfg['model']['physical_initialization_blocks'],
        depth_copies_verified=True,
        reader_copies_verified=True,
        per_site_native_input_verified=True,
        shared_c_verified=True,
        shared_context_grid_freqs_lengths_verified=True,
        reader_times_verified=True,
        reader_times_affect_output=True,
        no_noise_conditioning=True,
        physical_blocks_independent=True,
        native_start_equal=True,
        stages='128 physical generation + 272 joint',
        generation_gradient_paths=len(norms),
        resume_equal=True,
        frozen_native_unchanged=True,
        final_write_ratio=final['write_relative_rms_mean'])
    (ROOT / 'reports/training_path_tests.json').write_text(
        json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
