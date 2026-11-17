"""Actual small Wan: native start, depth mapping, PWCA contract and gradients."""
import copy
import importlib
from pathlib import Path
import sys
import warnings
import tempfile

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
from physical_content.checkpointing import save_checkpoint, restore_checkpoint
from physical_content.grounded_physics import native_block_with_read_state


def main():
    torch.set_num_threads(2)
    torch.manual_seed(73)
    warnings.filterwarnings('ignore', message='User provided device_type of.*')
    ensure_wan_namespace({'paths': {'wan_code': '/home/liuzhirui/model/Wan2.2'}})
    native = importlib.import_module('wan.modules.model')
    native.flash_attention = lambda q, k, v, **kw: F.scaled_dot_product_attention(
        q.transpose(1, 2).float(), k.transpose(1, 2).float(),
        v.transpose(1, 2).float()).transpose(1, 2)
    base = native.WanModel(model_type='t2v', dim=128, ffn_dim=256, num_heads=4,
                           num_layers=30, text_len=12, text_dim=64, freq_dim=16,
                           in_dim=4, out_dim=4, patch_size=(1, 2, 2)).eval()
    nn.init.normal_(base.head.head.weight, std=.04)
    cfg = load_config(ROOT / 'configs/r20it.yaml')
    cfg['model'].update(width=128, ffn_width=256, native_heads=4, qwen_width=32,
                        q_tokens=8, q_heads=4, grounded_heads=2)
    interface = dict(mean=torch.zeros(32), scale=torch.tensor(1.), native_scale=1.3)
    model = PhysicalModel(cfg, device='cpu', wan=copy.deepcopy(base), interface=interface)

    # The exposed read state must not change the official Wan computation.
    bs = torch.randn(1, 30, 128); bm = torch.randn(1, 30, 6, 128) * .05
    bl = torch.tensor([30]); bg = torch.tensor([[3, 2, 5]]); bc = torch.randn(1, 7, 128)
    with torch.no_grad():
        for site in model.injection_layers:
            original = base.blocks[site](bs, bm, bl, bg, base.freqs, bc, None)
            exposed, _ = native_block_with_read_state(
                base.blocks[site], bs, bm, bl, bg, base.freqs, bc)
            torch.testing.assert_close(original, exposed, atol=0, rtol=0)

    evidence = dict(words=torch.randn(6, 32), request_words=torch.randn(5, 32),
                    native_condition=torch.randn(6, 128) * .1,
                    word_groups=torch.tensor([0, 0, 0, 1, 1, 1]))
    sample = dict(latent=torch.randn(1, 4, 3, 4, 10), valid=torch.ones(4, 10),
                  times=torch.arange(3)[None] / 6, text=torch.randn(1, 7, 64))
    sigma, noise = torch.tensor(.7), torch.randn_like(sample['latent'])
    noisy = (1 - sigma) * sample['latent'] + sigma * noise

    with torch.no_grad():
        c, g = model.prepare_content(evidence, sample['text'])
        assert c['word_memory'][0].shape == evidence['words'].shape
        assert c['word_memory'][1].shape == evidence['native_condition'].shape
        native_y, _ = model.velocity(noisy, None, sample['text'], sigma, c, g,
                                     sample['times'], physical_mode='off')
        initial_y, _ = model.velocity(noisy, None, sample['text'], sigma, c, g,
                                      sample['times'])
        torch.testing.assert_close(initial_y, native_y, atol=0, rtol=0)
        # Zero-init PWCA must contribute exactly zero at step 0.
        physical = model.physical_blocks[0]
        word_zero = physical._word_attention(
            torch.randn(1, 30, 128), bg, c['word_memory'], True)
        torch.testing.assert_close(word_zero, torch.zeros_like(word_zero),
                                   atol=0, rtol=0)

    # Verify every copied parameter belongs to the corresponding native depth.
    assert tuple(model.injection_layers) == (0, 7, 14, 21, 28)
    for site, physical in zip(model.injection_layers, model.physical_blocks):
        source = dict(base.blocks[site].named_parameters())
        copied = dict(physical.native.named_parameters())
        source = {name: value for name, value in source.items() if 'lora' not in name}
        assert set(source) == set(copied), (site, set(source) ^ set(copied))
        for name, value in source.items():
            torch.testing.assert_close(value.float(), copied[name].float(),
                                       atol=0, rtol=0)
        torch.testing.assert_close(physical.word_query.weight,
                                   base.blocks[site].cross_attn.q.weight.float(),
                                   atol=0, rtol=0)
        torch.testing.assert_close(physical.word_query.bias,
                                   base.blocks[site].cross_attn.q.bias.float(),
                                   atol=0, rtol=0)
        torch.testing.assert_close(physical.conditioning.query.weight,
                                   base.blocks[site].cross_attn.q.weight.float(),
                                   atol=0, rtol=0)
        torch.testing.assert_close(physical.conditioning.key.weight,
                                   base.blocks[site].cross_attn.k.weight.float(),
                                   atol=0, rtol=0)
        torch.testing.assert_close(physical.conditioning.value.weight,
                                   base.blocks[site].cross_attn.v.weight.float(),
                                   atol=0, rtol=0)
        assert physical.native.self_attn.q.weight.shape == base.blocks[site].self_attn.q.weight.shape

    # PWCA K/V are frozen buffers, not optimizer parameters.
    for name, value in model.named_buffers():
        if name.endswith('.word_key') or name.endswith('.word_value'):
            assert not value.requires_grad
    assert not any(name.endswith('.word_key') or name.endswith('.word_value')
                   for name in model.learned_names)

    before = model.base_bank.hashes(full=True)
    opt = torch.optim.AdamW(model.optim_groups(), lr=1e-3)
    for group in opt.param_groups:
        group['lr'] = 1e-3
    norms = {}
    for step in (0, 1, 2, 128):
        phase = model.set_phase(step)
        opt.zero_grad(set_to_none=True)
        values = backward_micro(model, evidence, sample, sample['text'],
                                sigma, noise, .03, 1)
        if step < 128:
            assert all(p.grad is None for p in model.wan.parameters())
        else:
            assert model.wan.blocks[0].self_attn.q.lora_b.grad is not None
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                norms[name] = max(norms.get(name, 0.), float(parameter.grad.norm()))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        opt.step()
    required = [
        'enhancement.seed', 'enhancement.route_update.weight',
        'enhancement.execute_update.weight', 'enhancement.block.ffn.0.weight',
        'physical_blocks.0.conditioning.query.weight',
        'physical_blocks.0.conditioning.key.weight',
        'physical_blocks.0.conditioning.value.weight',
        'physical_blocks.0.query_update.weight', 'physical_blocks.4.key_update.weight',
        'physical_blocks.0.word_query.weight', 'physical_blocks.0.word_out.weight',
        'physical_blocks.4.word_query.weight', 'physical_blocks.4.word_out.weight',
        'physical_blocks.0.word_time.weight', 'physical_blocks.0.word_local.weight',
        'physical_blocks.0.word_logit_scale',
        'physical_blocks.0.native.self_attn.v.weight',
        'physical_blocks.0.native.cross_attn.k.weight',
        'physical_blocks.0.native.ffn.0.weight', 'physical_blocks.4.output.weight',
    ]
    for name in required:
        assert norms.get(name, 0.) > 0, (name, norms.get(name, 0.))
    model.assert_base_unchanged(before, full=True)
    assert all(not p.requires_grad for n, p in model.wan.named_parameters()
               if '.cross_attn.' in n)

    with torch.no_grad():
        c, g = model.prepare_content(evidence, sample['text'])
        plain, _ = model.velocity(noisy, None, sample['text'], sigma, c, g,
                                  sample['times'], physical_mode='off')
        empty, eg = model.prepare_content(dict(evidence, relations=['']), sample['text'])
        no_condition, _ = model.velocity(noisy, None, sample['text'], sigma, empty, eg,
                                         sample['times'])
        torch.testing.assert_close(plain, no_condition, atol=0, rtol=0)
        active, _ = model.velocity(noisy, None, sample['text'], sigma, c, g,
                                   sample['times'])
        assert (active - plain).abs().max() > 1e-6

    # Consistent permutation of word keys/values must not change PWCA.
    physical = model.physical_blocks[0]
    state = torch.randn(1, 30, 128)
    qwen, native_words = c['word_memory']
    permutation = torch.randperm(len(qwen))
    with torch.no_grad():
        original = physical._word_attention(state, bg, (qwen, native_words), True)
        permuted = physical._word_attention(
            state, bg, (qwen[permutation], native_words[permutation]), True)
        torch.testing.assert_close(original, permuted, atol=1e-5, rtol=1e-4)

    # C read still scales linearly with the execution input at a fixed state.
    with torch.no_grad():
        reader = physical.conditioning
        original = reader(state, c['routing'], c['execution'])
        weak = reader(state, c['routing'], c['execution'] * .01)
        torch.testing.assert_close(weak, original * .01, rtol=1e-4, atol=1e-7)

    # Asymmetric T/H/W RoPE coordinate test.
    probe = torch.randn(1, 30, 4, 32)
    rotated = native.rope_apply(probe, bg, base.freqs)
    expected = torch.empty_like(probe)
    split = [6, 5, 5]
    table = base.freqs.split(split, dim=1)
    for index in range(30):
        t = index // 10; h = (index // 5) % 2; windex = index % 5
        phase = torch.cat((table[0][t], table[1][h], table[2][windex]))
        value = torch.view_as_complex(probe[0, index].double().reshape(4, 16, 2))
        expected[0, index] = torch.view_as_real(value * phase).flatten(-2).float()
    torch.testing.assert_close(rotated, expected, atol=0, rtol=0)

    # Real serialization/restore path, including frozen word buffers.
    with tempfile.TemporaryDirectory() as folder:
        path = save_checkpoint(model, opt, cfg, folder, 400, 1600)
        saved = torch.load(path, map_location='cpu', weights_only=False)
        assert saved['physical_source_layers'] == list(model.injection_layers)
        assert any(name.endswith('.word_key') for name in saved['constants'])
        assert any(name.endswith('.word_value') for name in saved['constants'])
        restored = PhysicalModel(cfg, device='cpu', wan=copy.deepcopy(base),
                                 interface=interface)
        restored_opt = torch.optim.AdamW(restored.optim_groups(), lr=1e-3)
        restore_checkpoint(restored, restored_opt, path, cfg)
        with torch.no_grad():
            restored_c, restored_g = restored.prepare_content(evidence, sample['text'])
            restored_y, _ = restored.velocity(
                noisy, None, sample['text'], sigma, restored_c, restored_g,
                sample['times'])
            torch.testing.assert_close(restored_y, active, atol=0, rtol=0)

    print('PASS actual small Wan: native start; exact 0/7/14/21/28 copies; '
          'PWCA zero-init, frozen K/V, gradients, serialization, empty condition, '
          'C linearity and asymmetric RoPE. Quality is unproven.')


if __name__ == '__main__':
    main()
