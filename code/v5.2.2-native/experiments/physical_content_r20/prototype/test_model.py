"""Real-Wan CPU integration checks for the unsubmitted R20 prototype."""
import importlib
import json
from pathlib import Path
import sys
import warnings

import torch
from torch import nn
from torch.nn import functional as F

BASE = Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native')
ROOT = BASE/'experiments/physical_content_r20'
sys.path[:0] = [str(ROOT.parent), str(BASE/'experiments/physics_content_v10_impl')]
import bootstrap  # noqa: E402,F401
from physical_content.wan_loader import ensure_wan_namespace  # noqa: E402
from physical_content_r20.prototype.model import PrototypeVelocityModel  # noqa: E402


def cpu_attention(q, k, v, **kwargs):
    return F.scaled_dot_product_attention(
        q.transpose(1, 2).float(), k.transpose(1, 2).float(), v.transpose(1, 2).float(),
        scale=kwargs.get('softmax_scale'), is_causal=kwargs.get('causal', False)).transpose(1, 2)


def main():
    torch.set_num_threads(2)
    torch.manual_seed(28)
    warnings.filterwarnings('ignore', message='User provided device_type of.*')
    ensure_wan_namespace({'paths': {'wan_code': '/home/liuzhirui/model/Wan2.2'}})
    native = importlib.import_module('wan.modules.model')
    native.flash_attention = cpu_attention
    wan = native.WanModel(model_type='t2v', dim=128, ffn_dim=256, num_heads=4,
                          num_layers=3, text_len=12, text_dim=64, freq_dim=16,
                          in_dim=4, out_dim=4, patch_size=(1, 2, 2)).eval()
    nn.init.normal_(wan.head.head.weight, std=.04)
    nn.init.normal_(wan.head.head.bias, std=.01)
    before = {n: p.detach().clone() for n, p in wan.named_parameters()}
    model = PrototypeVelocityModel(wan, task='t2v', interface=dict(mean=torch.zeros(64), scale=torch.ones(64)),
                                   query_width=64, q_heads=4, grounded_heads=2, slots=8,
                                   injection_layers=(0, 2), checkpointing=True)
    evidence = dict(words=torch.randn(6, 64), request_words=torch.randn(5, 64),
                    native_condition=torch.randn(6, 128)*.1,
                    word_groups=torch.tensor([0, 0, 0, 1, 1, 1]))
    text = torch.randn(1, 7, 64)
    checks = []
    for task in ('t2v', 'i2v'):
        model.task = task
        for shape in [(3, 4, 6), (5, 4, 4)]:
            x = torch.randn(1, 4, *shape)
            first = torch.randn_like(x[:, :, :1]) if task == 'i2v' else None
            times = torch.arange(shape[0])[None]/6.
            spatial = (shape[1]//2)*(shape[2]//2)
            length = shape[0]*spatial
            for sigma in (.05, .95):
                t = torch.full((1, length), sigma*1000)
                if first is not None:
                    t[:, :spatial] = 0
                latent = x if first is None else torch.cat((first, x[:, :, 1:]), 2)
                with torch.no_grad():
                    expected = wan([latent[0]], t, [text[0]], length)[0][None]
                    content, groups = model.prepare_content(evidence, text)
                    actual, _ = model.velocity(x, first, text, torch.tensor(sigma), content, groups, times)
                error = float((expected-actual).abs().max())
                assert float(expected.abs().max()) > .01
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
                checks.append(dict(task=task, shape=shape, sigma=sigma, max_abs=error))
    model.task = 't2v'
    x = torch.randn(1, 4, 3, 4, 6)
    times = torch.arange(3)[None]/6.
    sigma = torch.tensor(.7)
    target = torch.randn_like(x)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    gradient_paths = {}
    for _ in range(6):
        optimizer.zero_grad(set_to_none=True)
        content, groups = model.prepare_content(evidence, text)
        prediction, _ = model.velocity(x, None, text, sigma, content, groups, times)
        loss = (prediction-target).square().mean()+.03*model.enhancement.routing_fidelity(content['raw'])
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                gradient_paths[name] = max(gradient_paths.get(name, 0), float(p.grad.norm()))
        optimizer.step()
    required = ['enhancement.seed', 'enhancement.block.cross.q.weight',
                'enhancement.route_update.weight', 'enhancement.execute_update.weight',
                'noise_modulation.project.1.weight', 'physics.0.direct.query_base.weight',
                'physics.0.direct_output.weight', 'physics.2.associated_output.weight']
    for name in required:
        assert gradient_paths.get(name, 0) > 0, name
    for name, value in wan.named_parameters():
        assert not value.requires_grad and value.grad is None, name
        assert torch.equal(value, before[name]), name
    with torch.no_grad():
        content, groups = model.prepare_content(evidence, text)
        actual, _ = model.velocity(x, None, text, sigma, content, groups, times)
        off, _ = model.velocity(x, None, text, sigma, content, groups, times, physical_mode='off')
        expected = wan([x[0]], torch.full((1, 18), 700.), [text[0]], 18)[0][None]
        torch.testing.assert_close(off, expected, atol=1e-6, rtol=1e-5)
        assert float((actual-off).abs().max()) > 1e-6
        modified = dict(evidence, native_condition=torch.flip(evidence['native_condition'], [0]))
        other, other_groups = model.prepare_content(modified, text)
        changed, _ = model.velocity(x, None, text, sigma, other, other_groups, times)
        assert float((actual-changed).abs().max()) > 1e-6
    report = dict(status='passed_unsubmitted_integration_prototype',
                  scope='Tiny real Wan CPU; native RoPE/time/head, same CPU SDPA on both paths; not fullGPU or quality evidence',
                  native_equality=checks, max_initial_error=max(c['max_abs'] for c in checks),
                  generation_gradient_paths={k: gradient_paths[k] for k in required},
                  unchanged_native=True, physical_effect_nonzero=True,
                  correct_condition_dependency=True, training_submitted=False)
    output = ROOT/'prototype/model_checks.json'
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(passed=True, native_cases=len(checks), max_error=report['max_initial_error'],
                          output=str(output))))


if __name__ == '__main__':
    main()
