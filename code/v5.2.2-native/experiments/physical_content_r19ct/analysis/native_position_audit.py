"""CPU algebraic audit against real Wan blocks, including native RoPE/time.

Only the CUDA attention kernel is replaced by identical float32 CPU SDPA on
both sides. Native modules, flattening, RoPE, time modulation and unpatchify
remain real implementations. This does not test trained full-model quality.
"""
import importlib
import json
from pathlib import Path
import sys
import types
import warnings

import torch
from torch import nn
from torch.nn import functional as F


BASE = Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native')
CT = BASE/'experiments/physical_content_r19ct'
SNAPSHOT = CT/'snapshots/20261114T060200Z'
sys.path.insert(0, str(SNAPSHOT))
import bootstrap  # noqa: E402,F401
from physical_content.wan_loader import ensure_wan_namespace  # noqa: E402


def load_package(label, directory):
    package = types.ModuleType(label)
    package.__path__ = [str(directory/'physical_content')]
    package.__package__ = label
    sys.modules[label] = package
    return importlib.import_module(label+'.model').PhysicalModel


def cpu_attention(q, k, v, **kwargs):
    return F.scaled_dot_product_attention(
        q.transpose(1, 2).float(), k.transpose(1, 2).float(),
        v.transpose(1, 2).float(), scale=kwargs.get('softmax_scale'),
        is_causal=kwargs.get('causal', False)).transpose(1, 2)


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    torch.manual_seed(23)
    warnings.filterwarnings('ignore', message='User provided device_type of.*')
    ensure_wan_namespace({'paths': {'wan_code': '/home/liuzhirui/model/Wan2.2'}})
    native = importlib.import_module('wan.modules.model')
    native.flash_attention = cpu_attention
    models = {
        'R18E': BASE/'experiments/physics_content_v10_impl',
        'R19CT': SNAPSHOT,
        'R19EI': BASE/'experiments/physical_content_r19ei/snapshots/20261114T065127Z',
        'R19DT': BASE/'experiments/physical_content_r19dt/snapshots/20261114T121915Z',
    }
    classes = {label: load_package('audit_'+label, path) for label, path in models.items()}
    wan = native.WanModel(model_type='t2v', dim=128, ffn_dim=256, num_heads=4,
                          num_layers=3, text_len=12, text_dim=64, freq_dim=16,
                          in_dim=4, out_dim=4, patch_size=(1, 2, 2)).eval()
    # Default zero head would make any equivalence check vacuous.
    nn.init.normal_(wan.head.head.weight, std=.04)
    nn.init.normal_(wan.head.head.bias, std=.01)
    rows = []
    for label, cls in classes.items():
        model = cls.__new__(cls)
        nn.Module.__init__(model)
        model.wan = wan
        model.cfg = {'model': {'injection_layers': [0, 1, 2]}}
        model.checkpointing = False
        model.mechanism = 1
        for task in ['t2v', 'i2v']:
            model.task = task
            for shape in [(5, 6, 8), (21, 4, 6)]:
                noisy = torch.randn(1, 4, *shape)
                first = torch.randn_like(noisy[:, :, :1]) if task == 'i2v' else None
                text = torch.randn(1, 7, 64)
                context, mask = model._context(text)
                content = dict(context=context, text_mask=mask,
                               routing=torch.zeros(1, 8, 128), execution=torch.zeros(1, 8, 128))
                x = torch.cat((first, noisy[:, :, 1:]), 2) if first is not None else noisy
                spatial = (shape[1]//2)*(shape[2]//2)
                length = shape[0]*spatial
                for sigma in [.01, .5, .98]:
                    t = torch.full((1, length), sigma*1000.)
                    if first is not None:
                        t[:, :spatial] = 0
                    expected = wan([x[0]], t, [text[0]], length)[0][None]
                    observed = model.velocity(noisy, first, text, torch.tensor(sigma), content,
                                              groups=torch.zeros(8, dtype=torch.long),
                                              times=torch.arange(shape[0])[None]/6.,
                                              physical_mode='off')[0]
                    error = observed-expected
                    row = dict(version=label, task=task, grid=list(shape), sigma=sigma,
                               max_abs=float(error.abs().max()),
                               relative_l2=float(error.norm()/expected.norm()),
                               reference_rms=float(expected.square().mean().sqrt()))
                    assert row['reference_rms'] > .01, row
                    torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-6)
                    rows.append(row)
    experts = importlib.import_module('audit_R19CT.grounded_experts')
    rotations = []
    for width in [32, 128]:
        value = torch.randn(2, 21, 3, width)
        freqs = torch.cat([native.rope_params(1024, width-4*(width//6)),
                           native.rope_params(1024, 2*(width//6)),
                           native.rope_params(1024, 2*(width//6))], dim=1)
        expected = native.rope_apply(value, torch.tensor([[21, 1, 1]]*2), freqs)
        observed = experts.temporal_rope(value.transpose(1, 2), torch.arange(21)[None].expand(2, -1)).transpose(1, 2)
        torch.testing.assert_close(observed, expected, rtol=1e-5, atol=3e-6)
        rotations.append(dict(head_width=width, max_abs=float((observed-expected).abs().max())))
    report = dict(scope=__doc__, cpu_only=True, full_gpu_equivalence_proven=False,
                  attention_kernel='same float32 CPU SDPA for both paths; no mask/padded video tokens in test',
                  native_forward_cases=rows, temporal_rope_same_integer_coordinates=rotations,
                  temporal_units='Native video RoPE uses latent-frame index; added branch uses elapsed seconds. Different units are intentional, not equivalent pretrained coordinates.',
                  passed=True)
    output = CT/'reports/native_position_audit.json'
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(passed=True, native_cases=len(rows), max_abs=max(r['max_abs'] for r in rows),
                          temporal_rotation=rotations, output=str(output))))


if __name__ == '__main__':
    main()
