"""Functional checks of the unsubmitted R20 components, not a score claim."""
import importlib
import json
from pathlib import Path
import sys
import types

import torch
from torch import nn

BASE = Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native')
ROOT = BASE/'experiments/physical_content_r20'
sys.path[:0] = [str(ROOT.parent), str(BASE/'experiments/physics_content_v10_impl')]
import bootstrap  # noqa: E402,F401
from physical_content.wan_loader import ensure_wan_namespace  # noqa: E402
from physical_content_r20.prototype.native_residual import (  # noqa: E402
    NativeAnchoredContent, DiffusionModulation, IndependentPhysicalResidual,
)


def main():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    ensure_wan_namespace({'paths': {'wan_code': '/home/liuzhirui/model/Wan2.2'}})
    from wan.modules.model import WanCrossAttention
    width = 24
    content = NativeAnchoredContent(32, width, 4, 2, 8, 2)
    words, request, native_words = torch.randn(6, 32), torch.randn(5, 32), torch.randn(6, width)
    groups = torch.tensor([0, 0, 0, 1, 1, 1])
    sources = content(words, request, native_words, groups)
    assert torch.equal(sources['route'], sources['native'])
    assert torch.equal(sources['execute'], sources['native'])
    assert content.routing_fidelity(sources) == 0
    native_attention = WanCrossAttention(width, 4).requires_grad_(False)
    native_before = {k: v.detach().clone() for k, v in native_attention.state_dict().items()}
    physical = IndependentPhysicalResidual(native_attention)
    modulation = DiffusionModulation(width)
    assert physical.direct.query_base.weight.requires_grad
    assert not physical.semantic_query.weight.requires_grad
    assert physical.direct.query_base.weight.data_ptr() != native_attention.q.weight.data_ptr()
    frames, spatial = 3, 4
    hidden = torch.randn(1, frames*spatial, width)
    normed = nn.functional.layer_norm(hidden, (width,))
    noise_embedding = torch.randn(1, frames, width)
    caption = torch.randn(1, 7, width)
    text_keys = physical.direct.split(native_attention.norm_k(native_attention.k(caption)))
    text_values = physical.direct.split(native_attention.v(caption))
    text_mask = torch.ones(1, 7, dtype=torch.bool)
    times = torch.tensor([[0., .2, .4]])

    def forward(known=0, **controls):
        source = content(words, request, native_words, groups)
        return physical(hidden, normed, source['route'], source['execute'], source['groups'],
                        text_keys, text_values, text_mask, times, modulation(noise_embedding),
                        spatial, known_tokens=known, semantic_hidden=normed, **controls), source

    for known in (0, spatial):
        initial, _ = forward(known)
        assert torch.equal(initial, torch.zeros_like(initial))
    parameters = [p for module in (content, physical, modulation) for p in module.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-3)
    target = torch.randn_like(hidden)*.2
    gradient_paths = {}
    for _ in range(4):
        optimizer.zero_grad(set_to_none=True)
        delta, source = forward()
        loss = (hidden+delta-target).square().mean()+.03*content.routing_fidelity(source)
        loss.backward()
        for prefix, module in [('content', content), ('physics', physical), ('noise', modulation)]:
            for name, parameter in module.named_parameters():
                if parameter.grad is not None:
                    gradient_paths[prefix+'.'+name] = max(gradient_paths.get(prefix+'.'+name, 0), float(parameter.grad.norm()))
        optimizer.step()
    for key in ['content.seed', 'content.block.cross.q.weight', 'content.execute_update.weight',
                'physics.direct.query_base.weight', 'physics.direct.key.weight', 'physics.direct.value.weight',
                'physics.state.weight', 'physics.direct_output.weight', 'physics.associated_output.weight',
                'noise.project.1.weight']:
        assert gradient_paths.get(key, 0) > 0, key
    full, _ = forward()
    direct, _ = forward(associated=False)
    associated, _ = forward(direct=False)
    torch.testing.assert_close(full, direct+associated, atol=1e-6, rtol=1e-5)
    off, _ = forward(direct=False, associated=False)
    assert torch.equal(off, torch.zeros_like(off))
    image_delta, _ = forward(spatial)
    assert torch.equal(image_delta[:, :spatial], torch.zeros_like(image_delta[:, :spatial]))
    assert float(image_delta[:, spatial:].abs().max()) > 0
    source = content(words, request, native_words, groups)
    zero_execute = physical(hidden, normed, source['route'], torch.zeros_like(source['execute']),
                            source['groups'], text_keys, text_values, text_mask, times,
                            modulation(noise_embedding), spatial, semantic_hidden=normed)
    assert torch.equal(zero_execute, torch.zeros_like(zero_execute))
    for name, value in native_attention.state_dict().items():
        assert torch.equal(value, native_before[name]), name
    assert all(p.grad is None for p in native_attention.parameters())
    report = dict(status='unsubmitted_prototype_functional_checks_passed',
                  scope='Small CPU components, real Wan attention projections; no trained generation quality claimed',
                  facts=['exact source content at initialization', 'independent physical Q and frozen native query',
                         'zero residual at initialization', 'trained gradients on all functional paths',
                         'direct+associated additive decomposition', 'known first frame excluded',
                         'zero execution content gives zero residual', 'original attention unchanged'],
                  gradient_paths=gradient_paths, training_submitted=False)
    output = ROOT/'prototype/functional_checks.json'
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(passed=True, output=str(output), checked_gradient_paths=len(gradient_paths))))


if __name__ == '__main__':
    main()
