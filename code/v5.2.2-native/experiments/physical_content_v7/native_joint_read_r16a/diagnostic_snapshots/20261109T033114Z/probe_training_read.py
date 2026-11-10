"""Frozen B/I, matched real training states: bank use and FM gradient credit.

No optimizer, generation, decoding, API, or checkpoint modification. The three
examples/noise levels were selected by training cursor, before inspecting this probe.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import math
import os
import sys
import time

import torch
from torch.nn import functional as F

ROOT = Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_v7')
OUT = ROOT / 'native_joint_read_r16a/reports/diagnostics'
SOURCES = {
    'B': ROOT / 'request_bound_fusion_r15b/snapshots/20261105T152223Z',
    'I': ROOT / 'joint_memory_fusion_r15i/snapshots/20261108T123450Z',
}
CHECKPOINTS = {
    'B': ROOT / 'request_bound_fusion_r15b/runs/physical_content_67829/step0400.pt',
    'I': ROOT / 'joint_memory_fusion_r15i/runs/physical_content_68312/step0400.pt',
}
CURSORS = [512, 1024, 1536]
SITES = ['2', '14', '29']


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['B', 'I'], required=True)
    args = parser.parse_args()
    source = SOURCES[args.model]
    folder = OUT / os.environ.get('DET_EXPERIMENT_ID', 'local') / args.model
    sys.path.insert(0, str(source))
    import bootstrap
    from physical_content.common import gpu_guard
    from physical_content.model import PhysicalModel
    from physical_content.teacher import Teacher
    from physical_content.supervision import generation_loss
    from physical_scaling.data import Data
    from physical_scaling.model import time_features

    gpu_guard(folder)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    save(folder / 'status.json', dict(stage='loading_frozen_model', model=args.model))
    saved = torch.load(CHECKPOINTS[args.model], map_location='cpu', mmap=True, weights_only=False)
    cfg = saved['config']
    model = PhysicalModel(cfg).eval().requires_grad_(False)
    parameters, buffers = dict(model.named_parameters()), dict(model.named_buffers())
    assert set(saved['model']) == set(model.learned_names)
    with torch.no_grad():
        for name, value in saved['model'].items(): parameters[name].copy_(value)
        for name, value in saved['constants'].items(): buffers[name].copy_(value)
    reference = torch.load(CHECKPOINTS['B'], map_location='cpu', mmap=True, weights_only=False)
    constant_equal = set(saved['constants']) == set(reference['constants']) and all(
        torch.equal(value, reference['constants'][name]) for name, value in saved['constants'].items())
    del saved, reference
    teacher, data = Teacher(cfg), Data(cfg)
    indices = {row['id']: index for index, row in enumerate(data.rows)}
    logs = [json.loads(line) for line in (CHECKPOINTS['I'].parent / 'train.jsonl').open()]
    micros = [row for row in logs if row['kind'] == 'micro']
    selected = [dict(cursor=cursor, **{key: micros[cursor][key] for key in ['sample', 'sigma', 'text_dropped', 'step']}) for cursor in CURSORS]
    active, rows, gradients = {}, {}, {}

    def before(site):
        def hook(fusion, inputs, kwargs):
            active['site'] = site
            if not active.get('record') or site not in SITES: return
            x, times, geometry, attention = inputs[0], inputs[3], inputs[4], inputs[5]
            known = geometry[1] * geometry[2]
            ids = torch.linspace(0, x.shape[1] - known - 1, 128, device=x.device).long()
            with torch.no_grad():
                temporal = fusion.time(time_features(times))[:, 1:].repeat_interleave(known, 1)
                native = attention.norm_q(attention.q(x[:, known + ids]) + temporal[:, ids])
                active['native_q'] = native.reshape(1, -1, attention.num_heads, attention.head_dim).transpose(1, 2)
                active['indices'] = ids
        return hook

    def output_hook(site, native):
        def hook(module, inputs, output):
            if not active.get('record') or site not in SITES: return
            with torch.no_grad():
                sample = output[:, active['indices'] + (output.shape[1] - active['future_tokens'] if native else 0)].float()
                key = (active['cursor'], site)
                if native:
                    active['text_output'] = sample
                else:
                    text = active['text_output']
                    rows[key]['write_rms'] = float(sample.square().mean().sqrt())
                    rows[key]['write_to_native_rms'] = float(sample.norm() / text.norm().clamp_min(1e-12))
                    rows[key]['write_native_cosine'] = float(F.cosine_similarity(sample, text, dim=-1).mean())
        return hook

    def observe_grad(site, name):
        key = (active['cursor'], site, name)
        def hook(grad):
            gradients.setdefault(key, []).append(grad.detach().float().cpu())
        return hook

    def instrumented(q, k, v, request_keys, reference, prior, mode='joint', *, relation_count=None):
        if mode != 'joint': raise ValueError('This diagnostic measures the joint reader only')
        site = active['site']
        kd, kb, vd, vb = k.clone(), k.clone(), v.clone(), v.clone()
        if site in SITES:
            for name, tensor in [('key_direct', kd), ('key_bound', kb), ('value_direct', vd), ('value_bound', vb)]:
                tensor.register_hook(observe_grad(site, name))
        scale = math.sqrt(q.shape[-1])
        binding = ((request_keys.float() @ kb.float().transpose(-1, -2)) / scale + prior[:-1]).softmax(-1)
        bound_values = binding.to(vb) @ vb
        log_relations = math.log(relation_count) if relation_count is not None else float(prior[:-1].float().logsumexp(-1))
        bound_prior = prior.new_full((request_keys.shape[-2],), log_relations - math.log(request_keys.shape[-2]))
        keys = torch.cat((kd, request_keys, reference), -2)
        values = torch.cat((vd, bound_values, torch.zeros_like(reference)), -2)
        physical_prior = torch.cat((prior[:-1], bound_prior), -1).float() - math.log(2.)
        logits = q.float() @ keys.float().transpose(-1, -2) / scale
        weights = (logits + torch.cat((physical_prior, prior[-1:]), -1)).softmax(-1)
        if active.get('record') and site in SITES:
            with torch.no_grad():
                ids = active['indices']; w = weights[:, :, ids]; count = k.shape[-2]
                direct, bound = w[..., :count], w[..., count:-1]
                zd = direct @ vd.float(); zb = bound @ bound_values.float()
                query = q[:, :, ids].float()
                b = binding.float(); mean = bound_values.float().mean(-2, keepdim=True)
                rows[(active['cursor'], site)] = dict(model=args.model, cursor=active['cursor'], sample=active['sample'], sigma=active['sigma'], layer=site,
                    direct_mass=float(direct.sum(-1).mean()), bound_mass=float(bound.sum(-1).mean()), null_mass=float(w[..., -1].mean()),
                    direct_contribution_rms=float(zd.square().mean().sqrt()), bound_contribution_rms=float(zb.square().mean().sqrt()),
                    direct_bound_cosine=float(F.cosine_similarity(zd, zb, dim=-1).mean()),
                    query_native_cosine=float(F.cosine_similarity(query, active['native_q'].float(), dim=-1).mean()),
                    binding_normalized_entropy=float(-(b * b.clamp_min(1e-20).log()).sum(-1).mean() / math.log(count)),
                    bound_common_energy_fraction=float(mean.square().mean() / bound_values.float().square().mean().clamp_min(1e-20)))
        result = weights.to(values) @ values
        if active.get('record') and site in SITES and not active.get('operator_checked'):
            with torch.no_grad():
                ids = active['indices']
                expected = original_reader(q[:, :, ids], k, v, request_keys, reference, prior, mode,
                                           relation_count=relation_count)
                actual = result[:, :, ids]
                torch.testing.assert_close(actual, expected, rtol=.01, atol=.005)
                active['operator_max_abs_difference'] = float((actual - expected).abs().max())
                active['operator_checked'] = True
        return result

    if args.model == 'B':
        path = ROOT / 'b400_joint_read_eval/snapshots/20261108T130636Z/joint_read.py'
        spec = importlib.util.spec_from_file_location('_b_joint_operator', path)
        joint = importlib.util.module_from_spec(spec); spec.loader.exec_module(joint)
        original_reader = joint.read_physical
        joint.read_physical = instrumented
        import types
        for fusion in model.fusions.values(): fusion.forward = types.MethodType(joint.fusion_forward, fusion)
    else:
        import physical_content.joint_read as joint
        original_reader = joint.read_physical
        joint.read_physical = instrumented
    handles = []
    for site, fusion in model.fusions.items():
        handles.append(fusion.register_forward_pre_hook(before(site), with_kwargs=True))
        if site in SITES:
            handles.append(model.wan.blocks[int(site)].cross_attn.o.register_forward_hook(output_hook(site, True)))
            handles.append(fusion.output.register_forward_hook(output_hook(site, False)))
    completed = []
    for choice in selected:
        start = time.monotonic()
        sample = data.get(indices[choice['sample']]); evidence = teacher.input(sample, drop=choice['text_dropped'])
        assert not choice['text_dropped'], 'The preselected cases use their normal text'
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
            content, groups = model.enhancement(evidence)
        content = {key: value.detach().requires_grad_(True) for key, value in content.items()}
        active.update(record=True, cursor=choice['cursor'], sample=choice['sample'], sigma=choice['sigma'], future_tokens=7800)
        sigma = torch.tensor(choice['sigma'], device='cuda')
        noise = torch.randn(sample['latent'].shape, device='cuda', generator=torch.Generator(device='cuda').manual_seed(42 * 2000003 + choice['cursor']))
        with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
            pred, _ = model.velocity((1 - sigma) * sample['latent'] + sigma * noise, sample['first'], sample['text'], sigma, content, sample['times'], groups)
            objective, _ = generation_loss(pred, noise, sample['latent'], sample['valid'], **cfg['loss']['generation'])
        active['record'] = False
        objective.backward()
        for site in SITES:
            row = rows[(choice['cursor'], site)]
            for kind in ['key', 'value']:
                dg = gradients[(choice['cursor'], site, kind + '_direct')]
                bg = gradients[(choice['cursor'], site, kind + '_bound')]
                a, b = dg[-1].flatten(), bg[-1].flatten()
                row[kind + '_gradient'] = dict(direct_norm=float(a.norm()), bound_norm=float(b.norm()),
                    direct_to_bound_norm=float(a.norm() / b.norm().clamp_min(1e-20)), cosine=float(F.cosine_similarity(a, b, dim=0)),
                    hook_counts=[len(dg), len(bg)])
                for name in [kind + '_direct', kind + '_bound']: del gradients[(choice['cursor'], site, name)]
        completed.append(choice['sample'])
        report = dict(model=args.model, checkpoint=str(CHECKPOINTS[args.model]), selected=selected,
                      frozen_interface_constants_equal_B=constant_equal, rows=list(rows.values()),
                      instrumented_operator_max_abs_difference=active.get('operator_max_abs_difference'),
                      training_updates=0, generated_videos=0,
                      limits='Three fixed training samples/noise levels, not a quality score or inference trajectory. Gradients diagnose two read routes at frozen weights; no causal training claim by themselves.')
        save(folder / 'result.json', report)
        save(folder / 'status.json', dict(stage='sample_complete', completed=completed))
        print(json.dumps(dict(model=args.model, sample=choice['sample'], seconds=time.monotonic() - start)), flush=True)
        del sample, evidence, content, groups, noise, pred, objective
    for handle in handles: handle.remove()
    save(folder / 'status.json', dict(stage='completed', completed=completed, training_updates=0))


if __name__ == '__main__': main()
