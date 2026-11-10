"""Read-only CPU comparison of H/B C on the already declared 16 inputs.

No Wan generation, GPU allocation, API request, training or cache mutation.
Statistics describe representations, not generation quality.
"""
import json
import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import bootstrap
sys.path.insert(0, str(ROOT/'data_tools'))
from physical_content.model import Enhancement
from condition_controls import InferenceCache

legacy_spec = importlib.util.spec_from_file_location('r15b_condition_controls',
    ROOT.parent/'request_bound_fusion_r15b/data_tools/condition_controls.py')
legacy_module = importlib.util.module_from_spec(legacy_spec)
legacy_spec.loader.exec_module(legacy_module)


def stats(value):
    x = value.detach().float().reshape(-1, value.shape[-1])
    normed = F.normalize(x, dim=-1)
    n = len(x)
    return dict(rms=float(x.square().mean().sqrt()),
                maximum_abs=float(x.abs().max()),
                mean_token_norm=float(x.norm(dim=-1).mean()),
                token_cosine=float((normed.sum(0).square().sum()-normed.square().sum()) /
                                   max(1, n*(n-1))),
                centered_energy=float((x-x.mean(0)).square().sum() /
                                      x.square().sum().clamp_min(1e-20)))


def compare(a, b):
    a, b = a.float(), b.float()
    return dict(rms_difference=float((a-b).square().mean().sqrt()),
                relative_difference=float((a-b).square().mean().sqrt() /
                                          b.square().mean().sqrt().clamp_min(1e-20)),
                flat_cosine=float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0)))


def input_for(cache, case, mode):
    folder = cache.root/f'videophy40_{case}'
    ready = json.loads((folder/'ready.json').read_text())
    sample = dict(id=ready['id'], caption=ready['caption'])
    return cache.input(sample, image=ready['image'], duration=ready['duration'],
                       drop=mode == 'negative', mode='real' if mode == 'negative' else mode)


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    plan = json.loads((ROOT/'reports/jobs/68243/plan.json').read_text())
    checkpoints = {
        'H': ROOT/'runs/physical_content_68243/step0400.pt',
        'B': ROOT.parent/'request_bound_fusion_r15b/runs/physical_content_67829/step0400.pt',
    }
    modes = ('real', 'reasoning_swapped', 'empty_reasoning', 'negative')
    cache_classes = {'H': InferenceCache, 'B': legacy_module.InferenceCache}
    cache_configs = {name: json.loads((path.parent/'config.json').read_text())
                     for name, path in checkpoints.items()}
    caches = {name: cache_classes[name](cfg['paths']['inference_evidence'], device='cpu')
              for name, cfg in cache_configs.items()}
    mismatches = []
    for case in plan['control_cases']:
        for mode in modes:
            inputs = {name: input_for(cache, case, mode) for name, cache in caches.items()}
            for key in ('words', 'word_groups', 'request_words'):
                a, b = inputs['H'][key], inputs['B'][key]
                if not torch.equal(a, b):
                    mismatches.append(dict(case=case, mode=mode, field=key,
                                           H_shape=list(a.shape), B_shape=list(b.shape)))
    if mismatches:
        out = ROOT/'review/condition_content_input_mismatches.json'
        out.write_text(json.dumps(mismatches, indent=2)+'\n')
        raise ValueError(f'Actual input tensors differ; inspect {out} before comparing models')
    report = dict(utc=datetime.now(timezone.utc).isoformat(), cases=plan['control_cases'],
                  checkpoints={k: str(v) for k, v in checkpoints.items()},
                  computation='CPU FP32 Enhancement only; not exact GPU rollout activations',
                  quality_proof=False, models={}, source_metadata={}, completed=False,
                  input_formats=dict(H='full_input_memory_v2', B='compact_first_image_request_v1'),
                  actual_H_B_prepared_feature_tensors_match=True)
    out = ROOT/'review/condition_content_stats.json'
    for name, checkpoint in checkpoints.items():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
        if saved['step'] != 400:
            raise ValueError('Compare complete400 checkpoints only')
        cfg = saved['config']
        model = Enhancement(cfg).eval().requires_grad_(False)
        values = {k.removeprefix('enhancement.'): v
                  for k, v in {**saved['model'], **saved['constants']}.items()
                  if k.startswith('enhancement.')}
        model.load_state_dict(values, strict=True)
        cache = caches[name]
        rows, means = [], {mode: {key: [] for key in ('routing', 'execution')} for mode in modes}
        for case in plan['control_cases']:
            contents = {}
            for mode in modes:
                evidence = input_for(cache, case, mode)
                if name == 'H' and case in ('VP027', 'VP034', 'VP038'):
                    report['source_metadata'][f'{case}/{mode}'] = {
                        k: evidence[k] for k in ('caption', 'relations', 'intervention')
                        if k in evidence and not torch.is_tensor(evidence[k])}
                content, groups = model(evidence)
                if any(not torch.isfinite(value).all() for value in content.values()):
                    raise ValueError(f'Nonfinite C: {name}/{case}/{mode}')
                anchor, _ = model(evidence, anchor_only=True)
                contents[mode] = {k: v.detach().clone() for k, v in content.items()}
                row = dict(case=case, mode=mode, word_count=len(evidence['words']),
                           relation_count=int(groups.max())+1,
                           fitted_input_words=stats((evidence['words']-model.anchor_mean)/model.anchor_scale),
                           native_anchor=stats(anchor['execution']), content={})
                for key, value in content.items():
                    row['content'][key] = dict(stats(value),
                        learned_residual_vs_anchor=compare(value, anchor[key]))
                    means[mode][key].append(value.mean(dim=1)[0].detach().clone())
                row['route_execution_alignment'] = float(F.cosine_similarity(
                    content['routing'].float(), content['execution'].float(), dim=-1).mean())
                rows.append(row)
            for row in rows[-len(modes):]:
                row['difference_from_own_real'] = {
                    key: compare(contents[row['mode']][key], contents['real'][key])
                    for key in ('routing', 'execution')}
                row['difference_from_negative_CFG_arm'] = {
                    key: compare(contents[row['mode']][key], contents['negative'][key])
                    for key in ('routing', 'execution')}
            print(json.dumps(dict(model=name, completed_case=case)), flush=True)
        aggregate = {}
        for mode in modes:
            selected = [r for r in rows if r['mode'] == mode]
            aggregate[mode] = {}
            for key in ('routing', 'execution'):
                aggregate[mode][key] = dict(
                    mean_rms=mean(r['content'][key]['rms'] for r in selected),
                    mean_within_sample_token_cosine=mean(r['content'][key]['token_cosine'] for r in selected),
                    across_sample_mean_cosine=stats(torch.stack(means[mode][key]))['token_cosine'],
                    mean_anchor_relative_correction=mean(r['content'][key]['learned_residual_vs_anchor']['relative_difference'] for r in selected))
        report['models'][name] = dict(rows=rows, aggregate=aggregate)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        del model, saved, values, rows, means, contents
    report['completed'] = True
    report['limitations'] = [
        'Different C magnitudes and cosine values are not measures of PC, SA or actual useful information.',
        'C differences between CFG arms are representation differences, not measured guided velocity differences.',
        'This does not run the video-state-dependent physical attention or output writer.',
        'Source reasoning is original/prepared content; the real label denotes the paired input, not a guarantee that every statement is correct.',
    ]
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(report=str(out), matched_cases=len(plan['control_cases']), modes=list(modes))), flush=True)


if __name__ == '__main__':
    main()
