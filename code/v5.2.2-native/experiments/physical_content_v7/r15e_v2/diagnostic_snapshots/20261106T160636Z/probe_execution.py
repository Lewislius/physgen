"""Read-only replay of two diagnosed R15E failures; no quality-score selection."""
from pathlib import Path
import argparse
import json
import math
import os
import sys
import time
import torch
from torch.nn import functional as F

ROOT = Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_v7/semantic_addressing_r15e')
SOURCE = ROOT / 'snapshots/20261106T101625Z'
CHECKPOINT = ROOT / 'runs/physical_content_67955/step0400.pt'


def rms(value):
    return float(value.float().square().mean().sqrt())


def cosine(a, b):
    return float(F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0))


def token_cosine(value):
    vectors = F.normalize(value.float().reshape(-1, value.shape[-1]), dim=-1)
    n = len(vectors)
    return float((vectors.sum(0).square().sum() - n) / (n * (n - 1)))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--describe', action='store_true')
    args = parser.parse_args()
    plan = dict(checkpoint=str(CHECKPOINT), source=str(SOURCE), cases=['VP010', 'VP034'],
        steps=[1, 15, 35, 50], layers=[2, 14, 29], seed=42, gamma=5,
        interventions=['no_c', 'semantic_routing_off', 'request_refinement_off', 'native_physical_query'],
        training_updates=0, output_videos=0, api_calls=0, model_downloads=0,
        selection='Two predeclared visual-review cases with observed stripe collapse; diagnostic selection, not a population sample.',
        scope='Replay normal full50 trajectory. Alternative velocities are computed at the SAME latent; paired16 runs separately assess their generated videos.',
        limitations='Conditional-minus-null includes both native text and Qwen conditions. Similarity to native guidance and C diversity are diagnostics, not optimization or quality targets.')
    if args.describe:
        print(json.dumps(plan)); return
    sys.path.insert(0, str(SOURCE))
    import bootstrap
    from physical_content.common import gpu_guard, atomic_json
    from physical_content.model import PhysicalModel, time_features, group_log_prior
    from condition_controls import InferenceCache
    from benchmark_inputs import reference, case_input, load_noise
    folder = ROOT / 'reports/execution_probe' / os.environ['DET_EXPERIMENT_ID']
    gpu_guard(folder)
    atomic_json(plan, folder / 'plan.json')
    torch.manual_seed(42); torch.cuda.manual_seed_all(42); torch.set_num_threads(4)
    saved = torch.load(CHECKPOINT, map_location='cpu', weights_only=False, mmap=True)
    assert saved['step'] == 400 and Path(saved['config']['source_snapshot']) == SOURCE
    cfg = saved['config']
    model = PhysicalModel(cfg).eval().requires_grad_(False)
    parameters, buffers = dict(model.named_parameters()), dict(model.named_buffers())
    assert set(saved['model']) == set(model.learned_names)
    for name, value in saved['model'].items(): parameters[name].copy_(value)
    for name, value in saved['constants'].items(): buffers[name].copy_(value)
    del saved
    active, native, physical = {}, {}, {}
    layer_rows, velocity_rows, content_rows = [], [], []

    def native_hook(site):
        def hook(module, inputs, output):
            if active.get('record'): native[site] = output
        return hook

    def output_hook(site):
        def hook(module, inputs, output):
            if active.get('record'): physical[site] = output
        return hook

    def fusion_hook(site):
        def hook(module, inputs, kwargs, output):
            if not active.get('record'): return
            x, context, c, times, geometry, attention, groups, *_ = inputs
            known = geometry[1] * geometry[2]
            delta, text = physical.pop(site), native.pop(site)[:, known:]
            indices = torch.linspace(0, delta.shape[1] - 1, 128, device=x.device).long()
            heads, dim = attention.num_heads, attention.head_dim
            split = lambda value: value.reshape(1, -1, heads, dim).transpose(1, 2)
            raw = attention.q(x[:, known:][:, indices])
            native_query = split(attention.norm_q(raw))
            time_input = module.time(time_features(times))[:, 1:].repeat_interleave(known, 1)[:, indices]
            query = split(module.query_norm(module.query(x[:, known:][:, indices]) + time_input))
            keys = split(module.key_norm(module.key(c['routing']))).to(query)
            values = split(module.value(c['execution'])).to(query)
            keys = torch.cat((keys, split(module.reference_key).to(query)), 2)
            base_prior = group_log_prior(groups).to(device=query.device, dtype=torch.float32)
            prior = base_prior
            if kwargs['text_length'] > 1:
                text_keys = split(attention.norm_k(attention.k(context[:, :kwargs['text_length']])))
                content_keys = split(attention.norm_k(attention.k(c['routing'])))
                prior = module.semantic_log_prior(native_query, text_keys, content_keys, base_prior)
            logits = query.float() @ keys.float().transpose(-1, -2) / math.sqrt(dim)
            weights = (logits + prior).softmax(-1)
            no_semantic = (logits + base_prior).softmax(-1)
            conditional = weights[..., :-1] / weights[..., :-1].sum(-1, keepdim=True).clamp_min(1e-30)
            pooled = (weights[..., :-1].to(values) @ values).transpose(1, 2).flatten(2)
            layer_rows.append(dict(case=active['case'], step=active['step'], sigma=active['sigma'],
                arm=active['arm'], layer=site, physical_rms=rms(delta), native_text_rms=rms(text),
                physical_to_native_text_rms=rms(delta) / max(rms(text), 1e-20),
                physical_native_cosine=cosine(delta[:, indices], text[:, indices]),
                physical_token_cosine=token_cosine(delta[:, indices]), read_value_token_cosine=token_cosine(pooled),
                physical_query_to_native_cosine=cosine(query, native_query),
                reference_mass=float(weights[..., -1].mean()),
                reference_mass_without_semantic_prior=float(no_semantic[..., -1].mean()),
                semantic_prior_attention_TV=float(.5 * (weights - no_semantic).abs().sum(-1).mean()),
                normalized_C_attention_entropy=float(-(conditional * conditional.clamp_min(1e-30).log()).sum(-1).mean() / math.log(c['routing'].shape[1])),
                physical_finite=bool(torch.isfinite(delta).all())))
        return hook

    handles = []
    for site in plan['layers']:
        handles.append(model.wan.blocks[site].cross_attn.o.register_forward_hook(native_hook(site)))
        handles.append(model.fusions[str(site)].output.register_forward_hook(output_hook(site)))
        handles.append(model.fusions[str(site)].register_forward_hook(fusion_hook(site), with_kwargs=True))
    reader = InferenceCache(cfg['paths']['inference_evidence'])
    cases = {case['id']: case for case in reference()['cases']}
    for case_id in plan['cases']:
        atomic_json(dict(status='execution_probe', case=case_id), folder / 'status.json')
        inputs, item = case_input(cases[case_id])
        sample = {key: value.cuda() for key, value in inputs.items()}
        sample['times'] = sample['times'][None]
        positive = reader.input(item, image=item['image'], duration=item['duration'])
        negative = reader.input(item, image=item['image'], duration=item['duration'], drop=True)
        with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
            normal_c, groups = model.enhancement(positive)
            normal_null, null_groups = model.enhancement(negative)
            old_c, _ = model.enhancement(positive, request_refinement=False)
            old_null, _ = model.enhancement(negative, request_refinement=False)
            content_rows.append(dict(case=case_id, content={
                name: {key: dict(rms=rms(value), token_cosine=token_cosine(value)) for key, value in content.items()}
                for name, content in [('positive', normal_c), ('negative', normal_null)]},
                request_refinement_relative_change={key:rms(normal_c[key] - old_c[key]) / max(rms(normal_c[key]), 1e-20) for key in normal_c}))
            x = load_noise(cases[case_id]).cuda().clone(); x[:, :, :1] = sample['first']
            grid = torch.linspace(1, 0, 51, device='cuda'); grid = 5 * grid / (1 + 4 * grid)
            for index in range(50):
                active.update(record=index + 1 in plan['steps'], case=case_id, step=index + 1, sigma=float(grid[index]))
                def predict(mode):
                    content, null = (old_c, old_null) if mode == 'request_refinement_off' else (normal_c, normal_null)
                    flags = dict(physical_enabled=mode != 'no_c', semantic_routing=mode != 'semantic_routing_off',
                                 native_physical_query=mode == 'native_physical_query')
                    active['arm'] = 'positive'
                    pos, _ = model.velocity(x, sample['first'], sample['text'], grid[index], content, sample['times'], groups, **flags)
                    active['arm'] = 'negative'
                    neg, _ = model.velocity(x, sample['first'], sample['null_text'], grid[index], null, sample['times'], null_groups, **flags)
                    return neg + 5 * (pos - neg), pos - neg
                full, difference = predict('normal')
                if index + 1 in plan['steps']:
                    active['record'] = False
                    alternatives = {mode: predict(mode) for mode in plan['interventions']}
                    base, base_difference = alternatives['no_c']
                    velocity_rows.append(dict(case=case_id, step=index + 1, sigma=float(grid[index]), latent_rms=rms(x[:, :, 1:]),
                        normal_velocity_rms=rms(full[:, :, 1:]), no_c_velocity_rms=rms(base[:, :, 1:]),
                        normal_velocity_finite=bool(torch.isfinite(full).all()),
                        full_native_conditional_difference_cosine=cosine(difference[:, :, 1:], base_difference[:, :, 1:]),
                        conditional_difference_change_to_native_rms=rms(difference[:, :, 1:] - base_difference[:, :, 1:]) / max(rms(base_difference[:, :, 1:]), 1e-20),
                        normal_clean_estimate_rms=rms(x[:, :, 1:] - grid[index] * full[:, :, 1:]),
                        no_c_clean_estimate_rms=rms(x[:, :, 1:] - grid[index] * base[:, :, 1:]),
                        intervention_effect_to_no_c_velocity_rms={mode:rms(full[:, :, 1:] - v[:, :, 1:]) / max(rms(base[:, :, 1:]), 1e-20)
                                                                for mode, (v, _) in alternatives.items()}))
                    del alternatives, base, base_difference
                x = x + (grid[index + 1] - grid[index]) * full; x[:, :, :1] = sample['first']
        atomic_json(dict(**plan, content=content_rows, layers=layer_rows, velocities=velocity_rows), folder / 'measurements.json')
        print(json.dumps(dict(kind='execution_case_complete', case=case_id, time=time.time())), flush=True)
        del x, sample, positive, negative, normal_c, normal_null, old_c, old_null
    for handle in handles: handle.remove()
    atomic_json(dict(status='completed', cases=plan['cases'], training_updates=0, output_videos=0), folder / 'status.json')


if __name__ == '__main__':
    main()
