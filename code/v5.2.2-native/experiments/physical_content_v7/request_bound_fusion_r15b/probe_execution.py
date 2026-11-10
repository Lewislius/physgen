"""Replay four cases without video decoding to measure actual physical writes over denoising."""
from pathlib import Path
import argparse
import json
import os
import sys
import time
import torch
from torch.nn import functional as F

ROOT = Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_v7/request_bound_fusion_r15b')
SOURCE = ROOT / 'snapshots/20261105T152223Z'
CASES = ['VP004', 'VP025', 'VP027', 'VP038']
STEPS = [1, 15, 35, 50]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def field_stats(value, temporal_axis):
    value = value.float()
    energy = value.square().mean()
    shared = value.mean(temporal_axis).square().mean()
    return dict(rms=float(energy.sqrt()), temporal_variation_energy_fraction=float(1 - shared / energy.clamp_min(1e-20)))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--describe', action='store_true')
    args = parser.parse_args()
    checkpoint = ROOT / 'runs/physical_content_67829/step0400.pt'
    plan = dict(checkpoint=str(checkpoint), source=str(SOURCE), cases=CASES, inspected_steps=STEPS,
                inspected_layers=[2, 14, 29], seed=42, gamma=5, output_videos=0, training_updates=0,
                api_calls=0, resource_pool='ada-48g',
                selection='Diagnostic selection after full40: failed fracture, SA-regressed rolling, nearly-static shaped dough, and roller deformation. Not an aggregate quality sample.',
                scope='Normal50-step trajectory replay; native/no_c and direct-read velocities evaluated at the identical latent and sigma at four steps. No alternate trajectory is advanced.')
    if args.describe:
        print(json.dumps(plan, ensure_ascii=False)); return
    folder = ROOT / 'reports/control_helpers' / os.environ['DET_EXPERIMENT_ID']
    count = torch.cuda.device_count()
    name = torch.cuda.get_device_name(0) if count else 'NONE'
    memory = torch.cuda.get_device_properties(0).total_memory if count else 0
    compact = ''.join(c for c in name.upper() if c.isalnum())
    accepted = count == 1 and 'H100' not in compact and memory >= 44 * 1024**3 and any(
        token in compact for token in ['L40', 'RTX6000ADA'])
    save(folder / 'gpu.json', dict(actual=name, accepted=accepted, resource_pool='ada-48g', visible_count=count))
    if not accepted: raise SystemExit(86)
    save(folder / 'plan.json', plan)
    sys.path.insert(0, str(SOURCE))
    import bootstrap
    from physical_content.model import PhysicalModel
    from condition_controls import InferenceCache
    from benchmark_inputs import reference, case_input, load_noise
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    cfg = saved['config']
    assert saved['step'] == 400 and Path(cfg['source_snapshot']) == SOURCE
    torch.manual_seed(42); torch.cuda.manual_seed_all(42); torch.set_num_threads(4)
    model = PhysicalModel(cfg).eval().requires_grad_(False)
    parameters = dict(model.named_parameters()); buffers = dict(model.named_buffers())
    assert set(saved['model']) == set(model.learned_names)
    for key, value in saved['model'].items(): parameters[key].copy_(value)
    for key, value in saved['constants'].items(): buffers[key].copy_(value)
    del saved
    active = {}; native = {}; layer_rows = []; velocity_rows = []
    def native_hook(site):
        def hook(module, inputs, output):
            if active.get('record'): native[site] = output
        return hook
    def physical_hook(site):
        def hook(module, inputs, output):
            if not active.get('record'): return
            text = native.pop(site)[:, -output.shape[1]:]
            indices = torch.linspace(0, output.shape[1] - 1, 128, device=output.device).long()
            a = output[:, indices].float(); t = text[:, indices].float()
            frames = active['frames']
            physical = field_stats(output.reshape(1, frames, -1, output.shape[-1]), 1)
            original = field_stats(text.reshape(1, frames, -1, text.shape[-1]), 1)
            layer_rows.append(dict(case=active['case'], step=active['step'], sigma=active['sigma'],
                                   arm=active['arm'], layer=site, physical=physical, native_text=original,
                                   physical_to_native_text_rms=physical['rms'] / max(original['rms'], 1e-12),
                                   sampled_physical_native_cosine=float(F.cosine_similarity(a, t, dim=-1).mean())))
        return hook
    handles = []
    for site in plan['inspected_layers']:
        handles.append(model.wan.blocks[site].cross_attn.o.register_forward_hook(native_hook(site)))
        handles.append(model.fusions[str(site)].output.register_forward_hook(physical_hook(site)))
    reader = InferenceCache(cfg['paths']['inference_evidence'])
    cases = {case['id']: case for case in reference()['cases']}
    for case_id in CASES:
        save(folder / 'status.json', dict(status='execution_probe', case=case_id, completed_cases=[r['case'] for r in velocity_rows[::len(STEPS)]]))
        inputs, item = case_input(cases[case_id])
        sample = {key: value.cuda() for key, value in inputs.items()}; sample['times'] = sample['times'][None]
        positive = reader.input(item, image=item['image'], duration=item['duration'], mode='real')
        negative = reader.input(item, image=item['image'], duration=item['duration'], drop=True)
        with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
            content, groups = model.enhancement(positive); null_content, null_groups = model.enhancement(negative)
            x = load_noise(cases[case_id]).cuda().clone(); x[:, :, :1] = sample['first']
            grid = torch.linspace(1, 0, cfg['inference']['steps'] + 1, device='cuda')
            shift = cfg['inference']['shift']; grid = shift * grid / (1 + (shift - 1) * grid)
            for index in range(cfg['inference']['steps']):
                active.update(record=index + 1 in STEPS, case=case_id, step=index + 1,
                              sigma=float(grid[index]), frames=x.shape[2] - 1)
                def predict(mode):
                    flags = dict(physical_enabled=mode != 'no_c', direct_physical_read=mode == 'direct')
                    active['arm'] = 'positive'
                    pos, _ = model.velocity(x, sample['first'], sample['text'], grid[index], content,
                                            sample['times'], groups, **flags)
                    active['arm'] = 'negative'
                    neg, _ = model.velocity(x, sample['first'], sample['null_text'], grid[index], null_content,
                                            sample['times'], null_groups, **flags)
                    return neg + 5. * (pos - neg)
                velocity = predict('normal')
                if index + 1 in STEPS:
                    active['record'] = False
                    baseline = predict('no_c'); direct = predict('direct')
                    full = velocity[:, :, 1:]; base = baseline[:, :, 1:]; direct = direct[:, :, 1:]
                    delta = full - base; direct_delta = direct - base
                    velocity_rows.append(dict(case=case_id, step=index + 1, sigma=float(grid[index]),
                        no_c=field_stats(base, 2), normal=field_stats(full, 2),
                        physical_effect=field_stats(delta, 2), direct_physical_effect=field_stats(direct_delta, 2),
                        normal_direct_difference=field_stats(full - direct, 2),
                        effect_to_no_c_rms=float(delta.square().mean().sqrt() / base.square().mean().sqrt().clamp_min(1e-12))))
                x = x + (grid[index + 1] - grid[index]) * velocity; x[:, :, :1] = sample['first']
        save(folder / 'measurements.json', dict(**plan, layers=layer_rows, velocities=velocity_rows,
            limitations='Instantaneous effects at the same sampled latent, not alternate-video quality or a retrained ablation. Temporal variation is in hidden/velocity fields, not an RGB motion reward. Ada replay is not claimed bitwise identical to RTX videos.'))
        del x, sample, positive, negative, content, null_content
        print(json.dumps(dict(kind='case_probe_complete', case=case_id, time=time.time())), flush=True)
    for handle in handles: handle.remove()
    save(folder / 'status.json', dict(status='completed', cases=CASES, output_videos=0))


if __name__ == '__main__':
    main()
