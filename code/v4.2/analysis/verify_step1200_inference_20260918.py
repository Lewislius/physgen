"""CPU audit of the actual step1200 EMA and both demo41 launchers; no generation."""
from argparse import Namespace
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import yaml

from inference.conditions import image_cache, text_cache
from inference.infer import build_plan, load_adapter, read_config
from physgen_v42.geometry import Geometry
from physgen_v42.runtime import code_fingerprint, digest, save_json, verify_files


def main():
    torch.set_num_threads(4)
    checkpoint = ROOT / 'checkpoints/stability_20260917T181001Z_a83129/step1200'
    cfg = read_config(checkpoint / 'config.json')
    complete = json.loads((checkpoint / 'complete.json').read_text())
    verify_files(checkpoint, {k: complete['files'][k] for k in ('config.json', 'ema.pt')}, full=True)
    assert complete['step'] == complete['ema_updates'] == 1200
    assert complete['version'] == cfg['model_version']
    assert digest(code_fingerprint()) == complete['identity']['code']
    plan = build_plan(cfg, Namespace(checkpoint=str(checkpoint), suite='demo', duration=5.,
                                     portrait=False, variant='full'))
    assert len(plan['cases']) == 41
    assert all('gt_record' not in c for c in plan['cases'])
    assert all(c['geometry']['frames'] == 121 and c['duration'] == 5. for c in plan['cases'])

    # Read v4's own case collector, including its image preference order.
    spec = importlib.util.spec_from_file_location('v4_audit_cases', ROOT.parent / 'v4/inference/cases.py')
    v4 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v4)
    v4_yaml = yaml.safe_load((ROOT.parent / 'v4/inference/infer_wisa_native_p_1x96g.yaml').read_text())
    old_env = dict(value.split('=', 1) for value in v4_yaml['environment']['environment_variables'])
    old_cases = v4.collect_cases(Namespace(suite=old_env['SUITE'], mode=old_env['MODE'],
        image=old_env['IMAGE'], prompt=old_env['PROMPT'], demo_root=old_env['DEMO_ROOT'],
        demo_modes=old_env['DEMO_MODES'], sample_ids=old_env['SAMPLE_IDS']))
    project = lambda c, id_key: (c[id_key], c['mode'], c['prompt'], c.get('image'))
    assert [project(c, 'id') for c in plan['cases']] == [project(c, 'case_id') for c in old_cases]

    launchers = {}
    for hardware, pool in (('1x96g', 'blk-96g'), ('1xada48g', 'ada-48g')):
        config = yaml.safe_load((ROOT / f'inference/infer_stability_{hardware}.yaml').read_text())
        env = dict(value.split('=', 1) for value in config['environment']['environment_variables'])
        assert env['CHECKPOINT'] == str(checkpoint) and env['SUITE'] == 'demo' and env['VARIANT'] == 'full'
        assert config['resources']['resource_pool'] == pool and config['resources']['slots_per_trial'] == 1
        clean = {k: v for k, v in os.environ.items() if k not in (
            'CHECKPOINT', 'SUITE', 'VARIANT', 'OUTPUT', 'CONFIG', 'EVALUATE', 'PORTRAIT')}
        script = ROOT / f'inference/infer_stability_{hardware}.sh'
        parsed = subprocess.check_output(['bash', str(script)], env=dict(clean, **env, PARSE_ONLY='1'), text=True)
        # Direct shell startup must choose the same checkpoint and output as YAML.
        direct = subprocess.check_output(['bash', str(script)], env=dict(clean, PARSE_ONLY='1'), text=True)
        assert parsed == direct
        commands = [shlex.split(line) for line in parsed.splitlines()]
        assert len(commands) == 3
        assert [c[c.index('--phase') + 1] for c in commands] == ['prepare', 'anchors', 'sample']
        assert commands[0][commands[0].index('--checkpoint') + 1] == str(checkpoint)
        outputs = [c[c.index('--output') + 1] for c in commands]
        assert len(set(outputs)) == 1
        launchers[hardware] = dict(pool=pool, output=outputs[0], commands=commands)
    assert launchers['1x96g']['output'] != launchers['1xada48g']['output']

    print('Loading and checking actual EMA tensors...', flush=True)
    adapter = load_adapter(plan, torch.device('cpu'))
    assert all(bool(torch.isfinite(v).all()) for v in adapter.state_dict().values())
    assert adapter.initializer == 'condition_trajectory_v1'
    assert all(c.writer.write_gate is not None for c in adapter.correctors.values())
    prompts = {c['prompt'] for c in plan['cases']} | {plan['negative']}
    for prompt in prompts:
        text = torch.load(text_cache(plan, prompt), map_location='cpu', weights_only=True)
        assert text.ndim == 2 and text.shape[1] == 4096 and bool(torch.isfinite(text).all())
    for case in plan['cases']:
        if case['mode'] != 'i2v':
            continue
        cached = torch.load(image_cache(plan, case, 'image'), map_location='cpu', weights_only=True)
        geo = case['geometry']
        assert cached['first'].shape == (1, 48, 1, geo['h'] // 16, geo['w'] // 16)
        assert cached['pixels'].shape == (1, 3, 1, geo['h'], geo['w'])
        anchor = torch.load(image_cache(plan, case, 'anchor'), map_location='cpu', weights_only=True)
        assert anchor.shape == (1, 576, 1664)
        assert all(bool(torch.isfinite(v).all()) for v in (cached['first'], cached['pixels'], anchor))

    initializer_checks = {}
    with torch.no_grad():
        for key in ('P01_i2v', 'P01_t2v'):
            case = next(c for c in plan['cases'] if c['key'] == key)
            geo = Geometry.build(case['geometry'], torch.device('cpu'))
            text = torch.load(text_cache(plan, case['prompt']), map_location='cpu', weights_only=True)[None].float()
            anchor = (torch.load(image_cache(plan, case, 'anchor'), map_location='cpu', weights_only=True).float()
                      if case['mode'] == 'i2v' else None)
            print(f'Running actual {key} trajectory initializer on CPU...', flush=True)
            p0 = adapter.initialize(text, geo, case['mode'], anchor, duration=case['duration'])
            assert p0.shape == (1, 16, 576, 1664) and bool(torch.isfinite(p0).all())
            initializer_checks[key] = dict(shape=list(p0.shape), finite=True,
                first_last_rms=float((p0[:, -1] - p0[:, 0]).square().mean().sqrt()))

    result = dict(status='passed', scope='CPU checkpoint/config/cache/initializer audit; no Wan GPU sampling',
        checkpoint=str(checkpoint), checkpoint_step=1200, ema_updates=1200,
        sha256_verified=['config.json', 'ema.pt'], strict_load=True,
        tensors=len(adapter.state_dict()), parameters=sum(p.numel() for p in adapter.parameters()),
        all_tensors_finite=True, exact_training_code_match=True,
        initializer=adapter.initializer, write_gate=True, initializer_checks=initializer_checks,
        cases=len(plan['cases']), modes=dict(Counter(c['mode'] for c in plan['cases'])),
        identical_v4_images_and_prompts=True, gt_future_required=False,
        geometries=dict(Counter(f"{c['geometry']['w']}x{c['geometry']['h']}x{c['geometry']['frames']}" for c in plan['cases'])),
        cached_texts=len(prompts), cached_first_images=21, cached_anchors=21,
        recipe=plan['recipe'], launchers=launchers, gpu_sampling_verified=False)
    output = ROOT / 'analysis/step1200_inference_audit_20260918'
    save_json(plan, output / 'cases.json')
    save_json(result, output / 'result.json')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
