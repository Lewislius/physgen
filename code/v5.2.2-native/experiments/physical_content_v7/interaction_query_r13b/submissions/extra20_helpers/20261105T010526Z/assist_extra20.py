"""Run only the declared additional20 early, then hand complete scores/videos to primary.

The primary training/40 runner stays unchanged. Helper outputs are isolated;
only a complete scored mode is copied atomically into an unstarted primary mode.
"""
from pathlib import Path
import argparse
import json
import os
import shutil
import sys
import yaml


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--parent-experiment', type=int, required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--describe', action='store_true')
    args = p.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    source = Path(cfg['source_snapshot'])
    root = Path(cfg['paths']['runs']).parent
    checkpoint = root / 'runs' / args.run_id / 'step0400.pt'
    target = root / 'inference_outputs' / args.run_id / 'step0400_original_fixed40_controls16_seed42'
    parent_plan = json.loads((root / 'reports/jobs' / str(args.parent_experiment) / 'plan.json').read_text())
    ids = parent_plan['extra_cases']
    modes = ['real']
    if len(ids) != 20 or set(ids) != {f'VP{i:03d}' for i in range(41, 61)}:
        raise ValueError('Use only the declared fixed additional20')
    plan = dict(parent_experiment=args.parent_experiment, run_id=args.run_id,
                checkpoint=str(checkpoint), source=str(source), primary_output=str(target),
                modes=modes, cases=ids, videos=len(modes) * len(ids), training_updates=0, api_calls=0,
                scope='Same frozen inference source/own400 weights, inputs, seed42 andgamma5 as primary.')
    if args.describe:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    sys.path.insert(0, str(source))
    import bootstrap
    from physical_content.common import gpu_guard, atomic_json
    import benchmark_inputs
    from pipeline import run_stage, evaluator_environment
    experiment = os.getenv('DET_EXPERIMENT_ID', 'local')
    folder = root / 'reports/control_helpers' / experiment
    gpu_guard(folder)
    atomic_json(plan, folder / 'plan.json')
    primary_status = root / 'reports/jobs' / str(args.parent_experiment) / 'status.json'
    status = json.loads(primary_status.read_text())
    allowed_start = {'original40_generation'}
    if not checkpoint.is_file() or status['status'] not in allowed_start:
        atomic_json(dict(status='leave_to_primary', reason='Own400 is absent or the primary has progressed beyond the early assistance window',
                         primary_stage=status['status']), folder / 'status.json')
        return
    completed40 = len(list((target / 'videophy40/cases').glob('*/planned.json')))
    if completed40 >= 24:
        atomic_json(dict(status='leave_to_primary', reason='Extra20 assistance needs enough original40 generation headstart',
                         completed40=completed40), folder / 'status.json')
        return
    output = root / 'helper_outputs' / args.run_id / experiment
    python = [sys.executable, '-I', '-B', '-u']
    transferred = []
    for mode in modes:
        destination = target / 'videophy60'
        if destination.exists():
            atomic_json(dict(status='leave_to_primary', mode=mode,
                             reason='Primary has already created this control mode', transferred=transferred), folder / 'status.json')
            continue
        atomic_json(dict(status=mode + '_generation', transferred=transferred), folder / 'status.json')
        run_stage([*python, source / 'infer.py', '--checkpoint', checkpoint, '--output', output,
                   '--suite', 'videophy60', '--modes', mode, '--cases', *ids], folder / (mode + '_generation.log'))
        generated = output / 'videophy60'
        atomic_json(dict(status=mode + '_scoring', transferred=transferred), folder / 'status.json')
        run_stage(['/home/liuzhirui/miniconda3/envs/physvid/bin/python', '-I', '-B', '-u',
                   bootstrap.BASE / 'tools/evaluate_videophy2.py', generated],
                  folder / (mode + '_scoring.log'), evaluator_environment())
        score = json.loads((generated / 'scores.json').read_text())
        if len(score['cases']) != 20 or {v['id'] for v in score['cases']} != set(ids):
            raise ValueError('Only a complete scored fixed20 may be handed to primary')
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.parent / ('.' + mode + '_helper_' + experiment)
            if temporary.exists():
                raise ValueError('An unfinished prior transfer requires explicit inspection')
            shutil.copytree(generated, temporary)
            try:
                temporary.rename(destination)
            except OSError:
                if not destination.exists():
                    raise
                # Another process owns the primary destination. Preserve its
                # contents and our isolated, complete helper evidence.
                shutil.rmtree(temporary)
            else:
                transferred.append(mode)
        atomic_json(dict(status=mode + '_complete', transferred=transferred,
                         isolated_result=str(generated / 'scores.json'),
                         primary_result=str(destination / 'scores.json')),
                    folder / (mode + '_handoff.json'))
    atomic_json(dict(status='completed', transferred=transferred, **plan), folder / 'status.json')


if __name__ == '__main__':
    main()
