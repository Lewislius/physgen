"""Run declared fixed16 controls early, then hand complete results to primary.

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
    p.add_argument('--modes', nargs='+', choices=['no_c', 'global_train_mean'],
                   default=['no_c'])
    p.add_argument('--describe', action='store_true')
    args = p.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    source = Path(cfg['source_snapshot'])
    root = Path(cfg['paths']['runs']).parent
    checkpoint = root / 'runs' / args.run_id / 'step0400.pt'
    target = root / 'inference_outputs' / args.run_id / 'step0400_original_fixed40_controls16_seed42'
    ids = [v['id'] for v in json.loads((source / 'subset16.json').read_text())['cases']]
    modes = args.modes
    if (len(ids) != 16 or len(modes) != 1 or len(modes) != len(set(modes))
            or not set(modes) <= set(cfg['evaluation_budget']['control_modes'])):
        raise ValueError('Assist only already-declared fixed16 controls')
    plan = dict(parent_experiment=args.parent_experiment, run_id=args.run_id,
                checkpoint=str(checkpoint), source=str(source), primary_output=str(target),
                modes=modes, cases=ids, videos=len(modes) * len(ids), training_updates=0, api_calls=0,
                scope='Same frozen inference source/own400 weights, inputs, seed42 andgamma5 as primary.',
                mean_prepare=('The same declared train64 positive/null, selection_seed463; canonical file reused by primary'
                              if 'global_train_mean' in modes else None))
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
    allowed_start = {'original40_generation', 'original40_scoring', 'original40_report',
                     'extra20_generation', 'extra20_scoring', 'full60_aggregate', 'full60_report'}
    if not checkpoint.is_file() or status['status'] not in allowed_start:
        atomic_json(dict(status='leave_to_primary', reason='Own400 is absent or the primary has progressed beyond the early assistance window',
                         primary_stage=status['status']), folder / 'status.json')
        return
    output = root / 'helper_outputs' / args.run_id / experiment
    python = [sys.executable, '-I', '-B', '-u']
    transferred = []
    for mode in modes:
        destination = target / 'controls' / mode
        if destination.exists():
            atomic_json(dict(status='leave_to_primary', mode=mode,
                             reason='Primary has already created this control mode', transferred=transferred), folder / 'status.json')
            continue
        mean = root / 'controls' / args.run_id / 'step0400_mean_content.pt'
        if mode == 'global_train_mean':
            atomic_json(dict(status='train_mean_prepare', transferred=transferred), folder / 'status.json')
            run_stage([*python, source / 'data_tools/prepare_mean_condition.py',
                       '--checkpoint', checkpoint, '--output', mean], folder / 'train_mean_prepare.log')
        atomic_json(dict(status=mode + '_generation', transferred=transferred), folder / 'status.json')
        command = [*python, source / 'infer.py', '--checkpoint', checkpoint, '--output', output,
                   '--suite', 'videophy40', '--modes', mode, '--cases', *ids]
        if mode == 'global_train_mean':
            command += ['--global-condition', mean]
        run_stage(command, folder / (mode + '_generation.log'))
        generated = output / 'controls' / mode
        atomic_json(dict(status=mode + '_scoring', transferred=transferred), folder / 'status.json')
        run_stage(['/home/liuzhirui/miniconda3/envs/physvid/bin/python', '-I', '-B', '-u',
                   bootstrap.BASE / 'tools/evaluate_videophy2.py', generated / 'videophy40'],
                  folder / (mode + '_scoring.log'), evaluator_environment())
        score = json.loads((generated / 'videophy40/scores.json').read_text())
        if len(score['cases']) != 16 or {v['id'] for v in score['cases']} != set(ids):
            raise ValueError('Only a complete scored fixed16 may be handed to primary')
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
                         isolated_result=str(generated / 'videophy40/scores.json'),
                         primary_result=str(destination / 'videophy40/scores.json')),
                    folder / (mode + '_handoff.json'))
    atomic_json(dict(status='completed', transferred=transferred, **plan), folder / 'status.json')


if __name__ == '__main__':
    main()
