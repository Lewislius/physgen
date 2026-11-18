"""Score the preserved own400 full40 and finish unassigned declared controls only."""
import argparse
import json
import os
from pathlib import Path
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
    root = Path(cfg['paths']['runs']).parent
    source = Path(cfg['source_snapshot'])
    checkpoint = root / 'runs' / args.run_id / 'step0400.pt'
    output = root / 'inference_outputs' / args.run_id / 'step0400_original_fixed40_controls16_seed42'
    plan = dict(parent_experiment=args.parent_experiment, run_id=args.run_id, source=str(source),
                checkpoint=str(checkpoint), primary_output=str(output), modes=cfg['evaluation_budget']['control_modes'],
                training_updates=0, new_main_videos=0, api_calls=0,
                scope='Existing completed original40 scoring and only missing, unassigned declared fixed16 controls')
    if args.describe:
        print(json.dumps(plan, ensure_ascii=False)); return
    sys.path.insert(0, str(source))
    import bootstrap
    import benchmark_inputs
    from physical_content.common import gpu_guard, atomic_json
    from pipeline import run_stage, evaluator_environment
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from allocated_gpu import bind_allocated_gpu
    experiment = os.environ['DET_EXPERIMENT_ID']
    folder = root / 'reports/control_helpers' / experiment
    gpu_guard(folder)
    atomic_json(plan, folder / 'plan.json')
    bind_allocated_gpu(folder, evaluator_environment)
    expected = {f'VP{i:03d}' for i in range(1, 41)}
    cases = list((output / 'videophy40/cases').glob('*/planned.json'))
    if {path.parent.name for path in cases} != expected or len(cases) != 40:
        raise ValueError('Resume scoring only after the same complete original40 exists')
    for path in cases:
        meta = json.loads(path.read_text())
        if not path.with_suffix('.mp4').is_file() or meta['checkpoint'] != str(checkpoint) or meta['inference_source'] != str(source):
            raise ValueError('Existing video provenance differs')
    def score(destination, name):
        atomic_json(dict(status=name + '_scoring'), folder / 'status.json')
        run_stage(['/home/liuzhirui/miniconda3/envs/physvid/bin/python', '-I', '-B', '-u',
                   bootstrap.BASE / 'tools/evaluate_videophy2.py', destination],
                  folder / (name + '_scoring.log'), evaluator_environment())
    score(output / 'videophy40', 'original40')
    python = [sys.executable, '-I', '-B', '-u']
    run_stage([*python, source / 'report_results.py', '--output', output], folder / 'original40_report.log')
    ids = [case['id'] for case in json.loads((source / 'subset16.json').read_text())['cases']]
    deferred = []
    for mode in cfg['evaluation_budget']['control_modes']:
        destination = output / 'controls' / mode / 'videophy40'
        if (destination / 'scores.json').is_file():
            continue
        owners = []
        for path in root.glob('*_controls_helper_prepared.json'):
            helper = json.loads(path.read_text())
            eid = helper.get('experiment_id')
            if mode not in helper.get('modes', []) or not eid or str(eid) == experiment:
                continue
            gpu = root / 'reports/control_helpers' / str(eid) / 'gpu.json'
            if gpu.is_file() and json.loads(gpu.read_text()).get('accepted'):
                owners.append(eid)
        if owners:
            deferred.append(dict(mode=mode, owners=owners))
            continue
        mean = root / 'controls' / args.run_id / 'step0400_mean_content.pt'
        if mode == 'global_train_mean' and not mean.is_file():
            raise ValueError('Prepared own400 mean cache is missing')
        atomic_json(dict(status=mode + '_generation', deferred=deferred), folder / 'status.json')
        command = [*python, source / 'infer.py', '--checkpoint', checkpoint, '--output', output,
                   '--suite', 'videophy40', '--modes', mode, '--cases', *ids]
        if mode == 'global_train_mean': command += ['--global-condition', mean]
        run_stage(command, folder / (mode + '_generation.log'))
        score(destination, mode)
    run_stage([*python, source / 'report_results.py', '--output', output], folder / 'final_report.log')
    atomic_json(dict(status='completed', deferred=deferred, **plan), folder / 'status.json')


if __name__ == '__main__':
    main()
