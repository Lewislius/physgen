"""Extend completed R13A40 to60 with the same own400 weights and20 new videos."""
import argparse
import json
import os
from pathlib import Path
import sys
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--describe', action='store_true')
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    source = Path(cfg['source_snapshot'])
    root = Path(cfg['paths']['runs']).parent
    checkpoint = root / 'runs' / args.run_id / 'step0400.pt'
    target = root / 'inference_outputs' / args.run_id / 'step0400_original_fixed40_controls16_seed42'
    protocol = json.loads((root.parent / 'videophy60/protocol.json').read_text())
    cases = protocol['extra_ids']
    if set(cases) != {f'VP{i:03d}' for i in range(41, 61)}:
        raise ValueError('Extend with only the original fixed20')
    experiment = os.getenv('DET_EXPERIMENT_ID', 'unsubmitted')
    output = root / 'helper_outputs' / args.run_id / experiment / 'extra20'
    record = dict(checkpoint=str(checkpoint), source=str(source),
                  completed_part=str(target / 'videophy40'),
                  additional_part=str(output / 'videophy60'),
                  merge_target=str(target / 'full60/videophy60'),
                  new_videos=20, cases_after_merge=60, extra_cases=cases,
                  new_training=False, new_api_calls=0, new_downloads=0, seed=42, gamma=5)
    if args.describe:
        print(json.dumps(record, ensure_ascii=False, indent=2))
        return
    scores = json.loads((target / 'videophy40/scores.json').read_text())
    if len(scores['cases']) != 40 or {v['id'] for v in scores['cases']} != {f'VP{i:03d}' for i in range(1, 41)}:
        raise ValueError('Complete the original40 before extending it')
    manifest = json.loads((target / 'videophy40/manifest.json').read_text())
    if Path(manifest['checkpoint']).resolve() != checkpoint.resolve() or not checkpoint.is_file():
        raise ValueError('Use the same completed own400 checkpoint')
    sys.path.insert(0, str(source))
    import bootstrap
    import benchmark_inputs
    from physical_content.common import atomic_json, gpu_guard
    from pipeline import run_stage, evaluator_environment
    folder = root / 'reports/extra20_helpers' / experiment
    gpu_guard(folder)
    atomic_json(record, folder / 'merge_record.json')

    def stage(name, command, environment=None):
        atomic_json(dict(status=name, **record), folder / 'status.json')
        if environment is None:
            run_stage(command, folder / (name + '.log'))
        else:
            run_stage(command, folder / (name + '.log'), environment)

    python = [sys.executable, '-I', '-B', '-u']
    stage('extra20_generation', [*python, source / 'infer.py', '--checkpoint', checkpoint,
          '--output', output, '--suite', 'videophy60', '--modes', 'real', '--cases', *cases])
    stage('extra20_scoring', ['/home/liuzhirui/miniconda3/envs/physvid/bin/python', '-I', '-B', '-u',
          bootstrap.BASE / 'tools/evaluate_videophy2.py', output / 'videophy60'], evaluator_environment())
    merge = root.parent / 'videophy60/snapshots/20261031T150803Z/merge_completed60.py'
    stage('aggregate_full60', [*python, merge, folder / 'merge_record.json'])
    complete = json.loads((target / 'full60/videophy60/scores.json').read_text())
    atomic_json(dict(status='completed', summary=complete['summary'], **record), folder / 'status.json')


if __name__ == '__main__':
    main()
