"""Generate the last eight already-declared mean-C cases on a spare RTX.

Primary still owns the fixed16 manifest and scoring. No scores or incomplete
videos are copied, and neither the training nor the inference source changes.
"""
from pathlib import Path
import argparse
import json
import os
import shutil
import sys
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--parent-experiment', type=int, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--describe', action='store_true')
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    source = Path(cfg['source_snapshot'])
    root = Path(cfg['paths']['runs']).parent
    checkpoint = root / 'runs' / args.run_id / 'step0400.pt'
    mean = root / 'controls' / args.run_id / 'step0400_mean_content.pt'
    target = root / 'inference_outputs' / args.run_id / 'step0400_original_fixed40_controls16_seed42'
    all_ids = [v['id'] for v in json.loads((source / 'subset16.json').read_text())['cases']]
    if len(all_ids) != 16 or 'global_train_mean' not in cfg['evaluation_budget']['control_modes']:
        raise ValueError('Require the existing declared fixed16 mean-C control')
    ids = all_ids[8:]
    plan = dict(parent_experiment=args.parent_experiment, run_id=args.run_id,
                source=str(source), checkpoint=str(checkpoint), global_condition=str(mean),
                mode='global_train_mean', cases=ids, videos=8, added_cases=0,
                primary_owns='Full16 manifest and independent scoring',
                training_updates=0, api_calls=0)
    if args.describe:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    sys.path.insert(0, str(source))
    import bootstrap
    import benchmark_inputs  # Adds the established evaluation pipeline path.
    from physical_content.common import gpu_guard, atomic_json
    from pipeline import run_stage
    experiment = os.getenv('DET_EXPERIMENT_ID', 'local')
    folder = root / 'reports/control_helpers' / experiment
    gpu_guard(folder)
    atomic_json(plan, folder / 'plan.json')
    status = json.loads((root / 'reports/jobs' / str(args.parent_experiment) / 'status.json').read_text())
    wrong = target / 'controls/reasoning_swapped/videophy40/cases'
    wrong_done = len(list(wrong.glob('*/planned.json')))
    early = status['status'] in {'control16_no_c_generation', 'control16_no_c_scoring',
                                'control16_reasoning_swapped_generation'}
    if not checkpoint.is_file() or not mean.is_file() or not early or wrong_done >= 8:
        atomic_json(dict(status='leave_to_primary', reason='Insufficient lead for a disjoint eight-case tail',
                         primary=status['status'], unrelated_done=wrong_done), folder / 'status.json')
        return
    output = root / 'helper_outputs' / args.run_id / experiment
    atomic_json(dict(status='mean_tail8_generation', cases=ids), folder / 'status.json')
    run_stage([sys.executable, '-I', '-B', '-u', source / 'infer.py',
               '--checkpoint', checkpoint, '--output', output, '--suite', 'videophy40',
               '--modes', 'global_train_mean', '--global-condition', mean, '--cases', *ids],
              folder / 'generation.log')
    generated = output / 'controls/global_train_mean/videophy40/cases'
    destination = target / 'controls/global_train_mean/videophy40/cases'
    destination.mkdir(parents=True, exist_ok=True)
    transferred = []
    for case in ids:
        metadata = json.loads((generated / case / 'planned.json').read_text())
        if (metadata['checkpoint'] != str(checkpoint) or metadata['mode'] != 'global_train_mean'
                or metadata['seed'] != 42 or metadata['global_condition'] != str(mean.resolve())
                or not (generated / case / 'planned.mp4').is_file()):
            raise ValueError('Tail provenance differs from primary inference')
        if (destination / case).exists():
            continue
        temporary = destination / ('.' + case + '_' + experiment)
        shutil.copytree(generated / case, temporary)
        try:
            temporary.rename(destination / case)
        except OSError:
            if not (destination / case).exists():
                raise
            shutil.rmtree(temporary)
        else:
            transferred.append(case)
    atomic_json(dict(status='completed', transferred=transferred, **plan), folder / 'status.json')


if __name__ == '__main__':
    main()
