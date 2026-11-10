"""Complete one declared R10 control using its frozen inference source."""
import argparse
import json
import os
from pathlib import Path
import sys

BASE = Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native')
ROOT = BASE / 'experiments/physical_content_v7/grounded_execution_r10b'
SOURCE = ROOT / 'snapshots/20261101T054248Z'
RUN = 'physical_content_67008'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['reasoning_swapped', 'empty_reasoning',
        'global_train_mean', 'core_only', 'anchor_only', 'no_c'], required=True)
    parser.add_argument('--describe', action='store_true')
    args = parser.parse_args()
    cases = [row['id'] for row in json.loads((SOURCE / 'subset16.json').read_text())['cases']]
    checkpoint = ROOT / 'runs' / RUN / 'step0400.pt'
    mean_content = ROOT / 'controls' / RUN / 'step0400_mean_content.pt'
    output = ROOT / 'control_dispatch/inference_outputs' / RUN / f'{args.mode}_subset16_seed42_gamma5'
    generated = output / 'controls' / args.mode
    canonical = ROOT / 'inference_outputs' / RUN / 'step0400_full40_controls16_physical5_seed42/controls' / args.mode
    owner_stage = ROOT / 'reports/jobs/67008' / f'{args.mode}.log'
    experiment = os.getenv('DET_EXPERIMENT_ID', 'local')
    report = ROOT / 'control_dispatch/reports/jobs' / experiment / args.mode
    command = [sys.executable, '-I', '-B', '-u', SOURCE / 'infer.py',
               '--checkpoint', checkpoint, '--output', output, '--suite', 'videophy40',
               '--physical-guidance', '5', '--seed', '42', '--modes', args.mode,
               '--cases', *cases]
    mean_command = None
    if args.mode == 'global_train_mean':
        command += ['--global-condition', mean_content]
        # The exact frozen preparer reuses a complete matching mean before any
        # model construction. The original runner later uses the same file.
        mean_command = [sys.executable, '-I', '-B', '-u', SOURCE / 'prepare_mean_condition.py',
                        '--checkpoint', checkpoint, '--output', mean_content]
    evaluation = ['/home/liuzhirui/miniconda3/envs/physvid/bin/python', '-I', '-B', '-u',
                  BASE / 'tools/evaluate_videophy2.py', generated / 'videophy40']
    plan = dict(mode=args.mode, checkpoint=str(checkpoint), source=str(SOURCE), cases=cases,
                output=str(output), canonical=str(canonical),
                generation_command=list(map(str, command)), evaluation_command=list(map(str, evaluation)),
                mean_preparation_command=list(map(str, mean_command)) if mean_command else None,
                scope='One already-declared fixed16 control; no additional cases, seeds, training or downloads')
    if args.describe:
        print(json.dumps(plan, indent=2)); return

    sys.path.insert(0, str(SOURCE))
    import bootstrap
    from physical_content.common import atomic_json, gpu_guard
    report.mkdir(parents=True, exist_ok=True)
    atomic_json(plan, report / 'plan.json')
    if owner_stage.exists() or canonical.exists():
        atomic_json(dict(status='left_to_original_owner', mode=args.mode), report / 'status.json')
        return
    original_mean_stage = ROOT / 'reports/jobs/67008/mean_content_prepare.log'
    if mean_command and not mean_content.exists() and original_mean_stage.exists():
        atomic_json(dict(status='mean_preparation_left_to_original_owner', mode=args.mode),
                    report / 'status.json')
        return
    gpu_guard(report)
    pinned = BASE / 'experiments/physical_scaling_q256/inference_snapshots/20261030T034046Z'
    sys.path.insert(0, str(pinned))
    from pipeline import evaluator_environment, run_stage
    if mean_command:
        run_stage(mean_command, report / 'mean_prepare.log')
    run_stage(command, report / 'generation.log')
    if owner_stage.exists() or canonical.exists():
        atomic_json(dict(status='original_owner_started', mode=args.mode,
                         preserved_output=str(output)), report / 'status.json')
        return
    run_stage(evaluation, report / 'evaluation.log', evaluator_environment())
    scores = json.loads((generated / 'videophy40/scores.json').read_text())
    if (scores['summary']['count'] != 16 or len(scores['cases']) != 16
            or {row['id'] for row in scores['cases']} != set(cases)):
        raise ValueError('Publish only the complete, originally declared control set')
    canonical.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Exclusive directory-link creation cannot replace another worker's output.
        canonical.symlink_to(generated, target_is_directory=True)
        status = 'completed_and_published'
    except FileExistsError:
        status = 'completed_separately_original_owner_retained'
    atomic_json(dict(status=status, mode=args.mode, output=str(output), canonical=str(canonical),
                     summary=scores['summary']), report / 'status.json')
    print(json.dumps(dict(status=status, mode=args.mode, summary=scores['summary'])), flush=True)


if __name__ == '__main__':
    main()
