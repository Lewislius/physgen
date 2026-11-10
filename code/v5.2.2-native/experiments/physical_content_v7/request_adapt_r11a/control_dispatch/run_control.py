"""Finish one already-declared R11A16 control using its exact frozen inference."""
import argparse
import json
import os
from pathlib import Path
import sys

BASE = Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native')
ROOT = BASE / 'experiments/physical_content_v7/request_adapt_r11a'
SOURCE = ROOT / 'snapshots/20261103T054429Z'
RUN = 'physical_content_67214'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['text_adapter_off', 'parallel_fusion'], required=True)
    p.add_argument('--describe', action='store_true')
    args = p.parse_args()
    cases = [r['id'] for r in json.loads((SOURCE / 'subset16.json').read_text())['cases']]
    checkpoint = ROOT / 'runs' / RUN / 'step0400.pt'
    status = json.loads((checkpoint.parent / 'status.json').read_text())
    if not checkpoint.is_file() or status['step'] != 400 or status['status'] not in ('paused', 'complete'):
        raise ValueError('The own400 checkpoint must already be complete')
    if len(cases) != 16:
        raise ValueError('Use only the original fixed16 cases')
    output = ROOT / 'control_dispatch/inference_outputs' / RUN / f'{args.mode}_subset16_seed42_gamma5'
    generated = output / 'controls' / args.mode
    canonical = ROOT / 'inference_outputs' / RUN / 'step0400_full40_controls16_physical5_seed42/controls' / args.mode
    owner_stage = ROOT / 'reports/jobs/67214' / f'{args.mode}.log'
    report = ROOT / 'control_dispatch/reports/jobs' / os.getenv('DET_EXPERIMENT_ID', 'local') / args.mode
    command = [sys.executable, '-I', '-B', '-u', SOURCE / 'infer.py', '--checkpoint', checkpoint,
        '--output', output, '--suite', 'videophy40', '--physical-guidance', '5', '--seed', '42',
        '--modes', args.mode, '--cases', *cases]
    evaluation = ['/home/liuzhirui/miniconda3/envs/physvid/bin/python', '-I', '-B', '-u',
        BASE / 'tools/evaluate_videophy2.py', generated / 'videophy40']
    plan = dict(mode=args.mode, checkpoint=str(checkpoint), source=str(SOURCE), cases=cases,
        output=str(output), canonical=str(canonical), generation_command=list(map(str, command)),
        evaluation_command=list(map(str, evaluation)),
        scope='Already-declared fixed16 intervention only; unchanged400 checkpoint, seed42, gamma5, prompts and evaluator')
    if args.describe:
        print(json.dumps(plan, indent=2)); return
    sys.path.insert(0, str(SOURCE))
    import bootstrap
    from physical_content.common import atomic_json, gpu_guard
    report.mkdir(parents=True, exist_ok=True); atomic_json(plan, report / 'plan.json')
    if owner_stage.exists() or canonical.exists():
        atomic_json(dict(status='left_to_original_owner', mode=args.mode), report / 'status.json'); return
    gpu_guard(report)
    sys.path.insert(0, str(BASE / 'experiments/physical_scaling_q256/inference_snapshots/20261030T034046Z'))
    from pipeline import evaluator_environment, run_stage
    run_stage(command, report / 'generation.log')
    if owner_stage.exists() or canonical.exists():
        atomic_json(dict(status='original_owner_started', mode=args.mode,
            preserved_output=str(output)), report / 'status.json'); return
    run_stage(evaluation, report / 'evaluation.log', evaluator_environment())
    scores = json.loads((generated / 'videophy40/scores.json').read_text())
    if (scores['summary']['count'] != 16 or len(scores['cases']) != 16
            or {r['id'] for r in scores['cases']} != set(cases)):
        raise ValueError('Publish only the completed original16 control')
    canonical.parent.mkdir(parents=True, exist_ok=True)
    try:
        canonical.symlink_to(generated, target_is_directory=True)
        result = 'completed_and_published'
    except FileExistsError:
        result = 'completed_separately_original_owner_retained'
    atomic_json(dict(status=result, mode=args.mode, output=str(output), canonical=str(canonical),
        summary=scores['summary']), report / 'status.json')
    print(json.dumps(dict(status=result, mode=args.mode, summary=scores['summary'])), flush=True)


if __name__ == '__main__': main()
