"""R19 evidence gathering on preserved R18 own400 weights, fixed paired16."""
from pathlib import Path
import argparse
import json
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap

from physical_content.common import atomic_json, gpu_guard
from pipeline import evaluator_environment, run_stage

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--normal', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', required=True)
    parser.add_argument('--control-cache', type=Path)
    args = parser.parse_args()
    job = args.output / 'jobs' / os.environ.get('DET_EXPERIMENT_ID', 'local')
    job.mkdir(parents=True, exist_ok=True)
    # No inherited permissive pool policy. Guard runs before any model loads.
    os.environ['GPU_POLICY'] = 'rtx6000'
    gpu_guard(job)
    plan = dict(checkpoint=str(args.checkpoint), normal=str(args.normal),
                output=str(args.output), cases=args.cases, source=str(ROOT),
                purpose='R19 design evidence; inference-only R18 checkpoint interventions',
                formal_R19_training=False, resource_pool='blk-96g', task='t2v' if 'r18g' in str(args.checkpoint).lower() else 'i2v')
    atomic_json(plan, args.output / 'plan.json')
    python = [sys.executable, '-I', '-B', '-u']
    control_args = ['--control-cache', str(args.control_cache)] if args.control_cache else []
    # Obtain module evidence first; do not delay it behind the three condition
    # interventions. Each phase shares one model load across its modes.
    for phase, modes in (
        ('modules', ['no_c', 'conditioned_ffn_off']),
        ('conditions', ['reasoning_swapped', 'empty_reasoning', 'global_train_mean']),
    ):
        atomic_json(dict(status=f'{phase}_generation', **plan), job / 'status.json')
        run_stage([*python, str(ROOT / 'infer.py'), '--checkpoint', str(args.checkpoint),
                   '--output', str(args.output), '--suite', 'videophy40',
                   '--controls', *modes, '--cases', *args.cases, *control_args], job / f'{phase}_generation.log')
        for mode in modes:
            folder = args.output / 'controls' / mode / 'videophy40'
            atomic_json(dict(status=f'{mode}_scoring', **plan), job / 'status.json')
            run_stage(['/home/liuzhirui/miniconda3/envs/physvid/bin/python', '-I', '-B', '-u',
                       str(bootstrap.BASE / 'tools/evaluate_videophy2.py'), str(folder)],
                      job / f'{mode}_scoring.log', evaluator_environment())
        run_stage([*python, str(ROOT / 'summarize_controls.py'), '--root', str(args.output)],
                  job / f'{phase}_summary.log')
    atomic_json(dict(status='completed', **plan), job / 'status.json')


if __name__ == '__main__':
    main()
