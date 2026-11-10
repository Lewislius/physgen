"""Frozen B400 joint-read full40 then disjoint20/full60; no new training."""
from pathlib import Path
import argparse
import json
import os
import sys

SOURCE = Path(__file__).resolve().parent
ROOT = Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_v7')
EVAL = ROOT / 'b400_joint_read_eval'
B_SOURCE = ROOT / 'request_bound_fusion_r15b/snapshots/20261105T152223Z'
sys.path[:0] = [str(SOURCE), str(B_SOURCE)]
import bootstrap
from physical_content.common import atomic_json
from benchmark_inputs import reference
from pipeline import run_stage, evaluator_environment
from gpu_runtime import gpu_guard
from allocated_gpu import bind_allocated_gpu
from reuse import reuse_probe16


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--describe', action='store_true')
    args = p.parse_args()
    checkpoint = ROOT / 'request_bound_fusion_r15b/runs/physical_content_67829/step0400.pt'
    output = EVAL / 'videophy_outputs'
    job = EVAL / 'reports/jobs' / os.getenv('DET_EXPERIMENT_ID', 'local')
    plan = dict(kind='B400_joint_read_evaluation_only', checkpoint=str(checkpoint),
                source=str(SOURCE), output=str(output), suites=[40, 60], reused_probe16=68309,
                new_videos=44, new_training_updates=0, not_trained_R15I=True,
                seed=42, steps=50, shift=5, gamma=5, resource_pool=os.getenv('B400_EVAL_POOL', 'ada-48g'))
    if args.describe:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    gpu_guard(job)
    bind_allocated_gpu(job, evaluator_environment)
    atomic_json(plan, job / 'plan.json')
    reuse = reuse_probe16(output / 'videophy40', reference(), checkpoint, SOURCE / 'joint_read.py')
    atomic_json(reuse, EVAL / 'reports/reused_probe16.json')
    python = [sys.executable, '-I', '-B', '-u']
    scorer = ['/home/liuzhirui/miniconda3/envs/physvid/bin/python', '-I', '-B', '-u',
              bootstrap.BASE / 'tools/evaluate_videophy2.py']
    stage_name = 'starting'

    def stage(name, command, environment=None):
        nonlocal stage_name
        stage_name = name
        atomic_json(dict(status='running', stage=name, **plan), job / 'status.json')
        if environment is None:
            run_stage(command, job / f'{name}.log')
        else:
            run_stage(command, job / f'{name}.log', environment)

    try:
        for name, suite, indices in [('main40', 'videophy40', range(1, 41)),
                                     ('extra20', 'videophy60', range(41, 61))]:
            stage(name + '_generate', [*python, SOURCE / 'infer_videophy.py', '--checkpoint', checkpoint,
                  '--output', output, '--suite', suite, '--cases', *[f'VP{i:03d}' for i in indices]])
            stage(name + '_score', [*scorer, output / suite], evaluator_environment())
            stage(name + '_report', [*python, SOURCE / 'report.py'])
        record = job / 'merge_record.json'
        atomic_json(dict(completed_part=str(output / 'videophy40'), additional_part=str(output / 'videophy60'),
                         merge_target=str(output / 'full60/videophy60')), record)
        stage('merge60', [*python, ROOT / 'videophy60/snapshots/20261031T150803Z/merge_completed60.py', record])
        stage('final_report', [*python, SOURCE / 'report.py'])
        atomic_json(dict(status='completed', stage='completed', **plan), job / 'status.json')
    except BaseException as error:
        atomic_json(dict(status='failed', stage=stage_name, error=str(error), **plan), job / 'status.json')
        raise


if __name__ == '__main__':
    main()
