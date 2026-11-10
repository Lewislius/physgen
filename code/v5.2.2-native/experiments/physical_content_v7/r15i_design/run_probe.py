"""R15B400 read-only fixed16 probe; no training or automatic candidate launch."""
from pathlib import Path
import argparse
import json
import os
import sys

SOURCE = Path(__file__).resolve().parent
ROOT = Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_v7/r15i_design')
B = ROOT.parent / 'request_bound_fusion_r15b'
B_SOURCE = B / 'snapshots/20261105T152223Z'
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(B_SOURCE))
import bootstrap
from physical_content.common import gpu_guard, atomic_json
from benchmark_inputs import reference
from pipeline import run_stage, evaluator_environment
from allocated_gpu import bind_allocated_gpu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--describe', action='store_true')
    args = parser.parse_args()
    experiment = os.getenv('DET_EXPERIMENT_ID', 'local')
    checkpoint = B / 'runs/physical_content_67829/step0400.pt'
    output = ROOT / 'probe_outputs' / experiment
    ids = [v['id'] for v in json.loads((SOURCE / 'subset16.json').read_text())['cases']]
    assert set(ids) <= {v['id'] for v in reference()['cases']}
    plan = dict(kind='R15B400_readonly_joint_physical_memory_probe',
                source=str(SOURCE), original_source=str(B_SOURCE), checkpoint=str(checkpoint),
                output=str(output), cases=ids, steps=50, shift=5, gamma=5, seed=42,
                training_updates=0, output_videos=16, new_api_calls=0, new_model_downloads=0,
                resource_pool='blk-96g', automatic_training=False,
                limitation='Not trained R15I scores; compare one declared read operator at fixed B weights.')
    if args.describe:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    folder = ROOT / 'reports/jobs' / experiment
    gpu_guard(folder)
    atomic_json(plan, folder / 'plan.json')
    bind_allocated_gpu(folder, evaluator_environment)

    def stage(name, command, environment=None):
        atomic_json(dict(status=name, **plan), folder / 'status.json')
        if environment is None:
            run_stage(command, folder / f'{name}.log')
        else:
            run_stage(command, folder / f'{name}.log', environment)

    python = [sys.executable, '-I', '-B', '-u']
    stage('generation', [*python, SOURCE / 'infer_probe.py', '--checkpoint', checkpoint,
                        '--output', output, '--suite', 'videophy40', '--cases', *ids])
    stage('scoring', ['/home/liuzhirui/miniconda3/envs/physvid/bin/python', '-I', '-B', '-u',
                      bootstrap.BASE / 'tools/evaluate_videophy2.py', output / 'videophy40'],
          evaluator_environment())
    stage('report', [*python, SOURCE / 'report_probe.py', '--output', output])
    stage('visual_panels', [*python, SOURCE / 'prepare_visual_review.py', '--output', output,
                           '--report', ROOT / 'reports' / experiment / 'visual_predeclared'])
    atomic_json(dict(status='completed', **plan), folder / 'status.json')


if __name__ == '__main__':
    main()
