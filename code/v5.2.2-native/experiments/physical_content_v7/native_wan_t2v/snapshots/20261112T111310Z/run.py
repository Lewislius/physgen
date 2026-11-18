"""Generate and score the native Wan2.2-TI2V-5B T2V baseline on fixed 40+20."""
from pathlib import Path
import argparse
import json
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap  # noqa: F401

from physical_scaling.common import atomic_json, gpu_guard
from pipeline import evaluator_environment, run_stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-id',
                        help='Reuse an existing output directory (e.g. after a '
                             'scoring-stage fix) instead of native_t2v_<experiment>')
    args = parser.parse_args()
    from infer import read_cfg, protocol
    cfg = read_cfg(args.config)
    experiment = os.getenv('DET_EXPERIMENT_ID', 'local')
    root = Path(cfg['paths']['output']) / (args.run_id or f'native_t2v_{experiment}')
    reports = Path(__file__).resolve().parent / 'reports' / 'jobs' / experiment
    gpu_guard(reports)
    protocol_value = protocol(cfg)
    extra = protocol_value['extra_ids']
    main_ids = [case['id'] for case in protocol_value['cases']
                if case['id'] not in extra]
    atomic_json(dict(candidate='NativeWan2.2-TI2V-5B', task='t2v',
                     first_frame_used=False, main_cases=main_ids,
                     extra_cases=extra, output=str(root)), reports / 'plan.json')
    python = [sys.executable, '-I', '-B', '-u']
    scorer = '/home/liuzhirui/miniconda3/envs/physvid/bin/python'

    def stage(name, command, environment=None):
        atomic_json(dict(status=name), reports / 'status.json')
        run_stage(command, reports / f'{name}.log', environment)

    def evaluate(name, suite, ids):
        stage(name + '_generation', [
            *python, Path(__file__).resolve().parent / 'infer.py',
            '--config', str(args.config), '--suite', suite,
            '--cases', *ids, '--output', str(root)])
        stage(name + '_scoring', [
            scorer, '-I', '-B', '-u',
            str(bootstrap.BASE / 'tools/evaluate_videophy2.py'),
            str(root / suite)], evaluator_environment())

    evaluate('original40', 'videophy40', main_ids)
    evaluate('extra20', 'videophy60', extra)
    stage('report', [*python, Path(__file__).resolve().parent / 'report_results.py',
                     '--output', str(root), '--require-extra20'])
    atomic_json(dict(status='declared_evaluation_completed',
                     candidate='NativeWan2.2-TI2V-5B', task='t2v',
                     first_frame_used=False, output=str(root)),
                reports / 'status.json')


if __name__ == '__main__':
    main()
