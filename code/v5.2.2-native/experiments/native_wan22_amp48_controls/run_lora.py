"""Train native Wan2.2 video LoRA on fixed1600, then evaluate fixed 40+20."""
from pathlib import Path
import argparse
import json
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap  # noqa: F401

from native_control.common import atomic_json, gpu_guard_amp48, load_yaml
from pipeline import evaluator_environment, run_stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    experiment = os.getenv('DET_EXPERIMENT_ID', 'local')
    run = Path(cfg['paths']['run_dir'])
    run.mkdir(parents=True, exist_ok=True)
    output = Path(cfg['paths']['output']) / f'wan22_video_lora400_{experiment}'
    reports = Path(__file__).resolve().parent / 'reports' / 'jobs' / experiment
    reports.mkdir(parents=True, exist_ok=True)
    gpu_guard_amp48(reports)
    protocol = json.loads(Path(cfg['paths']['eval_protocol']).read_text())
    extra = protocol['extra_ids']
    primary = [case['id'] for case in protocol['cases'] if case['id'] not in extra]
    plan = dict(
        candidate='Wan2.2-TI2V-5B + rank32 video LoRA (400 updates)',
        task='t2v', source=str(Path(__file__).resolve().parent),
        train_data='exact R20 fixed1600 sample_sequence and VAE/text cache',
        updates=400, effective_batch=4, lora='rank32/alpha32 on 180 video targets',
        text_frozen=True, physical_modules=False, first_frame_used=False,
        main_cases=primary, extra_cases=extra, output=str(output),
        checkpoint=str(run / 'step0400.pt'))
    atomic_json(plan, reports / 'plan.json')
    python = [sys.executable, '-I', '-B', '-u']

    def stage(name, command, environment=None):
        atomic_json(dict(status=name, **plan), reports / 'status.json')
        run_stage(command, reports / f'{name}.log', environment)

    checkpoint = run / 'step0400.pt'
    if not checkpoint.is_file():
        command = [*python, str(Path(__file__).resolve().parent / 'train_lora.py'),
                   '--config', str(args.config)]
        latest = run / 'latest.pt'
        if latest.is_file():
            command += ['--resume', str(latest)]
        stage('train_lora400', command)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    scorer = '/home/liuzhirui/miniconda3/envs/physvid/bin/python'

    def evaluate(name, suite, ids):
        stage(name + '_generation', [
            *python, Path(__file__).resolve().parent / 'infer_lora.py',
            '--config', str(args.config), '--suite', suite,
            '--cases', *ids, '--output', str(output),
            '--lora-checkpoint', str(checkpoint)])
        stage(name + '_scoring', [
            scorer, '-I', '-B', '-u',
            str(bootstrap.BASE / 'tools/evaluate_videophy2.py'),
            str(output / suite)], evaluator_environment())

    evaluate('original40', 'videophy40', primary)
    evaluate('extra20', 'videophy60', extra)
    stage('report', [*python,
                     str(Path(__file__).resolve().parent / 'report_lora.py'),
                     '--output', str(output), '--require-extra20'])
    atomic_json(dict(status='declared_evaluation_completed', **plan),
                reports / 'status.json')


if __name__ == '__main__':
    main()
