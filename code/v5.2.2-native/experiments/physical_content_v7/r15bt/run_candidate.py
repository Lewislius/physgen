"""Prepare evidence, fit interface, train 400 updates, score fixed40+20."""
from pathlib import Path
import argparse
import json
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap
from physical_content.common import (atomic_json, data_ready, gpu_guard,
                                     interface_ready, load_config)
from benchmark_inputs import reference
from pipeline import evaluator_environment, run_stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-id')
    parser.add_argument('--describe', action='store_true')
    parser.add_argument('--defer-extra20', action='store_true',
                        help='Review the fixed40 before running the disjoint20')
    parser.add_argument('--skip-train', action='store_true',
                        help='Evaluate an existing step0400 checkpoint only')
    args = parser.parse_args()
    cfg = load_config(args.config)
    source = Path(__file__).resolve().parent
    root = Path(cfg['paths']['runs']).parent
    experiment = os.getenv('DET_EXPERIMENT_ID', 'local')
    run_id = args.run_id or f'physical_content_{experiment}'
    run = root / 'runs' / run_id
    checkpoint = run / 'step0400.pt'
    output = root / 'inference_outputs' / run_id / 'step0400_fixed40_extra20_seed42'
    protocol = reference(cfg)
    extra = protocol['extra_ids']
    main_ids = [case['id'] for case in protocol['cases']
                if case['id'] not in extra]
    plan = dict(
        candidate='R15Bt', revision=cfg['revision'], source=str(source),
        run=run_id, checkpoint=str(checkpoint), data_revision=cfg['data_revision'],
        steps=400, samples=1600, generation_views_per_microbatch=1,
        native_lora_updates=cfg['stages']['joint'],
        physical_updates=sum(cfg['stages'].values()),
        training_stages=cfg['stages'],
        task='t2v', first_frame_used=False, known_latent_frames=0,
        training_sample_match='Same 1600 sample set as R15B; R18E/G filtered 1600-row loader and SampleOrder(seed42)',
        new_api_calls=0, new_model_downloads=0,
        main_cases=main_ids, extra_cases=extra, modes=['real'],
        automatic_next_candidate=False, extra20_enabled=not args.defer_extra20,
        output=str(output))
    if args.describe:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    folder = root / 'reports/jobs' / experiment
    gpu_guard(folder)
    atomic_json(plan, folder / 'plan.json')

    def stage(name, command, environment=None):
        atomic_json(dict(status=name, run=run_id, checkpoint=str(checkpoint)),
                    folder / 'status.json')
        run_stage(command, folder / f'{name}.log', environment)

    python = [sys.executable, '-I', '-B', '-u']
    # The existing v10 T2V processed cache is used as-is; no evidence build.
    stage('prepare_eval_evidence',
          [*python, source / 'prepare_inputs.py', '--config', args.config])
    if not interface_ready(cfg):
        stage('fit_interface',
              [*python, source / 'fit_interface.py', '--config', args.config])
    data_ready(cfg)
    if not checkpoint.exists() and not args.skip_train:
        command = [*python, source / 'train.py', '--config', args.config,
                   '--run-id', run_id, '--stop-after', '400']
        if (run / 'latest.pt').exists():
            command += ['--resume', str(run / 'latest.pt')]
        stage('training_to400', command)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    def evaluate(name, suite, ids):
        stage(name + '_generation', [
            *python, source / 'infer.py', '--checkpoint', str(checkpoint),
            '--output', str(output), '--suite', suite, '--cases', *ids])
        destination = output / suite
        stage(name + '_scoring', [
            '/home/liuzhirui/miniconda3/envs/physvid/bin/python', '-I', '-B', '-u',
            str(bootstrap.BASE / 'tools/evaluate_videophy2.py'), str(destination)],
            evaluator_environment())
        return destination

    evaluate('original40', 'videophy40', main_ids)
    command = [*python, source / 'report_results.py', '--output', str(output)]
    if not args.defer_extra20:
        evaluate('extra20', 'videophy60', extra)
        command.append('--require-extra20')
    else:
        command.append('--exclude-extra20')
    stage('report', command)
    atomic_json(dict(status='declared_evaluation_completed', **plan),
                folder / 'status.json')


if __name__ == '__main__':
    main()
