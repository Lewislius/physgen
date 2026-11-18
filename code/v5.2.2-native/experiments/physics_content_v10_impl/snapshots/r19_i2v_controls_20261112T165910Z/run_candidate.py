"""Prepare, fit interface, train 400 updates, generate fixed40+20 and score."""
from pathlib import Path
import argparse
import json
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap  # noqa: F401

from benchmark_inputs import reference
from physical_content.common import atomic_json, data_ready, gpu_guard, load_config
from pipeline import evaluator_environment, run_stage

ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--describe', action='store_true')
    parser.add_argument('--skip-train', action='store_true')
    parser.add_argument('--defer-extra20', action='store_true')
    args = parser.parse_args(argv)
    cfg = load_config(args.config, require_paths=True)
    updates = int(cfg['train']['max_updates'])
    run = Path(cfg['paths']['run_dir'])
    checkpoint = run / f'step{updates:04d}.pt'
    output = ROOT / 'inference_outputs' / run.name / f'step{updates:04d}_fixed40_extra20_seed42'
    protocol = reference()
    extra_ids = protocol['extra_ids']
    main_ids = [case['id'] for case in protocol['cases'] if case['id'] not in extra_ids]
    plan = dict(
        candidate=cfg['variant_id'], plan=cfg['plan'], design_plan=cfg['design_plan'],
        revision=cfg['revision'], source=str(ROOT), task=cfg['task'],
        mechanism=cfg['mechanism'], run=str(run),
        checkpoint=str(checkpoint), data_revision=cfg['train'].get('data_revision'),
        updates=updates, effective_batch=4, samples=updates * 4,
        training_stages={'joint': updates}, generation_views_per_microbatch=1,
        main_cases=main_ids, extra_cases=extra_ids, modes=['real'],
        automatic_next_candidate=False, extra20_enabled=not args.defer_extra20,
        output=str(output), new_api_calls=0,
    )
    if args.describe:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    folders = ROOT / 'reports/jobs' / os.getenv('DET_EXPERIMENT_ID', 'local')
    folders.mkdir(parents=True, exist_ok=True)
    gpu_guard(folders)
    atomic_json(plan, folders / 'plan.json')
    python = [sys.executable, '-I', '-B', '-u']

    def stage(name, command, environment=None):
        atomic_json(dict(status=name, candidate=cfg['variant_id']), folders / 'status.json')
        run_stage(command, folders / f'{name}.log', environment)

    if cfg['task'] == 't2v':
        ready = Path(cfg['paths']['input_cache']).parent / 'ready.json'
        if not ready.is_file():
            stage('t2v_evidence', [*python, str(ROOT / 'prepare_t2v_evidence.py'),
                                   '--config', str(args.config),
                                   '--device', 'cuda'])
        native_ready = Path(cfg['paths']['native_words_cache']).parent / 'native_ready.json'
        if not native_ready.is_file():
            stage('t2v_native_targets', [*python, str(ROOT / 'prepare_native_t2v.py'),
                                         '--config', str(args.config),
                                         '--device', 'cuda'])
    interface = Path(cfg['paths']['interface'])
    if not interface.is_file():
        stage('fit_interface', [*python, str(ROOT / 'fit_interface.py'),
                                '--config', str(args.config)])
    stage('prepare_inputs', [*python, str(ROOT / 'prepare_inputs.py'),
                             '--config', str(args.config)])
    data_ready(cfg)
    if not checkpoint.exists() and not args.skip_train:
        command = [*python, str(ROOT / 'train.py'), '--config', str(args.config)]
        if (run / 'latest.pt').exists():
            command += ['--resume', str(run / 'latest.pt')]
        stage(f'training_to{updates}', command)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    def evaluate(name, suite, ids):
        stage(name + '_generation', [
            *python, str(ROOT / 'infer.py'), '--checkpoint', str(checkpoint),
            '--output', str(output), '--suite', suite, '--cases', *ids])
        destination = output / suite
        stage(name + '_scoring', [
            '/home/liuzhirui/miniconda3/envs/physvid/bin/python', '-I', '-B', '-u',
            str(bootstrap.BASE / 'tools/evaluate_videophy2.py'), str(destination)],
            evaluator_environment())
        return destination

    part40 = evaluate('original40', 'videophy40', main_ids)
    report_folders = [str(part40)]
    if not args.defer_extra20:
        part20 = evaluate('extra20', 'videophy60', extra_ids)
        merged = output / 'full60/videophy60'
        record = folders / 'merge_record.json'
        atomic_json(dict(
            checkpoint=str(checkpoint), completed_part=str(part40),
            additional_part=str(part20), merge_target=str(merged),
            new_videos=len(extra_ids), cases_after_merge=len(main_ids) + len(extra_ids),
            seed=42, gamma=5), record)
        merge = (bootstrap.BASE / 'experiments/physical_content_v7/videophy60/'
                 'snapshots/20261031T150803Z/merge_completed60.py')
        stage('full60_aggregate', [*python, str(merge), str(record)])
        report_folders.append(str(merged))
    stage('report', [*python, str(ROOT / 'reports.py'), '--output',
                     str(output / 'report.json'), '--folders', *report_folders])
    atomic_json(dict(status='declared_evaluation_completed', **plan),
                folders / 'status.json')


if __name__ == '__main__':
    main()
