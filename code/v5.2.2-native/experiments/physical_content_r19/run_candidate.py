"""R19 own400 -> fixed40/60 -> required paired16; no automatic cross benchmark."""
from pathlib import Path
import argparse
import json
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap
from benchmark_inputs import reference
from physical_content.common import atomic_json, data_ready, gpu_guard, load_config
from pipeline import evaluator_environment, run_stage

ROOT = Path(__file__).resolve().parent
PAIRED = ['VP002','VP004','VP005','VP009','VP010','VP013','VP015','VP016',
          'VP017','VP018','VP024','VP025','VP026','VP027','VP034','VP038']
CONTROLS = ['no_c','official_executor','reasoning_swapped','empty_reasoning','global_train_mean']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--describe', action='store_true')
    parser.add_argument('--skip-train', action='store_true')
    args = parser.parse_args()
    cfg = load_config(args.config, require_paths=True)
    run = Path(cfg['paths']['run_dir'])
    checkpoint = run / 'step0400.pt'
    # Output stays in the workspace, including when the code lives in a snapshot.
    workspace = run.parent.parent
    output = workspace / 'inference_outputs' / run.name / 'step0400_seed42'
    protocol = reference()
    extra = protocol['extra_ids']
    primary = [case['id'] for case in protocol['cases'] if case['id'] not in extra]
    plan = dict(candidate=cfg['variant_id'], task=cfg['task'], source=str(ROOT),
                checkpoint=str(checkpoint), updates=400, stages=cfg['stages'],
                training_samples=1600, main_cases=primary, extra_cases=extra,
                paired_cases=PAIRED, controls=CONTROLS, output=str(output),
                automatic_next_candidate=False, automatic_cross_benchmark=False,
                cross_benchmark_requires='I2V40+60 gains over best matched references AND actual-video acceptance',
                catastrophic_visual_failure_allowed=False)
    if args.describe:
        print(json.dumps(plan, ensure_ascii=False, indent=2)); return
    job = workspace / 'reports/jobs' / os.getenv('DET_EXPERIMENT_ID', 'local')
    job.mkdir(parents=True, exist_ok=True)
    gpu_guard(job)
    if cfg['status'] != 'ready_for_training':
        raise ValueError('Formal training requires the finalized configuration')
    atomic_json(plan, job / 'plan.json')
    python = [sys.executable, '-I', '-B', '-u']

    def stage(name, command, env=None):
        atomic_json(dict(status=name, **plan), job / 'status.json')
        run_stage(command, job / f'{name}.log', env)

    if not Path(cfg['paths']['interface']).is_file():
        stage('fit_interface', [*python, str(ROOT/'fit_interface.py'), '--config', str(args.config)])
    stage('prepare_inputs', [*python, str(ROOT/'prepare_inputs.py'), '--config', str(args.config)])
    data_ready(cfg)
    if not checkpoint.exists() and not args.skip_train:
        command = [*python, str(ROOT/'train.py'), '--config', str(args.config)]
        if (run/'latest.pt').exists():
            command += ['--resume', str(run/'latest.pt')]
        stage('train400', command)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    def score(name, folder):
        stage(name, ['/home/liuzhirui/miniconda3/envs/physvid/bin/python', '-I', '-B', '-u',
                     str(bootstrap.BASE/'tools/evaluate_videophy2.py'), str(folder)],
              evaluator_environment())

    for name, suite, ids in [('original40','videophy40',primary), ('extra20','videophy60',extra)]:
        stage(name+'_generation', [*python, str(ROOT/'infer.py'), '--checkpoint', str(checkpoint),
                                  '--output', str(output), '--suite', suite, '--cases', *ids])
        score(name+'_scoring', output/suite)
    # Preserve the extra20 separately; do not present it as full60.
    merge = job/'merge.json'
    atomic_json(dict(checkpoint=str(checkpoint), completed_part=str(output/'videophy40'),
                     additional_part=str(output/'videophy60'),
                     merge_target=str(output/'full60/videophy60'), seed=42, gamma=5), merge)
    stage('merge60', [*python, str(ROOT/'merge60.py'), str(merge)])
    stage('main_report', [*python, str(ROOT/'reports.py'), '--output', str(output/'main_scores.json'),
                         '--folders', str(output/'videophy40'), str(output/'full60/videophy60')])
    stage('visual_review_material', [*python, str(ROOT/'visual_review.py'), '--root', str(output)])
    atomic_json(dict(status='pending_actual_video_review', eligible=False,
                     reason='No WMB or Physics-IQ submission without both40/60 gains and visual acceptance'),
                output/'cross_benchmark_gate.json')
    atomic_json(dict(normal=str(output/'videophy40'), checkpoint=str(checkpoint),
                     task=cfg['task'], cases=PAIRED), output/'plan.json')
    control_args = (['--control-cache', str(bootstrap.BASE/'experiments/physical_content_v7/'
                                         'r15b/prepared_inputs')] if cfg['task']=='i2v' else [])
    stage('paired16_generation', [*python, str(ROOT/'infer.py'), '--checkpoint', str(checkpoint),
                                 '--output', str(output), '--suite', 'videophy40',
                                 '--controls', *CONTROLS, '--cases', *PAIRED, *control_args])
    for mode in CONTROLS:
        score(mode+'_scoring', output/'controls'/mode/'videophy40')
    stage('paired16_report', [*python, str(ROOT/'summarize_controls.py'), '--root', str(output)])
    atomic_json(dict(status='declared_videophy_evaluation_completed', **plan), job/'status.json')


if __name__ == '__main__':
    main()
