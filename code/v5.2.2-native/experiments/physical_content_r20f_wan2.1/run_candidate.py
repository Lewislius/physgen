"""R20FT-Wan2.1 staged runner.

Resource split:
  ada-48g : --stage prepare-check  (cache, normalization, full-size GPU check)
  blk-96g : --stage train-eval     (formal 400 steps + fixed 40/60 + paired 16)

The train stage refuses to start unless the ada-48g check marker exists, passed,
and matches this exact design hash.
"""
from pathlib import Path
import argparse
import json
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap
from benchmark_inputs import reference
from physical_content.common import (atomic_json, data_ready, design_hash,
                                     gpu_guard, load_config)
from pipeline import evaluator_environment, run_stage

ROOT = Path(__file__).resolve().parent
PAIRED = ['VP002','VP004','VP005','VP009','VP010','VP013','VP015','VP016',
          'VP017','VP018','VP024','VP025','VP026','VP027','VP034','VP038']
CONTROLS = ['no_c','reasoning_swapped','video_binding_off']
STAGES = ('prepare', 'check', 'prepare-check', 'train', 'eval',
          'train-eval', 'all')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--stage', choices=STAGES, default='all')
    parser.add_argument('--gpu-policy', choices=('any48', 'ada48', 'rtx6000'),
                        default=None)
    parser.add_argument('--describe', action='store_true')
    parser.add_argument('--skip-train', action='store_true')
    args = parser.parse_args()
    cfg = load_config(args.config, require_paths=True)
    controls = CONTROLS
    run = Path(cfg['paths']['run_dir'])
    checkpoint = run / 'step0400.pt'
    workspace = run.parent.parent
    output = workspace / 'inference_outputs' / run.name / 'step0400_seed42'
    check_marker = workspace / 'preparation' / 'full_model_check.json'
    policy = args.gpu_policy or cfg['train'].get('gpu_policy', 'any48')
    protocol = reference()
    extra = protocol['extra_ids']
    primary = [case['id'] for case in protocol['cases'] if case['id'] not in extra]
    plan = dict(candidate=cfg['variant_id'], task=cfg['task'], source=str(ROOT),
                checkpoint=str(checkpoint), updates=400, stages=cfg['stages'],
                training_samples=1600, main_cases=primary, extra_cases=extra,
                paired_cases=PAIRED, controls=controls, output=str(output),
                stage=args.stage, gpu_policy=policy,
                check_marker=str(check_marker),
                automatic_next_candidate=False, automatic_cross_benchmark=False,
                cross_benchmark_requires='40+60 gains and actual-video acceptance',
                catastrophic_visual_failure_allowed=False)
    if args.describe:
        print(json.dumps(plan, ensure_ascii=False, indent=2)); return
    job = workspace / 'reports/jobs' / os.getenv('DET_EXPERIMENT_ID', 'local')
    job.mkdir(parents=True, exist_ok=True)
    gpu_guard(job, policy)
    if cfg['status'] != 'ready_for_training':
        raise ValueError('Formal training requires the finalized configuration')
    atomic_json(plan, job / 'plan.json')
    python = [sys.executable, '-I', '-B', '-u']

    def stage(name, command, env=None):
        atomic_json(dict(status=name, **plan), job / 'status.json')
        run_stage(command, job / f'{name}.log', env)

    def cache_counts():
        cache = Path(cfg['paths']['cache'])
        return (len(list((cache / 'base/vae').glob('*.pt'))),
                len(list(Path(cfg['paths']['native_words_cache']).glob('*.pt'))),
                len(list(Path(cfg['paths']['inference_evidence']).glob('*/positive.pt'))))

    def prepare():
        if cache_counts() != (1600, 3320, 60):
            stage('prepare_wan21_cache', [
                *python, str(ROOT / 'prepare_wan21_cache.py'),
                '--config', str(args.config), '--stage', 'all'])
            if cache_counts() != (1600, 3320, 60):
                raise RuntimeError(
                    f'Wan2.1 cache incomplete after preparation: {cache_counts()}')
        if not Path(cfg['paths']['interface']).is_file():
            stage('fit_interface', [*python, str(ROOT/'fit_interface.py'),
                                    '--config', str(args.config)])
        stage('prepare_inputs', [*python, str(ROOT/'prepare_inputs.py'),
                                 '--config', str(args.config)])
        data_ready(cfg)

    def check_ok():
        if not check_marker.is_file():
            return False
        try:
            value = json.loads(check_marker.read_text())
        except (OSError, ValueError):
            return False
        return (value.get('passed') is True
                and value.get('design_hash') == design_hash(cfg)
                and value.get('model') == cfg['base']['model']
                and list(value.get('latent_shape', [])) == list(cfg['data']['latent_shape']))

    def check():
        if cache_counts() != (1600, 3320, 60):
            raise RuntimeError('Run --stage prepare on ada-48g before the full-size check')
        data_ready(cfg)
        if not check_ok():
            stage('full_model_gradient_check', [
                *python, str(ROOT/'check_full_model.py'),
                '--config', str(args.config), '--output', str(check_marker),
                '--gpu-policy', 'ada48' if policy in ('any48', 'ada48') else policy])
        if not check_ok():
            raise RuntimeError(
                'Full-size check did not produce a matching passed marker; '
                'formal training on blk-96g is refused')

    def train():
        if args.skip_train:
            return
        if not check_ok():
            raise RuntimeError(
                'No matching ada-48g full-model check marker. Run '
                '--stage prepare-check on ada-48g before formal blk-96g training.')
        data_ready(cfg)
        if checkpoint.exists():
            return
        command = [*python, str(ROOT/'train.py'), '--config', str(args.config)]
        if (run/'latest.pt').exists():
            command += ['--resume', str(run/'latest.pt')]
        stage('train400', command)

    def evaluate():
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)

        def score(name, folder):
            stage(name, ['/home/liuzhirui/miniconda3/envs/physvid/bin/python',
                         '-I', '-B', '-u',
                         str(bootstrap.BASE/'tools/evaluate_videophy2.py'), str(folder)],
                  evaluator_environment())

        for name, suite, ids in [('original40','videophy40',primary),
                                 ('extra20','videophy60',extra)]:
            stage(name+'_generation', [*python, str(ROOT/'infer.py'),
                                      '--checkpoint', str(checkpoint),
                                      '--output', str(output), '--suite', suite,
                                      '--cases', *ids])
            score(name+'_scoring', output/suite)
        merge = job/'merge.json'
        atomic_json(dict(checkpoint=str(checkpoint),
                         completed_part=str(output/'videophy40'),
                         additional_part=str(output/'videophy60'),
                         merge_target=str(output/'full60/videophy60'),
                         seed=42, gamma=5), merge)
        stage('merge60', [*python, str(ROOT/'merge60.py'), str(merge)])
        stage('main_report', [*python, str(ROOT/'reports.py'),
                              '--output', str(output/'main_scores.json'),
                              '--folders', str(output/'videophy40'),
                              str(output/'full60/videophy60')])
        stage('visual_review_material', [*python, str(ROOT/'visual_review.py'),
                                          '--root', str(output)])
        atomic_json(dict(status='pending_actual_video_review', eligible=False,
                         reason='No external benchmark submission without both 40/60 gains '
                                'and actual-video acceptance'),
                    output/'cross_benchmark_gate.json')
        atomic_json(dict(normal=str(output/'videophy40'), checkpoint=str(checkpoint),
                         task=cfg['task'], cases=PAIRED), output/'plan.json')
        for mode in controls:
            stage(mode+'_generation', [*python, str(ROOT/'infer.py'),
                                      '--checkpoint', str(checkpoint),
                                      '--output', str(output), '--suite', 'videophy40',
                                      '--controls', mode, '--cases', *PAIRED])
            score(mode+'_scoring', output/'controls'/mode/'videophy40')
            stage('paired16_report', [*python, str(ROOT/'summarize_controls.py'),
                                     '--root', str(output)])
        atomic_json(dict(status='declared_videophy_evaluation_completed', **plan),
                    job/'status.json')

    if args.stage == 'prepare':
        prepare()
    elif args.stage == 'check':
        check()
    elif args.stage == 'prepare-check':
        prepare()
        check()
    elif args.stage == 'train':
        train()
    elif args.stage == 'eval':
        evaluate()
    elif args.stage == 'train-eval':
        train()
        evaluate()
    elif args.stage == 'all':
        prepare()
        check()
        train()
        evaluate()
    else:
        raise ValueError(args.stage)


if __name__ == '__main__':
    main()
