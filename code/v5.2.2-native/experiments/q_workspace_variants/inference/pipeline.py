"""Sequential demo21 -> VideoPhy2-40 -> independent scoring -> paired reports."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from jobs import JOBS, ROOT, NATIVE_ROOT, Q_REFERENCE
from compare import read, require, reference_groups, validate_manifest


def save(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temp.replace(path)


def require_amp80g():
    import torch
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
            'Run inside the allocated single-GPU amp-80g task')
    torch.cuda.set_device(0)
    props = torch.cuda.get_device_properties(0)
    require(props.total_memory >= 75*2**30, f'Expected amp-80g allocation, got {props.name}, {props.total_memory/2**30:.1f} GiB')
    return dict(name=props.name, memory_gib=props.total_memory/2**30, uuid=str(props.uuid))


def preflight(variant):
    import torch
    import generate
    from physgen_native.message_conditioning import MessageCondition
    job = JOBS[variant]
    state = generate.final_checkpoint(job['checkpoint'], variant, no_verifier_only=True)
    cfg = state['config']
    generate.check_resources(cfg)
    require(cfg['revision'] == job['revision'], 'Checkpoint/job revision differs')
    require(generate.training_seed(cfg, 42) == 42, 'Expected training seed 42')
    require(cfg['inference'] == {'steps': 50, 'shift': 5., 'guidance': 5., 'condition_strength': 1.},
            'Use the unchanged benchmark sampler')
    # Shape/name validation uses no GPU and does not materialize base-model weights.
    with torch.device('meta'):
        conditioner = MessageCondition(cfg)
    expected = dict(conditioner.named_parameters())
    stored = {key.removeprefix('conditioner.'): value for key, value in state['model'].items()
              if key.startswith('conditioner.')}
    require(set(expected) == set(stored), 'Checkpoint conditioner names differ from the dedicated variant loader')
    require(all(value.shape == stored[key].shape for key, value in expected.items()),
            'Checkpoint conditioner shapes differ')
    require(all(not key.startswith(('critic.', 'process_encoder.')) for key in state['model']),
            'Unexpected verifier/JEPA checkpoint parameters')
    demos = generate.collect_cases(generate.PREPARED_ROOT, cfg, list(generate.IDS), 42, 24.)
    cases, source = generate.collect_videophy_cases(generate.VIDEOPHY_ROOT, cfg, generate.VIDEOPHY_IDS, 24.)
    require(len(demos) == 21 and len(cases) == 40, 'Expected 21 + 40 cases')
    manifest = dict(cases=cases, settings=source['settings'], evaluation=source['evaluation'],
                    input_keys=list(generate.INPUT_KEYS), sampling=cfg['inference'], fps=24., frames=81,
                    noise_policy='saved per-case FP32 noise', checkpoint_step=224,
                    inference_reasoning_round=2, verifier={'kind': 'none'})
    validate_manifest(manifest)
    # Demo inputs, including the first latent and T5 conditions, match existing Q.
    demo_reference = read(Q_REFERENCE.parent / 'demo21_final224_round2_seed42/manifest.json')
    require(demos == demo_reference['cases'], 'Demo inputs differ from Q demo21')
    require(cfg['inference'] == demo_reference['sampling'], 'Demo sampler differs from Q')
    evaluation = source['evaluation']
    require(Path(evaluation['code'], 'template.py').is_file(), 'Missing existing VideoPhy2 evaluator source')
    evaluator = Path(evaluation['checkpoint'])
    for filename in ('config.json', 'tokenizer.model', 'preprocessor_config.json'):
        require((evaluator / filename).is_file(), f'Missing evaluator resource {filename}')
    index = evaluator / 'pytorch_model.bin.index.json'
    weights = set(read(index)['weight_map'].values()) if index.exists() else {'pytorch_model.bin'}
    require(all((evaluator / name).is_file() for name in weights), 'Missing evaluator weight shards')
    require(Path('/home/liuzhirui/miniconda3/envs/physvid/bin/python').is_file(), 'Missing existing evaluator interpreter')
    # The model definition imported above must be the immutable training snapshot.
    module_path = Path(sys.modules['physgen_native.message_conditioning'].__file__).resolve()
    require(module_path.is_relative_to(ROOT), 'Wrong model loader imported')
    result = dict(status='checked', **job, checkpoint_step=state['step'],
                  training_seed=42, inference_round=2, condition_slots=128,
                  demo_cases=21, videophy_cases=40, settings=source['settings'], evaluation=evaluation,
                  identical_Q_initial_conditions=True, loader=str(module_path),
                  conditioner_parameters=sum(v.numel() for v in expected.values()))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def evaluator_environment():
    env = dict(os.environ)
    prefix = '/home/liuzhirui/miniconda3/envs/physvid'
    for name in tuple(env):
        if name.startswith(('PYTHON', 'CONDA')) or name == 'VIRTUAL_ENV':
            env.pop(name)
    for name in ('PATH', 'LD_LIBRARY_PATH'):
        env[name] = ':'.join(p for p in env.get(name, '').split(':') if p and '/envs/' not in p)
    env['PATH'] = prefix + '/bin:' + env['PATH']
    env.update(CONDA_PREFIX=prefix, CONDA_DEFAULT_ENV='physvid', TOKENIZERS_PARALLELISM='false')
    return env


def run_stage(command, logfile, env=None):
    print('Starting: ' + ' '.join(map(str, command)), flush=True)
    with Path(logfile).open('a') as log:
        with subprocess.Popen(list(map(str, command)), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1, env=env) as process:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end='', flush=True)
            code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def pipeline(variant):
    device = require_amp80g()
    checked = preflight(variant)
    job = JOBS[variant]
    output = Path(job['output'])
    output.mkdir(exist_ok=True)
    # Protect reruns from concurrent writes to the same method output.
    import fcntl
    with (output / '.pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pinned = output / 'checkpoint.pt'
        if not pinned.exists():
            os.link(job['checkpoint'], pinned)
        import generate
        state = generate.final_checkpoint(pinned, variant, no_verifier_only=True)
        require(state['revision'] == checked['revision'], 'Pinned checkpoint variant differs')
        del state
        plan = dict(**checked, pinned_checkpoint=str(pinned), device=device,
                    inference_experiment_id=os.environ.get('DET_EXPERIMENT_ID'),
                    inference_trial_id=os.environ.get('DET_TRIAL_ID'))
        save(output / 'plan.json', plan)
        status = dict(status='running', variant=variant, device=device,
                      experiment_id=os.environ.get('DET_EXPERIMENT_ID'),
                      started_at=datetime.now(timezone.utc).isoformat(), completed_stages=[])
        save(output / 'pipeline_status.json', status)
        try:
            for suite, count in (('demo21', 21), ('videophy40', 40)):
                status['stage'] = suite
                save(output / 'pipeline_status.json', status)
                directory = output / suite
                command = [sys.executable, '-I', '-B', '-u', HERE / 'generate.py',
                           '--checkpoint', pinned, '--mode', variant, '--suite', suite,
                           '--training-experiment-id', str(job['training_experiment_id']),
                           '--output', directory, '--no-verifier-only', '--seed', '42',
                           '--fps', '24', '--steps', '50', '--shift', '5', '--guidance', '5']
                if directory.exists():
                    require((directory/'manifest.json').is_file() and (directory/'checkpoint.pt').is_file(),
                            f'Incomplete output initialization in {directory}')
                    command.append('--resume')
                run_stage(command, output / f'{suite}.log')
                result = read(directory / 'summary.json')
                require(result['status'] == 'completed' and result['completed_cases'] == count,
                        f'{suite}: generation incomplete')
                status['completed_stages'].append(suite)
            directory = output / 'videophy40'
            validate_manifest(read(directory / 'manifest.json'))
            status['stage'] = 'independent_autoeval'
            save(output / 'pipeline_status.json', status)
            # A fresh process frees all Wan/Qwen GPU allocations before scoring.
            run_stage(['/home/liuzhirui/miniconda3/envs/physvid/bin/python', '-I', '-B', '-u',
                       NATIVE_ROOT / 'tools/evaluate_videophy2.py', directory],
                      output / 'evaluation.log', env=evaluator_environment())
            status['completed_stages'].append('independent_autoeval')
            status['stage'] = 'comparison'
            save(output / 'pipeline_status.json', status)
            from compare import compare
            result = compare(directory, variant)
            status.update(status='completed', stage='completed',
                          completed_stages=[*status['completed_stages'], 'comparison'],
                          finished_at=datetime.now(timezone.utc).isoformat(),
                          demo_completed=21, videophy_completed=40, scalar_scores=80,
                          scores=result['summaries'][0], comparison=str(directory / 'comparison.json'))
        except BaseException as exc:
            status.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            save(output / 'pipeline_status.json', status)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', required=True, choices=JOBS)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.check_only:
        preflight(args.variant)
    else:
        pipeline(args.variant)
