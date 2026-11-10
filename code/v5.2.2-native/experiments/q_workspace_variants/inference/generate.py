"""Generate the existing demo21 or fixed VideoPhy2-40 inputs from native weights.

Reuse only the existing first-frame VAE and native T5 inputs. Qwen analysis,
message conditioning and Wan sampling run afresh with the selected checkpoint.
"""
import argparse
import copy
from datetime import datetime, timezone
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
ROOT = WORKSPACE_ROOT.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT))
import torch
import yaml
from PIL import Image
from physgen_native.config import validate, select_reasoning_exit
from physgen_native.data import move

IDS = ('reference', *(f'P{i:02d}' for i in range(1, 21)))
INPUT_KEYS = ('first', 'text', 'null_text', 'times', 'valid')
# Snapshot of the established H/Q benchmark runner; model imports resolve to
# this experiment's isolated implementation, including Q revisions v31-v35.
from physgen_native.q_variants import Q_WORKSPACE_VARIANTS
VARIANTS = {spec['workspace_variant']: revision
            for revision, spec in Q_WORKSPACE_VARIANTS.items()
            if spec['workspace_variant'] in ('b', 'c', 'd-s', 'd-i')}
PREPARED_ROOT = ROOT.parent / 'v5.2/inference_outputs/1d-v52-lora-demo21_20261010'
VIDEOPHY_ROOT = ROOT.parent / 'v5.2/inference_outputs/videophy2_lora_run64543_joint400_60_seed42'
VIDEOPHY_IDS = ROOT / 'analysis/videophy40_20261026/case_ids.json'


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def final_checkpoint(path, mode, *, no_verifier_only=False):
    state = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    cfg = validate(state['config'])
    revision = VARIANTS[mode]
    if state['revision'] != revision or cfg['revision'] != revision:
        raise ValueError(f'{mode} requires {revision}, got {state["revision"]}')
    if (cfg['verifier'] != {'kind': 'none'} or 'verifier_state' in state or
            cfg['inference'].get('latent_feedback', {}).get('enabled', False)):
        raise ValueError('Workspace inference requires the original no-verifier design')
    if state['step'] != 224 or state['step'] != sum(cfg['stages'].values()):
        raise ValueError('Use the final 224-step checkpoint')
    if (cfg['knowledge']['inference_reasoning_round'] != 2 or
            cfg['knowledge']['train_reasoning_rounds'] != [2]):
        raise ValueError('Use the trained R0 + two refinements')
    # Do not read optimizer tensor storage during generation.
    return {key: state[key] for key in ('revision', 'config', 'step', 'cursor', 'model', 'verifier_state') if key in state}


def select_ids(value):
    ids = list(IDS) if value == 'all' else [part.strip() for part in value.split(',')]
    if not ids or len(ids) != len(set(ids)) or not set(ids) <= set(IDS):
        raise ValueError('Use all or distinct IDs from reference,P01,...,P20')
    return ids


def training_seed(cfg, requested):
    seed = cfg['train']['seed']
    if requested is not None and requested != seed:
        raise ValueError(f'Inference seed must match checkpoint training seed {seed}, got {requested}')
    return seed


def load_inputs(folder, cfg, fps):
    # mmap avoids materializing historical task/JEPA-related storage. Only this
    # allowlist is copied or transferred; no future latent/video is a condition.
    stored = torch.load(folder / 'conditions.pt', map_location='cpu', weights_only=True, mmap=True)
    inputs = {key: stored[key].float().clone() for key in INPUT_KEYS}
    channels, length, height, width = cfg['data']['latent_shape']
    expected = {'first': (1, channels, 1, height, width), 'times': (length,), 'valid': (height, width)}
    for key, shape in expected.items():
        if tuple(inputs[key].shape) != shape:
            raise ValueError(f'{folder}: incompatible {key} geometry {tuple(inputs[key].shape)} != {shape}')
    for key in ('text', 'null_text'):
        tensor = inputs[key]
        if tensor.ndim != 3 or tensor.shape[0] != 1 or not 0 < tensor.shape[1] <= 512 or tensor.shape[2] != 4096:
            raise ValueError(f'{folder}: invalid native T5 {key} shape')
    if any(not bool(torch.isfinite(value).all()) for value in inputs.values()):
        raise ValueError(f'{folder}: nonfinite input')
    if not torch.allclose(inputs['times'], torch.arange(length).float() * (4 / fps), atol=1e-5, rtol=0):
        raise ValueError(f'{folder}: timestamp/FPS mismatch')
    # Historical fractional coverage has FP32 roundoff just above one.
    if not bool(((inputs['valid'] >= -1e-5) & (inputs['valid'] <= 1 + 1e-5)).all()) or not bool(inputs['valid'].any()):
        raise ValueError(f'{folder}: invalid canvas coverage')
    return inputs


def collect_cases(source, cfg, ids, seed, fps):
    available = {case['case_id']: case for case in json.loads((source / 'cases.json').read_text())}
    cases = []
    for cid in ids:
        original = available[cid]
        folder = (source / original['output']).with_suffix('.inputs')
        info = json.loads((folder / 'input.json').read_text())
        requirements = json.loads((folder / 'requirements.json').read_text())
        if requirements.get('input_scope') != 'caption_and_first_image':
            raise ValueError(f'{cid}: requirements must originate from caption and first image only')
        texts = requirements.get('requirements')
        if not isinstance(texts, list) or any(not isinstance(x, str) or not x.strip() for x in texts):
            raise ValueError(f'{cid}: invalid requirements')
        caption = info['prompt']
        if not caption.strip() or caption != original['prompt']:
            raise ValueError(f'{cid}: caption differs from the prepared manifest')
        if cid != 'reference':
            prompt = (ROOT.parent / 'v1/demo' / cid / f'{cid}-origin.txt').read_text(encoding='utf-8-sig').strip()
            if prompt != caption:
                raise ValueError(f'{cid}: prepared caption differs from the original demo')
        source_cfg = yaml.safe_load((folder / 'config.yaml').read_text())
        for key in ('wan_checkpoint', 'qwen_checkpoint'):
            if Path(source_cfg['paths'][key]).resolve() != Path(cfg['paths'][key]).resolve():
                raise ValueError(f'{cid}: prepared inputs use a different {key}')
        geometry = info['geometry']
        if ([geometry['h'], geometry['w']] != cfg['geometry']['rgb_size'] or
                info['frames'] != cfg['data']['frames'] or geometry['frames'] != cfg['data']['frames'] or
                info['fps'] != fps):
            raise ValueError(f'{cid}: prepared geometry/FPS differs from the requested native generation')
        for path in (Path(info['image']), folder / 'canvas.png'):
            with Image.open(path) as image:
                image.verify()
                if path.name == 'canvas.png' and image.size != (geometry['w'], geometry['h']):
                    raise ValueError(f'{cid}: wrong canvas size')
        load_inputs(folder, cfg, fps)
        cases.append(dict(id=cid, caption=caption, image=info['image'], prepared_inputs=str(folder),
                          geometry=geometry, seed=seed, output=f'{cid}/i2v/seed{seed}.mp4',
                          clauses=[dict(text=x, basis='both', phase='evolution') for x in texts]))
    return cases


def load_initial_noise(case, cfg):
    """Replay the saved comparison noise, without consulting any reference video."""
    if 'initial_noise' not in case:
        return None
    noise = torch.load(case['initial_noise'], map_location='cpu', weights_only=True, mmap=True)
    if list(noise.shape) != [1, *cfg['data']['latent_shape']] or noise.dtype != torch.float32:
        raise ValueError(f'{case["id"]}: incompatible saved FP32 video noise')
    if not bool(torch.isfinite(noise).all()):
        raise ValueError(f'{case["id"]}: nonfinite saved video noise')
    return noise


def collect_videophy_cases(source, cfg, ids_file, fps):
    original = json.loads((source / 'manifest.json').read_text())
    ids = json.loads(Path(ids_file).read_text())
    if len(ids) != 40 or len(set(ids)) != 40 or any(x != f'VP{i:03d}' for i, x in enumerate(ids, 1)):
        raise ValueError('Use the existing fixed VP001-VP040 selection')
    settings = original['settings']
    if ([settings['height'], settings['width']] != cfg['geometry']['rgb_size'] or
            settings['frames'] != cfg['data']['frames'] or settings['fps'] != fps):
        raise ValueError('VideoPhy2 source geometry/FPS differs from the checkpoint')
    available = {case['id']: case for case in original['cases']}
    cases = []
    for cid in ids:
        case = dict(available[cid])
        folder = source / 'cases' / cid
        info = json.loads((folder / 'input.json').read_text())
        requirements = json.loads((folder / 'requirements.json').read_text())
        if info['prompt'] != case['caption'] or requirements['prompt'] != case['caption']:
            raise ValueError(f'{cid}: saved caption differs from the manifest')
        if requirements.get('input_scope', 'caption_and_first_image') != 'caption_and_first_image':
            raise ValueError(f'{cid}: requirements must use only caption and first image')
        if Path(requirements['first_image']).resolve() != Path(info['image']).resolve():
            raise ValueError(f'{cid}: requirements and latent have different initial images')
        texts = requirements['requirements']
        if not isinstance(texts, list) or any(not isinstance(x, str) or not x.strip() for x in texts):
            raise ValueError(f'{cid}: invalid requirements')
        geometry = info['geometry']
        if ([geometry['h'], geometry['w']] != cfg['geometry']['rgb_size'] or
                info['frames'] != cfg['data']['frames'] or info['fps'] != fps):
            raise ValueError(f'{cid}: saved input geometry/FPS mismatch')
        with Image.open(folder / 'first.png') as image:
            image.verify()
            if image.size != (geometry['w'], geometry['h']):
                raise ValueError(f'{cid}: wrong initial image size')
        load_inputs(folder, cfg, fps)
        case.update(image=info['image'], prepared_inputs=str(folder), geometry=geometry,
                    initial_noise=str(folder / 'noise.pt'), canvas=str(folder / 'first.png'),
                    output=f'cases/{cid}/planned.mp4',
                    clauses=[dict(text=x, basis='both', phase='evolution') for x in texts])
        load_initial_noise(case, cfg)
        cases.append(case)
    return cases, original


def check_resources(cfg):
    for key, index_name in (('wan_checkpoint', 'diffusion_pytorch_model.safetensors.index.json'),
                            ('qwen_checkpoint', 'model.safetensors.index.json')):
        folder = Path(cfg['paths'][key])
        mapping = json.loads((folder / index_name).read_text())['weight_map']
        for shard in set(mapping.values()):
            if not (folder / shard).is_file():
                raise FileNotFoundError(folder / shard)
    vae = Path(cfg['paths']['wan_checkpoint']) / 'Wan2.2_VAE.pth'
    if not vae.is_file():
        raise FileNotFoundError(vae)
    for program in ('ffmpeg', 'ffprobe'):
        if shutil.which(program) is None:
            raise RuntimeError(f'{program} must be available in the existing runtime')


def verify_video(path, cfg, fps):
    result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
        '-show_entries', 'stream=width,height,nb_read_frames,avg_frame_rate', '-of', 'json', str(path)],
        check=True, capture_output=True, text=True)
    stream, = json.loads(result.stdout)['streams']
    height, width = cfg['geometry']['rgb_size']
    if (stream['width'], stream['height'], int(stream['nb_read_frames'])) != (width, height, cfg['data']['frames']):
        raise ValueError(f'{path}: encoded frame count or geometry is incorrect')
    if not math.isclose(float(Fraction(stream['avg_frame_rate'])), fps, rel_tol=1e-6):
        raise ValueError(f'{path}: encoded FPS differs')
    subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(path), '-f', 'null', '-'],
                   check=True, capture_output=True)
    return stream


@torch.no_grad()
def sample_staged(model, sample, seed, initial_noise=None):
    """Execute the same CFG conditions/trajectory with one backbone resident.

    Only inference residency changes. Positive and null C are calculated by the
    checkpoint's original encoders before Wan is moved onto the allocated GPU.
    """
    device = sample['first'].device
    model.generator.cpu()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    model.knowledge.to(device)
    model.conditioner.to(device)
    conditions = model.conditions(sample)
    first_rgb = model.decoder(conditions['first'], [0])
    positive = model.encode_condition(conditions, first_rgb)
    analysis = model.last_condition_analysis
    reasoning_round = model.last_condition_round
    if model.cfg['inference']['guidance'] == 1:
        negative = positive
    else:
        negative = model.encode_condition(dict(conditions, caption='', clauses=[]), first_rgb)
    model.knowledge.cpu()
    model.conditioner.cpu()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    model.generator.to(device)
    latent = model.sample(conditions, sample['text'], sample['null_text'], seed=seed,
                          token_override=positive, negative_token_override=negative, initial_noise=initial_noise)
    # Explicit tokens here are normal conditioning computed immediately above,
    # not a condition ablation. Preserve the actual analysis and message exit.
    model.last_sample_analysis = analysis
    model.last_sample_reasoning_round = reasoning_round
    model.last_sample_condition_tokens = positive.shape[1]
    model.generator.cpu()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return latent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--mode', choices=VARIANTS, required=True)
    parser.add_argument('--suite', choices=('demo21', 'videophy40'), default='demo21')
    parser.add_argument('--prepared-root', type=Path)
    parser.add_argument('--case-ids-file', type=Path, default=VIDEOPHY_IDS)
    parser.add_argument('--training-experiment-id', type=int)
    parser.add_argument('--no-verifier-only', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--sample-ids', default='all')
    parser.add_argument('--seed', type=int, help='Must equal the checkpoint training seed; default reads that seed')
    parser.add_argument('--fps', type=float, default=24.)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--shift', type=float)
    parser.add_argument('--guidance', type=float)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Resume the same manifest and pinned weights; verify completed videos')
    parser.add_argument('--exclude-gpu-uuid', action='append', default=[],
                        help='Reject an allocation with recorded repeated SIGKILL failures before loading models')
    parser.add_argument('--cpu-offload', action='store_true',
                        help='Stage Qwen, Wan and VAE work for 40/48GB inference without changing the sampler')
    args = parser.parse_args(argv)
    if not args.no_verifier_only:
        raise ValueError('Workspace generation always requires --no-verifier-only')
    source_checkpoint = args.checkpoint.expanduser().resolve()
    source = (args.prepared_root or (PREPARED_ROOT if args.suite == 'demo21' else VIDEOPHY_ROOT)).expanduser().resolve()
    if args.resume and args.output is None:
        parser.error('--resume requires --output')
    output = args.output.expanduser().resolve() if args.output else None
    checkpoint = output / 'checkpoint.pt' if args.resume else source_checkpoint
    state = final_checkpoint(checkpoint, args.mode, no_verifier_only=args.no_verifier_only)
    cfg = state['config']
    args.seed = training_seed(cfg, args.seed)
    if output is None:
        output = ROOT / 'inference_outputs' / (
            f'{source_checkpoint.parent.name}_{args.suite}_step0224_seed{args.seed}_'
            f'{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{os.getpid()}')
    policy = copy.deepcopy(cfg['inference'])
    for key in ('steps', 'shift', 'guidance'):
        if getattr(args, key) is not None:
            policy[key] = getattr(args, key)
    if (policy['steps'] <= 0 or not math.isfinite(policy['shift']) or policy['shift'] <= 0 or
            not math.isfinite(policy['guidance']) or policy['guidance'] <= 0 or
            not math.isfinite(args.fps) or args.fps <= 0 or not 0 <= args.seed < 2**63):
        raise ValueError('Invalid sampling settings')
    check_resources(cfg)
    if args.suite == 'demo21':
        cases = collect_cases(source, cfg, select_ids(args.sample_ids), args.seed, args.fps)
        original = None
    else:
        if args.sample_ids != 'all':
            raise ValueError('VideoPhy2 uses the fixed 40-case list, not --sample-ids')
        cases, original = collect_videophy_cases(source, cfg, args.case_ids_file, args.fps)
        if any(policy[k] != original['settings'][k] for k in ('steps', 'shift', 'guidance', 'condition_strength')):
            raise ValueError('VideoPhy2 sampling must match the saved comparison settings')
    round_index = cfg['knowledge']['inference_reasoning_round']
    manifest = dict(source_checkpoint=str(source_checkpoint), checkpoint_step=state['step'], revision=state['revision'],
        mode=cfg['knowledge']['reasoning_mode'], workspace_variant=args.mode, suite=args.suite, training_experiment_id=args.training_experiment_id,
        prepared_root=str(source), input_keys=list(INPUT_KEYS), verifier=cfg['verifier'],
        training_reasoning_rounds=cfg['knowledge']['train_reasoning_rounds'], inference_reasoning_round=round_index,
        message_block_calls_per_condition=round_index + 1,
        frames=cfg['data']['frames'], fps=args.fps, sampling=policy, seed=args.seed,
        training_seed=cfg['train']['seed'], cases=cases,
        noise_policy='saved per-case FP32 noise' if original else 'CUDA RNG with training seed')
    if original:
        manifest.update(settings=original['settings'], evaluation=original['evaluation'],
                        case_ids_file=str(args.case_ids_file.resolve()),
                        scope='Fixed 40 VideoPhy2 captions adapted to shared-first-image I2V; not official T2V leaderboard')
    if args.check_only:
        print(json.dumps(dict(status='checked', total_cases=len(cases), cpu_offload=args.cpu_offload, **manifest), ensure_ascii=False, indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError('Run inside an allocated GPU job, or use --check-only for CPU preflight')
    torch.cuda.set_device(0)
    from tools.run_resumable_training import check_allocated_device
    check_allocated_device(args.exclude_gpu_uuid)
    if args.resume:
        if json.loads((output / 'manifest.json').read_text()) != manifest:
            raise ValueError('Output belongs to different weights, inputs or sampling settings')
    else:
        output.mkdir(parents=True, exist_ok=False)
        # Pin the final training inode without copying or changing the checkpoint.
        os.link(source_checkpoint, output / 'checkpoint.pt')
        state = final_checkpoint(output / 'checkpoint.pt', args.mode, no_verifier_only=args.no_verifier_only)
        if state['config'] != cfg:
            raise ValueError('Checkpoint changed during preflight')
        cfg = state['config']
        save_json(output / 'manifest.json', manifest)
        (output / 'checkpoint_config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    summary = dict(status='loading_model', total_cases=len(cases), completed_cases=0, cpu_offload=args.cpu_offload,
                   suite=args.suite, mode=args.mode, training_experiment_id=args.training_experiment_id,
                   checkpoint=str(output / 'checkpoint.pt'), experiment_id=os.environ.get('DET_EXPERIMENT_ID'),
                   device=torch.cuda.get_device_name(0), device_uuid=str(torch.cuda.get_device_properties(0).uuid),
                   device_gib=torch.cuda.get_device_properties(0).total_memory / 2**30,
                   started_at=datetime.now(timezone.utc).isoformat(), cases=[])
    save_json(output / 'summary.json', summary)
    print(f'[{args.suite}] {args.mode}, final step {state["step"]}, round {round_index}, {len(cases)} cases -> {output}', flush=True)
    try:
        from physgen_native.generator import load_wan
        from physgen_native.system import PhysGenNative
        from physgen_native.runtime import restore
        from physgen_native.runtime import seed_all
        from train.train import export_video
        seed_all(args.seed)
        model = PhysGenNative(cfg, load_wan(cfg), device='cuda', training_observers=False)
        if args.cpu_offload:
            model.conditioner.cuda()
        else:
            model.cuda()
        completed, _ = restore(model, None, None, state, cfg)
        del state
        model.eval().requires_grad_(False)
        if args.no_verifier_only and (model.critic is not None or model.process_encoder is not None):
            raise RuntimeError('No-verifier inference must not construct a judge or JEPA encoder')
        # Apply sampling settings only after exact restoration. Learned C affine
        # and inherited Wan LoRA come from this checkpoint, never recalibration.
        cfg['inference'] = policy
        select_reasoning_exit(cfg, reasoning_round=round_index)
        summary['status'] = 'running'
        for index, case in enumerate(cases, 1):
            path = output / case['output']
            result = dict(id=case['id'], output=str(path), status='running')
            summary['cases'].append(result)
            save_json(output / 'summary.json', summary)
            if args.resume and path.is_file() and path.with_suffix('.json').is_file():
                metadata = json.loads(path.with_suffix('.json').read_text())
                expected = dict(id=case['id'], revision=cfg['revision'], optimizer_step=completed,
                                seed=case['seed'], latent_reasoning_steps=round_index, sampler=policy)
                if any(metadata.get(key) != value for key, value in expected.items()):
                    raise ValueError(f'{path}: saved video metadata differs from this run')
                result.update(status='completed', resumed=True, encoded_video=verify_video(path, cfg, args.fps))
            else:
                start = time.monotonic()
                torch.cuda.reset_peak_memory_stats()
                folder = Path(case['prepared_inputs'])
                inputs = load_inputs(folder, cfg, args.fps)
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(inputs, path.parent / 'conditions.pt')
                shutil.copyfile(case.get('canvas', folder / 'canvas.png'), path.parent / 'first_frame.png')
                save_json(path.parent / 'input.json', case)
                sample = move(dict(inputs, id=case['id'], caption=case['caption'], clauses=case['clauses']), 'cuda')
                noise = load_initial_noise(case, cfg)
                calls = [0]
                def progress(module, arguments, result):
                    calls[0] += 1
                    branches = 1 if policy['guidance'] == 1 else 2
                    if calls[0] % (10 * branches) == 0:
                        print(f'[{args.suite}] {index}/{len(cases)} {case["id"]} sampling {calls[0] // branches}/{policy["steps"]}; '
                              f'peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB', flush=True)
                hook = model.generator.register_forward_hook(progress)
                try:
                    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
                        latent = (sample_staged(model, sample, case['seed'], initial_noise=noise) if args.cpu_offload else
                                  model.sample(model.conditions(sample), sample['text'], sample['null_text'],
                                               seed=case['seed'], initial_noise=noise))
                    if not bool(torch.isfinite(latent).all()) or model.last_sample_reasoning_round != round_index:
                        raise RuntimeError('Nonfinite latent or unexpected actual message exit')
                    if model.last_sample_condition_tokens != 128:
                        raise RuntimeError('Q must retain its trained 128-slot workspace')
                    export_video(model, latent, sample, path, completed, case['seed'])
                    metadata = json.loads(path.with_suffix('.json').read_text())
                    metadata.update(checkpoint=str(output / 'checkpoint.pt'), source_checkpoint=str(source_checkpoint),
                        mode=cfg['knowledge']['reasoning_mode'], workspace_variant=args.mode, suite=args.suite, training_experiment_id=args.training_experiment_id,
                        inference_experiment_id=os.environ.get('DET_EXPERIMENT_ID'),
                        cpu_offload=args.cpu_offload, training_seed=cfg['train']['seed'],
                        seed_matches_training=case['seed'] == cfg['train']['seed'],
                        initial_noise=case.get('initial_noise'), verifier=cfg['verifier'],
                        training_reasoning_rounds=manifest['training_reasoning_rounds'],
                        message_block_calls_per_condition=manifest['message_block_calls_per_condition'],
                        prepared_inputs=str(folder), frames=cfg['data']['frames'])
                    save_json(path.with_suffix('.json'), metadata)
                    result.update(status='completed', seconds=time.monotonic() - start,
                                  peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30,
                                  encoded_video=verify_video(path, cfg, args.fps))
                finally:
                    hook.remove()
                del latent, sample, noise
            summary['completed_cases'] += 1
            save_json(output / 'summary.json', summary)
            print(f'[{args.suite}] completed {summary["completed_cases"]}/{len(cases)}: {path}', flush=True)
        summary.update(status='completed', finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        summary.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save_json(output / 'summary.json', summary)


if __name__ == '__main__':
    main()
