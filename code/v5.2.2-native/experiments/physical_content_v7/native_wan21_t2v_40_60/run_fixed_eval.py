"""Official Wan2.1-T2V-1.3B on the project's fixed VideoPhy2 40/60 cases."""

import argparse
import json
from math import prod
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parents[2]
REFERENCE = ROOT.parent / 'videophy60/protocol.json'
# Wan2.1 is evaluated at its native 16 fps. Wan2.2 candidates keep 24 fps.
INFER_FPS = 16.0
OUTPUT = ROOT / 'outputs/videophy60_16fps'
WAN = Path('/home/liuzhirui/model/Wan2.1')
CHECKPOINT = WAN / 'Wan2.1-T2V-1.3B'
MODEL = 'Wan2.1-T2V-1.3B'
sys.path.insert(0, str(ROOT.parent / 'prophy_wan21_t2v_1_3b'))
from common import aggregate, gpu_guard, read, save
from run import evaluator_environment


def protocol():
    fixed = read(REFERENCE)
    cases = [dict(id=c['id'], caption=c['caption'], generation_seed=c['seed'])
             for c in fixed['cases']]
    if [c['id'] for c in cases] != [f'VP{i:03d}' for i in range(1, 61)]:
        raise ValueError('Expected fixed VP001–VP060 in benchmark order')
    settings = fixed['settings']
    return dict(
        variant='native_wan21_t2v_1_3b', task='t2v', model=MODEL,
        checkpoint=str(CHECKPOINT), source_protocol=str(REFERENCE),
        scope='Official Wan2.1 T2V on the project fixed 40 + 20; request only.',
        sampling=dict(
            sampler='shifted_euler', steps=settings['steps'], shift=settings['shift'],
            cfg=settings['guidance'], width=settings['width'], height=settings['height'],
            frames=settings['frames'], fps=INFER_FPS, negative_prompt='',
            noise_policy='FP32 CUDA noise in Wan2.1 latent shape, seeded per case',
        ),
        evaluation=fixed['evaluation'], cases=cases,
    )


def sample(model, caption, seed, cfg, null_text, case_id):
    import torch

    temporal, vertical, horizontal = model.vae_stride
    shape = (model.vae.model.z_dim, (cfg['frames'] - 1) // temporal + 1,
             cfg['height'] // vertical, cfg['width'] // horizontal)
    seq_len = prod(size // patch for size, patch in zip(shape[1:], model.patch_size))
    device = model.device
    rng = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(shape, device=device, dtype=torch.float32, generator=rng)
    grid = torch.linspace(1, 0, cfg['steps'] + 1, device=device)
    grid = cfg['shift'] * grid / (1 + (cfg['shift'] - 1) * grid)
    with torch.no_grad():
        text = model.text_encoder([caption], device)
    with torch.no_grad(), torch.autocast('cuda', dtype=model.param_dtype):
        for step, sigma in enumerate(grid[:-1]):
            timestep = (sigma * model.num_train_timesteps).reshape(1)
            pos = model.model([x], t=timestep, context=text, seq_len=seq_len)[0]
            neg = model.model([x], t=timestep, context=null_text, seq_len=seq_len)[0]
            velocity = neg + cfg['cfg'] * (pos - neg)
            x = x + (grid[step + 1] - sigma) * velocity
            if (step + 1) % 10 == 0:
                print(json.dumps(dict(stage='sampling', case=case_id, step=step + 1)), flush=True)
        video = model.vae.decode([x])[0]
    expected = (3, cfg['frames'], cfg['height'], cfg['width'])
    if tuple(video.shape) != expected:
        raise ValueError(f'{case_id}: expected {expected}, got {tuple(video.shape)}')
    return ((video.float().clamp(-1, 1).permute(1, 2, 3, 0) + 1) * 127.5
            ).round().byte().cpu().numpy()


def write_video(frames, path):
    import imageio.v2 as imageio
    path = Path(path)
    temporary = path.with_suffix('.partial.mp4')
    try:
        with imageio.get_writer(
                str(temporary), fps=INFER_FPS, codec='libx264',
                pixelformat='yuv420p', macro_block_size=None,
                ffmpeg_params=['-crf', '18', '-profile:v', 'baseline']) as stream:
            for frame in frames:
                stream.append_data(frame)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def generate(plan, job):
    import torch

    gpu = gpu_guard(job)
    config = plan['sampling']
    if float(config['fps']) != INFER_FPS:
        raise ValueError(f'Wan2.1 native writer uses {INFER_FPS} fps')
    save(OUTPUT / 'manifest.json', plan)
    pending = []
    for case in plan['cases']:
        folder = OUTPUT / 'cases' / case['id']
        record = dict(case=case['id'], caption=case['caption'], seed=case['generation_seed'],
                      model=MODEL, checkpoint=str(CHECKPOINT), task='t2v', sampler=config)
        if (folder / 'planned.mp4').is_file() and (folder / 'planned.json').is_file():
            existing = read(folder / 'planned.json')
            if any(existing.get(key) != value for key, value in record.items()):
                raise ValueError(f'{case["id"]}: existing video has different inputs')
        else:
            pending.append((folder, record))
    status_path = OUTPUT / 'summary.json'
    total = len(plan['cases'])
    complete = total - len(pending)
    if pending:
        save(status_path, dict(status='loading_model', completed_cases=complete, requested=total,
                               gpu=gpu['actual']))
        sys.path.insert(0, str(WAN))
        from wan.configs import WAN_CONFIGS
        from wan.text2video import WanT2V

        torch.set_num_threads(4)
        model = WanT2V(WAN_CONFIGS['t2v-1.3B'], str(CHECKPOINT), device_id=0, t5_cpu=False)
        model.text_encoder.model.to(model.device)
        with torch.no_grad():
            null_text = model.text_encoder([config['negative_prompt']], model.device)
        for folder, record in pending:
            case_id = record['case']
            save(status_path, dict(status='running', completed_cases=complete,
                                   requested=total, current_case=case_id, gpu=gpu['actual']))
            begin = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            frames = sample(model, record['caption'], record['seed'], config, null_text, case_id)
            folder.mkdir(parents=True, exist_ok=True)
            write_video(frames, folder / 'planned.mp4')
            record.update(seconds=time.monotonic() - begin,
                          peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30)
            save(folder / 'planned.json', record)
            complete += 1
            print(json.dumps(dict(stage='video_complete', case=case_id, complete=complete,
                                  seconds=record['seconds'])), flush=True)
    save(status_path, dict(status='completed', completed_cases=complete, requested=total,
                           gpu=gpu['actual']))


def report(plan):
    rows = read(OUTPUT / 'scores.json')['cases']
    if [(r['id'], r['caption'], r['generation_seed']) for r in rows] != [
            (c['id'], c['caption'], c['generation_seed']) for c in plan['cases']]:
        raise ValueError('Scores do not cover the fixed prompt and seed sequence')
    comparison = {}
    for count in (40, 60):
        comparison[f'fixed{count}'] = dict(wan21=aggregate(rows[:count]))
    comparison['protocol'] = dict(
        fps=INFER_FPS, frames=81,
        note='Wan2.1 native 16 fps. Wan2.2 R20ET remains on its 24 fps protocol; '
             'no cross-fps delta is computed here. Compare this baseline only with '
             'the Wan2.1 R20FT results written at 16 fps.')
    save(ROOT / 'comparison_16fps.json', comparison)
    print(json.dumps(comparison, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('describe', 'generate', 'report'))
    args = parser.parse_args()
    plan = protocol()
    if args.stage == 'describe':
        print(json.dumps(dict(plan, cases=len(plan['cases']), fixed40=40), indent=2))
        return
    job = ROOT / 'jobs' / os.getenv('DET_EXPERIMENT_ID', 'local')
    if args.stage == 'generate':
        generate(plan, job)
        return
    if args.stage == 'report':
        report(plan)
        return

    python = [sys.executable, '-I', '-B', '-u']
    evaluator = '/home/liuzhirui/miniconda3/envs/physvid/bin/python'
    for stage, command, env in (
        ('generate', [*python, __file__, '--stage', 'generate'], None),
        ('score', [evaluator, '-I', '-B', '-u', BASE / 'tools/evaluate_videophy2.py', OUTPUT],
         evaluator_environment()),
        ('report', [*python, __file__, '--stage', 'report'], None),
    ):
        save(job / 'status.json', dict(status='running', stage=stage, model=MODEL))
        try:
            subprocess.run(command, env=env, check=True)
        except subprocess.CalledProcessError as exc:
            save(job / 'status.json', dict(status='failed', stage=stage, model=MODEL,
                                          exit_code=exc.returncode))
            raise
    save(job / 'status.json', dict(status='completed', model=MODEL))


if __name__ == '__main__':
    main()
