"""Read existing videos/logs; no model inference, training, or parameter sweep."""
import json
from collections import Counter
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'analysis/fullwidth3_step0300_flicker_20260918'
OUT.mkdir(exist_ok=True)
RUN = ROOT / 'inference_outputs/v42_fullwidth3_write_20260918T081555Z_1x96g'
LOG = ROOT / 'train/train_log/v42_v4_fullwidth3_without_prior_A1_1x96g_20260918T071253Z-9945c9d7/20260918T071253Z-9945c9d7'
V4 = ROOT.parent / 'v4/inference_outputs/wisa_native_p_1xada48g/final_20260913-070408-373326450'
LORA = ROOT.parent / 'v4-static-lora/inference_outputs/lora_20260917T140727Z_dae2a0_step0800_ema_demo_seed42'


def records(path):
    result = []
    for line in path.read_text().splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # Running jobs can have a partial final line.
    return result


def stats(values):
    x = np.asarray(values, dtype=float)
    return dict(n=len(x), mean=float(x.mean()), median=float(np.median(x)),
                min=float(x.min()), max=float(x.max())) if len(x) else None


def metrics_summary(rows, keys):
    return {k: stats([r['metrics'][k] for r in rows if k in r['metrics']]) for k in keys}


train_all = records(LOG / 'steps.jsonl')
train = [r for r in train_all if r['split'] == 'train' and r['step'] <= 300]
micro = [r for r in records(LOG / 'micro_rank0.jsonl') if r['step'] <= 300]
print('Micro variable fields:', list(micro[0].get('variables', {})), flush=True)
keys = [k for k in train[-1]['metrics'] if k.startswith(('loss/', 'grad/', 'prior/', 'P/'))]
keys += [f'A@{b}/{name}' for b in (5, 15, 25) for name in
         ('state_gate_mean', 'write_gate_mean', 'effective_update_rms', 'write_relative_rms',
          'p_before_rms', 'p_after_rms', 'struct_gain', 'target_cosine')]
report = dict(training_last_logged_step=max(r['step'] for r in train_all),
              checkpoint_step=300, training_windows={}, gradient_probes=[], validation=[])
for lo, hi in ((1, 25), (76, 100), (176, 200), (276, 300)):
    rows = [r for r in train if lo <= r['step'] <= hi]
    report['training_windows'][f'{lo}-{hi}'] = metrics_summary(rows, keys)
for r in train:
    m = r['metrics']
    if 'loss_grad/A@5/fm_norm' not in m:
        continue
    item = dict(step=r['step'], gt_conditioned=m.get('prior/gt_conditioned'), sites={})
    for b in (5, 15, 25):
        prefix = f'loss_grad/A@{b}/'
        d = {k.removeprefix(prefix): v for k, v in m.items() if k.startswith(prefix)}
        d['write_over_fm'] = d['write_norm'] / max(d['fm_norm'], 1e-30)
        d['struct_over_fm'] = d['struct_norm'] / max(d['fm_norm'], 1e-30)
        item['sites'][str(b)] = d
    report['gradient_probes'].append(item)
report['validation'] = [r for r in train_all if r['split'] != 'train' and r['step'] <= 300]
report['training_gt_counts'] = dict(Counter(str(r['metrics'].get('prior/gt_conditioned')) for r in train))
report['micro_fields'] = dict(metrics=list(micro[0]['metrics']), variables=micro[0].get('variables'))
report['videos'] = {}
report['trajectories'] = {}
thumbs = {}

for label, root, suffix in [('v42', RUN, 'demo20/{cid}/i2v/seed42_full.mp4'),
                            ('v4', V4, 'demo20/{cid}/i2v/seed42_full.mp4'),
                            ('lora', LORA, '{cid}_i2v/video.mp4')]:
    for n in range(1, 21):
        cid = f'P{n:02d}'
        path = root / suffix.format(cid=cid)
        if not path.exists():
            continue
        try:
            with imageio.get_reader(path, 'ffmpeg') as reader:
                frames = np.stack([frame for frame in reader])
        except Exception as error:
            print('Skip unfinished video', path, str(error), flush=True)
            continue
        x = frames.astype(np.float32) / 255
        means = x.mean(axis=(1, 2))
        differences = np.abs(np.diff(x, axis=0)).mean(axis=(1, 2, 3))
        color_differences = np.abs(np.diff(means, axis=0)).mean(axis=1)
        color_accel = np.abs(np.diff(means, n=2, axis=0)).mean(axis=1)
        key = f'{label}/{cid}'
        report['videos'][key] = dict(path=str(path), frames=len(frames),
            frame_rgb_mean=means.tolist(), adjacent_pixel_mae=differences.tolist(),
            color_jump_mean=float(color_differences[4:].mean()),
            color_jump_p95=float(np.quantile(color_differences[4:], .95)),
            color_acceleration_mean=float(color_accel[4:].mean()),
            adjacent_pixel_mae_mean=float(differences[4:].mean()),
            channel_temporal_std=means[5:].std(axis=0).tolist(),
            clipping_fraction=float(((x < .02) | (x > .98)).mean()),
            mean_from_first_mae=float(np.abs(x[1:] - x[:1]).mean()))
        indices = np.linspace(0, len(frames) - 1, 9).round().astype(int)
        tiles = []
        for index in indices:
            tile = Image.new('RGB', (192, 128), 'white')
            tile.paste(Image.fromarray(frames[index]).resize((192, 108)), (0, 20))
            ImageDraw.Draw(tile).text((4, 3), f'{key}  frame {index}', fill='black')
            tiles.append(tile)
        thumbs[key] = tiles
        if label == 'v42':
            trajectory_path = path.with_suffix('.steps.jsonl')
            trajectory = records(trajectory_path)
            report['trajectories'][cid] = {}
            for branch in ('cond', 'uncond'):
                report['trajectories'][cid][branch] = {
                    k: stats([r['metrics'][branch][k] for r in trajectory])
                    for k in trajectory[0]['metrics'][branch]
                    if any(s in k for s in ('gate_mean', 'write_relative_rms', 'effective_update_rms', 'p_before_rms'))}
        print(key, 'color_jump', round(report['videos'][key]['color_jump_mean'], 5), flush=True)

for group in range(4):
    names = [f'v42/P{n:02d}' for n in range(group * 5 + 1, group * 5 + 6) if f'v42/P{n:02d}' in thumbs]
    if not names:
        continue
    sheet = Image.new('RGB', (192 * 9, 128 * len(names)), 'white')
    for row, name in enumerate(names):
        for col, tile in enumerate(thumbs[name]):
            sheet.paste(tile, (192 * col, 128 * row))
    sheet.save(OUT / f'v42_frames_{group + 1}.jpg', quality=90)
for n in (1, 5, 9, 15):
    names = [f'{label}/P{n:02d}' for label in ('v42', 'v4', 'lora') if f'{label}/P{n:02d}' in thumbs]
    sheet = Image.new('RGB', (192 * 9, 128 * len(names)), 'white')
    for row, name in enumerate(names):
        for col, tile in enumerate(thumbs[name]):
            sheet.paste(tile, (192 * col, 128 * row))
    sheet.save(OUT / f'existing_video_P{n:02d}.jpg', quality=90)
(OUT / 'record_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print('Saved', OUT / 'record_audit.json', flush=True)
