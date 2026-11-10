"""Read-only review of the two no-verifier runs; write local evidence artifacts.

Run from the project root with tools/runtime_python.sh. No model/GPU is loaded.
RGB change scores below describe paired one-step reconstructions, not physics.
"""
import csv
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
RUNS = {'H': 'v522-h-content-message-no-verifier-simple_66438',
        'Q': 'v522-q-seed128-message-no-verifier-simple_66443'}
CASES = {'source0000042': 'Match structure / fire',
         'source0001596': 'Hands / bead-filled gel',
         'wisa0013758': 'Espresso pouring',
         'wisa0032606': 'Lake / mountain'}


def read_rows(path):
    # A live writer can leave a partial last line. Never modify its file.
    data = path.read_bytes()
    if not data.endswith(b'\n'):
        data = data[:data.rfind(b'\n') + 1]
    return [json.loads(line) for line in data.splitlines() if line.strip()]


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def write_csv(name, rows):
    with (OUT / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pooled(rows, hs):
    true = sum(r[f'rgb/h{h}_true_change_mse'] for r in rows for h in hs)
    pred = sum(r[f'rgb/h{h}_predicted_change_mse'] for r in rows for h in hs)
    error = sum(r[f'rgb/h{h}_change_error'] for r in rows for h in hs)
    return dict(change_skill=1 - error / max(true, 1e-8),
                change_rms_ratio=math.sqrt(pred / max(true, 1e-8)),
                change_cosine=(true + pred - error) / max(2 * math.sqrt(true * pred), 1e-8),
                rgb_mse=statistics.mean(r[f'rgb/h{h}_mse'] for r in rows for h in hs))


def corr(rows, a, b):
    return float(np.corrcoef([r[a] for r in rows], [r[b] for r in rows])[0, 1])


def build_records():
    summary = dict(captured_at=datetime.now(timezone.utc).isoformat(), runs={})
    aggregates, individual, windows, analysis_records = [], [], [], []
    all_training = {}
    for mode, name in RUNS.items():
        run = ROOT / 'checkpoints' / name
        metrics = read_rows(run / 'metrics.jsonl')
        paired = read_rows(run / 'validation_cases.jsonl')
        for suffix, rows in [('metrics', metrics), ('validation_cases', paired)]:
            (OUT / f'{mode}_{suffix}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
        train = [r for r in metrics if r['kind'] == 'train']
        all_training[mode] = train
        start = next(r for r in metrics if r['kind'] == 'start')
        summary['runs'][mode] = dict(directory=str(run), train_step=train[-1]['step'],
            train_cursor=train[-1]['cursor'], validation_steps=sorted({r['step'] for r in paired}),
            parameters=start['parameter_counts'], gpu=start['device_name'], revision=start['revision'],
            train_correlations={'loss/fm vs sigma': corr(train, 'loss/fm', 'sigma'),
                'loss/fm vs loss/flow_temporal': corr(train, 'loss/fm', 'loss/flow_temporal')})
        groups = defaultdict(list)
        for row in paired:
            sigma = round(row['sigma'], 2)
            groups[(row['step'], sigma)].append(row)
            case = dict(mode=mode, step=row['step'], sigma=sigma, sample=row['sample'],
                        title=CASES.get(row['sample'], row['sample']), fm=row['loss/fm'])
            for tag, hs in [('all', range(1, 21)), ('early', range(1, 6)), ('late', range(16, 21))]:
                case.update({f'{tag}_{key}': value for key, value in pooled([row], hs).items()})
                case[f'{tag}_true_energy'] = sum(row[f'rgb/h{h}_true_change_mse'] for h in hs)
            individual.append(case)
        for (step, sigma), rows in sorted(groups.items()):
            item = dict(mode=mode, step=step, sigma=sigma, samples=len(rows),
                        fm=statistics.mean(r['loss/fm'] for r in rows),
                        temporal_fm=statistics.mean(r['loss/flow_temporal'] for r in rows))
            for tag, hs in [('all', range(1, 21)), ('early', range(1, 6)), ('late', range(16, 21))]:
                item.update({f'{tag}_{key}': value for key, value in pooled(rows, hs).items()})
            aggregates.append(item)
        for lo, hi in [(1, 16), (17, 32), (33, 64), (65, 96), (97, 128), (129, 160), (161, 192), (193, 224)]:
            rows = [r for r in train if lo <= r['step'] <= hi]
            if not rows:
                continue
            item = dict(mode=mode, start=lo, end=rows[-1]['step'], count=len(rows))
            for key in ['loss/fm', 'loss/flow_temporal', 'loss', 'sigma', 'condition_tokens',
                        'seconds', 'peak_gib', 'native_text_dropped', 'C_text_dropped']:
                item[key] = statistics.mean(r[key] for r in rows)
            item['condition_preclip_norm'] = statistics.mean(r['clip_grad_norms']['condition'] for r in rows)
            item['condition_clip_count'] = sum(r['clip_grad_norms']['condition'] > 1 for r in rows)
            item['wan_grad_norm'] = statistics.mean(r['gradients']['wan_lora'] for r in rows)
            item['temporal_fraction_of_total'] = statistics.mean(r['weighted/flow_temporal'] / r['loss'] for r in rows)
            windows.append(item)
        texts = defaultdict(list)
        for path in sorted((ROOT / 'inference_outputs' / name).glob('*.json')):
            row = json.loads(path.read_text())
            if 'physical_dependency_analysis' in row:
                texts[row['id']].append(row)
        for sample, rows in texts.items():
            analysis_records.append(dict(mode=mode, sample=sample, records=len(rows),
                steps=sorted({r['optimizer_step'] for r in rows}),
                unique_analyses=list(dict.fromkeys(r['physical_dependency_analysis'] for r in rows)),
                caption=rows[0]['caption']))
    dump('summary.json', summary)
    dump('per_case_high_noise.json', [r for r in individual if r['sigma'] == .95])
    dump('analysis_text_consistency.json', analysis_records)
    write_csv('validation_summary.csv', aggregates)
    write_csv('validation_per_case.csv', individual)
    write_csv('training_windows.csv', windows)
    return summary, aggregates, individual, all_training


def plot_records(aggregates, individual, training):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    colors = {'H': '#1565c0', 'Q': '#d66b19'}
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), constrained_layout=True)
    for mode in RUNS:
        rows = [r for r in aggregates if r['mode'] == mode and r['sigma'] == .95]
        xs = [r['step'] for r in rows]
        for ax, key in zip(axes[0], ['fm', 'early_change_skill', 'late_change_skill']):
            ax.plot(xs, [r[key] for r in rows], 'o-', color=colors[mode], label=mode)
        rows = [r for r in individual if r['mode'] == mode and r['sigma'] == .95 and r['step'] == 160]
        by_id = {r['sample']: r for r in rows}
        axes[1, 0].bar(np.arange(4) + (-.18 if mode == 'H' else .18),
            [by_id[k]['late_change_skill'] for k in CASES], width=.36, color=colors[mode], label=mode)
        train = training[mode]
        for ax, key, logarithmic in [(axes[1, 1], 'condition', True), (axes[1, 2], 'seconds', False)]:
            vals = [r['clip_grad_norms']['condition'] if key == 'condition' else r[key] for r in train]
            smooth = [statistics.mean(vals[max(0, i-7):i+1]) for i in range(len(vals))]
            ax.plot([r['step'] for r in train], smooth, color=colors[mode], label=mode)
            if logarithmic:
                ax.set_yscale('log')
    titles = ['Fixed-noise FM loss (sigma = 0.95)', 'Early RGB change skill (frames 4-20)',
              'Late RGB change skill (frames 64-80)', 'Per-case late change skill at step 160',
              'C gradient norm before clipping (8-step mean)', 'Training seconds / optimizer step (8-step mean)']
    for i, ax in enumerate(axes.flat):
        ax.set_title(titles[i]); ax.legend(); ax.grid(alpha=.18)
        if i != 3:
            ax.axvline(64, color='gray', linestyle=':', linewidth=1)
            ax.set_xlabel('Optimizer step (joint starts after 64)')
    axes[1, 0].set_xticks(range(4), ['Fire', 'Gel', 'Espresso', 'Lake'])
    axes[1, 0].axhline(0, color='black', linewidth=.8)
    fig.suptitle('H 66438 / Q 66443: paired reconstruction diagnostics, 4 validation cases\n'
                 'Change skill = 1 - change error / true change energy; zero = no-change baseline. '
                 'These are not free-generation quality scores.', fontsize=12)
    fig.savefig(OUT / 'training_review.png', dpi=180)
    fig.savefig(OUT / 'training_review.pdf')
    plt.close(fig)


def frame_grids():
    width, height = 312, 180
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 17)
    for sample, seed in [('source0000042', 42), ('source0000086', 43)]:
        canvas = Image.new('RGB', (5 * width, 4 * (height + 28) + 32), 'white')
        draw = ImageDraw.Draw(canvas)
        for col, frame in enumerate([0, 20, 40, 60, 80]):
            draw.text((col * width + 8, 6), f'Frame {frame}', font=font, fill='black')
        sources = []
        for row, (mode, step) in enumerate([('H', 32), ('H', 160), ('Q', 32), ('Q', 160)]):
            path = ROOT / 'inference_outputs' / RUNS[mode] / f'step{step:04d}_{sample}_seed{seed}.mp4'
            filt = f'select=eq(n\\,0)+eq(n\\,20)+eq(n\\,40)+eq(n\\,60)+eq(n\\,80),scale={width}:{height}'
            result = subprocess.run(['ffmpeg', '-v', 'error', '-threads', '1', '-i', str(path),
                '-vf', filt, '-vsync', '0', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'],
                capture_output=True, check=True)
            frames = np.frombuffer(result.stdout, dtype=np.uint8).reshape(-1, height, width, 3)
            if len(frames) != 5:
                raise ValueError(f'Expected five frames: {path}')
            y = 32 + row * (height + 28)
            draw.text((8, y + 4), f'{mode} step {step} | {sample} | seed {seed}', font=font, fill='black')
            for col, frame in enumerate(frames):
                canvas.paste(Image.fromarray(frame), (col * width, y + 28))
            sources.append(str(path))
        canvas.save(OUT / f'{sample}_frames.png')
        dump(f'{sample}_frame_sources.json', dict(frames=[0, 20, 40, 60, 80], videos=sources))


if __name__ == '__main__':
    summary, aggregate, individual, training = build_records()
    plot_records(aggregate, individual, training)
    frame_grids()
    print(json.dumps({'captured_at': summary['captured_at'], 'steps': {
        mode: row['train_step'] for mode, row in summary['runs'].items()}, 'output': str(OUT)}))
