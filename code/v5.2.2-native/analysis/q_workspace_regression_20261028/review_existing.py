"""Read existing runs and videos only; no model training or generation."""
from pathlib import Path
import collections
import json
import statistics

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUNS = {
    'Q66443': ('checkpoints/v522-q-seed128-message-no-verifier-simple_66443',
               'inference_outputs/v522-q-seed128-message-no-verifier-simple_66443/demo21_final224_round2_seed42'),
    'B66523': ('experiments/q_workspace_variants/checkpoints/v522-q-workspace-b_66523',
               'experiments/q_workspace_variants/inference_outputs/v522-q-workspace-b_66523/final224_demo21_videophy40_amp80g/demo21'),
    'C66524': ('experiments/q_workspace_variants/checkpoints/v522-q-workspace-c_66524',
               'experiments/q_workspace_variants/inference_outputs/v522-q-workspace-c_66524/final224_demo21_videophy40_amp80g/demo21'),
    'DS66526': ('experiments/q_workspace_variants/checkpoints/v522-q-workspace-d-s_66526',
                'experiments/q_workspace_variants/inference_outputs/v522-q-workspace-d-s_66526/final224_demo21_videophy40_amp80g/demo21'),
    'DI66527': ('experiments/q_workspace_variants/checkpoints/v522-q-workspace-d-i_66527',
                'experiments/q_workspace_variants/inference_outputs/v522-q-workspace-d-i_66527/final224_demo21_videophy40_amp80g/demo21'),
}


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize():
    result = {}
    for name, (run, videos) in RUNS.items():
        rows = records(ROOT / run / 'metrics.jsonl')
        train = [r for r in rows if r['kind'] == 'train']
        start = rows[0]
        phases = {}
        for lo, hi in [(1, 32), (33, 64), (65, 96), (193, 224)]:
            subset = [r for r in train if lo <= r['step'] <= hi]
            phases[f'{lo}-{hi}'] = {
                key: statistics.mean(r[key] for r in subset)
                for key in ['loss/fm', 'loss/flow_temporal', 'main_grad_norm']
            }
        validation = [{k: v for k, v in r.items() if k in ('kind', 'step') or '/loss' in k}
                      for r in rows if r['kind'] == 'validation']
        paired = records(ROOT / run / 'validation_cases.jsonl')
        fixed_first = [{k: v for k, v in r.items() if k in ('kind', 'step', 'sample', 'sigma', 'loss/fm', 'loss/flow_temporal')}
                       for r in paired if r['kind'] == 'paired_validation' and r['sample'] == 'source0000042']
        video_records = []
        for path in sorted((ROOT / videos).glob('*/i2v/seed42.json')):
            meta = json.loads(path.read_text())
            mp4 = path.with_suffix('.mp4')
            cap = cv2.VideoCapture(str(mp4))
            video_records.append(dict(case=path.parent.parent.name, video=str(mp4),
                frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), fps=cap.get(cv2.CAP_PROP_FPS),
                metadata=meta))
            cap.release()
        result[name] = dict(run=str(ROOT / run), video_root=str(ROOT / videos),
            initializer=start['initialized_generator'], revision=start['revision'],
            parameter_counts=start['parameter_counts'], training_steps=len(train),
            consumed_samples=train[-1]['cursor'], training_phases=phases,
            condition_clip_count=sum(r['clip_grad_norms']['condition'] > 1 for r in train),
            truncation_sum=sum(r['analysis_truncated'] for r in train),
            active_counts={key: collections.Counter(r.get(key) for r in train)
                for key in ['wan_forward_count', 'condition_use_active', 'sequence_active', 'reasoning_round']},
            validation=validation, same_case_same_noise_validation=fixed_first, videos=video_records)
    (OUT / 'existing_evidence.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def contact_sheets():
    indices = [0, 8, 16, 24, 40, 60, 80]
    width, height, label = 320, 185, 24
    for case in ['P01', 'P02', 'P03', 'P04', 'P05']:
        canvas = Image.new('RGB', (width * len(indices), (height + label) * len(RUNS)), '#151922')
        draw = ImageDraw.Draw(canvas)
        for row, (name, (_, video_root)) in enumerate(RUNS.items()):
            path = ROOT / video_root / case / 'i2v/seed42.mp4'
            cap = cv2.VideoCapture(str(path))
            chosen = {}
            for i in range(81):
                ok, frame = cap.read()
                if not ok:
                    break
                if i in indices:
                    chosen[i] = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            for col, i in enumerate(indices):
                x, y = col * width, row * (height + label)
                draw.text((x + 5, y + 5), f'{case} {name} | f{i:02d} {i/24:.2f}s', fill='white')
                if i in chosen:
                    canvas.paste(chosen[i].resize((width, height), Image.Resampling.LANCZOS), (x, y + label))
        canvas.save(OUT / f'{case}_comparison.jpg', quality=94)


def plot_training(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, data in result.items():
        rows = records(Path(data['run']) / 'metrics.jsonl')
        train = [r for r in rows if r['kind'] == 'train']
        x = np.array([r['loss/fm'] for r in train])
        axes[0].plot(np.arange(16, len(x) + 1), np.convolve(x, np.ones(16)/16, mode='valid'), label=name)
        fixed = [r for r in data['same_case_same_noise_validation'] if r.get('sigma', 0) > .9]
        axes[1].plot([r['step'] for r in fixed], [r['loss/fm'] for r in fixed], 'o-', label=name)
    axes[0].set_title('Training FM, 16-step moving average')
    axes[1].set_title('Same validation case/noise: source42, sigma .95')
    for ax in axes:
        ax.set_xlabel('Optimizer step'); ax.set_ylabel('FM MSE'); ax.grid(alpha=.25)
        ax.axvline(64, color='gray', linestyle='--', alpha=.5)
    axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / 'training_curves.png', dpi=180)


if __name__ == '__main__':
    result = summarize()
    contact_sheets()
    plot_training(result)
    for name, data in result.items():
        fixed = [r for r in data['same_case_same_noise_validation'] if r['step'] == 224]
        print(name, 'videos', len(data['videos']), 'steps', data['training_steps'],
              'fixed-case final', fixed)
