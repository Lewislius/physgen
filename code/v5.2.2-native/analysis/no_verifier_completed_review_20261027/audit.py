"""Recompute the completed H/Q runs from immutable training records on CPU.

No training sources, checkpoints, previous reviews, or GPU jobs are modified.
RGB metrics are fixed-noise, one-step reconstruction diagnostics, not physics
scores. The pooled change score uses summed true RGB change as its denominator.
"""
from collections import Counter
import importlib.util
import json
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
spec = importlib.util.spec_from_file_location(
    'previous_review', ROOT/'analysis/no_verifier_review_20261027/analyze_records.py')
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)
review.OUT = OUT
summary, aggregates, individual, training = review.build_records()
final = [r for r in aggregates if r['step'] == 224]
review.write_csv('final_validation.csv', final)
review.write_csv('final_cases_high_noise.csv', [r for r in individual
    if r['step'] == 224 and r['sigma'] == .95])

manifest = json.loads((ROOT.parents[1]/'data_cache/wisa_4000/base/manifest.json').read_text())['records']
splits = Counter(r['split'] for r in manifest)
summary['data_splits'] = dict(splits)
summary['scope'] = ('Both runs completed at 224 optimizer steps. All RGB scores '
                    'are paired one-step reconstruction, not free-generation physics quality.')
summary['metric_definition'] = {
    'change_skill': '1 - sum(squared predicted-minus-reference RGB change) / sum(squared reference RGB change)',
    'zero_baseline': 'predict zero inter-frame RGB change; not a physical-correctness baseline',
    'early': 'five differences at RGB frames 4,8,12,16,20, each relative to four frames earlier',
    'late': 'five differences at RGB frames 64,68,72,76,80, each relative to four frames earlier',
    'change_rms_ratio': 'sqrt(sum(predicted RGB change energy) / sum(reference RGB change energy))',
}
for mode, train in training.items():
    assert [r['step'] for r in train] == list(range(1, 225))
    records = review.read_rows(ROOT/'checkpoints'/review.RUNS[mode]/'metrics.jsonl')
    assert records[-1]['kind'] == 'complete' and records[-1]['step'] == 224
    item = summary['runs'][mode]
    item['record_kinds'] = dict(Counter(r['kind'] for r in records))
    item['train_fraction_one_epoch'] = train[-1]['cursor']/splits['train']
    item['validation_case_count'] = len({r['sample'] for r in individual if r['mode'] == mode})
    item['seconds_sum'] = sum(r['seconds'] for r in train)
    item['seconds_median'] = statistics.median(r['seconds'] for r in train)
    item['peak_allocated_gib'] = max(r['peak_gib'] for r in train)
    item['condition_tokens_mean'] = statistics.mean(r['condition_tokens'] for r in train)
    item['condition_clip_steps'] = [r['step'] for r in train if r['clip_grad_norms']['condition'] > 1]
    item['analysis_truncated_micro_count'] = sum(r['analysis_truncated']*4 for r in train)
    item['temporal_fraction_mean'] = statistics.mean(r['weighted/flow_temporal']/r['loss'] for r in train)
    item['condition_gradient_early_mean'] = statistics.mean(r['clip_grad_norms']['condition'] for r in train[:16])
    item['condition_gradient_late_mean'] = statistics.mean(r['clip_grad_norms']['condition'] for r in train[-32:])
    item['wan_lora_interface_max_gradient'] = max(r['gradients']['wan_lora'] for r in train[:64])
    item['wan_lora_joint_mean_gradient'] = statistics.mean(r['gradients']['wan_lora'] for r in train[64:])
    item['joint_fm_temporal_correlation'] = review.corr(train[64:], 'loss/fm', 'loss/flow_temporal')
    item['native_text_dropped_micro_count'] = sum(r['native_text_dropped']*4 for r in train)
    item['C_text_dropped_micro_count'] = sum(r['C_text_dropped']*4 for r in train)
    a = next(r for r in aggregates if r['mode'] == mode and r['step'] == 64 and r['sigma'] == .95)
    b = next(r for r in final if r['mode'] == mode and r['sigma'] == .95)
    item['high_noise_FM_relative_reduction_step64_to224'] = 1-b['fm']/a['fm']
    item['final_high_noise'] = b
    last_cases = [r for r in individual if r['mode'] == mode and r['step'] == 224 and r['sigma'] == .95]
    total_energy = sum(r['late_true_energy'] for r in last_cases)
    item['late_true_energy_shares'] = {r['sample']: r['late_true_energy']/total_energy for r in last_cases}
    item['final_late_negative_skill_cases'] = [r['sample'] for r in last_cases if r['late_change_skill'] < 0]
    videos = list((ROOT/'inference_outputs'/review.RUNS[mode]).glob('step*.json'))
    metadata = [json.loads(p.read_text()) for p in videos]
    item['exported_video_count'] = len(videos)
    item['exported_video_unique_cases'] = sorted({r['id'] for r in metadata})
    item['exported_video_seeds_by_case'] = {name: sorted({r['seed'] for r in metadata if r['id'] == name})
                                          for name in item['exported_video_unique_cases']}
summary['training_random_draw_comparison'] = {
    'equal_sigma_step_means': sum(a['sigma'] == b['sigma'] for a, b in zip(training['H'], training['Q'])),
    'equal_native_dropout_step_means': sum(a['native_text_dropped'] == b['native_text_dropped']
                                         for a, b in zip(training['H'], training['Q'])),
    'note': 'Equal training seed does not pair per-step random draws across differently initialized architectures.'
}
review.dump('summary.json', summary)

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
colors = {'H': '#1764ad', 'Q': '#d16817'}
for mode in ('H', 'Q'):
    rows = [r for r in aggregates if r['mode'] == mode and r['sigma'] == .95]
    xs = [r['step'] for r in rows]
    axes[0, 0].plot(xs, [r['fm'] for r in rows], 'o-', color=colors[mode], label=mode)
    for part, line in [('early', '-'), ('late', '--')]:
        axes[0, 1].plot(xs, [r[f'{part}_change_skill'] for r in rows], marker='o',
                        linestyle=line, color=colors[mode], label=f'{mode} {part}')
    axes[1, 0].plot(xs, [r['late_change_rms_ratio'] for r in rows], 'o-', color=colors[mode], label=mode)
    cases = {r['sample']: r for r in individual if r['mode'] == mode and r['sigma'] == .95 and r['step'] == 224}
    axes[1, 1].bar(np.arange(4)+(-.18 if mode == 'H' else .18),
                  [cases[k]['late_change_skill'] for k in review.CASES], width=.36, color=colors[mode], label=mode)
titles = ['Fixed-noise FM loss', 'Early / late RGB change skill',
          'Late predicted / reference RGB change RMS', 'Final per-case late change skill']
for i, ax in enumerate(axes.flat):
    ax.set_title(titles[i]); ax.grid(alpha=.15); ax.legend()
    if i < 3:
        ax.axvline(64, color='gray', linestyle=':', linewidth=1)
        ax.set_xlabel('Optimizer step; joint training starts after 64')
axes[1, 1].set_xticks(range(4), ['Fire', 'Gel', 'Espresso', 'Lake'])
axes[1, 1].axhline(0, color='black', linewidth=.7)
fig.suptitle('Completed H 66438 / Q 66443 | sigma = 0.95 | 4 validation cases\n'
             'Paired one-step reconstruction diagnostics, not free-generation physics scores', fontsize=12)
fig.savefig(OUT/'completed_training_review.png', dpi=170)
fig.savefig(OUT/'completed_training_review.pdf')
plt.close(fig)

font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
width, height = 312, 180
for sample, seed in [('source0000042', 42), ('source0000086', 43)]:
    canvas = Image.new('RGB', (5*width, 32+4*(height+28)), 'white')
    draw = ImageDraw.Draw(canvas)
    for col, frame in enumerate([0, 20, 40, 60, 80]):
        draw.text((col*width+8, 6), f'Frame {frame}', font=font, fill='black')
    sources = []
    for row, (mode, step) in enumerate([('H', 64), ('H', 224), ('Q', 64), ('Q', 224)]):
        path = ROOT/'inference_outputs'/review.RUNS[mode]/f'step{step:04d}_{sample}_seed{seed}.mp4'
        filt = f'select=eq(n\\,0)+eq(n\\,20)+eq(n\\,40)+eq(n\\,60)+eq(n\\,80),scale={width}:{height}'
        result = subprocess.run(['ffmpeg', '-v', 'error', '-threads', '1', '-i', str(path), '-vf', filt,
            '-vsync', '0', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'], capture_output=True, check=True)
        frames = np.frombuffer(result.stdout, dtype=np.uint8).reshape(-1, height, width, 3)
        assert len(frames) == 5
        y = 32+row*(height+28)
        draw.text((8, y+4), f'{mode} step {step} | {sample} | seed {seed}', font=font, fill='black')
        for col, frame in enumerate(frames):
            canvas.paste(Image.fromarray(frame), (col*width, y+28))
        sources.append(str(path))
    canvas.save(OUT/f'{sample}_final_frames.png')
    review.dump(f'{sample}_frame_sources.json', {'frames': [0, 20, 40, 60, 80], 'videos': sources})

print(json.dumps({'output': str(OUT), 'runs': {mode: {
    key: value for key, value in row.items() if key not in ('parameters', 'late_true_energy_shares')}
    for mode, row in summary['runs'].items()}}, ensure_ascii=False, indent=2))
