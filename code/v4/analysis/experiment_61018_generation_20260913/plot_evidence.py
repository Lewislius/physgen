"""Plot saved CPU video measurements and training logs; no model execution."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
inventory = json.loads((OUT / 'video_inventory_metrics.json').read_text())
by_name = {r['name']: r for r in inventory['videos']}
fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
for name, label in [('smoke2_reference_i2v', 'Reference, 2 steps'),
                    ('formal50_reference_i2v', 'Reference, 50 steps'),
                    ('formal50_P02_i2v', 'P02 I2V, 50 steps'),
                    ('formal50_P02_t2v', 'P02 T2V, 50 steps')]:
    row = by_name[name]
    axes[0].plot(np.arange(row['frames']) / 24, row['luma_by_frame'], label=label)
axes[0].set(ylabel='Frame mean luma (0–1)', xlabel='Video time (s)',
            title='Observed global brightness: descriptive, not a physical-quality score')
axes[0].legend(ncol=2)
ref = by_name['formal50_reference_i2v']
for c, color in enumerate(['red', 'green', 'blue']):
    axes[1].plot(np.arange(ref['frames']) / 24, np.asarray(ref['rgb_by_frame'])[:, c],
                 color=color, label=color)
axes[1].set(xlabel='Video time (s)', ylabel='Frame mean channel (0–1)',
            title='Reference, 50 steps: global RGB fluctuations')
axes[1].legend(ncol=3)
fig.savefig(OUT / 'video_color_temporal.png', dpi=160)
plt.close(fig)

log = ROOT / 'train/train_log/wisa_native_p_A1_4x96g_20260912T151800Z-e189c27c/20260912T151800Z-e189c27c/steps.jsonl'
records = [json.loads(line) for line in log.read_text().splitlines()]
records = [r for r in records if r['split'] == 'train']
steps = [r['step'] for r in records]
fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
for block in [5, 10, 15, 20, 25, 30]:
    label = f'A@{block}'
    axes[0].semilogy(steps, [r['metrics'][f'{label}/state_gate_mean'] for r in records], label=label)
    values = np.asarray([r['metrics'][f'{label}/write_relative_rms'] for r in records]) * 100
    axes[1].plot(steps[9:], np.convolve(values, np.ones(10) / 10, mode='valid'), label=label)
axes[0].set(ylabel='Mean state gate (log scale)', title='State updates close in five layers during training')
axes[0].legend(ncol=6)
axes[1].set(xlabel='Optimizer step', ylabel='Write / hidden RMS (%)',
            title='Writes remain active (10-step moving average)')
axes[1].legend(ncol=6)
fig.savefig(OUT / 'training_gates_and_writes.png', dpi=160)
plt.close(fig)

font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
with imageio.get_reader(ref['path']) as reader:
    frames = [frame for frame in reader]
luma = np.asarray(ref['luma_by_frame'])
indices = sorted(set([0, 4, 12, 20, 25, 30, 37, 45, 50, 60, 70, 75, 85, 90, 100,
                      int(luma.argmin()), int(luma.argmax()), int(np.abs(np.diff(luma)).argmax()) + 1]))
tw, th, cols = 320, 344, 5
sheet = Image.new('RGB', (cols * tw, ((len(indices) + cols - 1) // cols) * th), '#161616')
draw = ImageDraw.Draw(sheet)
for j, i in enumerate(indices):
    x, y = (j % cols) * tw, (j // cols) * th
    sheet.paste(Image.fromarray(frames[i]).resize((tw, tw)), (x, y + 24))
    draw.text((x + 4, y + 2), f'f{i:03d} / {i/24:.2f}s / Y={luma[i]:.3f}', font=font, fill='white')
sheet.save(OUT / 'reference_detail_grid.jpg', quality=94)
print('Saved color plot, gate plot, and reference detail grid.')
print('Reference max adjacent luma jump:', int(np.abs(np.diff(luma)).argmax()),
      '->', int(np.abs(np.diff(luma)).argmax()) + 1)
