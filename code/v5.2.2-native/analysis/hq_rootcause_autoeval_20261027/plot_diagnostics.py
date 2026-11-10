"""Plot observed gradients and held-out FM; these are not causal ablations."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
gradients = json.loads((root / 'gradient_audit.json').read_text())
validation = json.loads((root / 'validation_fm_audit.json').read_text())
fig, axes = plt.subplots(2, 1, figsize=(10, 7.5), constrained_layout=True)
colors = {'H': '#d34e43', 'Q': '#287fb8'}
for run, values in gradients.items():
    label = 'H' if '-h-' in run else 'Q'
    rows = values['feedback_rows']
    axes[0].plot([r['step'] for r in rows], [r['feedback_to_paired_C_grad_ratio'] for r in rows],
                 '.-', label=f"{label}: median {values['ratio_median']:.1f}x", color=colors[label], alpha=.85)
axes[0].axhline(1, color='#666666', ls='--', lw=1)
axes[0].set(yscale='log', ylabel='Weighted verifier / paired C gradient norm',
            title='Feedback often dominates the C gradient before clipping')
axes[0].legend(loc='upper left')
for label, values in validation.items():
    points = values['points']
    start = next(r for r in points if r['step'] == 64)
    for sigma, linestyle in [('0.25', ':'), ('0.65', '--'), ('0.95', '-')]:
        axes[1].plot([r['step'] for r in points],
                     [100 * (r['fm'][sigma] / start['fm'][sigma] - 1) for r in points],
                     linestyle, color=colors[label], label=f'{label}, sigma={sigma}')
axes[1].axhline(0, color='#666666', lw=1)
axes[1].set(xlabel='Optimizer step', ylabel='FM error change from step 64 (%)',
            title='Held-out denoising error after interface adaptation (lower is better)')
axes[1].legend(ncol=2)
for ax in axes:
    ax.set_xlim(60, 226)
    ax.grid(alpha=.18)
fig.suptitle('H / Q training diagnostics\nGradient norms are pre-clip; FM error is not video quality.', fontsize=13)
fig.savefig(root / 'diagnostics.png', dpi=160)
fig.savefig(root / 'diagnostics.pdf')
print(root / 'diagnostics.png')
