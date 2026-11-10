"""Plot the two completed R6 representation measurements; no model/evaluator calls."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parent
    sources = {step: root / 'content_modulation_r6/reports' / f'content_step{step:04d}.json'
               for step in [16, 128]}
    reports = {step: json.loads(path.read_text()) for step, path in sources.items()}
    if reports[16]['cases'] != reports[128]['cases'] or len(reports[16]['cases']) != 16:
        raise ValueError('The two measurements must use the same fixed16 inputs')
    components = ['raw_query_update', 'preprojection_content_update', 'learned_residual', 'C']
    labels = ['Raw Q\nupdate\n$U$', 'Content\nmodulation\n$A\\odot U$',
              'Projected\ncorrection\n$\\Delta C$', 'Complete\ncondition\n$C$']
    panels = [('mean_components', 'Within-sample token cosine'),
              ('cross_sample_mean_vectors', 'Cosine between sample-mean vectors')]
    destination = root / 'reports/figures'
    destination.mkdir(parents=True, exist_ok=True)
    stem = destination / 'r6_content_endpoints16_128'
    rows = []
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    figure, axes = plt.subplots(1, 2, figsize=(12.2, 4.6), sharey=True)
    x = np.arange(len(components))
    for axis, (field, title) in zip(axes, panels):
        for index, (step, color) in enumerate([(16, '#3978a8'), (128, '#d17c37')]):
            values = [reports[step][field][component]['pair_cosine'] for component in components]
            bars = axis.bar(x + (index - .5) * .36, values, width=.34,
                            label=f'Step {step}', color=color)
            axis.bar_label(bars, labels=[f'{value:.4f}' for value in values], padding=4, fontsize=8.5)
            rows.extend(dict(panel=field, component=component, step=step, pair_cosine=value,
                             case_count=16, source=str(sources[step]))
                        for component, value in zip(components, values))
        axis.set_title(title, fontsize=12, pad=12)
        axis.set_xticks(x, labels)
        axis.set_ylim(0, 1.075)
        axis.set_yticks(np.arange(0, 1.01, .2))
        axis.set_axisbelow(True)
        axis.grid(axis='y', alpha=.2)
        axis.spines[['top', 'right']].set_visible(False)
        axis.tick_params(axis='x', length=0, pad=9)
    axes[0].set_ylabel('Mean pairwise cosine')
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc='upper center', bbox_to_anchor=(.5, .985),
                  ncol=2, frameon=False)
    figure.subplots_adjust(left=.065, right=.985, top=.82, bottom=.25, wspace=.14)
    figure.text(.065, .075,
                "Same 16 normal-condition inputs. Right: average each sample's tokens first, then compare those means.\n"
                'These are representation measurements, not measures of physical correctness or generation quality.',
                ha='left', va='center', fontsize=9, color='#444444')
    figure.savefig(stem.with_suffix('.pdf'), bbox_inches='tight')
    figure.savefig(stem.with_suffix('.png'), dpi=170, bbox_inches='tight')
    plt.close(figure)
    with stem.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps(dict(pdf=str(stem.with_suffix('.pdf')), png=str(stem.with_suffix('.png')),
                          csv=str(stem.with_suffix('.csv')), measurements=16)))


if __name__ == '__main__':
    main()
