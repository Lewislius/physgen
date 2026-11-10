"""Export completed paired results; no generation, evaluator call or new bootstrap."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'reports/figures'


def read(relative, *keys):
    source = ROOT / relative
    result = json.loads(source.read_text())
    for key in keys:
        result = result[key]
    return result, str(source)


def plot_comparisons(name, title, rows, count, note):
    for _, result, _ in rows:
        if result['count'] != count:
            raise ValueError('Do not mix fixed40 main results with fixed16 controls')
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 2.8 + .72 * len(rows)), sharey=True)
    fig.subplots_adjust(left=.32, right=.98, top=.80, bottom=.26, wspace=.16)
    colors = {'pc': '#2468A0', 'sa': '#BE622A'}
    bound = max(abs(value) for _, result, _ in rows for metric in ['pc', 'sa']
                for value in result['metrics'][metric]['paired_bootstrap_95_interval'])
    limit = max(.40 if count == 40 else .60, float(np.ceil(bound/.2)*.2))
    export = []
    for ax, metric, heading in zip(axes, ['pc', 'sa'], ['Physical commonsense (PC)', 'Semantic adherence (SA)']):
        ax.axvline(0, color='#555555', linewidth=1)
        ax.set_axisbelow(True)
        ax.grid(axis='x', color='#DCDCDC', linewidth=.6)
        for index, (label, result, source) in enumerate(rows):
            value = result['metrics'][metric]
            delta = value['difference']
            lower, upper = value['paired_bootstrap_95_interval']
            ax.errorbar(delta, index, xerr=[[delta-lower], [upper-delta]],
                        fmt='o', color=colors[metric], capsize=4, markersize=6,
                        linewidth=1.7, zorder=3)
            formatted = f'{delta:+.4f}' if count == 16 else f'{delta:+.3f}'
            ax.text(.98, index+.28,
                    f"{formatted}   W/T/L {value['wins']}/{value['ties']}/{value['losses']}",
                    transform=ax.get_yaxis_transform(), ha='right', va='center',
                    fontsize=8.5, color='#424242')
            export.append(dict(comparison=label.replace('\n', '; '), metric=metric,
                cases=count, candidate_mean=value['candidate'], baseline_mean=value['baseline'],
                difference=delta, ci95_lower=lower, ci95_upper=upper,
                wins=value['wins'], ties=value['ties'], losses=value['losses'], source=source))
        ax.set_title(heading, fontsize=12, fontweight='bold', pad=13)
        ax.set_xlim(-limit, limit)
        ax.set_xticks(np.arange(-limit, limit+.001, .2))
        ax.set_ylim(len(rows)-.45, -.55)
        ax.set_xlabel('Paired mean difference (candidate minus reference)', fontsize=9)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([label for label, _, _ in rows], fontsize=10)
        ax.tick_params(axis='y', length=0, pad=12)
        for side in ['top', 'right', 'left']:
            ax.spines[side].set_visible(False)
    fig.suptitle(title, x=.32, y=.96, ha='left', fontsize=14, fontweight='bold')
    fig.text(.32, .895, 'Seed 42; native text guidance 5; active physical guidance 5.',
             ha='left', fontsize=9, color='#555555')
    fig.text(.02, .04, note, ha='left', va='bottom', fontsize=8.5, color='#444444')
    paths = []
    for extension in ['pdf', 'png']:
        path = OUT / f'{name}.{extension}'
        fig.savefig(path, dpi=180, facecolor='white')
        paths.append(str(path))
    plt.close(fig)
    with (OUT / f'{name}.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(export[0]))
        writer.writeheader()
        writer.writerows(export)
    return dict(name=name, count=count, files=paths, comparisons=len(rows), data=export)


def final_candidates():
    for label, directory in [('R9', 'r1_content_reader_r9'), ('R9.1', 'content_value_r9_1'), ('R10-A', 'execution_value_r10'), ('R10-B', 'grounded_execution_r10b')]:
        metadata = ROOT / directory / 'prepared.json'
        if not metadata.exists():
            continue
        run = json.loads(metadata.read_text()).get('successful_candidate_run')
        if not run:
            continue
        path = ROOT / directory / 'reports' / run / 'generation_review.json'
        if path.exists():
            yield label, json.loads(path.read_text()), str(path)


def plot_module_matrix():
    """Separate branch effects from condition-content effects on the same16."""
    specifications = [
        ('R1-400', 'module_controls16/r1_step0400_guidance5_seed42/reports/module_effects.json', 'correct_vs_'),
        ('R4-64', 'grounded_read_r4/reports/generation_review.json', None),
        ('R6-128', 'content_modulation_r6/reports/physical_content_66910/generation_review.json', 'normal_vs_'),
        ('R7-128', 'native_coordinate_r7/reports/physical_content_66920/generation_review.json', 'normal_vs_'),
        ('R8-128', 'content_reader_r8/reports/physical_content_66926/generation_review.json', 'normal_vs_'),
    ]
    datasets = []
    all_modes = ['no_c', 'reasoning_swapped', 'empty_reasoning', 'global_train_mean',
                 'anchor_only', 'mean_query_update', 'mean', 'shared_only', 'routing_only', 'core_only']
    for label, relative, prefix in specifications:
        report, source = read(relative)
        pairs = report['paired_comparisons']
        mapping = ({mode: prefix + mode for mode in all_modes} if prefix else
                   dict(no_c='correct_vs_no_c', reasoning_swapped='correct_vs_unrelated',
                        anchor_only='learned_correction'))
        datasets.append((label, {mode: pairs[key] for mode, key in mapping.items() if key in pairs}, source))
    for label, report, source in final_candidates():
        pairs = report['paired_comparisons']
        values = {mode: pairs['normal_vs_' + mode] for mode in all_modes if 'normal_vs_' + mode in pairs}
        if values:
            datasets.append((label + '-400', values, source))
    expected = {row['id'] for row in json.loads((ROOT / 'content_value_r9_1/subset16.json').read_text())['cases']}
    export = []
    for label, values, source in datasets:
        for mode, result in values.items():
            if result['count'] != 16 or {row['id'] for row in result['cases']} != expected:
                raise ValueError('Module figure requires the same preselected16: ' + label + '/' + mode)
            for metric in ['pc', 'sa']:
                value = result['metrics'][metric]
                lower, upper = value['paired_bootstrap_95_interval']
                export.append(dict(model=label, intervention=mode, metric=metric, cases=16,
                    normal_mean=value['candidate'], control_mean=value['baseline'], difference=value['difference'],
                    ci95_lower=lower, ci95_upper=upper, wins=value['wins'], ties=value['ties'],
                    losses=value['losses'], source=source))
    columns = [('no_c', 'Normal minus no injection'),
               ('reasoning_swapped', 'Normal minus unrelated reasoning'),
               ('global_train_mean', 'Normal minus train-mean C')]
    bound = max(abs(value) for _, values, _ in datasets for mode, _ in columns if mode in values
                for metric in ['pc', 'sa'] for value in values[mode]['metrics'][metric]['paired_bootstrap_95_interval'])
    limit = max(.6, float(np.ceil(bound/.3)*.3))
    fig, axes = plt.subplots(2, 3, figsize=(14, 4.0 + .65 * len(datasets)), sharex=True, sharey=True)
    fig.subplots_adjust(left=.10, right=.98, top=.82, bottom=.18, hspace=.25, wspace=.12)
    for row, (metric, color, metric_label) in enumerate([('pc', '#2468A0', 'PC'), ('sa', '#BE622A', 'SA')]):
        for col, (mode, title) in enumerate(columns):
            ax = axes[row, col]
            ax.axvline(0, color='#555555', linewidth=1)
            ax.grid(axis='x', color='#DDDDDD', linewidth=.6)
            for y, (_, values, _) in enumerate(datasets):
                if mode not in values:
                    ax.text(0, y, 'not evaluated', ha='center', va='center', color='#888888', fontsize=8)
                    continue
                value = values[mode]['metrics'][metric]
                delta = value['difference']; lower, upper = value['paired_bootstrap_95_interval']
                ax.errorbar(delta, y, xerr=[[delta-lower], [upper-delta]], fmt='o',
                            color=color, capsize=3, markersize=5, linewidth=1.4)
                ax.text(.97, y+.21, f'{delta:+.4f}', transform=ax.get_yaxis_transform(),
                        ha='right', va='center', fontsize=8, color=color)
            ax.set_xlim(-limit, limit); ax.set_xticks(np.arange(-limit, limit+.001, .3))
            ax.set_ylim(len(datasets)-.45, -.55)
            ax.set_yticks(range(len(datasets))); ax.set_yticklabels([label for label, _, _ in datasets])
            if row == 0:
                ax.set_title(title, fontsize=10, pad=12)
            else:
                ax.set_xlabel('Paired mean difference', fontsize=9)
            if col == 0:
                ax.set_ylabel(metric_label, fontsize=11, fontweight='bold')
            for side in ['top', 'right']:
                ax.spines[side].set_visible(False)
    fig.suptitle('Branch benefit and useful condition content are different claims',
                 x=.10, y=.97, ha='left', fontsize=14, fontweight='bold')
    fig.text(.10, .91, 'Same fixed16 per intervention; seed42; gamma5. Positive values favor the normal condition.', fontsize=9)
    fig.text(.02, .035,
        'Lines: existing paired bootstrap95% intervals, not across-seed uncertainty. Each row intervenes within one checkpoint.\n'
        'Rows have different training budgets and recipes; differences between rows are not controlled architecture effects.\n'
        'Train-mean C fixes both CFG conditions and changes amplitude. Unrelated reasoning changes the positive reasoning only.\n'
        'All available controls, including empty reasoning and anchor-only, are retained in the companion CSV; missing cells are not zero effects.',
        fontsize=8, color='#444444')
    name = 'cross_version_module_effects16'; paths = []
    for extension in ['pdf', 'png']:
        path = OUT / f'{name}.{extension}'; fig.savefig(path, dpi=180, facecolor='white'); paths.append(str(path))
    plt.close(fig)
    with (OUT / f'{name}.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(export[0])); writer.writeheader(); writer.writerows(export)
    return dict(name=name, count=16, files=paths, data=export, models=len(datasets),
                comparisons=sum(len(values) for _, values, _ in datasets))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42,
                         'font.size': 10, 'axes.unicode_minus': False})
    main_rows = []
    for label, relative, keys in [
        ('R1-400 vs native400\n400 total updates each',
         'reports/videophy400_comparison.json', []),
        ('R1 request anchor vs R1-400\nSame trained weights',
         'request_anchor_r1/reports/generation_review.json', ['paired_comparisons', 'request_anchored_vs_original']),
        ('R1-1000 vs R1-400\nTraining continuation',
         'reports/r1_400_to_1000_full40.json', []),
        ('R4-64 vs native64\n64 total updates each',
         'grounded_read_r4/reports/generation_review.json', ['paired_comparisons', 'full40_matched_native64']),
        ('R6-128 vs native128\n128 total updates each',
         'content_modulation_r6/reports/physical_content_66910/generation_review.json', ['paired_comparisons', 'normal_vs_native128']),
        ('R7-128 vs native128\n128 total updates each',
         'native_coordinate_r7/reports/physical_content_66920/generation_review.json', ['paired_comparisons', 'normal_vs_native128']),
        ('R8-128 vs native128\n128 total updates each',
         'content_reader_r8/reports/physical_content_66926/generation_review.json', ['paired_comparisons', 'normal_vs_native128']),
    ]:
        result, source = read(relative, *keys)
        main_rows.append((label, result, source))
    for label, report, source in final_candidates():
        result = report['paired_comparisons'].get('normal_vs_native400')
        if result:
            main_rows.append((label + '-400 vs native400\n400 total updates each', result, source))
    figures = [plot_comparisons('paired_generation_effects40',
        'Completed generation comparisons: fixed 40 cases', main_rows, 40,
        'Units: points on the 1–5 scale. Lines: existing case-paired bootstrap 95% intervals; they do not estimate variation across seeds.\n'
        'R1/R9/R9.1-400:128 interface+272 LoRA; R4-64:16+48; R6/R7/R8-128:16+112. Native baselines train LoRA on every update.\n'
        'The continuation row deliberately compares different training budgets. These are shared-first-image I2V results, not the full T2V benchmark.')]
    controls, source = read('module_controls16/r1_step0400_guidance5_seed42/reports/module_effects.json')
    control_rows = []
    for mode, label in [
        ('no_c', 'Normal minus no injection'),
        ('reasoning_swapped', 'Normal minus unrelated reasoning'),
        ('empty_reasoning', 'Normal minus empty reasoning'),
        ('mean', 'Normal minus own-token mean C'),
        ('global_train_mean', 'Normal minus fixed train-mean C'),
        ('anchor_only', 'Normal minus anchor-only C'),
    ]:
        control_rows.append((label, controls['paired_comparisons'][f'correct_vs_{mode}'], source))
    figures.append(plot_comparisons('paired_module_effects16',
        'R1-400 condition interventions: fixed 16 cases', control_rows, 16,
        'Same checkpoint, first image, native T5 request and initial noise. Units: points on the 1–5 scale; existing paired bootstrap 95% intervals.\n'
        'Empty reasoning retains the own image/request. Own-token mean retains sample identity; fixed train-mean C removes Qwen sample information.\n'
        'Mean interventions also change amplitude. This fixed16 subset must not be presented as the full40 main comparison.'))
    r6, source = read('content_modulation_r6/reports/physical_content_66910/generation_review.json')
    r6_rows = []
    for mode, label in [
        ('no_c', 'Normal minus no injection'),
        ('reasoning_swapped', 'Normal minus unrelated reasoning'),
        ('global_train_mean', 'Normal minus fixed train-mean C'),
        ('mean_query_update', 'Normal minus fixed train-mean Q update'),
        ('anchor_only', 'Normal minus anchor-only C'),
    ]:
        r6_rows.append((label, r6['paired_comparisons'][f'normal_vs_{mode}'], source))
    figures.append(plot_comparisons('r6_paired_module_effects16',
        'R6-128 condition interventions: fixed 16 cases', r6_rows, 16,
        'Same checkpoint, first image, native T5 request and initial noise. Existing case-paired bootstrap 95% intervals; one seed only.\n'
        'Mean Q update retains each sample anchor. Fixed mean C removes Qwen sample information; mean controls also change amplitude.\n'
        'Small gains over anchor or mean Q do not establish net gain over no injection or unrelated reasoning. Main full40 PC equals native128.'))
    reference_rows = []
    for label, relative, key in [
        ('R4-64 vs R1-400', 'grounded_read_r4/reports/generation_review.json', 'full40_preserved_r1_400'),
        ('R6-128 vs R1-400', 'content_modulation_r6/reports/physical_content_66910/generation_review.json', 'normal_vs_r1_reference'),
        ('R7-128 vs R1-400', 'native_coordinate_r7/reports/physical_content_66920/generation_review.json', 'normal_vs_r1_reference'),
        ('R8-128 vs R1-400', 'content_reader_r8/reports/physical_content_66926/generation_review.json', 'normal_vs_r1_reference'),
    ]:
        result, source = read(relative, 'paired_comparisons', key)
        reference_rows.append((label, result, source))
    for label, report, source in final_candidates():
        result = report['paired_comparisons'].get('normal_vs_r1_reference')
        if result:
            reference_rows.append((label + '-400 vs R1-400\n400 total updates each', result, source))
    figures.append(plot_comparisons('paired_r1_reference_effects40',
        'Generation relative to the preserved R1-400 reference', reference_rows, 40,
        'Fixed40; paired mean differences and existing95% intervals. R1-400 reference:PC3.625/SA2.950.\n'
        'R4/R6/R7/R8 have shorter budgets and different recipes; these rows assess replacement quality, not isolated architecture effects.\n'
        'The final400 candidates appear only after complete scores. A SA gain cannot compensate for a PC loss.'))
    figures.append(plot_module_matrix())
    index = dict(recorded_utc=datetime.now(timezone.utc).isoformat(), seed=42, gamma=5,
        computation='Only plotting existing completed summaries and intervals; no re-scoring or new bootstrap',
        figures=figures)
    (OUT / 'manifest.json').write_text(json.dumps(index, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({f['name']: f['files'] for f in figures}))


if __name__ == '__main__':
    main()
