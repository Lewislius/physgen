"""Describe both execution paths using the already declared paired16 videos.

This only reads completed evaluation scores. It performs no generation, scoring,
training or model selection, and does not alter the submitted candidate.
"""
from pathlib import Path
import argparse
import json
import statistics


def joint(row):
    return int(row['sa'] >= 4 and row['pc'] >= 4)


def comparison(after, before):
    result = {}
    for metric in ('sa', 'pc'):
        changes = [after[k][metric] - before[k][metric] for k in after]
        result[metric] = dict(
            mean_change=statistics.mean(changes),
            wins=sum(v > 0 for v in changes),
            ties=sum(v == 0 for v in changes),
            losses=sum(v < 0 for v in changes),
            changed=[dict(case=k, before=before[k][metric], after=after[k][metric])
                     for k in after if after[k][metric] != before[k][metric]])
    result['joint'] = dict(
        gains=[k for k in after if joint(after[k]) and not joint(before[k])],
        losses=[k for k in after if joint(before[k]) and not joint(after[k])])
    return result


def analyze(output, subset, first_off, second_off):
    ids = [v['id'] for v in json.loads(subset.read_text())['cases']]
    if len(ids) != 16 or len(set(ids)) != 16:
        raise ValueError('Use the original fixed16')
    paths = {
        'neither': output / 'controls/no_c/videophy40/scores.json',
        'first_only': output / 'controls' / second_off / 'videophy40/scores.json',
        'second_only': output / 'controls' / first_off / 'videophy40/scores.json',
        'both': output / 'videophy40/scores.json'}
    report = dict(
        status='pending', cases=ids,
        source_scores={k: str(v) for k, v in paths.items()},
        first_path_disabled_by=first_off, second_path_disabled_by=second_off,
        scope='Same own400 checkpoint, fixed16 paired inference interventions. '
              'These are not independently retrained architecture ablations. '
              'Numeric interactions are descriptive ordinal-score differences; '
              'actual video review and complete40/60 remain necessary.')
    missing = [k for k, v in paths.items() if not v.is_file()]
    if missing:
        report['missing'] = missing
        return report
    groups = {}
    for name, path in paths.items():
        rows = json.loads(path.read_text())['cases']
        by_id = {v['id']: v for v in rows}
        expected = {f'VP{i:03d}' for i in range(1, 41)} if name == 'both' else set(ids)
        if len(rows) != len(expected) or set(by_id) != expected:
            raise ValueError(f'Incomplete or mismatched score set: {path}')
        groups[name] = {k: by_id[k] for k in ids}
    for case in ids:
        reference = groups['both'][case]
        for rows in groups.values():
            if any(rows[case][key] != reference[key] for key in ('caption', 'generation_seed')):
                raise ValueError(f'Unpaired prompt/seed: {case}')
    report['status'] = 'complete'
    report['summaries'] = {
        name: dict(sa=statistics.mean(v['sa'] for v in rows.values()),
                   pc=statistics.mean(v['pc'] for v in rows.values()),
                   joint=sum(joint(v) for v in rows.values()), count=16)
        for name, rows in groups.items()}
    pairs = {
        'first_without_second': ('first_only', 'neither'),
        'first_with_second': ('both', 'second_only'),
        'second_without_first': ('second_only', 'neither'),
        'second_with_first': ('both', 'first_only'),
        'both_vs_neither': ('both', 'neither')}
    report['effects'] = {k: comparison(groups[a], groups[b]) for k, (a, b) in pairs.items()}
    report['interactions'] = {}
    for metric in ('sa', 'pc', 'joint'):
        value = joint if metric == 'joint' else lambda row: row[metric]
        per_case = {k: value(groups['both'][k]) - value(groups['first_only'][k])
                      - value(groups['second_only'][k]) + value(groups['neither'][k]) for k in ids}
        report['interactions'][metric] = dict(
            definition='both - first_only - second_only + neither',
            mean=statistics.mean(per_case.values()), per_case=per_case)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--subset', type=Path, default=Path(__file__).parent / 'subset16.json')
    p.add_argument('--first-off', default='query_modulation_off')
    p.add_argument('--second-off', default='key_modulation_off')
    args = p.parse_args()
    report = analyze(args.output, args.subset, args.first_off, args.second_off)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('status', 'summaries', 'missing') if k in report}))


if __name__ == '__main__':
    main()
