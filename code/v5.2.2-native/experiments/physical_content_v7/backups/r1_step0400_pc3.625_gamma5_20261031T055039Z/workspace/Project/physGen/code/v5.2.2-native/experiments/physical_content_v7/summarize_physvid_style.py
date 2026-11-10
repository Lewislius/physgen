"""Reaggregate completed VideoPhy2-40 scores using PhysVid's >=4 pass rates.

Reads existing evaluator outputs only; never regenerates videos or overwrites scores.
"""
import csv
import datetime
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parents[1]
SOURCE = 'https://github.com/5aurabhpathak/PhysVid/blob/main/physvid/evaluation/videophy2_eval.py'


def aggregate(rows):
    n = len(rows)
    if n != 40 or len({r['id'] for r in rows}) != n:
        raise ValueError('Expected 40 distinct videos')
    if any(r[k] not in range(1, 6) for r in rows for k in ('sa', 'pc')):
        raise ValueError('Expected original 1-to-5 evaluator ratings')
    counts = {k + '_pass_count': sum(r[k] >= 4 for r in rows) for k in ('sa', 'pc')}
    counts['joint_count'] = sum(r['sa'] >= 4 and r['pc'] >= 4 for r in rows)
    return dict(count=n, sa_mean=sum(r['sa'] for r in rows) / n,
        pc_mean=sum(r['pc'] for r in rows) / n, **counts,
        sa_pass_rate=counts['sa_pass_count'] / n,
        pc_pass_rate=counts['pc_pass_count'] / n,
        joint_pass_rate=counts['joint_count'] / n)


def label(path):
    s = str(path)
    if 'native_lora_66781/' in s:
        return '当前匹配基线 Wan+LoRA，400步'
    if 'physical_content_66778/step0400_seed42_r1/' in s:
        return '当前增强版 R1，400步'
    for prefix, name in [('v5_ce_', 'V5-CE'), ('v6_assoc_', 'V6-assoc'), ('v6_write_', 'V6-write')]:
        if prefix in s:
            run = s.split(prefix, 1)[1].split('/', 1)[0]
            steps = 1000 if 'final1000_' in s else 224
            return f'{name}，{steps}步，Search {run}'
    return path.parent.parent.parent.name + ' / ' + path.parent.parent.name


def main():
    paths = subprocess.check_output(['rg', '--files', '-g', 'scores.json', '.'], cwd=BASE, text=True).splitlines()
    methods = []
    for relative in sorted(paths):
        if 'videophy40' not in relative or '/snapshots/' in relative:
            continue
        path = BASE / relative
        data = json.loads(path.read_text())
        rows = data.get('cases', [])
        if len(rows) != 40:
            continue
        metrics = aggregate(rows)
        for key in ('sa_mean', 'pc_mean', 'joint_pass_rate'):
            if key in data.get('summary', {}) and abs(metrics[key] - data['summary'][key]) > 1e-12:
                raise ValueError(f'Aggregate differs from saved scores: {path}, {key}')
        methods.append(dict(label=label(path), source=str(path), source_type='per_video_scores', **metrics))

    # Older experiments outside this revision have already archived exact counts.
    historical = BASE / 'experiments/q_workspace_variants/reports/videophy40_all13_20261028/comparison.json'
    data = json.loads(historical.read_text())
    for row in data['summaries']:
        if row['arm'].startswith(('v522_workspace_', 'v522_native_')):
            continue  # These six experiments were recomputed from per-video files above.
        n = row['n']
        if n != 40:
            raise ValueError('Historical suite is not the full40')
        methods.append(dict(label=row['label'], source=str(historical),
            source_type='archived_exact_counts', count=n, sa_mean=row['sa_mean'], pc_mean=row['pc_mean'],
            sa_pass_count=row['sa_pass_count'], pc_pass_count=row['pc_pass_count'], joint_count=row['joint_count'],
            sa_pass_rate=row['sa_pass_count']/n, pc_pass_rate=row['pc_pass_count']/n,
            joint_pass_rate=row['joint_count']/n))

    pair = json.loads((ROOT / 'reports/videophy400_comparison.json').read_text())
    paired = {}
    for name in ('sa', 'pc', 'joint'):
        def passed(row, arm):
            x = row[arm]
            return x['sa'] >= 4 and x['pc'] >= 4 if name == 'joint' else x[name] >= 4
        transitions = [int(passed(r, 'candidate')) - int(passed(r, 'baseline')) for r in pair['cases']]
        paired[name] = dict(new_passes=sum(x == 1 for x in transitions),
            lost_passes=sum(x == -1 for x in transitions), unchanged=sum(x == 0 for x in transitions),
            delta=sum(transitions)/40,
            new_pass_ids=[r['id'] for r, d in zip(pair['cases'], transitions) if d == 1],
            lost_pass_ids=[r['id'] for r, d in zip(pair['cases'], transitions) if d == -1])
    report = dict(recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        definition=dict(sa='count(SA >= 4) / N', pc='count(PC >= 4) / N',
            joint='count(SA >= 4 AND PC >= 4) / N', threshold=4, author_code=SOURCE,
            normalization='Pass rates, not mean divided by5 or (mean-1)/4'),
        scope='Fixed40 shared-first-image I2V. Same aggregation formula as PhysVid; not identical to its590-prompt T2V and five-seed benchmark protocol.',
        methods=methods, current400_paired_pass_transitions=paired,
        optimization_target='Retain/improve R1 PC mean3.625 and pass rate0.45 while improving SA and physical-process execution; use paired subset references before full40 confirmation. A sacrifice in PC is not accepted merely for an SA increase.')
    output = ROOT / 'reports/videophy40_physvid_style.json'
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    with output.with_suffix('.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(methods[0]))
        writer.writeheader()
        writer.writerows(methods)
    print(json.dumps(dict(output=str(output), method_count=len(methods), current400_paired=paired), ensure_ascii=False))


if __name__ == '__main__':
    main()
