"""Summarize existing R10 score files; no generation, rescoring, or model calls."""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / 'reports'
MODELS = {
    'native400': (None, 'native_lora_66781', 'native400'),
    'R1': (None, 'physical_content_66778', 'R1_400'),
    'R10-A': ('execution_value_r10', 'physical_content_67003', 'R10_physical_content_67003'),
    'R10-B': ('grounded_execution_r10b', 'physical_content_67008', 'R10B_physical_content_67008'),
}


def load(path, count):
    if not path.exists():
        return None
    document = json.loads(path.read_text())
    rows = {r['id']: r for r in document['cases']}
    if len(rows) != count or len(document['cases']) != count:
        raise ValueError(f'Incomplete or repeated cases: {path}')
    return rows


def summary(rows):
    n = len(rows)
    return dict(count=n, sa_mean=sum(r['sa'] for r in rows.values()) / n,
                pc_mean=sum(r['pc'] for r in rows.values()) / n,
                sa_pass=sum(r['sa'] >= 4 for r in rows.values()),
                pc_pass=sum(r['pc'] >= 4 for r in rows.values()),
                joint_pass=sum(r['sa'] >= 4 and r['pc'] >= 4 for r in rows.values()))


def paired(baseline, candidate):
    if baseline.keys() != candidate.keys():
        raise ValueError('Pair only matching case sets')
    metrics, changed = {}, []
    for metric in ('sa', 'pc'):
        differences = {k: candidate[k][metric] - baseline[k][metric] for k in baseline}
        metrics[metric] = dict(difference=sum(differences.values()) / len(differences),
                              wins=[k for k, v in differences.items() if v > 0],
                              ties=[k for k, v in differences.items() if v == 0],
                              losses=[k for k, v in differences.items() if v < 0])
    joint = dict(gained=[], retained=[], lost=[])
    transition = {f'sa_{s}_pc_{p}': [] for s in ('up', 'same', 'down') for p in ('up', 'same', 'down')}
    sign = lambda value: 'up' if value > 0 else 'down' if value < 0 else 'same'
    for key in baseline:
        a, b = baseline[key], candidate[key]
        ds, dp = b['sa'] - a['sa'], b['pc'] - a['pc']
        transition[f'sa_{sign(ds)}_pc_{sign(dp)}'].append(key)
        ja, jb = a['sa'] >= 4 and a['pc'] >= 4, b['sa'] >= 4 and b['pc'] >= 4
        if ja or jb:
            joint['retained' if ja and jb else 'gained' if jb else 'lost'].append(key)
        if ds or dp:
            changed.append(dict(id=key, caption=a['caption'], baseline_sa=a['sa'], baseline_pc=a['pc'],
                                candidate_sa=b['sa'], candidate_pc=b['pc'], sa_delta=ds, pc_delta=dp))
    return dict(count=len(baseline), baseline=summary(baseline), candidate=summary(candidate),
                metrics=metrics, joint=joint, cross_metric=transition, changed=changed)


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    main_rows, control_rows, transitions = [], [], []
    main_data, controls, paths, pending, comparison = {}, {}, {}, [], {}
    ids = {r['id'] for r in json.loads((ROOT / 'grounded_execution_r10b/subset16.json').read_text())['cases']}
    for model, (directory, run, extension) in MODELS.items():
        parent = ROOT / directory if directory else ROOT
        stem = 'step0400_full40_controls16_physical5_seed42' if directory else 'step0400_seed42_r1'
        final = parent / 'inference_outputs' / run / stem
        locations = {'original40': final / 'videophy40/scores.json'}
        for scope in ('extra20', 'full60'):
            locations[scope] = ROOT / 'videophy60/inference_outputs' / extension / f'{scope}_gamma5_seed42/videophy60/scores.json'
        for scope, path in locations.items():
            rows = load(path, {'original40': 40, 'extra20': 20, 'full60': 60}[scope])
            key = f'{model}/{scope}'
            paths[key] = str(path)
            if rows is None:
                pending.append(key)
                continue
            main_data[key] = rows
            main_rows.append(dict(model=model, scope=scope, **summary(rows)))
        if directory:
            controls[model] = {'normal': {k: v for k, v in main_data[f'{model}/original40'].items() if k in ids}}
            modes = ['reasoning_swapped', 'empty_reasoning', 'global_train_mean',
                     'routing_only' if model == 'R10-A' else 'core_only', 'anchor_only', 'no_c']
            for mode in modes:
                path = final / 'controls' / mode / 'videophy40/scores.json'
                rows = load(path, 16)
                paths[f'{model}/{mode}'] = str(path)
                if rows is None:
                    pending.append(f'{model}/{mode}')
                elif rows.keys() != ids:
                    raise ValueError(f'Unexpected control subset: {path}')
                else:
                    controls[model][mode] = rows
            for mode, rows in controls[model].items():
                control_rows.append(dict(model=model, mode=mode, **summary(rows)))
                if mode != 'normal':
                    comparison[f'{model}/normal_minus_{mode}'] = paired(rows, controls[model]['normal'])
            for before, after in [('no_c', 'anchor_only'), ('anchor_only', 'routing_only'), ('anchor_only', 'core_only')]:
                if before in controls[model] and after in controls[model]:
                    comparison[f'{model}/{after}_minus_{before}'] = paired(controls[model][before], controls[model][after])
    for scope in ('original40', 'extra20', 'full60'):
        for before, after in [('R1', 'R10-A'), ('R1', 'R10-B'), ('R10-A', 'R10-B'), ('native400', 'R10-A'), ('native400', 'R10-B')]:
            if f'{before}/{scope}' in main_data and f'{after}/{scope}' in main_data:
                key = f'{scope}/{after}_minus_{before}'
                comparison[key] = paired(main_data[f'{before}/{scope}'], main_data[f'{after}/{scope}'])
                transitions.extend(dict(comparison=key, **row) for row in comparison[key]['changed'])
    record = dict(recorded_utc=datetime.now(timezone.utc).isoformat(), source='Existing raw scores.json case records',
                  protocol='Same400, gamma5, seed42, shared first-image I2V. Full60 contains original40 plus extra20.',
                  main=main_rows, controls=control_rows, paired=comparison, pending=pending, source_paths=paths,
                  interpretation='Descriptive paired changes. No new generation, rescoring, bootstrap or model tests. Inference controls act on jointly trained checkpoints; no_c is not independently trained native LoRA.')
    REPORT.mkdir(exist_ok=True)
    (REPORT / 'r10_final_evidence.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    write_csv(REPORT / 'r10_main400.csv', main_rows)
    write_csv(REPORT / 'r10_controls16.csv', control_rows)
    write_csv(REPORT / 'r10_transitions400.csv', transitions)
    print(json.dumps(dict(main_suites=len(main_rows), completed_control_rows=len(control_rows), pending=pending)))


if __name__ == '__main__':
    main()
