"""Strict 40-case comparisons against Q, H and the seven archived methods."""
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from jobs import NATIVE_ROOT, Q_REFERENCE, REFERENCE_ANALYSIS


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(arm, label, rows):
    n = len(rows)
    require(n == 40, f'{arm}: expected 40 scores')
    require(all(type(r[t]) is int and 1 <= r[t] <= 5 for r in rows for t in ('sa', 'pc')),
            f'{arm}: invalid score')
    sa, pc = (sum(r[t] for r in rows) for t in ('sa', 'pc'))
    sa_pass = sum(r['sa'] >= 4 for r in rows)
    pc_pass = sum(r['pc'] >= 4 for r in rows)
    joint = sum(r['sa'] >= 4 and r['pc'] >= 4 for r in rows)
    return dict(arm=arm, label=label, n=n, sa_total=sa, pc_total=pc,
                total_score=sa + pc, scalar_scores=2*n,
                sa_mean=sa/n, pc_mean=pc/n, mean_sa_pc=(sa+pc)/(2*n),
                sa_pass_count=sa_pass, sa_pass_percent=100*sa_pass/n,
                pc_pass_count=pc_pass, pc_pass_percent=100*pc_pass/n,
                joint_count=joint, joint_percent=100*joint/n)


def reference_groups():
    baseline = read(REFERENCE_ANALYSIS / 'comparison.json')
    with (REFERENCE_ANALYSIS / 'scores_by_case_40.csv').open(encoding='utf-8-sig') as handle:
        wide = list(csv.DictReader(handle))
    require([r['case_id'] for r in wide] == baseline['case_ids'], 'Archived IDs/order differ')
    groups = {}
    for expected in baseline['summaries']:
        arm = expected['arm']
        rows = [dict(id=r['case_id'], caption=r['caption'],
                     sa=int(r[f'{arm}_sa']), pc=int(r[f'{arm}_pc'])) for r in wide]
        require(summarize(arm, expected['label'], rows) == expected, f'{arm}: archived summary differs')
        groups[arm] = rows
    require(len(groups) == 9, 'Expected all nine archived comparisons')
    return baseline, groups


def validate_manifest(manifest):
    reference = read(Q_REFERENCE / 'manifest.json')
    for key in ('cases', 'settings', 'evaluation', 'input_keys', 'sampling', 'fps', 'frames', 'noise_policy'):
        require(manifest[key] == reference[key], f'Candidate {key} differs from Q initial conditions/protocol')
    require(manifest['checkpoint_step'] == 224 and manifest['inference_reasoning_round'] == 2,
            'Expected final224, round2')
    require(manifest['verifier'] == {'kind': 'none'}, 'Generation verifier must be absent')
    require(manifest['noise_policy'] == 'saved per-case FP32 noise', 'Must replay saved noise')
    baseline, groups = reference_groups()
    require(manifest['evaluation'] == baseline['evaluation'], 'Independent evaluator differs')
    require([(r['id'], r['caption']) for r in manifest['cases']] ==
            [(r['id'], r['caption']) for r in groups['v522_native_q']], 'Reference captions/order differ')
    return baseline, groups


def compare(directory, variant):
    root = Path(directory)
    manifest = read(root / 'manifest.json')
    baseline, groups = validate_manifest(manifest)
    require(manifest['workspace_variant'] == variant, 'Candidate model variant differs')
    generated = read(root / 'summary.json')
    require(generated['status'] == 'completed' and generated['completed_cases'] == 40,
            'Generation must finish all 40 videos before comparison')
    scored = read(root / 'scores.json')
    rows = []
    for case in manifest['cases']:
        row = dict(id=case['id'], caption=case['caption'])
        for task in ('sa', 'pc'):
            record = read(root / 'cases' / case['id'] / f'planned.{task}.json')
            require(record['case_id'] == case['id'] and record['task'] == task, 'Score provenance differs')
            row[task] = record['score']
        rows.append(row)
    require([(r['id'], r['caption'], r['sa'], r['pc']) for r in scored['cases']] ==
            [(r['id'], r['caption'], r['sa'], r['pc']) for r in rows], 'scores.json differs from raw judgments')
    arm = f'v522_workspace_{variant.replace("-", "_")}'
    candidate = summarize(arm, f'v5.2.2-native Q workspace {variant.upper()}, final224, round2', rows)
    for key in ('sa_mean', 'pc_mean'):
        require(candidate[key] == scored['summary'][key], 'Score summary differs')
    require(candidate['joint_percent']/100 == scored['summary']['joint_pass_rate'], 'Joint summary differs')
    summaries = [candidate, *baseline['summaries']]
    deltas, paired = [], []
    for reference in baseline['summaries']:
        ref = reference['arm']
        wins = ties = losses = 0
        for case, old in zip(rows, groups[ref]):
            delta = (case['sa'] + case['pc'] - old['sa'] - old['pc'])/2
            wins += delta > 0
            ties += delta == 0
            losses += delta < 0
            paired.append(dict(case_id=case['id'], caption=case['caption'], reference=ref,
                               sa=case['sa'], pc=case['pc'], reference_sa=old['sa'], reference_pc=old['pc'],
                               sa_delta=case['sa']-old['sa'], pc_delta=case['pc']-old['pc'], mean_delta=delta,
                               joint_delta=int(case['sa']>=4 and case['pc']>=4)-int(old['sa']>=4 and old['pc']>=4)))
        deltas.append(dict(reference=ref, label=reference['label'],
                           sa_delta=round(candidate['sa_mean']-reference['sa_mean'], 4),
                           pc_delta=round(candidate['pc_mean']-reference['pc_mean'], 4),
                           mean_delta=round(candidate['mean_sa_pc']-reference['mean_sa_pc'], 4),
                           joint_delta_pp=round(candidate['joint_percent']-reference['joint_percent'], 4),
                           average_wins=wins, average_ties=ties, average_losses=losses))
    wide = []
    for i, row in enumerate(rows):
        item = dict(case_id=row['id'], caption=row['caption'])
        for key, values in {arm: rows, **groups}.items():
            score = values[i]
            item.update({f'{key}_sa': score['sa'], f'{key}_pc': score['pc'],
                         f'{key}_mean': (score['sa']+score['pc'])/2,
                         f'{key}_joint': int(score['sa']>=4 and score['pc']>=4)})
        wide.append(item)
    result = dict(created_at=datetime.now(timezone.utc).isoformat(), variant=variant,
                  scope=baseline['scope'], mean_definition=baseline['mean_definition'],
                  joint_definition=baseline['joint_definition'], source_checkpoint=manifest['source_checkpoint'],
                  training_experiment_id=manifest['training_experiment_id'],
                  evaluation=manifest['evaluation'], identical_initial_conditions=True,
                  summaries=summaries, candidate_minus_reference=deltas)
    (root / 'comparison.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    write_csv(root / 'summary.csv', summaries)
    write_csv(root / 'scores_by_case_40.csv', wide)
    write_csv(root / 'candidate_minus_reference.csv', deltas)
    write_csv(root / 'paired_differences.csv', paired)
    lines = ['方案,SA,PC,总平均分,联合通过率',
             *[f"{r['label']},{r['sa_mean']:.3f},{r['pc_mean']:.3f},{r['mean_sa_pc']:.4f},{r['joint_percent']:.1f}%" for r in summaries]]
    (root / 'comparison.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    parser.add_argument('--variant', required=True, choices=('b', 'c', 'd-s', 'd-i'))
    args = parser.parse_args()
    compare(args.directory, args.variant)
