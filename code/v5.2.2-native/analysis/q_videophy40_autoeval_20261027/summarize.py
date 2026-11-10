"""Validate Q/H judgments and compare all nine methods on fixed VP001-VP040."""
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
Q_ROOT = ROOT / 'inference_outputs/v522-q-seed128-message-no-verifier-simple_66443/videophy40_final224_round2_shared_noise'
H_ROOT = ROOT / 'inference_outputs/v522-h-content-message-no-verifier-simple_66438/videophy40_final224_round2_shared_noise'
PROTOCOL = ROOT / 'analysis/videophy40_20261026/protocol.json'
HISTORY = ROOT.parent / 'v5.2-occam/analysis/videophy2_seven_arms_20261011/scores_420.csv'
LABELS = {
    'v522_native_q': 'v5.2.2-native Q v30，无 verifier，224 步，2 轮',
    'v522_native_h': 'v5.2.2-native H v29，无 verifier，224 步，2 轮',
    'joint_lora_only': '联合 LoRA，关闭模块',
    'no_intensity': 'Occam no_intensity + LoRA',
    'learned_gate': 'Occam learned_gate + LoRA',
    'ssim_control': 'Occam SSIM + LoRA',
    'independent_lora': '独立 LoRA-only',
    'occam_no_lora': 'Occam 无 LoRA',
    'v52_lora': 'v5.2 + LoRA r32',
}


def read(path):
    return json.loads(path.read_text())


def write_csv(name, rows):
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(arm, rows):
    n = len(rows)
    sa = sum(row['sa'] for row in rows)
    pc = sum(row['pc'] for row in rows)
    joint = sum(row['sa'] >= 4 and row['pc'] >= 4 for row in rows)
    sa_pass = sum(row['sa'] >= 4 for row in rows)
    pc_pass = sum(row['pc'] >= 4 for row in rows)
    return dict(arm=arm, label=LABELS[arm], n=n, sa_total=sa, pc_total=pc,
                total_score=sa + pc, scalar_scores=2 * n,
                sa_mean=sa / n, pc_mean=pc / n,
                mean_sa_pc=(sa + pc) / (2 * n),
                sa_pass_count=sa_pass, sa_pass_percent=100 * sa_pass / n,
                pc_pass_count=pc_pass, pc_pass_percent=100 * pc_pass / n,
                joint_count=joint, joint_percent=100 * joint / n)


def main():
    protocol = read(PROTOCOL)
    expected = {row['id']: row['caption'] for row in protocol['cases']}
    assert len(expected) == 40
    groups = defaultdict(list)
    compatibility = {}
    for arm, root in [('v522_native_q', Q_ROOT), ('v522_native_h', H_ROOT)]:
        manifest = read(root / 'manifest.json')
        assert manifest['evaluation'] == protocol['evaluation']
        assert manifest['settings'] == protocol['settings']
        assert {row['id']: row['caption'] for row in manifest['cases']} == expected
        compatibility[arm] = dict(manifest=str(root / 'manifest.json'),
                                  evaluation_match=True, settings_match=True)
        scored = read(root / 'scores.json')
        for row in scored['cases']:
            case_id = row['id']
            assert row['caption'] == expected[case_id]
            for task in ('sa', 'pc'):
                raw = read(root / 'cases' / case_id / f'planned.{task}.json')
                assert raw['case_id'] == case_id and raw['task'] == task
                assert type(raw['score']) is int and 1 <= raw['score'] <= 5
                assert row[task] == raw['score']
            groups[arm].append(dict(row))
        recalculated = summarize(arm, groups[arm])
        assert recalculated['sa_mean'] == scored['summary']['sa_mean']
        assert recalculated['pc_mean'] == scored['summary']['pc_mean']
        assert recalculated['joint_count'] / 40 == scored['summary']['joint_pass_rate']
    with HISTORY.open(encoding='utf-8-sig') as handle:
        for row in csv.DictReader(handle):
            if row['case_id'] in expected:
                assert row['caption'] == expected[row['case_id']]
                if row['arm'] not in compatibility:
                    old_path = Path(row['video']).parents[2] / 'manifest.json'
                    old = read(old_path)
                    assert old['evaluation'] == protocol['evaluation']
                    assert old['settings'] == protocol['settings']
                    compatibility[row['arm']] = dict(manifest=str(old_path),
                                                     evaluation_match=True, settings_match=True)
                groups[row['arm']].append(dict(id=row['case_id'], caption=row['caption'],
                                              sa=int(row['sa']), pc=int(row['pc'])))
    assert set(groups) == set(LABELS)
    for arm, rows in groups.items():
        assert len(rows) == 40 and {row['id'] for row in rows} == set(expected), arm
        assert all(1 <= row['sa'] <= 5 and 1 <= row['pc'] <= 5 for row in rows)
        rows.sort(key=lambda row: row['id'])
    summaries = [summarize(arm, groups[arm]) for arm in LABELS]
    candidate = summaries[0]
    deltas = []
    for ref in summaries[1:]:
        differences = [(a['sa'] + a['pc'] - b['sa'] - b['pc']) / 2
                       for a, b in zip(groups['v522_native_q'], groups[ref['arm']])]
        deltas.append(dict(reference=ref['arm'], label=ref['label'],
                           sa_delta=round(candidate['sa_mean'] - ref['sa_mean'], 4),
                           pc_delta=round(candidate['pc_mean'] - ref['pc_mean'], 4),
                           mean_delta=round(candidate['mean_sa_pc'] - ref['mean_sa_pc'], 4),
                           joint_delta_pp=round(candidate['joint_percent'] - ref['joint_percent'], 4),
                           average_wins=sum(d > 0 for d in differences),
                           average_ties=sum(d == 0 for d in differences),
                           average_losses=sum(d < 0 for d in differences)))
    wide = []
    lookups = {arm: {row['id']: row for row in rows} for arm, rows in groups.items()}
    for case_id, caption in expected.items():
        row = dict(case_id=case_id, caption=caption)
        for arm in LABELS:
            score = lookups[arm][case_id]
            row.update({f'{arm}_sa': score['sa'], f'{arm}_pc': score['pc'],
                        f'{arm}_mean': (score['sa'] + score['pc']) / 2,
                        f'{arm}_joint': int(score['sa'] >= 4 and score['pc'] >= 4)})
        wide.append(row)
    result = dict(created_at=datetime.now(timezone.utc).isoformat(),
                  evaluation_experiment_id=66519, evaluation_trial_id=67899,
                  training_experiment_id=66443, generation_experiment_id=66517,
                  candidate_directory=str(Q_ROOT), h_directory=str(H_ROOT),
                  h_evaluation_experiment_id=66518, historical_csv=str(HISTORY),
                  scope=protocol['scope'], case_ids=list(expected),
                  mean_definition='sum of all 40 SA scores and 40 PC scores / 80; range 1 to 5; custom mean, not joint pass rate',
                  joint_definition='SA >= 4 and PC >= 4 for the same video',
                  evaluation=protocol['evaluation'], compatibility=compatibility,
                  summaries=summaries, q_minus_reference=deltas,
                  inference_limit='Descriptive comparison for these fixed 40 cases and saved generation noise; no repeated-training or repeated-seed claim.')
    (OUT / 'comparison.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    write_csv('summary.csv', summaries)
    write_csv('q_minus_reference.csv', deltas)
    write_csv('scores_by_case_40.csv', wide)
    print(json.dumps(dict(summaries=summaries, deltas=deltas), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
