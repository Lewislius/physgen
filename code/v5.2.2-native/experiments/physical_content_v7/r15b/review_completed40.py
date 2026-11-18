"""Complete fixed40 and paired16 review, including the useful R9 reference."""
from pathlib import Path
from datetime import datetime, timezone
import json

from report_results import paired, summarize

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'inference_outputs/physical_content_67829/step0400_original_fixed40_controls16_seed42'
EXPECTED = {f'VP{i:03d}' for i in range(1, 41)}
CONTROLS = ('no_c', 'direct_physical_read', 'state_value_off', 'reasoning_swapped',
            'empty_reasoning', 'global_train_mean')


def read_scores(path, expected):
    data = json.loads(path.read_text())
    assert len(data['cases']) == len(expected)
    assert {case['id'] for case in data['cases']} == expected
    return data


def changes(after, before):
    old = {case['id']: case for case in before['cases']}
    return [dict(id=case['id'], sa=case['sa'] - old[case['id']]['sa'],
                 pc=case['pc'] - old[case['id']]['pc'],
                 joint_after=case['sa'] >= 4 and case['pc'] >= 4,
                 joint_before=old[case['id']]['sa'] >= 4 and old[case['id']]['pc'] >= 4)
            for case in after['cases']]


def main():
    actual = read_scores(OUTPUT / 'videophy40/scores.json', EXPECTED)
    ids16 = {case['id'] for case in json.loads((ROOT / 'subset16.json').read_text())['cases']}
    normal16 = dict(cases=[case for case in actual['cases'] if case['id'] in ids16])
    assert len(normal16['cases']) == 16
    refs = json.loads((ROOT.parent / 'reports/r10_final_evidence.json').read_text())['source_paths']
    refs = {name: Path(refs[name + '/original40']) for name in ('native400', 'R1', 'R10-A')}
    refs['R9'] = ROOT.parent / 'r1_content_reader_r9/inference_outputs/physical_content_66968/step0400_full40_controls16_physical5_seed42/videophy40/scores.json'
    refs['R15A'] = ROOT.parent / 'grounded_fusion_r15a/inference_outputs/physical_content_67762/step0400_original_fixed40_controls16_seed42/videophy40/scores.json'
    report = dict(utc=datetime.now(timezone.utc).isoformat(), experiment=67829,
                  full40=summarize(actual['cases']), normal16=summarize(normal16['cases']),
                  references={}, controls16={}, pending_controls=[])
    for name, path in refs.items():
        reference = read_scores(path, EXPECTED)
        report['references'][name] = dict(summary=summarize(reference['cases']),
            candidate_minus_reference=paired(actual, reference), cases=changes(actual, reference), source=str(path))
    for mode in CONTROLS:
        path = OUTPUT / 'controls' / mode / 'videophy40/scores.json'
        if not path.is_file():
            report['pending_controls'].append(mode)
            continue
        control = read_scores(path, ids16)
        report['controls16'][mode] = dict(summary=summarize(control['cases']),
            normal_minus_control=paired(normal16, control), cases=changes(normal16, control), source=str(path))
    s = report['full40']
    report['targets'] = dict(pc=s['pc_mean'] >= 3.8, sa=s['sa_mean'] >= 3.3,
                             joint=s['joint_pass_count'] >= 12)
    report['numeric_primary_accepted'] = all(report['targets'].values())
    report['overall_goal_achieved'] = False
    report['limitations'] = [
        'Fixed40 I2V evaluation, not the full official 590-caption T2V benchmark.',
        'Controls intervene on this jointly trained checkpoint; no_c is not the separately trained native400 baseline.',
        'Training mean fixes C but retains the own request keys and relation prior; content and amplitude both change.',
        'Full cross-benchmark and actual-video evidence are required in addition to these scores.'
    ]
    path = ROOT / 'reports/main40_review.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ['full40', 'normal16', 'targets', 'pending_controls']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
