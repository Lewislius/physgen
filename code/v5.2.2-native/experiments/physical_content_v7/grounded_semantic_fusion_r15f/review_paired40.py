"""Post-generation review against useful references; never modifies scores or a job."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from report_results import paired, summarize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
IDS = {f'VP{i:03d}' for i in range(1, 41)}


def read40(path):
    data = json.loads(path.read_text())
    if len(data['cases']) != 40 or {v['id'] for v in data['cases']} != IDS:
        raise ValueError(f'Incomplete or different fixed40: {path}')
    if any(v[k] not in range(1, 6) for v in data['cases'] for k in ('sa', 'pc')):
        raise ValueError(f'Invalid raw score: {path}')
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='R15F inference output root containing videophy40/scores.json')
    args = parser.parse_args()
    source = args.output / 'videophy40/scores.json'
    current = read40(source)
    refs = json.loads((ROOT / 'reports/r10_final_evidence.json').read_text())['source_paths']
    paths = {name: Path(refs[key + '/original40']) for name, key in
             [('Wan+LoRA', 'native400'), ('R1', 'R1'), ('R10A', 'R10-A')]}
    for name, folder, experiment, suite in [
        ('R9', 'r1_content_reader_r9', 66968, 'step0400_full40_controls16_physical5_seed42'),
        ('R15B', 'request_bound_fusion_r15b', 67829, 'step0400_original_fixed40_controls16_seed42'),
        ('R15D', 'denoising_matched_fusion_r15d', 67920, 'step0400_original_fixed40_controls16_seed42'),
        ('R15E', 'semantic_addressing_r15e', 67955, 'step0400_original_fixed40_controls16_seed42'),
    ]:
        paths[name] = ROOT / folder / f'inference_outputs/physical_content_{experiment}' / suite / 'videophy40/scores.json'
    fixed16 = {v['id'] for v in json.loads((HERE / 'subset16.json').read_text())['cases']}
    review = dict(candidate='R15F', utc=datetime.now(timezone.utc).isoformat(),
                  source=str(source), full40=summarize(current['cases']),
                  fixed16=summarize([v for v in current['cases'] if v['id'] in fixed16]),
                  remaining24=summarize([v for v in current['cases'] if v['id'] not in fixed16]),
                  references={}, cases=current['cases'],
                  visual_predeclared=['VP004', 'VP010', 'VP027', 'VP034', 'VP038'],
                  visual_review_complete=False, quality_goal_achieved=False,
                  scope='Same fixed I2V40, seed42 and original prompts; paired scores are descriptive. '
                        '16 and remaining24 are separated to expose subset bias. '
                        'Score gains require actual video inspection, including new Joint passes. '
                        'Cross-benchmark acceptance remains separately required.')
    for name, path in paths.items():
        reference = read40(path)
        before = {v['id']: v for v in reference['cases']}
        for case in current['cases']:
            for field in ('caption', 'generation_seed'):
                if case.get(field) != before[case['id']].get(field):
                    raise ValueError(f'{name}/{case["id"]}: unmatched {field}')
        comparison = paired(current, reference)
        comparison.update(source=str(path), summary=summarize(reference['cases']),
                          pc_gain_with_sa_loss=[], sa_gain_with_pc_loss=[], both_improved=[])
        for case in current['cases']:
            previous = before[case['id']]
            ds, dp = case['sa'] - previous['sa'], case['pc'] - previous['pc']
            if dp > 0 and ds < 0:
                comparison['pc_gain_with_sa_loss'].append(case['id'])
            if ds > 0 and dp < 0:
                comparison['sa_gain_with_pc_loss'].append(case['id'])
            if ds > 0 and dp > 0:
                comparison['both_improved'].append(case['id'])
        review['references'][name] = comparison
    review['pc_pass_but_sa_fail'] = [v['id'] for v in current['cases'] if v['pc'] >= 4 and v['sa'] < 4]
    review['sa_pass_but_pc_fail'] = [v['id'] for v in current['cases'] if v['sa'] >= 4 and v['pc'] < 4]
    review['joint_gain_visual_checks'] = {
        name: value['joint']['gains'] for name, value in review['references'].items()}
    destination = args.output / 'paired40_review.json'
    destination.write_text(json.dumps(review, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(report=str(destination), full40=review['full40'],
                          fixed16=review['fixed16'], remaining24=review['remaining24']), ensure_ascii=False))


if __name__ == '__main__':
    main()
