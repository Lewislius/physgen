"""Collect completed first-wave evidence and case-paired comparisons.

This is read-only with respect to videos, checkpoints, manifests and scores.
It never launches jobs, downloads models, or treats partial suites as complete.
"""
from pathlib import Path
import argparse
import csv
import datetime
import json
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from compare_iteration import compare

MODES = ['real', 'reasoning_swapped', 'empty_reasoning', 'mean', 'no_c', 'anchor_only']
CONTROL_IDS = ['VP004', 'VP010', 'VP015', 'VP034']


def inspect_suite(path, expected):
    result = dict(path=str(path), expected_cases=expected, complete=False)
    videos = {p.parent.name for p in path.glob('cases/*/planned.mp4')
              if p.with_suffix('.json').is_file()}
    result['generated_cases'] = len(videos)
    if not (path / 'manifest.json').is_file():
        return dict(result, pending='generation and manifest')
    plan = json.loads((path / 'manifest.json').read_text())
    planned = [row['id'] for row in plan['cases']]
    if len(planned) != expected or len(set(planned)) != expected:
        raise ValueError(f'{path}: wrong case set for this suite')
    if expected == 4 and sorted(planned) != CONTROL_IDS:
        raise ValueError(f'{path}: content controls must retain the preselected cases')
    result['missing_videos'] = sorted(set(planned) - videos)
    if not (path / 'scores.json').is_file():
        return dict(result, pending='complete generation and independent scoring')
    scores = json.loads((path / 'scores.json').read_text())
    rows = {row['id']: row for row in scores['cases']}
    if len(scores['cases']) != expected or set(rows) != set(planned) or result['missing_videos']:
        raise ValueError(f'{path}: scores do not cover the complete generated suite')
    for case in plan['cases']:
        row = rows[case['id']]
        if row['caption'] != case['caption']:
            raise ValueError(f'{path}: scoring caption differs from generation caption')
        for task in ['sa', 'pc']:
            record = json.loads((path / 'cases' / case['id'] / f'planned.{task}.json').read_text())
            if record['case_id'] != case['id'] or record['score'] != row[task] or row[task] not in range(1, 6):
                raise ValueError(f'{path}: inconsistent per-case evaluator result')
    threshold = plan['evaluation']['joint_threshold']
    summary = dict(count=expected,
        sa_mean=sum(x['sa'] for x in rows.values()) / expected,
        pc_mean=sum(x['pc'] for x in rows.values()) / expected,
        joint_pass_rate=sum(x['sa'] >= threshold and x['pc'] >= threshold for x in rows.values()) / expected)
    for key, value in summary.items():
        if abs(scores['summary'][key] - value) > 1e-12:
            raise ValueError(f'{path}: aggregate {key} does not agree with case results')
    summary.update(sa_pass_rate=sum(x['sa'] >= threshold for x in rows.values()) / expected,
        pc_pass_rate=sum(x['pc'] >= threshold for x in rows.values()) / expected)
    return dict(result, complete=True, summary=summary,
        checkpoint_step=plan['checkpoint_step'], seed=plan['generation_seed'], mode=plan['mode'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate1000-run', default='physical_content_66856')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/first_wave_evidence.json')
    args = parser.parse_args()
    outputs = ROOT / 'inference_outputs'
    paths = {
        'native400_heldout8': (outputs / 'native_lora_66781/step0400_real_heldout8_seed42/heldout8', 8),
        'native400_videophy40': (outputs / 'native_lora_66781/step0400_seed42_r1/videophy40', 40),
        'candidate400_heldout8': (outputs / 'physical_content_66778/step0400_real_heldout8_seed42/heldout8', 8),
        'candidate400_videophy40': (outputs / 'physical_content_66778/step0400_seed42_r1/videophy40', 40),
        'candidate1000_heldout8': (outputs / args.candidate1000_run / 'step1000_real_heldout8_seed42/heldout8', 8),
        'candidate1000_videophy40': (outputs / args.candidate1000_run / 'step1000_seed42_r1/videophy40', 40),
    }
    for step, parent in [(400, outputs / 'physical_content_66778/step0400_controls_seed42_r1'),
                         (1000, outputs / args.candidate1000_run / 'step1000_controls_seed42_r1')]:
        for mode in MODES:
            path = parent / 'videophy40' if mode == 'real' else parent / 'controls' / mode / 'videophy40'
            paths[f'candidate{step}_control_{mode}'] = path, 4
    evidence = {name: inspect_suite(path, count) for name, (path, count) in paths.items()}
    comparisons = {}
    for suite in ['heldout8', 'videophy40']:
        for first, second, label in [
            (f'native400_{suite}', f'candidate400_{suite}', f'matched400_{suite}'),
            (f'candidate400_{suite}', f'candidate1000_{suite}', f'candidate_scaling_{suite}')]:
            if evidence[first]['complete'] and evidence[second]['complete']:
                comparisons[label] = compare(paths[first][0], paths[second][0])
                comparisons[label]['training_comparison'] = (
                    'Same 400 total updates; candidate has128 interface updates and272 Wan-LoRA updates, native baseline400 Wan-LoRA updates.'
                    if label.startswith('matched') else
                    'Within-candidate400-to1000 continuation; not a matched1000 baseline comparison.')
    for step in [400, 1000]:
        own = f'candidate{step}_control_real'
        for mode in MODES[1:]:
            other = f'candidate{step}_control_{mode}'
            if evidence[own]['complete'] and evidence[other]['complete']:
                comparisons[f'candidate{step}_correct_vs_{mode}'] = compare(paths[other][0], paths[own][0])
    result = dict(recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        first_wave_complete=all(item['complete'] for item in evidence.values()),
        pending_suites=[name for name, item in evidence.items() if not item['complete']],
        suites=evidence, comparisons=comparisons,
        interpretation={
            'videophy40': 'Fixed40 shared-first-image I2V protocol; not official T2V leaderboard.',
            'heldout8': 'Preselected real validation first images; separate long-caption completion budget.',
            'mean_control': 'Own sample token-mean C, not a global mean across different samples.',
            'reasoning_controls': 'Own image/request/native T5 are retained; empty reasoning still has image/request information.',
            'no_c': 'Physical outputs disabled in both CFG branches in the same co-adapted checkpoint.',
            'anchor_only': 'Learned C projection removed; grounded semantic anchor path retained.',
            'statistics': 'One seed. Four-case controls are descriptive; inspect processes and stability alongside SA/PC/Joint.',
            'goal': 'Completion of this evaluation batch does not establish the generation-improvement goal.',
            'next_revision': 'Latest user correction: use the complete400-step evidence to revise now;1000-step continuation is a scaling reference, not a prerequisite. Require short-budget paired generation before extending new revisions.'})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(args.output)
    table = args.output.with_suffix('.cases.csv')
    with table.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['comparison', 'id', 'caption', 'reference_sa', 'reference_pc',
            'candidate_sa', 'candidate_pc', 'reference_joint', 'candidate_joint'])
        writer.writeheader()
        for name, comparison in comparisons.items():
            for row in comparison['cases']:
                writer.writerow(dict(comparison=name, id=row['id'], caption=row['caption'],
                    reference_sa=row['baseline']['sa'], reference_pc=row['baseline']['pc'],
                    candidate_sa=row['candidate']['sa'], candidate_pc=row['candidate']['pc'],
                    reference_joint=row['baseline_joint'], candidate_joint=row['candidate_joint']))
    print(json.dumps(dict(first_wave_complete=result['first_wave_complete'],
        completed_suites=sum(x['complete'] for x in evidence.values()), total_suites=len(evidence),
        pending_suites=result['pending_suites'], output=str(args.output)), ensure_ascii=False))


if __name__ == '__main__':
    main()
