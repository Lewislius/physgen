"""Review short-run generation and PC preservation against completed references."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import importlib.util
import json

ROOT = Path(__file__).resolve().parents[1]
EARLY = ROOT / 'early_execution_r3'
spec = importlib.util.spec_from_file_location('physical_content_comparison', ROOT / 'compare_iteration.py')
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def summarize(folder):
    score_file = folder / 'scores.json'
    result = dict(path=str(folder), generated=len(list((folder / 'cases').glob('*/planned.mp4'))))
    if not score_file.is_file():
        return dict(**result, status='awaiting_completed_evaluation')
    scores = json.loads(score_file.read_text())
    plan = json.loads((folder / 'manifest.json').read_text())
    rows = scores['cases']
    if len({x['id'] for x in rows}) != len(rows) or {x['id'] for x in rows} != {x['id'] for x in plan['cases']}:
        raise ValueError(f'Incomplete or duplicate evaluation: {folder}')
    if any(x[k] not in [1, 2, 3, 4, 5] for x in rows for k in ['sa', 'pc']):
        raise ValueError(f'Invalid VideoPhy2 score: {folder}')
    n = len(rows)
    return dict(**result, status='completed', count=n,
        sa_mean=sum(x['sa'] for x in rows)/n, pc_mean=sum(x['pc'] for x in rows)/n,
        sa_pass_rate=sum(x['sa'] >= 4 for x in rows)/n,
        pc_pass_rate=sum(x['pc'] >= 4 for x in rows)/n,
        joint_pass_rate=sum(x['sa'] >= 4 and x['pc'] >= 4 for x in rows)/n)


def assess_reference(reference, candidate):
    pc_keys = ['pc_mean', 'pc_pass_rate']
    sa_keys = ['sa_mean', 'sa_pass_rate', 'joint_pass_rate']
    pc_preserved = all(candidate[k] >= reference[k] for k in pc_keys)
    semantic_preserved = all(candidate[k] >= reference[k] for k in sa_keys)
    improved = any(candidate[k] > reference[k] for k in pc_keys + sa_keys)
    return dict(pc_preserved_on_this_set=pc_preserved,
        semantic_and_joint_preserved_on_this_set=semantic_preserved,
        at_least_one_metric_improved=improved,
        eligible_for_full40_confirmation=pc_preserved and semantic_preserved and improved,
        automatic_replacement=False,
        note='Subset means are compared with the identical subset; these are screening results, not a full40 or cross-seed benefit claim.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate', default='physical_content_66870')
    p.add_argument('--baseline', default='native_lora_66866')
    p.add_argument('--output', type=Path, default=EARLY / 'reports/generation_review.json')
    args = p.parse_args()
    candidate_root = EARLY / 'inference_outputs' / args.candidate / 'step0064_subset16_physical5_seed42'
    baseline_root = EARLY / 'inference_outputs' / args.baseline / 'step0064_subset16_physical1_seed42'
    # This historical baseline runner has no physical branch; native text CFG is5.
    if not baseline_root.exists():
        baseline_root = EARLY / 'inference_outputs' / args.baseline / 'step0064_subset16_physical5_seed42'
    folders = dict(
        baseline64=baseline_root / 'videophy40',
        candidate64=candidate_root / 'videophy40',
        candidate64_no_c=candidate_root / 'controls/no_c/videophy40',
        candidate64_unrelated=candidate_root / 'controls/reasoning_swapped/videophy40',
        r1_reference16=ROOT / 'module_controls16/r1_step0400_guidance5_seed42/references/real',
        r1_reference40=ROOT / 'inference_outputs/physical_content_66778/step0400_seed42_r1/videophy40',
    )
    suites = {key:summarize(path) for key,path in folders.items()}
    results = {}
    for label, reference, candidate in [
        ('same_budget64', 'baseline64', 'candidate64'),
        ('r1_reference_preservation16', 'r1_reference16', 'candidate64'),
        ('correct_vs_no_c', 'candidate64_no_c', 'candidate64'),
        ('correct_vs_unrelated', 'candidate64_unrelated', 'candidate64'),
    ]:
        if all(suites[k]['status'] == 'completed' for k in [reference, candidate]):
            results[label] = comparison.compare(folders[reference], folders[candidate])
    content_effects = {}
    for label in ['correct_vs_no_c', 'correct_vs_unrelated']:
        if label in results:
            metrics = results[label]['metrics']
            content_effects[label] = dict(
                positive_pc_difference=metrics['pc']['difference'] > 0,
                positive_sa_difference=metrics['sa']['difference'] > 0,
                pc_regression=metrics['pc']['difference'] < 0,
                equal_discrete_scores=all(metrics[k]['wins'] == 0 and metrics[k]['losses'] == 0 for k in ['sa','pc']),
                warning='A difference after an intervention is not by itself useful physical-content benefit; also compare the native baseline and the generated process.')
    eligibility = (assess_reference(suites['r1_reference16'], suites['candidate64'])
        if suites['candidate64']['status'] == 'completed' else dict(status='pending'))
    visual_path = EARLY / 'reports/visual_process64.json'
    visual = (json.loads(visual_path.read_text()) if visual_path.exists()
        else dict(status='pending_new_generation'))
    report = dict(recorded_utc=datetime.now(timezone.utc).isoformat(), gamma=5,
        policy='Preserve the R1 PC benefit; seek further PC and SA gains. SA-only gains with PC regressions do not replace R1.',
        full40_reference=suites['r1_reference40'], suites=suites, paired_comparisons=results,
        subset_reference_review=eligibility, content_interventions=content_effects,
        actual_process_and_visual_review=dict(path=str(visual_path),
            status=visual['status'], reviewed_cases=visual.get('reviewed_cases',0),
            interpretation=visual.get('interpretation')),
        automatic_training_extension=False, automatic_replacement=False,
        distinction='Baseline64 comparison tests the short revised recipe; R1-400 reference comparison tests usability at a different training budget. This is not a single-component training ablation.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'paired_comparisons'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
