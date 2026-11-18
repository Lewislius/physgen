"""Refresh factual control results and progress references without accepting a candidate."""
from pathlib import Path
from datetime import datetime, timezone
import json

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'inference_outputs/physical_content_67955/step0400_original_fixed40_controls16_seed42'


def main():
    comparison = json.loads((OUTPUT / 'comparison_report.json').read_text())
    observation = json.loads((ROOT / 'reports/latest_runtime_observation.json').read_text())
    expected = json.loads((ROOT / 'reports/jobs/67955/plan.json').read_text())['modes']
    path = ROOT / 'reports/main40_review.json'
    review = json.loads(path.read_text())
    review.update(updated_utc=datetime.now(timezone.utc).isoformat(),
                  completed_paired16_controls=comparison['controls'],
                  pending_control_modes=[mode for mode in expected if mode not in comparison['controls']],
                  controls_complete=set(expected) <= set(comparison['controls']))
    path.write_text(json.dumps(review, ensure_ascii=False, indent=2) + '\n')
    updates = dict(updated_utc=review['updated_utc'], current_stage=observation['stage'],
                   candidate_stage=observation['stage'], candidate_last_verified_utc=observation['utc'],
                   next_check_utc=observation['next_check_utc'],
                   next_routine_check_utc=observation['next_check_utc'], candidate_review=str(path),
                   completed_r15e_control_modes=list(comparison['controls']),
                   pending_r15e_control_modes=review['pending_control_modes'],
                   candidate_accepted=False, candidate_goal_achieved=False, goal_complete=False)
    if review['controls_complete']:
        updates['status'] = 'r15e_main40_failed_controls_completed_review_pending'
    for name in ['plan.json', 'latest_goal_turn.json']:
        target = ROOT.parent / 'r11_r12' / name
        document = json.loads(target.read_text()); document.update(updates)
        target.write_text(json.dumps(document, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(normal16=comparison['normal_fixed16'],
                         controls={name:row['summary'] for name,row in comparison['controls'].items()},
                         pending=review['pending_control_modes']), ensure_ascii=False))


if __name__ == '__main__':
    main()
