"""Persist the last compact observation; no cluster calls, loops or dispatch."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parent
observation=json.loads((ROOT/'reports/latest_runtime_observation.json').read_text())
step=observation.get('step')
run=ROOT/'runs'/f'physical_content_{observation["experiment"]}'
training=json.loads((run/'status.json').read_text()) if (run/'status.json').is_file() else {}
completed=training.get('status')=='complete' and (run/'step0400.pt').is_file()
prepared_path=ROOT/'prepared.json'
prepared=json.loads(prepared_path.read_text())
prepared.update(verified_step=step,verified_utc=observation['utc'],next_check_utc=observation['next_check_utc'],
                phase=training.get('phase'),training_complete=completed)
if completed:prepared['status']='own400_evaluation_active' if observation['state']=='STATE_RUNNING' else observation['state']
prepared_path.write_text(json.dumps(prepared,ensure_ascii=False,indent=2)+'\n')
for name in ('plan.json','latest_goal_turn.json'):
    path=ROOT.parent/'r11_r12'/name;data=json.loads(path.read_text())
    data.update(updated_utc=observation['utc'],candidate_last_verified_optimizer_step=step,
        candidate_last_verified_utc=observation['utc'],candidate_phase=training.get('phase'),
        candidate_state=observation['state'],candidate_stage=observation.get('stage'),
        candidate_training_complete=completed,next_routine_check_utc=observation['next_check_utc'],
        next_check_utc=observation['next_check_utc'],candidate_runtime_observation=str(ROOT/'reports/latest_runtime_observation.json'),
        main40_generated_at_last_check=observation.get('counts',{}).get('40',0))
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(step=step,training_complete=completed,next_check=observation['next_check_utc'])))
