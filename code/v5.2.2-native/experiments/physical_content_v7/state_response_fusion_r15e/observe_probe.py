"""Single compact read-only observation; no loop, dispatch or background worker."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,subprocess

ROOT=Path(__file__).resolve().parent
metadata=ROOT/'readonly_probe_prepared.json';prepared=json.loads(metadata.read_text())
eid=prepared['experiment_id']
result=subprocess.run(['det','-m','http://10.130.130.5:4096','experiment','describe',str(eid),'--json'],
    capture_output=True,text=True,check=True)
experiment=json.loads(result.stdout)[0]['experiment']
folder=ROOT/'reports/probe_jobs'/str(eid);output=Path(prepared['output'])
read=lambda path:json.loads(path.read_text()) if path.exists() else {}
events=output/'generation.jsonl';latest={}
if events.exists():
    for line in events.read_text().splitlines()[-4:]:
        try:latest=json.loads(line)
        except json.JSONDecodeError:pass
now=datetime.now(timezone.utc)
observation=dict(utc=now.isoformat(),experiment=eid,trials=experiment.get('trialIds'),state=experiment['state'],
    gpu=read(folder/'gpu.json'),stage=read(folder/'status.json').get('status'),
    generated=sum(p.with_suffix('.json').exists() for p in (output/'videophy40/cases').glob('*/planned.mp4')),
    expected=16,latest_event=latest,scores=read(output/'videophy40/scores.json').get('summary'),
    formal_R15E_training=False,next_check_utc=(now+timedelta(seconds=900)).isoformat())
prepared.update(last_observed_utc=observation['utc'],state=observation['state'],
    trial_ids=observation['trials'],actual_gpu=observation['gpu'],next_check_utc=observation['next_check_utc'])
metadata.write_text(json.dumps(prepared,ensure_ascii=False,indent=2)+'\n')
(ROOT/'reports/latest_probe_observation.json').write_text(json.dumps(observation,ensure_ascii=False,indent=2)+'\n')
for name in ['plan.json','latest_goal_turn.json']:
    path=ROOT.parent/'r11_r12'/name;state=json.loads(path.read_text())
    state.update(updated_utc=observation['utc'],next_routine_check_utc=observation['next_check_utc'],
        next_check_utc=observation['next_check_utc'],current_readonly_probe=str(metadata),
        active_jobs=[dict(experiment=eid,kind='readonly_R10A400_execution_probe16',formal_training=False,
                         state=observation['state'])])
    path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(observation,ensure_ascii=False))
