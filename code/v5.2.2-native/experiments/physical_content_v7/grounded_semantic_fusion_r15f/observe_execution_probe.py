"""One compact check of the submitted frozen-checkpoint replay; no job actions."""
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def read(path,default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def main():
    prepared_path=ROOT/'execution_probe_prepared.json'; prepared=read(prepared_path)
    experiment=prepared['experiment']
    response=subprocess.run(['det','-m','http://10.130.130.5:4096','experiment','describe',str(experiment),'--json'],
                            text=True,capture_output=True,timeout=35)
    response.check_returncode(); actual=json.loads(response.stdout)[0]['experiment']
    folder=ROOT/'reports/execution_probe'/str(experiment)
    status=read(folder/'status.json',{}); data=read(folder/'measurements.json',{})
    gpu=read(folder/'gpu.json',{}); now=datetime.now(timezone.utc)
    terminal=actual['state'] in ['STATE_COMPLETED','STATE_ERROR','STATE_CANCELED','STATE_DELETED']
    observation=dict(utc=now.isoformat(),experiment=experiment,state=actual['state'],trial_ids=actual.get('trialIds'),
        end_utc=actual.get('endTime'),gpu={k:gpu.get(k) for k in ['actual','accepted','visible_count']},
        status=status,recorded_layer_rows=len(data.get('layers',[])),recorded_velocity_rows=len(data.get('velocities',[])),
        next_check_utc=None if terminal else (now+timedelta(seconds=900)).isoformat())
    output=ROOT/'reports/latest_execution_probe_observation.json'; previous=read(output,{})
    output.write_text(json.dumps(observation,ensure_ascii=False,indent=2)+'\n')
    prepared.update(last_observation=observation,status=actual['state']);prepared_path.write_text(json.dumps(prepared,ensure_ascii=False,indent=2)+'\n')
    for name in ['plan.json','latest_goal_turn.json']:
        path=ROOT.parent/'r11_r12'/name; document=read(path,{})
        if document.get('candidate_experiment')==68038:
            document.update(active_execution_probe=observation,next_check_utc=observation['next_check_utc'],
                            next_routine_check_utc=observation['next_check_utc'])
            path.write_text(json.dumps(document,ensure_ascii=False,indent=2)+'\n')
    changed={k:v for k,v in observation.items() if k not in ['utc','next_check_utc'] and previous.get(k)!=v}
    print(json.dumps(dict(utc=observation['utc'],experiment=experiment,changed=changed,next_check_utc=observation['next_check_utc']),ensure_ascii=False))


if __name__=='__main__':
    main()
