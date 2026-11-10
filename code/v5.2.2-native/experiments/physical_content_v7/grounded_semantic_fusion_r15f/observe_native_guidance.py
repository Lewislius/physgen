"""One compact observation of the frozen-checkpoint guidance intervention."""
import json
import statistics
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def read(path,default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def main():
    path=ROOT/'native_guidance_prepared.json';prepared=read(path)
    experiment=prepared['experiment']
    response=subprocess.run(['det','-m','http://10.130.130.5:4096','experiment','describe',str(experiment),'--json'],
        text=True,capture_output=True,timeout=35)
    response.check_returncode();actual=json.loads(response.stdout)[0]['experiment']
    folder=ROOT/'reports/native_guidance16'/str(experiment)
    output=Path(prepared['output']);suite=output/'controls/native_guidance/videophy40'
    gpu=read(folder/'gpu.json',{});now=datetime.now(timezone.utc)
    terminal=actual['state'] in ['STATE_COMPLETED','STATE_ERROR','STATE_CANCELED','STATE_DELETED']
    observation=dict(utc=now.isoformat(),experiment=experiment,state=actual['state'],trial_ids=actual.get('trialIds'),
        gpu={k:gpu.get(k) for k in ['actual','accepted','visible_count']},status=read(folder/'status.json',{}),
        generated=len(list((suite/'cases').glob('*/planned.json'))),
        next_check_utc=None if terminal else (now+timedelta(seconds=900)).isoformat())
    log=output/'generation.jsonl'
    if log.is_file():
        lines=log.read_text().splitlines()
        if lines:observation['last_event']=json.loads(lines[-1])
    scores=read(suite/'scores.json',{})
    if scores.get('cases'):
        rows=scores['cases'];observation['scores']=dict(count=len(rows),sa=statistics.mean(x['sa'] for x in rows),
            pc=statistics.mean(x['pc'] for x in rows),joint=sum(x['sa']>=4 and x['pc']>=4 for x in rows))
    report=ROOT/'reports/latest_native_guidance_observation.json';previous=read(report,{})
    report.write_text(json.dumps(observation,indent=2)+'\n')
    prepared.update(status=actual['state'],last_observation=observation,next_check_utc=observation['next_check_utc'])
    path.write_text(json.dumps(prepared,indent=2)+'\n')
    for name in ['plan.json','latest_goal_turn.json']:
        mirror=ROOT.parent/'r11_r12'/name;value=read(mirror,{})
        value.update(active_guidance_probe=observation,next_check_utc=observation['next_check_utc'],
            next_routine_check_utc=observation['next_check_utc'])
        mirror.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    changed={k:v for k,v in observation.items() if k not in ['utc','next_check_utc'] and previous.get(k)!=v}
    print(json.dumps(dict(utc=observation['utc'],experiment=experiment,changed=changed,
                         next_check_utc=observation['next_check_utc'])))


if __name__=='__main__':main()
