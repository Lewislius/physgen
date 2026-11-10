"""One authoritative observation of the two-video failure-isolation job."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,subprocess

ROOT=Path(__file__).resolve().parent
def read(path,default=None):return json.loads(path.read_text()) if path.exists() else default

def main():
    prepared=read(ROOT/'prepared.json');terminal={'STATE_COMPLETED','STATE_ERROR','STATE_CANCELED','STATE_DELETED'}
    if prepared.get('state') in terminal:
        print(json.dumps(dict(reused_terminal_observation=prepared['last_observation'])));return
    id=prepared['experiment'];now=datetime.now(timezone.utc)
    response=subprocess.run(['det','-m','http://10.130.130.5:4096','experiment','describe',str(id),'--json'],
                            capture_output=True,text=True,check=True,timeout=35)
    actual=json.loads(response.stdout)[0]['experiment'];job=ROOT/'reports/jobs'/str(id)
    output=ROOT/'outputs';generated={}
    for label,path in [('real',output/'videophy40'),('no_c',output/'controls/no_c/videophy40')]:
        generated[label]=len(list((path/'cases').glob('*/planned.json')))
    observation=dict(utc=now.isoformat(),experiment=id,state=actual['state'],trial_ids=actual.get('trialIds'),
        gpu=read(job/'gpu.json',{}),stage=read(job/'status.json',{}).get('status'),generated=generated,
        next_check_utc=None if actual['state'] in terminal else (now+timedelta(seconds=900)).isoformat())
    prepared.update(state=actual['state'],last_observation=observation,next_check_utc=observation['next_check_utc'])
    (ROOT/'prepared.json').write_text(json.dumps(prepared,indent=2)+'\n')
    master=ROOT.parent.parent/'r11_r12'
    for name in ['plan.json','latest_goal_turn.json']:
        path=master/name;value=read(path,{})
        value.update(active_auxiliary_jobs=[] if actual['state'] in terminal else [id],
                     auxiliary_runtime_isolation=dict(prepared=str(ROOT/'prepared.json'),observation=observation))
        path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(observation))

if __name__=='__main__':main()
