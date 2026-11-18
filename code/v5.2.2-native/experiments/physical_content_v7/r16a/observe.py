"""One compact authoritative observation. Does not wait, submit or cancel jobs."""
import json,statistics,subprocess,sys
from datetime import datetime,timedelta,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def read(path,default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def active_tasks(experiment):
    response=subprocess.run(['det','-m','http://10.130.130.5:4096','task','list','--json'],
        text=True,capture_output=True,timeout=35)
    response.check_returncode()
    tasks=json.loads(response.stdout)
    return [task_id for task_id in tasks if task_id.startswith(str(experiment)+'.')]


def tail_event(path,kind=None):
    if not path.is_file():return None
    with path.open('rb') as stream:
        stream.seek(max(0,path.stat().st_size-16000));lines=stream.read().decode(errors='replace').splitlines()
    for line in reversed(lines):
        try:value=json.loads(line)
        except json.JSONDecodeError:continue
        if kind is None or value.get('kind')==kind or value.get('event')==kind:return value
    return None


def summary(path):
    value=read(path,{})
    if not value.get('cases'):return None
    rows=value['cases']
    return dict(count=len(rows),sa=statistics.mean(x['sa'] for x in rows),pc=statistics.mean(x['pc'] for x in rows),
        joint=sum(x['sa']>=4 and x['pc']>=4 for x in rows))


def main():
    local_only='--local-only' in sys.argv
    prepared_path=ROOT/'prepared.json';prepared=read(prepared_path)
    experiment=prepared['experiment'];now=datetime.now(timezone.utc)
    response=subprocess.run(['det','-m','http://10.130.130.5:4096','experiment','describe',str(experiment),'--json'],
        text=True,capture_output=True,timeout=35)
    response.check_returncode();actual=json.loads(response.stdout)[0]['experiment']
    run_id=prepared.get('run_id',f'physical_content_{experiment}')
    job=ROOT/'reports/jobs'/str(experiment);run=ROOT/'runs'/run_id
    output=ROOT/'inference_outputs'/run_id/'step0400_original_fixed40_controls16_seed42'
    terminal=actual['state'] in ['STATE_COMPLETED','STATE_ERROR','STATE_CANCELED','STATE_DELETED']
    gpu=read(job/'gpu.json',{});preflight=read(job/'gpu_preflight/status.json',{})
    observation=dict(utc=now.isoformat(),experiment=experiment,state=actual['state'],trial_ids=actual.get('trialIds'),
        active_task_ids=active_tasks(experiment),
        gpu={k:gpu.get(k) for k in ['actual','accepted','visible_count']},
        stage=read(job/'status.json',{}).get('status'),training=read(run/'status.json',{}),
        preflight={k:preflight.get(k) for k in ['status','error','parameters'] if k in preflight},
        next_check_utc=None if terminal else (now+timedelta(seconds=1800)).isoformat())
    train=tail_event(run/'train.jsonl','train')
    if train:observation['last_train']={k:train.get(k) for k in ['step','phase','seconds','peak_gib','generation']}
    generated={};scores={}
    for label,folder in [('main40',output/'videophy40'),('extra20',output/'videophy60'),
                         *[(x.parent.name,x) for x in (output/'controls').glob('*/videophy40')]]:
        count=len(list((folder/'cases').glob('*/planned.json')))
        if count:generated[label]=count
        score=summary(folder/'scores.json')
        if score:scores[label]=score
    full60=summary(output/'full60/videophy60/scores.json')
    if full60:scores['full60']=full60
    observation.update(generated=generated,scores=scores)
    event=tail_event(output/'generation.jsonl')
    if event:
        observation['last_generation']={k:event.get(k) for k in ['kind','case','mode','step','time'] if k in event}
        if event.get('time'):
            observation['seconds_since_generation_event']=round(now.timestamp()-event['time'])
    destination=ROOT/'reports/latest_observation.json';previous=read(destination,{})
    destination.parent.mkdir(parents=True,exist_ok=True);destination.write_text(json.dumps(observation,indent=2)+'\n')
    training_complete=observation['training'].get('status')=='complete' and observation['training'].get('step')==400
    work_status=('terminal_'+actual['state'].removeprefix('STATE_').lower() if terminal else
                 observation.get('stage') or 'waiting_allocation_or_startup')
    prepared.update(last_observation=observation,state=actual['state'],status=work_status,
        training_step_observed=observation['training'].get('step'),training_complete=training_complete,
        next_check_utc=observation['next_check_utc'])
    prepared_path.write_text(json.dumps(prepared,indent=2)+'\n')
    if not local_only:
        for name in ['plan.json','latest_goal_turn.json']:
            path=ROOT.parent/'r11_r12'/name;value=read(path,{})
            value.update(candidate='R16A',candidate_experiment=experiment,candidate_runtime=observation,
                status='R16A_'+work_status,active_experiment_id=None if terminal else experiment,
                active_jobs=[] if terminal else [experiment],active_trial_ids=[] if terminal else actual.get('trialIds',[]),
                candidate_training_complete=training_complete,training_step_observed=observation['training'].get('step'),
                next_check_utc=observation['next_check_utc'],next_routine_check_utc=observation['next_check_utc'])
            path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    changed={k:v for k,v in observation.items() if k not in ['utc','next_check_utc'] and previous.get(k)!=v}
    print(json.dumps(dict(utc=observation['utc'],experiment=experiment,changed=changed,next_check_utc=observation['next_check_utc'])))


if __name__=='__main__':main()
