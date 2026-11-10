"""One compact observation of final supplementary evaluation, no job actions."""
import collections,json,subprocess
from datetime import datetime,timedelta,timezone
from pathlib import Path
from observe import active_tasks

ROOT=Path(__file__).resolve().parent
BENCH=ROOT.parent/'cross_benchmark_70_60'


def read(path,default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def main():
    path=ROOT/'cross_prepared.json';prepared=read(path);experiment=prepared['experiment']
    response=subprocess.run(['det','-m','http://10.130.130.5:4096','experiment','describe',str(experiment),'--json'],
        text=True,capture_output=True,timeout=35)
    response.check_returncode();actual=json.loads(response.stdout)[0]['experiment'];now=datetime.now(timezone.utc)
    terminal=actual['state'] in ['STATE_COMPLETED','STATE_ERROR','STATE_CANCELED','STATE_DELETED']
    output=BENCH/'outputs/r15g/seed42';protocol=read(Path(prepared['source'])/'protocol.json')
    by_id={x['id']:x['benchmark'] for x in protocol['cases']}
    done=[p.parent.name for p in (output/'cases').glob('*/video.json')]
    counts=collections.Counter(by_id[x] for x in done)
    gpu=read(BENCH/'allocations'/str(experiment)/'gpu.json',{})
    observation=dict(utc=now.isoformat(),experiment=experiment,state=actual['state'],trial_ids=actual.get('trialIds'),
        active_task_ids=active_tasks(experiment),
        gpu={k:gpu.get(k) for k in ['name','pool','accepted','visible_devices']},
        stage=read(BENCH/'jobs/r15g/status.json',{}),generated=dict(counts),total_generated=len(done),
        next_check_utc=None if terminal else (now+timedelta(seconds=900)).isoformat())
    video_paths=list((output/'cases').glob('*/video.json'))
    if video_paths:
        latest=max(p.stat().st_mtime for p in video_paths)
        observation['last_video_metadata_utc']=datetime.fromtimestamp(latest,timezone.utc).isoformat()
    wmb=read(output/'worldmodelbench_summary.json',{});piq=read(output/'physics_iq_summary.json',{})
    if wmb:observation['WMB70']=wmb['overall']
    if piq:observation['PhysicsIQ60']=dict(count=piq['video_count'],verified=piq['scores']['final_score_view'])
    result=ROOT/'reports/latest_cross_observation.json';previous=read(result,{})
    result.write_text(json.dumps(observation,indent=2)+'\n')
    prepared.update(status=actual['state'],last_observation=observation,next_check_utc=observation['next_check_utc'])
    path.write_text(json.dumps(prepared,indent=2)+'\n')
    for name in ['plan.json','latest_goal_turn.json']:
        p=ROOT.parent/'r11_r12'/name;value=read(p,{})
        value['r15g_cross_runtime']=observation
        main_next=read(ROOT/'prepared.json',{}).get('next_check_utc')
        choices=[x for x in [main_next,observation['next_check_utc']] if x]
        value['next_check_utc']=value['next_routine_check_utc']=min(choices) if choices else None
        p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    changed={k:v for k,v in observation.items() if k not in ['utc','next_check_utc'] and previous.get(k)!=v}
    print(json.dumps(dict(utc=observation['utc'],experiment=experiment,changed=changed,next_check_utc=observation['next_check_utc'])))


if __name__=='__main__':main()
