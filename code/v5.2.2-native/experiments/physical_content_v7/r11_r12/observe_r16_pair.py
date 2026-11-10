"""One due observation per active R16 candidate; no loop, submission or cancellation."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json, subprocess

MASTER=Path(__file__).resolve().parent
ROOT=MASTER.parent
CANDIDATES={'R16A':ROOT/'native_joint_read_r16a',
            'R16A-NOZERO':ROOT/'native_joint_read_r16a_nozero',
            'R16B':ROOT/'native_budget_joint_read_r16b'}
TERMINAL={'STATE_COMPLETED','STATE_ERROR','STATE_CANCELED','STATE_DELETED'}


def read(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def main():
    now=datetime.now(timezone.utc);events={};states={};deadlines=[];capacity_waits={}
    for label,root in CANDIDATES.items():
        prepared=read(root/'prepared.json')
        if not prepared.get('experiment'):
            states[label]={'status':prepared.get('status','preparing'),'experiment':None};continue
        deadline=prepared.get('next_check_utc')
        due=deadline is None or datetime.fromisoformat(deadline)<=now
        if prepared.get('state') not in TERMINAL and due:
            response=subprocess.run(['/usr/bin/python3',str(root/'observe.py'),'--local-only'],
                                    capture_output=True,text=True,timeout=80)
            if response.returncode:
                # Observation failure is not evidence of a terminated job.
                events[label]={'observation_error':response.stderr[-1500:]}
            else:
                events[label]=json.loads(response.stdout)
            prepared=read(root/'prepared.json')
        observation=prepared.get('last_observation',{})
        if prepared.get('status')=='waiting_RTX_capacity':
            capacity_deadline=prepared.get('next_capacity_check_utc')
            if capacity_deadline is None or datetime.fromisoformat(capacity_deadline)<=now:
                response=subprocess.run(['det','-m','http://10.130.130.5:4096','slot','list','--json'],
                                        capture_output=True,text=True,timeout=35)
                response.check_returncode()
                rows=[v for v in json.loads(response.stdout)
                      if v.get('resource_pools')=='blk-96g']
                free=[v for v in rows if v.get('allocation_id')=='FREE'
                      and 'RTX' in v.get('device','') and '6000' in v.get('device','')
                      and v.get('enabled') and not v.get('draining')]
                capacity=dict(utc=now.isoformat(),slots=rows,free_RTX6000=len(free),
                              ready_for_same_source_resubmission=bool(free))
                (root/'reports/latest_capacity.json').write_text(json.dumps(capacity,indent=2)+'\n')
                prepared['last_capacity']=capacity
                prepared['next_capacity_check_utc']=(now+timedelta(seconds=1800)).isoformat()
                (root/'prepared.json').write_text(json.dumps(prepared,indent=2)+'\n')
                events[label+'_capacity']={k:capacity[k] for k in ['utc','free_RTX6000','ready_for_same_source_resubmission']}
            deadlines.append(prepared['next_capacity_check_utc'])
            capacity_waits[label]=dict(prepared=str(root/'prepared.json'),
                                      next_check_utc=prepared['next_capacity_check_utc'],
                                      last_capacity=prepared.get('last_capacity'))
        states[label]=dict(experiment=prepared['experiment'],state=prepared.get('state'),
                          status=prepared.get('status'),prepared=str(root/'prepared.json'),
                          training_step=prepared.get('training_step_observed'),
                          observation=observation)
        if prepared.get('state') not in TERMINAL and prepared.get('next_check_utc'):
            deadlines.append(prepared['next_check_utc'])
    active=[v['experiment'] for v in states.values()
            if v.get('experiment') and v.get('state') not in TERMINAL]
    # Consolidate the two candidates into one roughly30-minute observation;
    # joining a new job may delay the first older-job check by a few minutes.
    next_check=max(deadlines) if deadlines else None
    result=dict(utc=now.isoformat(),candidates=states,active_jobs=active,
                capacity_waits=capacity_waits,next_check_utc=next_check)
    (MASTER/'reports/r16_pair_latest_observation.json').write_text(json.dumps(result,indent=2)+'\n')
    for name in ['plan.json','latest_goal_turn.json']:
        path=MASTER/name;value=read(path)
        value.update(status='R16_pair_active' if active else 'R16_pair_requires_review',
                     authorized_candidates=list(CANDIDATES),candidate_runtimes=states,
                     active_jobs=active,active_experiment_id=active[-1] if active else None,
                     capacity_waits=capacity_waits,
                     active_trial_ids=[trial for v in states.values() if v.get('experiment') in active
                                       for trial in v.get('observation',{}).get('trial_ids',[])],
                     next_check_utc=next_check,next_routine_check_utc=next_check)
        path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(utc=now.isoformat(),observed=events,active_jobs=active,
                         capacity_waits=list(capacity_waits),next_check_utc=next_check)))


if __name__=='__main__':main()
