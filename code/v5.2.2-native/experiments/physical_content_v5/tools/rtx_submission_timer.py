#!/usr/bin/env python3
"""Every 20 minutes, obtain an RTX6000 allocation for this V5 run only."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import fcntl
import json
import math
import os
import re
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports'
MANIFEST=REPORTS/'latest_submission.json'
STATE=REPORTS/'rtx_submission_timer.json'
DET='/home/liuzhirui/.local/bin/det'
MASTER='http://10.130.130.5:4096'
INTERVAL=20*60


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path,default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def save(path,value):
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    os.replace(temporary,path)


def log(event,**values):
    print(json.dumps(dict(time=utc(),event=event,**values),ensure_ascii=False),flush=True)


def det(*args,as_json=False):
    result=subprocess.run([DET,'-m',MASTER,*map(str,args)],capture_output=True,text=True,timeout=90)
    if result.returncode:
        # Do not dump configurations, credentials, or unrelated task logs.
        raise RuntimeError(f'det {args[0]} {args[1]} returned {result.returncode}')
    return json.loads(result.stdout) if as_json else result.stdout


def allowed(name):
    name=re.sub('[^A-Z0-9]','',name.upper())
    return 'H100' not in name and ('RTXPRO6000' in name or 'RTX6000BLACKWELL' in name)


def free_rtx_slots():
    rows=det('slot','list','--json',as_json=True)
    if isinstance(rows,dict):rows=rows['slots']
    return [r for r in rows if 'blk-96g' in r.get('resource_pools','')
            and r.get('enabled') and not r.get('draining')
            and r.get('allocation_id') in (None,'','FREE') and allowed(r.get('device',''))]


def experiment_state(exp_id):
    rows=det('experiment','describe',exp_id,'--json',as_json=True)
    entry=rows[0] if isinstance(rows,list) else rows
    return entry['experiment']


def training_rows(path):
    result=[]
    if not path.exists():return result
    for line in path.read_text().splitlines():
        try:row=json.loads(line)
        except json.JSONDecodeError:continue  # Concurrent final line append.
        result.append(row)
    return result


def cycle():
    """One capacity check; allow startup/rejection handling within this cycle."""
    deadline=time.monotonic()+10*60
    while True:
        manifest=read(MANIFEST)
        if manifest.get('status')=='completed_rtx6000_running':return True
        if manifest.get('status')=='needs_reconciliation_submission':
            log('submission_outcome_needs_reconciliation')
            return False
        attempt=manifest['attempts'][-1]
        exp_id=attempt['experiment_id'];run=Path(attempt['run'])
        gpu=read(run/'gpu.json')
        task=experiment_state(exp_id)
        attempt['trial_id']=(task.get('trialIds') or [None])[0]
        manifest['last_checked_at_utc']=utc()

        if gpu and gpu.get('accepted') and gpu.get('visible_count')==1 and allowed(gpu.get('actual','')):
            attempt['gpu']=gpu
            status=read(run/'status.json',{})
            rows=training_rows(run/'train.jsonl')
            updates=[r for r in rows if r.get('kind')=='train']
            finite=all(math.isfinite(r[k]) for r in updates[-2:] for k in ('fm','aux','weighted_aux'))
            finite=finite and all(math.isfinite(x) for r in updates[-2:] for x in r['gradients'].values())
            live=task['state'] in ('STATE_RUNNING','STATE_COMPLETED')
            if len(updates)>=2 and finite and live and status.get('status') in ('training','complete'):
                attempt.update(status='training_confirmed',first_optimizer_update=updates[0],latest_confirmed_update=updates[-1])
                manifest.update(status='completed_rtx6000_running',active_experiment_id=exp_id,
                                active_run=str(run),verified_at_utc=utc())
                save(MANIFEST,manifest)
                checks=read(REPORTS/'checks.json',{})
                checks.update(gpu_integration_run=True,gpu_verified_at_utc=utc(),
                    gpu_integration_scope='At least two optimizer updates; finite losses and gradients; actual single RTX6000 and live/completed task. Not a video-quality validation.')
                save(REPORTS/'checks.json',checks)
                log('complete',experiment_id=exp_id,gpu=gpu['actual'],step=updates[-1]['step'])
                return True
            if status.get('status')=='failed' or task['state'] in ('STATE_ERROR','STATE_CANCELED'):
                manifest['status']='needs_attention_non_gpu_training_failure'
                save(MANIFEST,manifest)
                log('training_failure',experiment_id=exp_id,task_state=task['state'])
                return False
            manifest['status']='rtx6000_starting';attempt['status']='accepted_loading'
            save(MANIFEST,manifest)
            if time.monotonic()>=deadline:return False
            time.sleep(30)
            continue

        rejected=bool(gpu and (not gpu.get('accepted') or not allowed(gpu.get('actual',''))))
        if rejected:
            if attempt.get('status')!='rejected_gpu_killed':
                det('experiment','kill',exp_id)
                attempt.update(status='rejected_gpu_killed',gpu=gpu)
                save(MANIFEST,manifest)
                log('rejected_and_killed',experiment_id=exp_id,gpu=gpu.get('actual'))
            free=free_rtx_slots()
            if not free:
                manifest['status']='waiting_for_free_rtx6000';save(MANIFEST,manifest)
                log('no_free_rtx6000',experiment_id=exp_id)
                return False
            # Exactly the authorized CLI form. Existing queued jobs are never duplicated.
            manifest['status']='submission_in_progress';save(MANIFEST,manifest)
            try:
                output=det('experiment','create',manifest['yaml'],manifest['submission_context'])
                found=re.search(r'Created experiment\s+(\d+)',output)
                if not found:raise RuntimeError('No experiment ID returned')
            except Exception:
                manifest['status']='needs_reconciliation_submission';save(MANIFEST,manifest)
                raise RuntimeError('Create result uncertain; will not create duplicate jobs')
            new_id=int(found.group(1))
            manifest['attempts'].append(dict(experiment_id=new_id,submitted_at_utc=utc(),
                run=str(ROOT/'runs'/f'v5_ce_{new_id}'),status='awaiting_gpu'))
            manifest['status']='submitted_awaiting_gpu';save(MANIFEST,manifest)
            log('resubmitted',experiment_id=new_id)
            time.sleep(15)
            continue

        if task['state'] in ('STATE_ERROR','STATE_CANCELED','STATE_DELETED'):
            manifest['status']='needs_attention_unexpected_exit';save(MANIFEST,manifest)
            log('unexpected_exit',experiment_id=exp_id,state=task['state'])
            return False
        free=free_rtx_slots()
        manifest['status']='queued' if task['state']=='STATE_QUEUED' else 'awaiting_gpu_policy'
        save(MANIFEST,manifest)
        log('existing_task',experiment_id=exp_id,state=task['state'],free_rtx6000=len(free))
        if (free or task['state']=='STATE_RUNNING') and time.monotonic()<deadline:
            time.sleep(30)
            continue
        return False


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--once',action='store_true')
    args=parser.parse_args()
    REPORTS.mkdir(parents=True,exist_ok=True)
    with (REPORTS/'rtx_submission_timer.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            log('already_running');return
        state=dict(pid=os.getpid(),interval_seconds=INTERVAL,started_at_utc=utc(),status='running',
            completion='Actual single RTX6000; at least two finite optimizer updates; task still running or completed.')
        save(STATE,state)
        while True:
            start=time.monotonic();state.update(status='checking',last_check_at_utc=utc());save(STATE,state)
            try:
                done=cycle()
            except Exception as error:
                log('check_error',error=str(error));state['last_error']=str(error);done=False
            if done:
                state.update(status='completed',completed_at_utc=utc(),next_check_at_utc=None)
                save(STATE,state);return
            delay=max(1,INTERVAL-(time.monotonic()-start))
            state.update(status='waiting',last_result=read(MANIFEST)['status'],
                next_check_at_utc=datetime.fromtimestamp(time.time()+delay,timezone.utc).isoformat())
            save(STATE,state)
            if args.once:return
            time.sleep(delay)


if __name__=='__main__':main()
