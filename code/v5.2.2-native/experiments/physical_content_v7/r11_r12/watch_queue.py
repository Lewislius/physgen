"""User-requested 30-minute queue checks; no submission, training or recovery."""
from datetime import datetime,timezone
from pathlib import Path
import fcntl
import json
import os
import subprocess
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
JOBS={67214:ROOT/'request_adapt_r11a',67213:HERE/'r10a_controls16'}
INTERVAL=1800


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def main():
    folder=HERE/'queue_watch';folder.mkdir(exist_ok=True)
    lock=(folder/'owner.lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        print('An existing queue watcher owns this task.',flush=True);return
    next_check=time.time()+INTERVAL
    write(folder/'status.json',dict(status='waiting',pid=os.getpid(),interval_seconds=INTERVAL,
        experiments=list(JOBS),next_check_utc=datetime.fromtimestamp(next_check,timezone.utc).isoformat(),
        maximum_checks=48,automatic_submission=False,restore_r9=False))
    print(json.dumps(dict(status='waiting',pid=os.getpid(),interval_seconds=INTERVAL,
        next_check_utc=datetime.fromtimestamp(next_check,timezone.utc).isoformat())),flush=True)
    for check in range(1,49):
        # Sleep in short chunks; stop file supports quiet, bounded cancellation.
        while time.time()<next_check:
            if (folder/'STOP').exists():
                write(folder/'status.json',dict(status='stopped',pid=os.getpid()));return
            time.sleep(min(30,max(0,next_check-time.time())))
        records=[]
        for eid,root in JOBS.items():
            try:
                result=subprocess.run(['det','-m','http://10.130.130.5:4096','experiment','describe',str(eid),'--json'],capture_output=True,text=True,timeout=60)
                if result.returncode:raise RuntimeError(f'det exit {result.returncode}')
                e=json.loads(result.stdout)[0]['experiment']
                record=dict(experiment_id=eid,state=e['state'],trial_ids=e.get('trialIds',[]))
                gpu=root/'reports/jobs'/str(eid)/'gpu.json'
                if gpu.exists():
                    g=json.loads(gpu.read_text());record.update(actual_gpu=g.get('actual'),gpu_accepted=g.get('accepted'))
                stage=root/'reports/jobs'/str(eid)/'status.json'
                if stage.exists():record['stage']=json.loads(stage.read_text()).get('status')
                records.append(record)
            except Exception as error:records.append(dict(experiment_id=eid,error=str(error)))
        done=all('error' not in row and row.get('state') not in ('STATE_QUEUED','STATE_PENDING','STATE_PAUSED') for row in records)
        next_check=time.time()+INTERVAL
        record=dict(status='queue_resolved' if done else 'waiting',pid=os.getpid(),check=check,
            checked_utc=datetime.now(timezone.utc).isoformat(),jobs=records,interval_seconds=INTERVAL,
            next_check_utc=None if done else datetime.fromtimestamp(next_check,timezone.utc).isoformat())
        write(folder/'status.json',record)
        with (folder/'checks.jsonl').open('a') as stream:stream.write(json.dumps(record,ensure_ascii=False)+'\n')
        print(json.dumps(record,ensure_ascii=False),flush=True)
        if done:return
    write(folder/'status.json',dict(status='24h_limit_reached',pid=os.getpid(),jobs=records))


if __name__=='__main__':main()
