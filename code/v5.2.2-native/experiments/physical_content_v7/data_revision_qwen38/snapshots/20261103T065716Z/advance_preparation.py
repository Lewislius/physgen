"""Finish the authorized400 data preparation; never start model training."""
from datetime import datetime,timezone
from pathlib import Path
import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from annotate import save
RUNTIME=Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native/tools/runtime_python.sh')
MASTER='http://10.130.130.5:4096'


def annotations_complete(output):
    selection=json.loads((output/'selection.json').read_text())
    if selection['scope']!='actual400_training_only' or len(selection['ids'])!=1600:
        raise ValueError('Only actual400 data preparation is authorized')
    return all((output/'records'/sample/'caption.json').is_file()
               and all((output/'reasoning'/f'{sample}_{arm}.json').is_file() for arm in ('text','null'))
               for sample in selection['ids'])


def api_progress(output):
    ids=set(json.loads((output/'selection.json').read_text())['ids'])
    captions={p.parent.name for p in (output/'records').glob('*/caption.json')} & ids
    positive={p.name.removesuffix('_text.json') for p in (output/'reasoning').glob('*_text.json')} & ids
    negative={p.name.removesuffix('_null.json') for p in (output/'reasoning').glob('*_null.json')} & ids
    return dict(captions=len(captions),positive_reasoning=len(positive),negative_reasoning=len(negative),
        complete=len(captions & positive & negative),total=len(ids),
        errors=len(list((output/'errors').glob('*.json'))))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--yaml',type=Path,required=True)
    p.add_argument('--wait-pid',type=int,required=True)
    p.add_argument('--describe',action='store_true')
    args=p.parse_args();output=args.output.resolve();folder=output/'preparation_manager'
    if args.describe:
        print(json.dumps(dict(scope='actual400 only,1600 samples',wait_for_existing_api_pid=args.wait_pid,
            reuse_completed_api_stages=True,encoding_yaml=str(args.yaml),gpu='blk-96g single RTX6000; reject H100 before models',
            queue_check_seconds=1800,after_encoding='CPU fresh interface fitting and readiness; no training submission')))
        return
    folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'owner.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('Another process owns this data preparation')
    deadline=time.monotonic()+48*3600
    def status(stage,**kwargs):
        value=dict(stage=stage,pid=os.getpid(),utc=datetime.now(timezone.utc).isoformat(),
            samples=1600,starts_training=False,changes_r11a=False,**kwargs)
        save(folder/'status.json',value);print(json.dumps(value),flush=True)
    def pause(seconds):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            if (folder/'STOP').exists():raise RuntimeError('Preparation manager explicitly stopped')
            if time.monotonic()>deadline:raise RuntimeError('Preparation manager reached its48h limit')
            time.sleep(min(30,end-time.monotonic()))
    try:
        status('waiting_for_existing_api_batch',api_pid=args.wait_pid)
        while True:
            try:command=Path(f'/proc/{args.wait_pid}/cmdline').read_bytes()
            except FileNotFoundError:break
            if b'run_dataset.py' not in command or b'train400' not in command:break
            status('api_processing',api_pid=args.wait_pid,progress=api_progress(output),
                local_progress_interval_seconds=60)
            pause(60)
        for retry in range(3):
            if annotations_complete(output):break
            if retry==2:raise RuntimeError('Some API records remain incomplete after two resumable retry passes')
            status('retrying_only_missing_api_stages',retry=retry+1)
            with (folder/f'api_retry{retry+1}.log').open('a') as log:
                subprocess.run(['bash',str(RUNTIME),str(args.snapshot/'run_dataset.py'),
                    '--output',str(output),'--scope','train400','--workers','16'],stdout=log,stderr=subprocess.STDOUT)
        status('api_complete_encoding_pending')
        attempts_path=folder/'encoding_submissions.json'
        attempts=json.loads(attempts_path.read_text()) if attempts_path.exists() else []
        eid=attempts[-1]['experiment_id'] if attempts else None
        empty=folder/'empty_context';empty.mkdir(exist_ok=True)
        while True:
            if eid is None:
                status('submitting_local_encoding')
                result=subprocess.run(['det','-m',MASTER,'experiment','create',str(args.yaml),str(empty)],
                    capture_output=True,text=True,timeout=120)
                match=re.search(r'Created experiment\s+(\d+)',result.stdout+'\n'+result.stderr)
                if not match:
                    raise RuntimeError(f'Encoding submission returned no identifiable experiment; exit{result.returncode}. No blind resubmission.')
                eid=int(match.group(1));attempts.append(dict(experiment_id=eid,yaml=str(args.yaml),utc=datetime.now(timezone.utc).isoformat()))
                save(attempts_path,attempts)
                status('encoding_submitted',experiment_id=eid,queue_check_seconds=1800)
            pause(1800)
            query=subprocess.run(['det','-m',MASTER,'experiment','describe',str(eid),'--json'],capture_output=True,text=True,timeout=60)
            if query.returncode:
                status('encoding_status_unavailable',experiment_id=eid,queue_check_seconds=1800);continue
            experiment=json.loads(query.stdout)[0]['experiment']
            state=experiment['state'];gpu_path=output/'allocations'/str(eid)/'gpu.json'
            gpu=json.loads(gpu_path.read_text()) if gpu_path.exists() else {}
            status('encoding_status',experiment_id=eid,state=state,gpu=gpu.get('actual'),accepted=gpu.get('accepted'))
            if state=='STATE_COMPLETED':
                encoded=json.loads((output/'encoding_status.json').read_text())
                if not encoded['complete']:raise RuntimeError('Encoding job ended without complete paired caches')
                break
            if state in ('STATE_CANCELED','STATE_DELETED','STATE_DELETING'):
                raise RuntimeError('Encoding was canceled; no automatic restoration')
            if state=='STATE_ERROR':
                if gpu.get('accepted') is False:
                    status('rejected_non_rtx_allocation',experiment_id=eid);eid=None;continue
                raise RuntimeError('RTX encoding failed; inspect job before retrying')
        status('fitting_fresh_training_input_interface',experiment_id=eid)
        with (folder/'interface_fit.log').open('a') as log:
            result=subprocess.run(['bash',str(RUNTIME),str(args.snapshot/'finish_preparation.py'),
                '--output',str(output)],stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError('Final preparation failed; see interface_fit.log')
        ready=json.loads((output/'encoded/readiness.json').read_text())
        if not ready['training_ready']:raise RuntimeError('Readiness was not established')
        status('ready_for_next_candidate400',experiment_id=eid,readiness=str(output/'encoded/readiness.json'))
    except Exception as error:
        status('needs_attention',error=str(error));raise


if __name__=='__main__':main()
