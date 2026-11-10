"""At30min intervals wait for complete data, then submit this one frozen candidate.

Never alter the existing data-preparation manager, failed R11A or historical jobs.
Never start an automatic next architecture. Explicit cancellation is respected.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,fcntl,json,os,re,subprocess,time

MASTER='http://10.130.130.5:4096'
RUNTIME='/home/liuzhirui/Project/physGen/code/v5.2.2-native/tools/runtime_python.sh'


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');os.replace(temporary,path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--describe',action='store_true')
    args=p.parse_args();prepared=json.loads(args.prepared.read_text())
    root=args.prepared.parent;folder=root/'submission_manager';source=Path(prepared['snapshot'])
    data=Path(prepared['data_preparation']);config=source/'configs/physical_content.yaml'
    readiness=data/'encoded/readiness.json'
    if args.describe:
        print(json.dumps(dict(candidate='R11A-rework',readiness=str(readiness),snapshot=str(source),yaml=prepared['yaml'],
            poll_seconds=1800,steps=400,evaluation='40+20/60, five16controls',automatic_next_candidate=False)));return
    if not prepared['code_checks_passed'] or prepared['submitted']:
        # A previously submitted job is recovered through this manager own receipt.
        if not (folder/'submissions.json').exists():raise ValueError('Candidate is not in a reviewed submit-ready state')
    folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'owner.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('Another process owns candidate submission')
    deadline=time.monotonic()+48*3600
    def status(stage,**kwargs):
        value=dict(stage=stage,utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),candidate='R11A-rework',poll_seconds=1800,**kwargs)
        write(folder/'status.json',value);print(json.dumps(value,ensure_ascii=False),flush=True)
    def pause():
        end=time.monotonic()+1800
        while time.monotonic()<end:
            if (folder/'STOP').exists():raise RuntimeError('Candidate manager was explicitly stopped')
            if time.monotonic()>deadline:raise RuntimeError('Candidate manager reached48h limit')
            time.sleep(min(30,max(0,end-time.monotonic())))
    try:
        while not readiness.exists():
            existing=data/'preparation_manager/status.json'
            progress=json.loads(existing.read_text()) if existing.exists() else {}
            status('waiting_for_paired_data',data_stage=progress.get('stage'),api_progress=progress.get('progress'))
            if progress.get('stage')=='needs_attention':raise RuntimeError('Data manager needs attention before training')
            pause()
        ready=json.loads(readiness.read_text())
        if not ready.get('training_ready') or ready.get('unique_training_samples')!=1600 or ready.get('local_contexts')!=3200:
            raise ValueError('Data readiness is incomplete')
        status('validating_prepared_training_configuration')
        with (folder/'preflight.log').open('a') as log:
            result=subprocess.run(['bash',RUNTIME,str(source/'train.py'),'--config',str(config),'--check-only'],stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError('Prepared data/config check failed; see preflight.log')
        status('preparing_benchmark_api_reasoning',cases=60,first_image_request_only=True)
        with (folder/'benchmark_reasoning.log').open('a') as log:
            result=subprocess.run(['bash',RUNTIME,str(source/'prepare_inputs.py'),'--config',str(config),'--stage','reasoning'],stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError('Benchmark reasoning preparation failed; successes remain resumable')
        receipt=folder/'submissions.json';attempts=json.loads(receipt.read_text()) if receipt.exists() else []
        eid=attempts[-1]['experiment_id'] if attempts else None
        empty=folder/'empty_context';empty.mkdir(exist_ok=True)
        while True:
            if eid is None:
                status('submitting_reviewed_candidate')
                creation=subprocess.run(['det','-m',MASTER,'experiment','create',prepared['yaml'],str(empty)],capture_output=True,text=True,timeout=120)
                match=re.search(r'Created experiment\s+(\d+)',creation.stdout+'\n'+creation.stderr)
                if not match:raise RuntimeError(f'No identifiable experiment ID (exit{creation.returncode}); no blind duplicate submission')
                eid=int(match.group(1));attempts.append(dict(experiment_id=eid,yaml=prepared['yaml'],snapshot=str(source),utc=datetime.now(timezone.utc).isoformat()))
                write(receipt,attempts);prepared.update(submitted=True,experiment_id=eid);write(args.prepared,prepared)
                status('submitted',experiment_id=eid)
            pause()
            response=subprocess.run(['det','-m',MASTER,'experiment','describe',str(eid),'--json'],capture_output=True,text=True,timeout=60)
            if response.returncode:status('remote_status_unavailable',experiment_id=eid);continue
            experiment=json.loads(response.stdout)[0]['experiment'];state=experiment['state']
            job=root/'reports/jobs'/str(eid);gpu=json.loads((job/'gpu.json').read_text()) if (job/'gpu.json').exists() else {}
            stage=json.loads((job/'status.json').read_text()) if (job/'status.json').exists() else {}
            status('candidate_status',experiment_id=eid,state=state,job_stage=stage.get('status'),gpu=gpu.get('actual'),gpu_accepted=gpu.get('accepted'))
            if state=='STATE_COMPLETED':
                if stage.get('status')!='declared_evaluation_completed':raise RuntimeError('Job ended without all declared evaluations')
                status('finished_waiting_for_result_review',experiment_id=eid,automatic_next_candidate=False);return
            if state in ('STATE_CANCELED','STATE_DELETED','STATE_DELETING'):
                status('explicitly_stopped_no_automatic_restore',experiment_id=eid,state=state);return
            if state=='STATE_ERROR':
                if gpu.get('accepted') is False:
                    status('non_rtx_rejected_before_models',experiment_id=eid);eid=None;continue
                raise RuntimeError('Candidate failed on accepted/unknown allocation; inspect before retrying')
    except Exception as error:status('needs_attention',error=str(error));raise


if __name__=='__main__':main()
