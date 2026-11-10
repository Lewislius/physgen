"""Submit the reviewed own400 candidate; bounded allocation check, no watcher."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,re,subprocess,time

ROOT=Path(__file__).resolve().parent
PREPARED=ROOT/'prepared.json'
MASTER='http://10.130.130.5:4096'


def describe(identifier):
    result=subprocess.run(['det','-m',MASTER,'experiment','describe',str(identifier),'--json'],
                          capture_output=True,text=True,check=True,timeout=30)
    # Never print this object: server config includes private registry credentials.
    return json.loads(result.stdout)[0]['experiment']


def main():
    prepared=json.loads(PREPARED.read_text())
    if not prepared.get('final_design_ready'):
        raise RuntimeError('Finish the declared evidence review before submitting formal400')
    source=Path(prepared['snapshot'])
    manifest=json.loads((source/'source_manifest.json').read_text())['sha256']
    for name,digest in manifest.items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Frozen submitted source has changed: '+name)
    def save():
        temporary=PREPARED.with_suffix('.tmp')
        temporary.write_text(json.dumps(prepared,ensure_ascii=False,indent=2)+'\n');temporary.replace(PREPARED)
    if prepared.get('experiment_id') is not None:
        previous=describe(prepared['experiment_id'])
        path=ROOT/'reports/jobs'/str(prepared['experiment_id'])/'gpu.json'
        gpu=json.loads(path.read_text()) if path.exists() else {}
        if previous['state']!='STATE_ERROR' or gpu.get('accepted') or 'H100' not in gpu.get('actual',''):
            raise RuntimeError('An existing candidate must not be duplicated; inspect its actual state')
    for _ in range(4):
        result=subprocess.run(['det','-m',MASTER,'experiment','create',prepared['yaml'],prepared['empty_context']],
                              capture_output=True,text=True,check=True,timeout=60)
        match=re.search(r'Created experiment (\d+)',result.stdout)
        if not match:raise RuntimeError('Submission response has no experiment ID; inspect server state before retrying')
        identifier=int(match.group(1));entry=dict(experiment_id=identifier,submitted_utc=datetime.now(timezone.utc).isoformat(),accepted=None)
        prepared.setdefault('attempts',[]).append(entry)
        prepared.update(experiment_id=identifier,status='submitted_pending_actual_gpu',formal_training_submitted=True)
        save();print(json.dumps(dict(submitted=identifier,steps=400,pool='blk-96g')),flush=True)
        gpu_path=ROOT/'reports/jobs'/str(identifier)/'gpu.json';rejected=False
        for _ in range(8):
            time.sleep(5)
            if not gpu_path.exists():continue
            gpu=json.loads(gpu_path.read_text());actual=describe(identifier)
            entry.update(actual_gpu=gpu.get('actual'),accepted=gpu.get('accepted'),state=actual['state'],trial_ids=actual.get('trialIds'))
            save()
            if gpu.get('accepted'):
                if actual['state'] in ('STATE_ERROR','STATE_CANCELED','STATE_STOPPING_ERROR','STATE_STOPPING_CANCELED'):
                    prepared.update(status='startup_failed_on_allowed_gpu',last_state=actual['state']);save()
                    raise RuntimeError('Allowed GPU was assigned but startup failed; inspect before retrying')
                prepared.update(status='running_actual_rtx_preflight_then_fresh400',actual_gpu=gpu['actual'],
                    trial_ids=actual.get('trialIds'),verified_utc=datetime.now(timezone.utc).isoformat())
                save();print(json.dumps(dict(accepted=identifier,trials=actual.get('trialIds'),gpu=gpu['actual'])),flush=True);return
            if actual['state']=='STATE_ERROR' and 'H100' in gpu.get('actual',''):
                entry['reason']='H100 rejected before loading foundation models';prepared['status']='terminal_gpu_rejected'
                save();rejected=True;print(json.dumps(dict(rejected=identifier,gpu=gpu['actual'])),flush=True);break
            if actual['state'] in ('STATE_ERROR','STATE_CANCELED'):
                raise RuntimeError('Unexpected startup failure; inspect before any retry')
        if not rejected:
            actual=describe(identifier)
            prepared.update(trial_ids=actual.get('trialIds'),last_state=actual['state']);save()
            print(json.dumps(dict(experiment=identifier,trials=actual.get('trialIds'),state=actual['state'],
                                 reason='Bounded startup check ended; preserve this job, no duplicate')),flush=True);return


if __name__=='__main__':main()
