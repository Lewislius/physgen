"""One-shot handoff after own400; no polling and no new training."""
from pathlib import Path
from datetime import datetime,timedelta,timezone
import argparse,json,os,re,shlex,subprocess
import yaml

HERE=Path(__file__).resolve().parent
ROOT=Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_v7')
DET=['/home/liuzhirui/.local/bin/det','-m','http://10.130.130.5:4096']
RUNTIME=ROOT/'r16_evaluation_runtime/snapshots/20261109T060601Z'


def invoke(args):
    env=os.environ.copy()
    # Use the existing system CLI, outside the isolated training interpreter.
    env.pop('PYTHONPATH',None);env.pop('PYTHONNOUSERSITE',None)
    return subprocess.run(args,check=True,text=True,capture_output=True,env=env,timeout=90)


def select_pool(slots):
    allowed=['amp-80g','ada-48g','amp-48g']
    for pool in allowed:
        if any(v.get('resource_pools')==pool and v.get('allocation_id')=='FREE'
               and v.get('enabled') and not v.get('draining')
               and 'H100' not in v.get('device','') for v in slots):return pool
    return 'amp-48g'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--run-id',required=True)
    args=p.parse_args();cfg=yaml.safe_load(args.config.read_text())
    root=Path(cfg['paths']['runs']).parent;run=root/'runs'/args.run_id
    record=run/'evaluation_submission.json'
    if record.exists():print(record.read_text());return
    status=json.loads((run/'status.json').read_text())
    if status.get('status')!='complete' or status.get('step')!=400 or not (run/'step0400.pt').is_file():
        raise ValueError('Require the completed own400 checkpoint')
    audit=root/'reports'/args.run_id/'completed_training_match.json';audit.parent.mkdir(parents=True,exist_ok=True)
    invoke(['/usr/bin/python3',str(ROOT/'native_joint_read_r16a/audit_completed_training.py'),
            '--run-id',args.run_id,'--run-directory',str(run),'--output',str(audit)])
    result=json.loads(audit.read_text())
    if not result['data_noise_dropout_and_schedule_match']:raise ValueError('Training audit failed')
    slots=json.loads(invoke([*DET,'slot','list','--json']).stdout);pool=select_pool(slots)
    template=ROOT/'r16_evaluation_runtime/submissions/r16a_nozero_evaluation_pending_amp48.yaml'
    spec=yaml.safe_load(template.read_text());stamp=datetime.now(timezone.utc)
    spec.update(name=f'physical-r16b-own400-eval40-60-controls16-{stamp:%Y%m%dT%H%M%SZ}',
                description='Native R16B own400 evaluation only; exact frozen model and matched protocol. No further training.')
    spec['resources']['resource_pool']=pool
    spec['entrypoint']=shlex.join(['bash',str(ROOT.parents[1]/'tools/runtime_python.sh'),
        str(RUNTIME/'resume_evaluation.py'),'--config',str(args.config),'--run-id',args.run_id,'--pool',pool])
    path=root/'submissions'/f'evaluation_{stamp:%Y%m%dT%H%M%SZ}.yaml';path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(yaml.safe_dump(spec,sort_keys=False))
    empty=root/'empty_context';empty.mkdir(exist_ok=True)
    response=invoke([*DET,'experiment','create',str(path),str(empty)])
    match=re.search(r'Created experiment\s+(\d+)',response.stdout)
    if match is None:raise RuntimeError('Submission response did not contain experiment ID: '+response.stdout)
    experiment=int(match[1]);prepared=root/'prepared.json';value=json.loads(prepared.read_text())
    training=value.get('training_experiment',value.get('experiment'))
    event=dict(utc=stamp.isoformat(),experiment=experiment,training_experiment=training,run_id=args.run_id,
               pool=pool,yaml=str(path),runtime=str(RUNTIME),audit=str(audit),submitted_once=True)
    record.write_text(json.dumps(event,indent=2)+'\n')
    value.update(training_experiment=training,training_complete=True,training_step_observed=400,
                 experiment=experiment,job_scope='evaluation_only',status='evaluation_submitted',
                 state='STATE_ACTIVE',last_observation={},evaluation_yaml=str(path),pool=pool,
                 next_check_utc=(stamp+timedelta(minutes=15)).isoformat())
    temporary=prepared.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(prepared)
    print(json.dumps(event))


if __name__=='__main__':main()
