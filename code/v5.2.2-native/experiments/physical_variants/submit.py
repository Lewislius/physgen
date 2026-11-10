#!/usr/bin/env python3
"""Submit only these requested jobs; retry an allocation only after GPU rejection."""
from pathlib import Path
import argparse
import datetime
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import yaml
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from physical_training.common import load_config,audit,atomic_json,VARIANTS

DET='/home/liuzhirui/.local/bin/det'
MASTER='http://10.130.130.5:4096'


def cli_environment():
    # Training's isolation variables must NOT reach the system det launcher:
    # user site contains the established 0.32.1, system site also has old 0.22.2.
    env=os.environ.copy()
    for key in ('PYTHONNOUSERSITE','PYTHONPATH','PYTHONHOME','PYTHONUSERBASE',
                'PYTHONSTARTUP','PYTHONEXECUTABLE'):
        env.pop(key,None)
    return env


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--variants',nargs='+',choices=VARIANTS,default=list(VARIANTS))
    parser.add_argument('--resume',help='New-revision checkpoint; only with one variant')
    args=parser.parse_args()
    if args.resume and len(args.variants)!=1:
        raise ValueError('Resume requires exactly one selected variant')
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    release=ROOT/'snapshots'/stamp
    release.mkdir(parents=True,exist_ok=False)
    for name in ['physical_training','configs']:
        shutil.copytree(ROOT/name,release/name,ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(ROOT.parents[1]/'physgen_native',release/'physgen_native',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(ROOT/'train.py',release/'train.py')
    report=ROOT/'reports'/f'submissions_{stamp}.json'
    state=dict(created_at=stamp,release=str(release),attempts=[],accepted={})
    atomic_json(state,report)
    jobs={}

    def submit(variant,number):
        cfg=load_config(ROOT/'configs'/f'{variant}.yaml')
        audit(cfg)
        run_id=f'{variant}_{stamp}_attempt{number}'
        run=Path(cfg['paths']['runs'])/run_id
        job=yaml.safe_load((ROOT/'jobs'/f'{variant}_1x96g.yaml').read_text())
        if job['resources']['slots_per_trial']!=1 or job['resources']['resource_pool']!='blk-96g':
            raise ValueError('Only one blk-96g slot is authorized per experiment')
        command=['bash',str(ROOT.parents[1]/'tools/runtime_python.sh'),str(release/'train.py'),
                 '--config',str(release/'configs'/f'{variant}.yaml'),'--run-id',run_id]
        if args.resume:
            command+=['--resume',str(Path(args.resume).resolve())]
        job['name']=f'physical-{variant}-{stamp}-attempt{number}-rtx6000'
        job['entrypoint']=shlex.join(command)
        jobpath=release/f'{variant}_attempt{number}.yaml'
        jobpath.write_text(yaml.safe_dump(job,sort_keys=False))
        attempt=dict(variant=variant,attempt=number,run=str(run),job=str(jobpath),status='submitting')
        state['attempts'].append(attempt);atomic_json(state,report)
        completed=subprocess.run([DET,'-m',MASTER,'experiment','create',str(jobpath),str(ROOT/'submission_context')],
                                 text=True,capture_output=True,timeout=180,env=cli_environment())
        attempt['submission_output']=completed.stdout+completed.stderr
        match=re.search(r'Created experiment\s+(\d+)',attempt['submission_output'],re.I)
        if completed.returncode or not match:
            attempt['status']='submission_failed_or_unconfirmed';atomic_json(state,report)
            raise RuntimeError(attempt['submission_output'])
        attempt.update(experiment_id=int(match.group(1)),status='awaiting_gpu_guard',submitted_at=time.time())
        atomic_json(state,report)
        print(json.dumps(attempt,ensure_ascii=False),flush=True)
        return attempt

    for variant in args.variants:
        jobs[variant]=submit(variant,1)
    while len(state['accepted'])<len(args.variants):
        for variant,attempt in list(jobs.items()):
            if variant in state['accepted']:
                continue
            guard=Path(attempt['run'])/'gpu.json'
            if guard.is_file():
                result=json.loads(guard.read_text())
                if result['accepted']:
                    attempt['status']='rtx6000_confirmed'
                    attempt['gpu']=result['actual']
                    state['accepted'][variant]={k:attempt[k] for k in ('experiment_id','run','gpu')}
                    print(json.dumps(dict(kind='accepted',variant=variant,**state['accepted'][variant])),flush=True)
                else:
                    # The training process exits 86 before loading any model.
                    attempt['status']='rejected_gpu';attempt['gpu']=result['actual']
                    subprocess.run([DET,'-m',MASTER,'experiment','kill',str(attempt['experiment_id'])],
                                   capture_output=True,text=True,timeout=60,env=cli_environment())
                    print(json.dumps(dict(kind='retry_rejected_gpu',variant=variant,actual=result['actual'])),flush=True)
                    jobs[variant]=submit(variant,attempt['attempt']+1)
                atomic_json(state,report)
            elif time.time()-attempt['submitted_at']>1800:
                raise RuntimeError(f'GPU startup not confirmed after 30 minutes; inspect experiment {attempt["experiment_id"]}. Report: {report}')
        if len(state['accepted'])<len(args.variants):
            time.sleep(10)
    atomic_json(dict(report=str(report),**state),ROOT/'reports/latest_submission.json')
    print(json.dumps(dict(kind='all_rtx6000_confirmed',report=str(report),accepted=state['accepted']),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
