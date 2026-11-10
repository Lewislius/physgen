"""One assigned GPU: shared input shard, own model generation, then AutoEval."""
from pathlib import Path
import argparse
import os
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import ROOT, BASE, save, gpu_guard


def evaluator_environment():
    env=dict(os.environ)
    for name in tuple(env):
        if name.startswith(('PYTHON','CONDA')) or name=='VIRTUAL_ENV':env.pop(name)
    prefix='/home/liuzhirui/miniconda3/envs/physvid'
    for name in ('PATH','LD_LIBRARY_PATH'):
        env[name]=':'.join(p for p in env.get(name,'').split(':') if p and '/envs/' not in p)
    env['PATH']=prefix+'/bin:'+env['PATH'];env['CONDA_PREFIX']=prefix
    return env


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);a=p.parse_args();gpu_guard(a.model)
    root=ROOT/'jobs'/a.model;root.mkdir(parents=True,exist_ok=True);source=Path(__file__).resolve().parent
    stage='starting'
    def run(script,extra=(),evaluate=False):
        nonlocal stage
        stage=script.removesuffix('.py')+('_'+extra[1] if script=='prepare_inputs.py' else '')
        save(root/'status.json',dict(model=a.model,status='running',stage=stage,experiment_id=os.getenv('DET_EXPERIMENT_ID'),time=time.time()))
        interpreter='/home/liuzhirui/miniconda3/envs/physvid/bin/python' if evaluate else sys.executable
        cmd=[interpreter,'-I','-B','-u',str(source/script),'--model',a.model,*extra]
        with (root/(stage+'.log')).open('a') as log:
            result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=evaluator_environment() if evaluate else None)
        if result.returncode:raise subprocess.CalledProcessError(result.returncode,cmd)
    try:
        # The first real requested case verifies the full path and is reused.
        # No extra diagnostic videos or extra scoring samples are generated.
        for phase in ('images','encode','evidence'):run('prepare_inputs.py',('--stage',phase,'--limit','1'))
        run('generate.py',('--limit','1'));run('evaluate.py',('--partial',),evaluate=True)
        for phase in ('images','encode','evidence'):run('prepare_inputs.py',('--stage',phase))
        run('generate.py');run('evaluate.py',evaluate=True)
        save(root/'status.json',dict(model=a.model,status='completed',time=time.time(),source=str(source)))
    except BaseException as exc:
        save(root/'status.json',dict(model=a.model,status='failed',stage=stage,error=str(exc),time=time.time()))
        raise


if __name__=='__main__':main()
