"""Supplementary130-video benchmark per preserved model, on separate ada/amp slots."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import ROOT,BENCH,plan,save,gpu_guard,shared

def evaluator_environment():
    env=dict(os.environ)
    for key in tuple(env):
        if key.startswith(('PYTHON','CONDA')) or key=='VIRTUAL_ENV':env.pop(key)
    for key in ('PATH','LD_LIBRARY_PATH'):
        env[key]=':'.join(p for p in env.get(key,'').split(':') if p and '/envs/' not in p)
    env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',MPLBACKEND='Agg',OMP_NUM_THREADS='4')
    return env

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',required=True);p.add_argument('--describe',action='store_true');a=p.parse_args()
    protocol=plan();spec=protocol['models'][a.model]
    if a.describe:
        print(json.dumps(dict(model=a.model,checkpoint=spec,pool=spec['resource_pool'],cases=len(protocol['cases']),
            new_training_steps=0,main_training_priority=True,
            reuse_existing_shared_inputs=spec.get('reuse_shared_inputs',False)),indent=2));return
    reuse = spec.get('reuse_shared_inputs', False)
    if reuse:
        if not Path(spec['checkpoint']).is_file():
            raise FileNotFoundError('Submit only after the declared own400 checkpoint is complete')
        for case in protocol['cases']:
            for filename in ('ready.json','first.png','conditions.pt','positive.pt','negative.pt'):
                if not (shared(case)/filename).is_file():
                    raise FileNotFoundError(f'Missing preserved shared input: {case["id"]}/{filename}')
    gpu_guard(a.model)
    job=ROOT/'jobs'/a.model;job.mkdir(parents=True,exist_ok=True);source=Path(__file__).resolve().parent
    from allocated_gpu import bind_benchmark_runtimes
    bind_benchmark_runtimes(job,evaluator_environment,[BENCH/'.venv-vila/bin/python',BENCH/'.venv-physicsiq/bin/python'])
    stage='starting'
    def run(script,extra=(),interpreter=None):
        nonlocal stage
        stage=script.removesuffix('.py')+('_'+extra[1] if script=='prepare_inputs.py' else '')
        save(job/'status.json',dict(status='running',stage=stage,experiment_id=os.getenv('DET_EXPERIMENT_ID'),time=time.time(),model=a.model))
        cmd=[interpreter or sys.executable,'-I','-B','-u',str(source/script),'--model',a.model,*extra]
        with (job/(stage+'.log')).open('a') as log:
            result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=evaluator_environment() if interpreter else None)
        if result.returncode:raise subprocess.CalledProcessError(result.returncode,cmd)
    try:
        # First actual requested video validates memory/runtime and is reused.
        if not reuse:
            for phase in ('images','encode','evidence'):run('prepare_inputs.py',('--stage',phase,'--limit','1'))
        run('generate.py',('--limit','1'))
        if not reuse:
            for phase in ('images','encode','evidence'):run('prepare_inputs.py',('--stage',phase))
        run('generate.py')
        run('evaluate_worldmodel.py',interpreter=str(BENCH/'.venv-vila/bin/python'))
        run('evaluate_physics_iq.py',interpreter=str(BENCH/'.venv-physicsiq/bin/python'))
        save(job/'status.json',dict(status='completed',model=a.model,time=time.time(),source=str(source)))
    except BaseException as exc:
        save(job/'status.json',dict(status='failed',stage=stage,error=str(exc),time=time.time(),model=a.model))
        raise

if __name__=='__main__':main()
