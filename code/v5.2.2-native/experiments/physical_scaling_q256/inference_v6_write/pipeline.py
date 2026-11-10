"""One single-GPU allocation: demo21 -> VideoPhy2-40 -> AutoEval -> comparison."""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from settings import JOBS, NATIVE_ROOT, TRAIN_SOURCE, activate_training
from protocol import checkpoint_state, check_inputs, read, require, manifest_for


def save(path,value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


def preflight(variant):
    activate_training()
    import torch
    from physical_scaling.model import Enhancement, Fusion
    state = checkpoint_state(JOBS[variant]['checkpoint'],variant)
    cfg = state['config']
    with torch.device('meta'):
        enhancement = Enhancement(cfg)
    actual = {k.removeprefix('enhancement.'):v for k,v in state['model'].items() if k.startswith('enhancement.')}
    require(set(actual)==set(enhancement.state_dict()),'Checkpoint C enhancement names differ')
    require(all(actual[k].shape==v.shape for k,v in enhancement.state_dict().items()),'Checkpoint C enhancement dimensions differ')
    expected_sites = {str(i) for i in range(2,30,3)}
    actual_sites = {k.split('.')[1] for k in state['model'] if k.startswith('fusions.')}
    require(actual_sites==expected_sites,'Unexpected physical fusion layers')
    # Check full K/V for V5 versus base+LoRA K/V for V6 against the actual class.
    from types import SimpleNamespace
    width = cfg['model']['width']
    with torch.device('meta'):
        attention = SimpleNamespace(dim=width,
            q=torch.nn.Linear(width,width,bias=False),
            k=torch.nn.Linear(width,width,bias=False),
            v=torch.nn.Linear(width,width,bias=False),
            norm_k=torch.nn.RMSNorm(width,eps=1e-6))
        fusion = Fusion(attention,variant,cfg['model']['lora_rank'])
    expected = {n:p.shape for n,p in fusion.named_parameters() if p.requires_grad}
    for site in expected_sites:
        prefix = f'fusions.{site}.'
        actual = {n.removeprefix(prefix):p.shape for n,p in state['model'].items() if n.startswith(prefix)}
        require(actual==expected,f'Checkpoint fusion parameter ownership/shapes differ: layer {int(site)+1}')
    require(('bridge.weight' in state['model'])==(variant=='v5_ce'),'Semantic bridge ownership differs')
    if variant=='v5_ce':
        require(tuple(state['model']['bridge.weight'].shape)==(cfg['model']['qwen_width'],width),'Semantic bridge shape differs')
    for key,index in [('wan_checkpoint','diffusion_pytorch_model.safetensors.index.json'),('qwen_checkpoint','model.safetensors.index.json')]:
        root = Path(cfg['paths'][key])
        require(all((root/p).is_file() for p in set(read(root/index)['weight_map'].values())),f'Missing {key} shards')
    require((Path(cfg['paths']['wan_checkpoint'])/'Wan2.2_VAE.pth').is_file(),'Missing VAE')
    counts = check_inputs()
    from compare import validate_manifest
    baseline,_ = validate_manifest(manifest_for(variant,'videophy40'))
    evaluation = baseline['evaluation']
    evaluator = Path(evaluation['checkpoint'])
    index = evaluator/'pytorch_model.bin.index.json'
    weights = set(read(index)['weight_map'].values()) if index.exists() else {'pytorch_model.bin'}
    require(all((evaluator/name).is_file() for name in weights),'Missing evaluator weights')
    require(Path(evaluation['code']).is_dir(),'Missing evaluator implementation')
    require(Path('/home/liuzhirui/miniconda3/envs/physvid/bin/python').is_file(),'Missing evaluator interpreter')
    require(shutil.which('ffmpeg') and shutil.which('ffprobe'),'Missing video encoding tools')
    result = dict(status='checked',**JOBS[variant],**counts,checkpoint_step=state['step'],
        parameter_tensors=len(state['model']),parameters=sum(v.numel() for v in state['model'].values()),
        comparison_references=len(baseline['summaries']),evaluation=evaluation)
    return result


def evaluator_environment():
    env = dict(os.environ)
    prefix = '/home/liuzhirui/miniconda3/envs/physvid'
    for name in tuple(env):
        if name.startswith(('PYTHON','CONDA')) or name=='VIRTUAL_ENV':
            env.pop(name)
    for name in ('PATH','LD_LIBRARY_PATH'):
        env[name] = ':'.join(p for p in env.get(name,'').split(':') if p and '/envs/' not in p)
    env['PATH'] = prefix+'/bin:'+env['PATH']
    env.update(CONDA_PREFIX=prefix,CONDA_DEFAULT_ENV='physvid',TOKENIZERS_PARALLELISM='false')
    return env


def run_stage(command,logfile,env=None):
    print('Starting: '+' '.join(map(str,command)),flush=True)
    with Path(logfile).open('a') as log:
        with subprocess.Popen(list(map(str,command)),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                              text=True,bufsize=1,env=env) as process:
            for line in process.stdout:
                log.write(line);log.flush();print(line,end='',flush=True)
            code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code,command)


def pipeline(variant):
    activate_training()
    from physical_scaling.common import gpu_guard
    job = JOBS[variant]
    root = Path(job['output'])
    root.mkdir(parents=True,exist_ok=True)
    allocation = root/'allocations'/os.environ.get('DET_EXPERIMENT_ID','local')
    gpu = gpu_guard(allocation)  # H100 exits86 BEFORE any foundation loading.
    with (root/'.pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        status = dict(status='running',variant=variant,experiment_id=os.getenv('DET_EXPERIMENT_ID'),
            gpu=gpu,started_at=datetime.now(timezone.utc).isoformat(),stage='preflight',completed_stages=[])
        save(root/'pipeline_status.json',status)
        try:
            save(root/'preflight.json',preflight(variant))
            status['stage']='demo21_then_videophy40'
            save(root/'pipeline_status.json',status)
            run_stage([sys.executable,'-I','-B','-u',HERE/'generate.py','--variant',variant],root/'generation.log')
            for suite,count in [('demo21',21),('videophy40',40)]:
                value=read(root/suite/'summary.json')
                require(value['status']=='completed' and value['completed_cases']==count,f'{suite}: incomplete')
                status['completed_stages'].append(suite)
            status['stage']='independent_autoeval'
            save(root/'pipeline_status.json',status)
            # Generation process exited: the evaluator gets the entire same GPU.
            run_stage(['/home/liuzhirui/miniconda3/envs/physvid/bin/python','-I','-B','-u',
                NATIVE_ROOT/'tools/evaluate_videophy2.py',root/'videophy40'],root/'evaluation.log',evaluator_environment())
            status['completed_stages'].append('independent_autoeval')
            from compare import compare
            result = compare(root/'videophy40',variant)
            status.update(status='completed',stage='completed',demo_completed=21,videophy_completed=40,
                scalar_scores=80,scores=result['summaries'][0],finished_at=datetime.now(timezone.utc).isoformat())
            status['completed_stages'].append('comparison')
        except BaseException as error:
            status.update(status='failed',error=repr(error))
            raise
        finally:
            save(root/'pipeline_status.json',status)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--variant',choices=JOBS,required=True)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    if args.check_only:
        print(json.dumps(preflight(args.variant),ensure_ascii=False,indent=2))
    else:
        pipeline(args.variant)
