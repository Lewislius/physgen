"""Fresh content revision and matched native-LoRA training."""
from pathlib import Path
import argparse
import json
import os
import random
import time
import numpy as np
import torch
from .common import load_config, audit, gpu_guard, atomic_json, atomic_torch, cpu_tree, Log
from physical_scaling.data import Data, SampleOrder
from .teacher import Teacher
from .model import PhysicalModel
from physical_scaling.schedule import rates, auxiliary_coefficient
from physical_scaling.supervision import flow_matching
from .supervision import generation_loss
from .objectives import backward_sample


def save(model,optimizer,cfg,run,step,cursor):
    named = dict(model.named_parameters())
    state = dict(revision=cfg['revision'],variant=cfg['variant'],config=cfg,step=step,cursor=cursor,
                 model={n:named[n].detach().cpu() for n in model.learned_names},
                 constants=model.constant_state(),optimizer=cpu_tree(optimizer.state_dict()),
                 rng=dict(python=random.getstate(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()))
    atomic_torch(state,run/'latest.pt')
    if step in cfg['train']['keep_steps']:
        alias = run/f'step{step:04d}.pt'
        if alias.exists():
            alias.unlink()
        os.link(run/'latest.pt',alias)


def restore(model,optimizer,path,cfg):
    state = torch.load(path,map_location='cpu',weights_only=False)
    if state.get('revision')!=cfg['revision'] or state.get('variant')!=cfg['variant'] or state['config']!=cfg:
        raise ValueError('Resume requires this exact new variant/config, never an old project initializer')
    named = dict(model.named_parameters())
    if set(state['model'])!=set(model.learned_names):
        raise ValueError('Checkpoint parameter ownership differs')
    with torch.no_grad():
        for n,v in state['model'].items():
            if named[n].shape!=v.shape:
                raise ValueError(f'Checkpoint shape differs: {n}')
            named[n].copy_(v)
        buffers=dict(model.named_buffers())
        for n,v in state['constants'].items():buffers[n].copy_(v)
    optimizer.load_state_dict(state['optimizer'])
    random.setstate(state['rng']['python'])
    torch.set_rng_state(state['rng']['torch'])
    torch.cuda.set_rng_state_all(state['rng']['cuda'])
    return state['step'],state['cursor']


@torch.no_grad()
def sample_video(model,teacher,sample,null,seed=42):
    cfg = model.cfg
    positive = teacher.input(sample)
    negative = teacher.input(sample,drop=True)
    c,groups = model.enhancement(positive)
    nc,ngroups = model.enhancement(negative)
    request=model.enhancement.request_memory(positive)
    null_request=model.enhancement.request_memory(negative)
    rng = torch.Generator(device=sample['first'].device).manual_seed(seed)
    x = torch.randn((1,48,21,30,52),device=sample['first'].device,generator=rng)
    policy = cfg['inference']
    grid = torch.linspace(1,0,policy['steps']+1,device=x.device)
    grid = policy['shift']*grid/(1+(policy['shift']-1)*grid)
    x[:,:,:1] = sample['first']
    for i in range(policy['steps']):
        pos,_ = model.velocity(x,sample['first'],sample['text'],grid[i],c,sample['times'],groups,request_memory=request)
        neg,_ = model.velocity(x,sample['first'],null,grid[i],nc,sample['times'],ngroups,request_memory=null_request)
        v = neg+policy['guidance']*(pos-neg)
        x = x+(grid[i+1]-grid[i])*v
        x[:,:,:1] = sample['first']
    return x


@torch.no_grad()
def validate(model,teacher,data,null,run,step,log,decoder):
    state = (torch.get_rng_state(),torch.cuda.get_rng_state_all(),random.getstate())
    try:
        values = []
        with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
            for i in range(min(4,len(data))):
                sample = data.get(i)
                evidence = teacher.input(sample)
                c,groups = model.enhancement(evidence)
                request=model.enhancement.request_memory(evidence)
                sigma = torch.tensor(.5,device='cuda')
                noise = torch.randn(sample['latent'].shape,device='cuda',generator=torch.Generator(device='cuda').manual_seed(42000+i))
                noisy = (1-sigma)*sample['latent']+sigma*noise
                pred,_ = model.velocity(noisy,sample['first'],sample['text'],sigma,c,sample['times'],groups,request_memory=request)
                values.append(float(flow_matching(pred,noise,sample['latent'],sample['valid'])))
            sample = data.get(0,targets=False)
            generated = sample_video(model,teacher,sample,null,42)
            if decoder[0] is None:
                from physgen_native.decoder import VideoDecoder
                decoder[0] = VideoDecoder(model.cfg)
            frames = decoder[0](generated)
        import imageio.v2 as imageio
        frames = ((frames.permute(0,2,3,1).clamp(-1,1)+1)*127.5).round().byte().cpu().numpy()
        out = run/'videos'/f'val_{sample["id"]}_step{step:04d}_seed42.mp4'
        out.parent.mkdir(parents=True,exist_ok=True)
        fps = 80/float(sample['times'][0,-1])
        imageio.mimwrite(out,frames,fps=fps,codec='libx264',macro_block_size=None,ffmpeg_params=['-crf','18'])
        atomic_json(dict(sample=sample['id'],caption=sample['caption'],step=step,seed=42,fps=fps,
                         inference_inputs='first frame and request only',sampler=model.cfg['inference']),out.with_suffix('.json'))
        log('validation',step=step,fm=float(np.mean(values)),video=str(out),sample=sample['id'])
    finally:
        torch.set_rng_state(state[0]);torch.cuda.set_rng_state_all(state[1]);random.setstate(state[2])


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--run-id')
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--resume')
    parser.add_argument('--stop-after',type=int)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.check_only:
        print(json.dumps(audit(cfg),indent=2));return
    run_id = args.run_id or f'{cfg["variant"]}_{os.getenv("DET_EXPERIMENT_ID",str(int(time.time())))}'
    run = Path(cfg['paths']['runs'])/run_id
    run.mkdir(parents=True,exist_ok=True)
    gpu = gpu_guard(run)
    log = Log(run)
    if (run/'config.json').exists() and not args.resume:
        raise ValueError('Run directory already initialized; use explicit resume or a new run ID')
    atomic_json(cfg,run/'config.json')
    atomic_json(dict(status='loading',gpu=gpu,variant=cfg['variant']),run/'status.json')
    random.seed(cfg['train']['seed']);np.random.seed(cfg['train']['seed']);torch.manual_seed(cfg['train']['seed'])
    torch.cuda.manual_seed_all(cfg['train']['seed'])
    torch.set_num_threads(4)
    log('audit',**audit(cfg))
    try:
        log('load_official_wan')
        model = PhysicalModel(cfg).train()
        model.wan.eval()
        log('use_frozen_input_evidence_cache')
        teacher = Teacher(cfg,log=log)
        optimizer = torch.optim.AdamW(model.optim_groups(),betas=(.9,.999),weight_decay=.01,foreach=False)
        data,heldout = Data(cfg),Data(cfg,'validation')
        null = torch.load(Path(cfg['paths']['cache'])/'base/null_text.pt',weights_only=True,map_location='cpu').float()[None].cuda()
        completed,cursor = restore(model,optimizer,args.resume,cfg) if args.resume else (0,0)
        order = iter(SampleOrder(len(data),cfg['train']['seed'],cursor))
        total = sum(cfg['stages'].values())
        stop = min(total,args.stop_after or total)
        named = dict(model.named_parameters())
        log('model_ready',variant=cfg['variant'],parameters=sum(named[n].numel() for n in model.learned_names),
            groups={g['name']:sum(p.numel() for p in g['params']) for g in optimizer.param_groups},
            stages=cfg['stages'],initialization='official foundations only',verifier=False)
        atomic_json(dict(status='training',step=completed,variant=cfg['variant']),run/'status.json')
        decoder = [None]
        for step in range(completed,stop):
            phase = model.set_phase(step)
            rates(optimizer,cfg,step)
            coefficient = auxiliary_coefficient(cfg,step)
            if cfg['variant']=='native_lora' and phase=='interface':
                for _ in range(cfg['train']['accumulation']):next(order);cursor+=1
                completed=step+1
                if completed==cfg['stages']['interface']:
                    log('frozen_interface_skipped',step=completed,cursor=cursor,
                        reason='Native baseline has no physical interface parameters; joint data/noise positions remain matched')
                    save(model,optimizer,cfg,run,completed,cursor)
                continue
            optimizer.zero_grad(set_to_none=True)
            sums = {key:0. for key in ('fm','generation','tail','aux','weighted_aux')}
            aux_count = 0
            start = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            for micro in range(cfg['train']['accumulation']):
                sample = data.get(next(order))
                draws = torch.rand(2,generator=torch.Generator().manual_seed(cfg['train']['seed']*1000003+cursor))
                drop = bool(draws[0]<cfg['train']['text_dropout'])
                sigma = (.001+.998*draws[1]).to('cuda')
                with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
                    evidence = teacher.input(sample,drop=drop)
                    noise = torch.randn(sample['latent'].shape,device='cuda',generator=torch.Generator(device='cuda').manual_seed(cfg['train']['seed']*2000003+cursor))
                    values=backward_sample(model,evidence,sample,null if drop else sample['text'],
                        sigma,noise,coefficient,cfg['train']['accumulation'])
                for key in sums:sums[key]+=values[key]
                aux_count += 1
                cursor += 1
                log('micro',step=step+1,phase=phase,micro=micro+1,sample=sample['id'],
                    sigma=float(sigma),text_dropped=drop,aux_active=True,aux_coefficient=coefficient,
                    gradient_ownership=cfg['train']['gradient_ownership'],
                    **values,
                    peak_gib=torch.cuda.max_memory_allocated()/2**30)
                del sample,evidence,noise
            grad,clip_scales = {},{}
            for group in optimizer.param_groups:
                parameters = [p for p in group['params'] if p.grad is not None]
                threshold = cfg['train']['grad_clip'][group['name']]
                norm = torch.nn.utils.clip_grad_norm_(parameters,threshold,error_if_nonfinite=True) if parameters else 0.
                grad[group['name']] = float(norm)
                clip_scales[group['name']] = min(1.,threshold/(float(norm)+1e-6))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            completed = step+1
            torch.cuda.synchronize()
            log('train',step=completed,phase=phase,cursor=cursor,seconds=time.monotonic()-start,
                **{k:v/cfg['train']['accumulation'] for k,v in sums.items()},aux_samples=aux_count,
                gradients=grad,clip_scales=clip_scales,aux_coefficient=coefficient,
                learning_rates={g['name']:g['lr'] for g in optimizer.param_groups},
                peak_gib=torch.cuda.max_memory_allocated()/2**30)
            atomic_json(dict(status='training',step=completed,phase=phase,variant=cfg['variant']),run/'status.json')
            if completed%cfg['train']['save_every']==0 or completed in cfg['train']['keep_steps'] or completed==stop:
                save(model,optimizer,cfg,run,completed,cursor)
                log('checkpoint',step=completed,path=str(run/'latest.pt'))
            if completed in cfg['train']['validation_steps']:
                validate(model,teacher,heldout,null,run,completed,log,decoder)
        status = 'complete' if completed==total else 'paused'
        atomic_json(dict(status=status,step=completed,checkpoint=str(run/'latest.pt')),run/'status.json')
        log(status,step=completed,checkpoint=str(run/'latest.pt'))
    except BaseException as error:
        atomic_json(dict(status='failed',error=repr(error)),run/'status.json')
        log('failed',error=repr(error))
        raise


if __name__ == '__main__':
    main()
