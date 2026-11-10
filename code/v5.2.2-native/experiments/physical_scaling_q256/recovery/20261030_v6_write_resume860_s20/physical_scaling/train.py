"""Independent Q256 scaling: 128 interface + 872 joint optimizer updates."""
from pathlib import Path
import argparse
import json
import os
import random
import time
import numpy as np
import torch
from .common import load_config, audit, gpu_guard, atomic_json, atomic_torch, cpu_tree, Log
from .data import Data, SampleOrder
from .teacher import Teacher
from .model import PhysicalModel
from .schedule import rates, auxiliary_coefficient
from .supervision import flow_matching


def save(model,optimizer,cfg,run,step,cursor):
    named = dict(model.named_parameters())
    state = dict(revision=cfg['revision'],variant=cfg['variant'],config=cfg,step=step,cursor=cursor,
                 model={n:named[n].detach().cpu() for n in model.learned_names},
                 optimizer=cpu_tree(optimizer.state_dict()),
                 rng=dict(python=random.getstate(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()))
    atomic_torch(state,run/'latest.pt')
    if step in cfg['train']['keep_steps']:
        alias = run/f'step{step:04d}.pt'
        if alias.exists():
            alias.unlink()
        os.link(run/'latest.pt',alias)


def restore(model,optimizer,path,cfg):
    # Map CPU checkpoint storage instead of allocating another 4.5 GB copy.
    state = torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    if state.get('revision')!=cfg['revision'] or state.get('variant')!=cfg['variant'] or state['config']!=cfg:
        raise ValueError('Resume requires this exact new variant/config, never an old project initializer')
    named = dict(model.named_parameters())
    if set(state['model'])!=set(model.learned_names):
        raise ValueError('Checkpoint parameter ownership differs')
    if state['cursor'] != state['step'] * cfg['train']['accumulation']:
        raise ValueError('Checkpoint cursor is not at a complete optimizer update')
    saved_groups = state['optimizer']['param_groups']
    if [(g['name'],len(g['params'])) for g in saved_groups] != [(g['name'],len(g['params'])) for g in optimizer.param_groups]:
        raise ValueError('Checkpoint optimizer groups differ')
    with torch.no_grad():
        for n,v in state['model'].items():
            if named[n].shape!=v.shape:
                raise ValueError(f'Checkpoint shape differs: {n}')
            named[n].copy_(v)
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
    rng = torch.Generator(device=sample['first'].device).manual_seed(seed)
    x = torch.randn((1,48,21,30,52),device=sample['first'].device,generator=rng)
    policy = cfg['inference']
    grid = torch.linspace(1,0,policy['steps']+1,device=x.device)
    grid = policy['shift']*grid/(1+(policy['shift']-1)*grid)
    x[:,:,:1] = sample['first']
    for i in range(policy['steps']):
        pos,_ = model.velocity(x,sample['first'],sample['text'],grid[i],c,sample['times'],groups)
        neg,_ = model.velocity(x,sample['first'],null,grid[i],nc,sample['times'],ngroups)
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
                sigma = torch.tensor(.5,device='cuda')
                noise = torch.randn(sample['latent'].shape,device='cuda',generator=torch.Generator(device='cuda').manual_seed(42000+i))
                noisy = (1-sigma)*sample['latent']+sigma*noise
                pred,_ = model.velocity(noisy,sample['first'],sample['text'],sigma,c,sample['times'],groups)
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
    parser.add_argument('--checkpoint-every',type=int,default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    save_every = args.checkpoint_every or cfg['train']['save_every']
    if save_every < 1:
        raise ValueError('Checkpoint interval must be positive')
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
        log('load_frozen_qwen')
        teacher = Teacher(cfg,log=log)
        if any(p.requires_grad for p in teacher.knowledge.backbone.parameters()):
            raise RuntimeError('Qwen must be frozen')
        optimizer = torch.optim.AdamW(model.optim_groups(),betas=(.9,.999),weight_decay=.01,foreach=False)
        data,heldout = Data(cfg),Data(cfg,'validation')
        null = torch.load(Path(cfg['paths']['cache'])/'base/null_text.pt',weights_only=True,map_location='cpu').float()[None].cuda()
        completed,cursor = restore(model,optimizer,args.resume,cfg) if args.resume else (0,0)
        if args.resume:
            log('resumed',checkpoint=str(Path(args.resume).resolve()),step=completed,cursor=cursor,
                optimizer_restored=True,rng_restored=True,configuration_identical=True,
                checkpoint_every=save_every)
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
            optimizer.zero_grad(set_to_none=True)
            sums = {'fm':0.,'aux':0.,'weighted_aux':0.}
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
                    target = None if drop else teacher.local_target(sample,evidence,model)
                    c,groups = model.enhancement(evidence)
                    noise = torch.randn(sample['latent'].shape,device='cuda',generator=torch.Generator(device='cuda').manual_seed(cfg['train']['seed']*2000003+cursor))
                    noisy = (1-sigma)*sample['latent']+sigma*noise
                    pred,aux = model.velocity(noisy,sample['first'],null if drop else sample['text'],sigma,c,sample['times'],groups,target,sample['valid'])
                    fm = flow_matching(pred,noise,sample['latent'],sample['valid'])
                    # Weak local labels supervise a specific real-video branch.
                    # Preserve (1-sigma) in V6; use it for revised association too.
                    weight = coefficient*(1-sigma)
                    objective = fm+(weight*aux if target is not None else 0)
                if not torch.isfinite(objective):
                    raise FloatingPointError(f'Nonfinite objective: {sample["id"]}')
                (objective/cfg['train']['accumulation']).backward()
                fm_value,aux_value = float(fm.detach()),float(aux.detach())
                weighted_aux = float(weight.detach())*aux_value
                del objective,fm,aux,pred,c,noisy,noise
                sums['fm'] += fm_value;sums['aux'] += aux_value;sums['weighted_aux'] += weighted_aux
                aux_count += int(target is not None)
                cursor += 1
                log('micro',step=step+1,phase=phase,micro=micro+1,sample=sample['id'],fm=fm_value,
                    aux=aux_value,weighted_aux=weighted_aux,sigma=float(sigma),text_dropped=drop,
                    aux_active=target is not None,aux_coefficient=coefficient,
                    peak_gib=torch.cuda.max_memory_allocated()/2**30)
                del sample,evidence,target
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
            if completed%save_every==0 or completed in cfg['train']['keep_steps'] or completed==stop:
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
