"""Train the three requested physical baselines from official pretrained bases."""
from pathlib import Path
import argparse
import json
import math
import os
import random
import sys
import time
import numpy as np
import torch
from .common import ROOT, load_config, audit, gpu_guard, atomic_json, atomic_torch, cpu_tree, Log
from .data import Data, SampleOrder
from .teacher import Teacher
from .model import PhysicalModel, semantic_ce, enable_frozen_decoder_checkpointing
from .supervision import flow_matching


def rates(optimizer,cfg,step):
    t = cfg['train']
    interface,joint = cfg['stages']['interface'],cfg['stages']['joint']
    if step<interface:
        if step<t['warmup']:
            new = t['new_lr']*(step+1)/t['warmup']
        else:
            fraction = (step-t['warmup'])/max(1,interface-t['warmup']-1)
            new = t['joint_lr']+(t['new_lr']-t['joint_lr'])*.5*(1+math.cos(math.pi*fraction))
        wan = 0.
    else:
        j = step-interface
        floor = t['minimum_lr_ratio']
        new = t['joint_lr']*(floor+(1-floor)*.5*(1+math.cos(math.pi*j/max(1,joint-1))))
        wan = t['wan_lora_lr']*min(1.,(j+1)/t['wan_warmup'])
        if j>=t['wan_warmup']:
            frac = (j-t['wan_warmup'])/max(1,joint-t['wan_warmup']-1)
            wan *= floor+(1-floor)*.5*(1+math.cos(math.pi*frac))
    for group in optimizer.param_groups:
        group['lr'] = wan if group['name']=='wan_lora' else new


def save(model,optimizer,cfg,run,step,cursor):
    named = dict(model.named_parameters())
    state = dict(revision=cfg['revision'],variant=cfg['variant'],config=cfg,step=step,cursor=cursor,
                 model={n:named[n].detach().cpu() for n in model.learned_names},
                 optimizer=cpu_tree(optimizer.state_dict()),
                 rng=dict(python=random.getstate(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()))
    atomic_torch(state,run/'latest.pt')
    if step in (cfg['stages']['interface'],sum(cfg['stages'].values())):
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
        log('load_frozen_qwen')
        teacher = Teacher(cfg,log=log)
        if cfg['variant']=='v5_ce':
            enable_frozen_decoder_checkpointing(teacher.knowledge.backbone)
        if any(p.requires_grad for p in teacher.knowledge.backbone.parameters()):
            raise RuntimeError('Qwen must be frozen')
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
                    target,answer = None,None
                    if not drop:
                        if cfg['variant']=='v5_ce':
                            answer = teacher.semantic_target(sample)
                        else:
                            target = teacher.local_target(sample,evidence,model)
                    c,groups = model.enhancement(evidence)
                    noise = torch.randn(sample['latent'].shape,device='cuda',generator=torch.Generator(device='cuda').manual_seed(cfg['train']['seed']*2000003+cursor))
                    noisy = (1-sigma)*sample['latent']+sigma*noise
                    pred,aux = model.velocity(noisy,sample['first'],null if drop else sample['text'],sigma,c,sample['times'],groups,target,sample['valid'])
                    fm = flow_matching(pred,noise,sample['latent'],sample['valid'])
                    # Weak local labels supervise a specific real-video branch.
                    # Preserve (1-sigma) in V6; use it for revised association too.
                    weight = cfg['loss']['aux_weight']*(1-sigma)
                    objective = fm+(weight*aux if target is not None else 0)
                if not torch.isfinite(objective):
                    raise FloatingPointError(f'Nonfinite objective: {sample["id"]}')
                (objective/cfg['train']['accumulation']).backward()
                fm_value,aux_value = float(fm.detach()),float(aux.detach())
                weighted_aux = float(weight.detach())*aux_value
                del objective,fm,aux,pred,c,noisy,noise
                if answer is not None:
                    # Recompute tiny C graph after freeing the Wan graph: same
                    # mathematical CE gradient without simultaneous activations.
                    with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
                        ce = semantic_ce(model,teacher.knowledge,evidence,answer)
                    if not torch.isfinite(ce):
                        raise FloatingPointError('Nonfinite C semantic CE')
                    (cfg['loss']['aux_weight']*ce/cfg['train']['accumulation']).backward()
                    aux_value = float(ce.detach());weighted_aux = cfg['loss']['aux_weight']*aux_value
                    del ce
                sums['fm'] += fm_value;sums['aux'] += aux_value;sums['weighted_aux'] += weighted_aux
                aux_count += int(not drop)
                cursor += 1
                log('micro',step=step+1,phase=phase,micro=micro+1,sample=sample['id'],fm=fm_value,
                    aux=aux_value,weighted_aux=weighted_aux,sigma=float(sigma),text_dropped=drop,
                    peak_gib=torch.cuda.max_memory_allocated()/2**30)
                del sample,evidence,target,answer
            grad = {}
            for group in optimizer.param_groups:
                parameters = [p for p in group['params'] if p.grad is not None]
                norm = torch.nn.utils.clip_grad_norm_(parameters,cfg['train']['grad_clip'],error_if_nonfinite=True) if parameters else 0.
                grad[group['name']] = float(norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            completed = step+1
            torch.cuda.synchronize()
            log('train',step=completed,phase=phase,cursor=cursor,seconds=time.monotonic()-start,
                **{k:v/cfg['train']['accumulation'] for k,v in sums.items()},aux_samples=aux_count,
                gradients=grad,learning_rates={g['name']:g['lr'] for g in optimizer.param_groups},
                peak_gib=torch.cuda.max_memory_allocated()/2**30)
            atomic_json(dict(status='training',step=completed,phase=phase,variant=cfg['variant']),run/'status.json')
            if completed%cfg['train']['save_every']==0 or completed in (cfg['stages']['interface'],stop):
                save(model,optimizer,cfg,run,completed,cursor)
                log('checkpoint',step=completed,path=str(run/'latest.pt'))
            if completed in (cfg['stages']['interface'],total):
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
