"""Paired inference interventions requested in the scaling audit; frozen weights."""
from pathlib import Path
import sys
import json
import time
import os
import copy
import torch
import numpy as np

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
PINNED=BASE/'experiments/physical_scaling_q256/inference_snapshots/20261030T034046Z'
sys.path.insert(0,str(PINNED))
from settings import activate_training, JOBS
activate_training()
from protocol import checkpoint_state,restore_parameters,manifest_for,load_inputs,load_noise,verify_video
from physical_scaling.model import PhysicalModel
from physical_scaling.common import gpu_guard
from physgen_native.decoder import VideoDecoder
from generate import write_video

OUT=HERE/'gpu_results'
CACHE=BASE/'experiments/physical_variants/inference_cache/physical_r1_first_image_request/input'


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2));temp.replace(path)


def log(kind,**kw):
    record=dict(kind=kind,time=time.time(),**kw)
    print(json.dumps(record),flush=True)
    with (OUT/'events.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')


def evidence(case,null=False):
    d=torch.load(CACHE/f'videophy40_{case}_{"null" if null else "text"}.pt',map_location='cpu',weights_only=True)
    return {k:v.cuda() if torch.is_tensor(v) else v for k,v in d.items()}


@torch.no_grad()
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    gpu_guard(OUT/'allocations'/os.getenv('DET_EXPERIMENT_ID','local'))
    torch.set_num_threads(4);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    state=checkpoint_state(JOBS['v6_assoc']['checkpoint'],'v6_assoc')
    cfg=state['config'];model=PhysicalModel(cfg);restore_parameters(model,state);del state
    decoder=VideoDecoder(cfg).eval()
    original_outputs={k:m.output.weight.detach().clone() for k,m in model.fusions.items()}
    # Keep the native T5 request, first image, Wan LoRA, sampler and noise identical.
    manifest=manifest_for('v6_assoc','videophy40')
    cases=[c for c in manifest['cases'] if c['id'] in ['VP014','VP015']]
    donor={'VP014':'VP025','VP015':'VP032'}
    for mode in ['swapped','image_only','no_c']:
        plan=copy.deepcopy(manifest);plan['cases']=cases
        plan['audit_intervention']=mode;plan['audit_reference']=JOBS['v6_assoc']['output']
        save(OUT/mode/'manifest.json',plan)
        for key,m in model.fusions.items():m.output.weight.copy_(original_outputs[key] if mode!='no_c' else torch.zeros_like(original_outputs[key]))
        for case in cases:
            path=OUT/mode/'cases'/case['id']/'planned.mp4'
            if path.exists() and path.with_suffix('.json').exists():continue
            start=time.monotonic();path.parent.mkdir(parents=True,exist_ok=True)
            values,_=load_inputs(case);sample={k:v.cuda() for k,v in values.items()};sample['times']=sample['times'][None]
            source=donor[case['id']] if mode=='swapped' else case['id']
            with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
                c,g=model.enhancement(evidence(source,null=mode=='image_only'))
                nc,ng=model.enhancement(evidence(source,null=True))
                x=load_noise(case).cuda().clone();x[:,:,:1]=sample['first']
                grid=torch.linspace(1,0,51,device='cuda');grid=5*grid/(1+4*grid)
                for i in range(50):
                    pos,_=model.velocity(x,sample['first'],sample['text'],grid[i],c,sample['times'],g)
                    neg,_=model.velocity(x,sample['first'],sample['null_text'],grid[i],nc,sample['times'],ng)
                    if mode=='swapped' and i in [0,9,24,39,49]:
                        # Same current x: an actual diffusion-state causal sensitivity probe.
                        rc,rg=model.enhancement(evidence(case['id']))
                        rn,rng=model.enhancement(evidence(case['id'],True))
                        rp,_=model.velocity(x,sample['first'],sample['text'],grid[i],rc,sample['times'],rg)
                        rv,_=model.velocity(x,sample['first'],sample['null_text'],grid[i],rn,sample['times'],rng)
                        a=(neg+5*(pos-neg))[:,:,1:];b=(rv+5*(rp-rv))[:,:,1:]
                        log('same_state_sensitivity',case=case['id'],step=i+1,sigma=float(grid[i]),
                            reference='real C',intervention='unrelated C',
                            relative_cfg_velocity_difference=float((a-b).norm()/b.norm()),
                            relative_positive_velocity_difference=float((pos-rp)[:,:,1:].norm()/rp[:,:,1:].norm()),
                            c_relative_difference=float((c-rc).norm()/rc.norm()))
                        del rc,rn,rp,rv,a,b
                    x=x+(grid[i+1]-grid[i])*(neg+5*(pos-neg));x[:,:,:1]=sample['first']
                    if (i+1)%10==0:log('sampling',mode=mode,case=case['id'],step=i+1)
                latent=x.detach().cpu()
                frames=decoder(x)
            frames=((frames.permute(0,2,3,1).clamp(-1,1)+1)*127.5).round().byte().cpu().numpy()
            write_video(frames,path);verify_video(path)
            torch.save(latent,path.with_suffix('.latent.pt'))
            save(path.with_suffix('.json'),dict(case=case['id'],mode=mode,donor=source if mode=='swapped' else None,
                seed=42,steps=50,guidance=5,shift=5,checkpoint=JOBS['v6_assoc']['checkpoint'],
                native_t5_unchanged=True,first_image_unchanged=True,noise=case['initial_noise'],
                source_input_scope='cached first image and request only',elapsed=time.monotonic()-start))
            log('completed_video',mode=mode,case=case['id'],seconds=time.monotonic()-start)
            del frames,latent,x,c,nc,sample,values,pos,neg
    save(OUT/'status.json',dict(status='generated',cases=[c['id'] for c in cases],modes=['swapped','image_only','no_c']))


if __name__=='__main__':main()
