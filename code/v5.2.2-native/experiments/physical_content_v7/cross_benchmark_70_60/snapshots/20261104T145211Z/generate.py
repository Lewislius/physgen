"""Replay each preserved400 model on the fixed WMB70/PhysicsIQ60 I2V protocol."""
from pathlib import Path
import argparse
import json
import os
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import ROOT, activate, plan, owned_cases, shared, save, gpu_guard, case_seed


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',required=True);parser.add_argument('--limit',type=int)
    a=parser.parse_args();gpu_guard(a.model)
    import torch
    import imageio.v2 as imageio
    torch.set_num_threads(4);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    protocol=plan();s=protocol['sampling'];spec=protocol['models'][a.model]
    activate(spec['source'])
    from physical_content.model import PhysicalModel
    from physgen_native.decoder import VideoDecoder
    state=torch.load(spec['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
    if state['step']!=400 or state['config']['source_snapshot']!=spec['source']:raise ValueError('Wrong frozen checkpoint/source')
    cfg=state['config'];model=PhysicalModel(cfg);parameters=dict(model.named_parameters());buffers=dict(model.named_buffers())
    if set(state['model'])!=set(model.learned_names):raise ValueError('Model/checkpoint parameter mismatch')
    with torch.no_grad():
        for name,value in state['model'].items():parameters[name].copy_(value)
        for name,value in state['constants'].items():buffers[name].copy_(value)
    del state,parameters,buffers
    model.eval().requires_grad_(False);decoder=VideoDecoder(cfg).eval();native=cfg['variant']=='native_lora'
    root=ROOT/'outputs'/a.model;root.mkdir(parents=True,exist_ok=True)
    cases=owned_cases(a.model,a.limit) if a.limit else protocol['cases']
    for seed in protocol['generation_seeds']:
        out=root/f'seed{seed}';out.mkdir(parents=True,exist_ok=True)
        save(out/'manifest.json',dict(model=a.model,checkpoint=spec,seed=seed,sampling=s,cases=protocol['cases'],scope=protocol['scope']))
        pending=[c for c in cases if not (out/'cases'/c['id']/'video.json').exists()]
        while pending:
            ready=next((c for c in pending if (shared(c)/'ready.json').exists()),None)
            if ready is None:
                failed=[]
                for path in (ROOT/'jobs').glob('*/status.json'):
                    status=json.loads(path.read_text())
                    if status.get('status')=='failed' and status.get('stage','').startswith('prepare_'):failed.append(status)
                if failed:raise RuntimeError('A shared preparation owner failed; preserve generated results and resume after correction: '+str(failed))
                print(json.dumps(dict(stage='waiting_shared_inputs',remaining=len(pending))),flush=True)
                time.sleep(30);continue
            case=ready;folder=shared(case);destination=out/'cases'/case['id'];destination.mkdir(parents=True,exist_ok=True)
            start=time.monotonic();inputs=torch.load(folder/'conditions.pt',map_location='cpu',weights_only=True)
            inputs={k:v.cuda() for k,v in inputs.items()};times=inputs['times'][None]
            if native:
                positive=negative=dict(memory=inputs['first'].new_zeros((1,1,1)))
            else:
                def evidence(arm):
                    value=torch.load(folder/(arm+'.pt'),map_location='cpu',weights_only=True)
                    return {k:(v.cuda() if torch.is_tensor(v) else v) for k,v in value.items()}
                positive=evidence('positive');negative=evidence('negative')
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
                c,groups=model.enhancement(positive);nc,ngroups=model.enhancement(negative)
                actual_seed=case_seed(seed,case['index'])
                x=torch.randn((1,48,21,30,52),generator=torch.Generator(device='cpu').manual_seed(actual_seed),dtype=torch.float32).cuda()
                x[:,:,:1]=inputs['first']
                grid=torch.linspace(1,0,s['steps']+1,device='cuda');grid=s['shift']*grid/(1+(s['shift']-1)*grid)
                for index in range(s['steps']):
                    neg,_=model.velocity(x,inputs['first'],inputs['null_text'],grid[index],nc,times,ngroups)
                    pos,_=model.velocity(x,inputs['first'],inputs['text'],grid[index],c,times,groups)
                    if native or s['physical_guidance']==s['text_guidance']:
                        velocity=neg+s['text_guidance']*(pos-neg)
                    else:
                        neutral,_=model.velocity(x,inputs['first'],inputs['text'],grid[index],nc,times,ngroups)
                        velocity=neg+s['text_guidance']*(neutral-neg)+s['physical_guidance']*(pos-neutral)
                        del neutral
                    x=x+(grid[index+1]-grid[index])*velocity;x[:,:,:1]=inputs['first']
                    if (index+1)%10==0:print(json.dumps(dict(stage='sampling',model=a.model,case=case['id'],step=index+1)),flush=True)
                frames=decoder(x)
            rgb=((frames.permute(0,2,3,1).clamp(-1,1)+1)*127.5).round().byte().cpu().numpy()
            if rgb.shape!=(81,480,832,3):raise ValueError('Unexpected decoded video dimensions')
            rgb=rgb[:case['output_frames']]
            path=destination/'video.partial.mp4'
            imageio.mimwrite(path,rgb,fps=s['fps'],codec='libx264',macro_block_size=None,ffmpeg_params=['-crf','18','-pix_fmt','yuv420p'])
            path.replace(destination/'video.mp4')
            exported=out/case['benchmark']/case['output_filename'];exported.parent.mkdir(parents=True,exist_ok=True)
            if not exported.exists():exported.symlink_to(destination/'video.mp4')
            record=dict(case=case['id'],caption=case['caption'],model=a.model,checkpoint=spec['checkpoint'],seed=seed,noise_seed=actual_seed,
                        first_image=str(folder/'first.png'),original_image=case['image'],output_frames=case['output_frames'],sampling=s,seconds=time.monotonic()-start)
            save(destination/'video.json',record);pending.remove(case)
            save(out/'generation_status.json',dict(status='running',complete=len(cases)-len(pending),requested=len(cases),model=a.model))
            print(json.dumps(dict(stage='video_complete',**record)),flush=True)
            del inputs,positive,negative,c,nc,x,pos,neg,velocity,frames,rgb
        save(out/'generation_status.json',dict(status='partial' if a.limit else 'completed',complete=len(cases),requested=len(protocol['cases']),model=a.model))


if __name__=='__main__':main()
