"""Generate demo21, then fixed VideoPhy2-40 with each checkpoint's exact model."""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from settings import ROOT, JOBS, activate_training
activate_training()
import numpy as np
import torch
from PIL import Image
from physical_training.model import PhysicalModel
from physical_training.teacher import Teacher
from physical_training.common import atomic_json, gpu_guard
from physgen_native.knowledge import QwenKnowledge
from physgen_native.decoder import VideoDecoder
from protocol import (checkpoint_state, restore_parameters, manifest_for, load_inputs,
                      load_noise, verify_video, read, require, LATENT_SHAPE)


class InputEncoder:
    """Use the trained input encoder verbatim; no training-target preparation."""
    input, cache, _report = Teacher.input, Teacher.cache, Teacher._report

    def __init__(self,cfg,log):
        self.cfg,self.device,self.log = cfg,'cuda',log
        self.knowledge = QwenKnowledge(cfg,device='cuda').eval().requires_grad_(False)
        self.root = ROOT/'inference_cache/physical_r1_first_image_request'
        self.root.mkdir(parents=True,exist_ok=True)


@torch.no_grad()
def sample_latent(model,sample,positive,negative,initial_noise=None,seed=42,progress=None):
    """Same shifted Euler/CFG as training validation, plus exact saved-noise replay."""
    device = sample['first'].device
    c,groups = model.enhancement(positive)
    nc,ngroups = model.enhancement(negative)
    x = (torch.randn(LATENT_SHAPE,device=device,generator=torch.Generator(device=device).manual_seed(seed))
         if initial_noise is None else initial_noise.to(device).clone())
    policy = model.cfg['inference']
    grid = torch.linspace(1,0,policy['steps']+1,device=device)
    grid = policy['shift']*grid/(1+(policy['shift']-1)*grid)
    x[:,:,:1] = sample['first']
    for i in range(policy['steps']):
        pos,_ = model.velocity(x,sample['first'],sample['text'],grid[i],c,sample['times'],groups)
        neg,_ = model.velocity(x,sample['first'],sample['null_text'],grid[i],nc,sample['times'],ngroups)
        x = x+(grid[i+1]-grid[i])*(neg+policy['guidance']*(pos-neg))
        x[:,:,:1] = sample['first']
        if progress is not None and ((i+1)%10==0 or i+1==policy['steps']):
            progress(i+1)
    return x


def write_video(frames,path):
    import imageio.v2 as imageio
    temporary = path.with_suffix('.partial.mp4')
    try:
        with imageio.get_writer(str(temporary),fps=24.,codec='libx264',pixelformat='yuv420p',
                macro_block_size=None,quality=None,ffmpeg_params=['-crf','18','-profile:v','baseline']) as stream:
            for frame in frames:
                stream.append_data(frame)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def generate(variant):
    job = JOBS[variant]
    root = Path(job['output'])
    gpu_guard(root/'allocations'/os.environ.get('DET_EXPERIMENT_ID','local'))
    state = checkpoint_state(job['checkpoint'],variant)
    cfg = state['config']
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    torch.set_num_threads(4)
    def log(kind,**values):
        record = dict(kind=kind,time=time.time(),variant=variant,**values)
        print(json.dumps(record,ensure_ascii=False),flush=True)
        with (root/'generation.jsonl').open('a') as handle:
            handle.write(json.dumps(record,ensure_ascii=False)+'\n')
    log('loading_final_model',checkpoint=job['checkpoint'])
    model = PhysicalModel(cfg)
    restore_parameters(model,state)
    del state
    encoder = InputEncoder(cfg,log)
    decoder = VideoDecoder(cfg).eval()
    log('model_ready',checkpoint_step=224,enhancement_passes=2)
    for suite in ('demo21','videophy40'):
        directory = root/suite
        directory.mkdir(parents=True,exist_ok=True)
        manifest = manifest_for(variant,suite)
        if (directory/'manifest.json').exists():
            require(read(directory/'manifest.json')==manifest,'Resume manifest differs')
        else:
            atomic_json(manifest,directory/'manifest.json')
        summary = dict(status='running',suite=suite,variant=variant,completed_cases=0,cases=[])
        atomic_json(dict(stage=suite,variant=variant,status='running'),root/'generation_status.json')
        try:
            for index,case in enumerate(manifest['cases'],1):
                path = directory/case['output']
                path.parent.mkdir(parents=True,exist_ok=True)
                expected = dict(id=case['id'],variant=variant,checkpoint=job['checkpoint'],
                    optimizer_step=224,seed=case['seed'],sampler=cfg['inference'],
                    initial_noise=case.get('initial_noise'),enhancement_passes=2)
                if path.is_file() and path.with_suffix('.json').is_file():
                    saved = read(path.with_suffix('.json'))
                    require(all(saved.get(k)==v for k,v in expected.items()),'Resume video provenance differs')
                    encoded = verify_video(path)
                    result = dict(id=case['id'],status='completed',resumed=True,encoded=encoded)
                else:
                    start = time.monotonic()
                    torch.cuda.reset_peak_memory_stats()
                    inputs,canvas = load_inputs(case)
                    sample = {k:v.to('cuda') for k,v in inputs.items()}
                    sample['times'] = sample['times'][None]
                    rgb = np.array(Image.open(canvas).convert('RGB'),copy=True)
                    sample.update(id=f'{suite}_{case["id"]}',caption=case['caption'],
                        first_rgb=torch.from_numpy(rgb).permute(2,0,1)[None].cuda().float()/127.5-1)
                    atomic_json(case,path.parent/'input.json')
                    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
                        positive = encoder.input(sample)
                        negative = encoder.input(sample,drop=True)
                        def progress(step):
                            log('sampling',suite=suite,case=case['id'],index=index,step=step,total_steps=50)
                        latent = sample_latent(model,sample,positive,negative,load_noise(case),case['seed'],progress)
                        require(bool(torch.isfinite(latent).all()),'Nonfinite sampled latent')
                        frames = decoder(latent)
                    frames = ((frames.permute(0,2,3,1).clamp(-1,1)+1)*127.5).round().byte().cpu().numpy()
                    write_video(frames,path)
                    encoded = verify_video(path)
                    metadata = dict(expected,caption=case['caption'],source_canvas=str(canvas),
                        condition_scope='first image and request only; native T5 retained; no historical clauses',
                        analysis=positive['analysis'],relations=positive['relations'],
                        null_analysis=negative['analysis'],revision=cfg['revision'],
                        inference_experiment_id=os.getenv('DET_EXPERIMENT_ID'),
                        frames=81,fps=24.,condition_tokens=128)
                    atomic_json(metadata,path.with_suffix('.json'))
                    result = dict(id=case['id'],status='completed',seconds=time.monotonic()-start,
                        peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,encoded=encoded)
                    del latent,frames,sample,positive,negative
                summary['completed_cases'] += 1
                summary['cases'].append(result)
                atomic_json(summary,directory/'summary.json')
                log('video_complete',suite=suite,count=summary['completed_cases'],total=len(manifest['cases']),**result)
            summary['status'] = 'completed'
        except BaseException as error:
            summary.update(status='failed',error=repr(error))
            raise
        finally:
            atomic_json(summary,directory/'summary.json')
    atomic_json(dict(status='completed',demo_completed=21,videophy_completed=40),root/'generation_status.json')


if __name__=='__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--variant',choices=JOBS,required=True)
    generate(parser.parse_args().variant)
