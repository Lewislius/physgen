"""Prepare only VP041-VP060 with the original frozen first-image/request encoder."""
from pathlib import Path
import argparse
import copy
import json
import os
import random
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
import numpy as np
import torch
import yaml
from PIL import Image
from physical_scaling.teacher import Teacher as InputTeacher
from physical_scaling.common import gpu_guard,atomic_json,atomic_torch

ROOT=bootstrap.BASE/'experiments/physical_content_v7'
HERE=ROOT/'videophy60'
PINNED=bootstrap.BASE/'experiments/physical_scaling_q256/inference_snapshots/20261030T034046Z'


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-inputs-only',action='store_true')
    args=parser.parse_args()
    sys.path.insert(0,str(PINNED))
    from protocol import load_inputs,load_noise
    plan=json.loads((HERE/'protocol.json').read_text())
    cases=plan['cases'][40:]
    if [c['id'] for c in cases]!=[f'VP{i:03d}' for i in range(41,61)]:
        raise ValueError('Use the existing fixed extra20 without outcome-based selection')
    if args.check_inputs_only:
        for case in cases:load_inputs(case);load_noise(case)
        print(json.dumps(dict(status='existing_extra20_inputs_readable',count=20,model_loads=0)))
        return
    gpu_guard(HERE/'allocations'/os.getenv('DET_EXPERIMENT_ID','local'))
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    torch.set_num_threads(4)
    cfg=copy.deepcopy(yaml.safe_load((ROOT/'configs/physical_content.yaml').read_text()))
    cfg['paths']['supervision_cache']=str(HERE/'qwen_inputs')
    sys.path.insert(0,str(ROOT/'content_reader_r8/snapshots/20261031T123319Z'))
    from physical_content.evidence import recover_words,recover_request_words
    # The original InputTeacher.input generates/re-reads the same128-token
    # physical analysis used in R1. Its video-target methods are never called.
    teacher=InputTeacher(cfg)
    completed=[]
    for case in cases:
        folder=HERE/'inputs'/case['id'];folder.mkdir(parents=True,exist_ok=True)
        inputs,canvas=load_inputs(case)
        rgb=np.array(Image.open(canvas).convert('RGB'),copy=True)
        sample=dict(id='videophy60_'+case['id'],caption=case['caption'],
            times=inputs['times'][None].cuda(),
            first_rgb=torch.from_numpy(rgb).permute(2,0,1)[None].cuda().float()/127.5-1)
        duration=float(inputs['times'][-1])
        for arm,drop in [('positive',False),('negative',True)]:
            path=folder/(arm+'.pt')
            if path.is_file():
                previous=torch.load(path,map_location='cpu',weights_only=True)
                expected_caption='' if drop else case['caption']
                if previous['sample']!=sample['id'] or previous['caption']!=expected_caption:
                    raise ValueError('Existing evidence belongs to a different case')
                continue
            with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
                evidence=teacher.input(sample,drop=drop)
            evidence={k:(v.cpu() if torch.is_tensor(v) else v) for k,v in evidence.items()}
            evidence.update(recover_words(evidence,duration,teacher.knowledge.tokenizer))
            evidence.update(recover_request_words(evidence,duration,teacher.knowledge.tokenizer))
            evidence['input_canvas']=str(canvas)
            atomic_torch(evidence,path)
        completed.append(case['id'])
        atomic_json(dict(status='preparing',completed=completed,total=20),HERE/'preparation_status.json')
        print(json.dumps(dict(case=case['id'],status='prepared',completed=len(completed))),flush=True)
    atomic_json(dict(status='completed',completed=completed,total=20,
        scope='Only original first image/request and presentation interval; unchanged128-token encoder; shared by R1/R9/R10.'),HERE/'preparation_status.json')


if __name__=='__main__':main()
