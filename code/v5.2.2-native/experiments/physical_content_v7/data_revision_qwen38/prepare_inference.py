"""Optional API reasoning -> unchanged local Qwen encoder for new inference inputs."""
from pathlib import Path
import argparse
import json
import os
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from annotate import Client, input_reasoning, save


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=['reasoning','encode'],required=True)
    p.add_argument('--image',type=Path,required=True)
    p.add_argument('--caption-file',type=Path,required=True)
    p.add_argument('--duration',type=float,required=True)
    p.add_argument('--id',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--controls',nargs='*',choices=['reasoning_swapped','empty_reasoning'],default=[])
    p.add_argument('--reasoning-donor',type=Path)
    args=p.parse_args()
    if args.controls and args.stage!='encode':
        p.error('Controls reuse existing reasoning text during local encoding; no extra API calls')
    if 'reasoning_swapped' in args.controls and args.reasoning_donor is None:
        p.error('reasoning_swapped requires a recorded same-relation-count donor JSON')
    from PIL import Image
    caption=args.caption_file.read_text().strip()
    with Image.open(args.image) as image:rgb=image.convert('RGB')
    if args.stage=='reasoning':
        client=Client()
        for arm,text in [('positive',caption),('negative','')]:
            path=args.output/f'{arm}.json'
            if path.exists():
                existing=json.loads(path.read_text())
                if (existing['caption']!=text or existing['first_image']!=str(args.image.resolve())
                        or existing['duration']!=args.duration or existing['sample']!=args.id):
                    raise ValueError('Existing reasoning belongs to different input')
                continue
            value=input_reasoning(rgb,text,args.duration,client)
            value.update(sample=args.id,first_image=str(args.image.resolve()),duration=args.duration)
            save(path,value)
        return
    from encode import SOURCE,encode_input
    import torch
    import numpy as np
    import yaml
    from physical_scaling.common import gpu_guard,atomic_torch
    from physgen_native.knowledge import QwenKnowledge
    gpu_guard(args.output/'allocations'/os.getenv('DET_EXPERIMENT_ID','local'))
    cfg=yaml.safe_load((SOURCE/'configs/physical_content.yaml').read_text())
    knowledge=QwenKnowledge(cfg).eval()
    image=torch.from_numpy(np.array(rgb,copy=True)).permute(2,0,1)[None].cuda().float()/127.5-1
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for arm,text in [('positive',caption),('negative','')]:
            reasoning=json.loads((args.output/f'{arm}.json').read_text())
            if (reasoning['first_image']!=str(args.image.resolve()) or reasoning['duration']!=args.duration
                    or reasoning['sample']!=args.id):
                raise ValueError('Input image or duration changed since reasoning generation')
            evidence,words=encode_input(knowledge,image,text,args.duration,reasoning,sample=args.id,
                revision='qwen38_first_image_request_v1')
            evidence.update(words)
            evidence.update(first_image=str(args.image.resolve()),duration=args.duration)
            atomic_torch(evidence,args.output/f'{arm}.pt')
        if args.controls:
            from condition_controls import encode_control
            own=json.loads((args.output/'positive.json').read_text())
            donor=json.loads(args.reasoning_donor.read_text()) if args.reasoning_donor else None
            for mode in args.controls:
                evidence=encode_control(knowledge,image,caption,args.duration,own,
                    sample=args.id,mode=mode,donor=donor)
                evidence.update(first_image=str(args.image.resolve()),duration=args.duration)
                atomic_torch(evidence,args.output/'controls'/f'{mode}.pt')
    save(args.output/'ready.json',dict(id=args.id,caption=caption,duration=args.duration,
        image=str(args.image.resolve()),scope='first_image_and_request_only',
        text_producer='qwen3.8-flash',feature_encoder=cfg['paths']['qwen_checkpoint'],
        width=4096,no_future_video=True,native_wan_prompt_unchanged=True,
        controls=args.controls,control_policy='Own image/request/positive relation count retained; negative branch unchanged'))


if __name__=='__main__':main()
