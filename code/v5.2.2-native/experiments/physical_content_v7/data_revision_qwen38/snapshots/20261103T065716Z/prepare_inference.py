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
    args=p.parse_args()
    from PIL import Image
    caption=args.caption_file.read_text().strip()
    with Image.open(args.image) as image:rgb=image.convert('RGB')
    if args.stage=='reasoning':
        client=Client()
        for arm,text in [('positive',caption),('negative','')]:
            path=args.output/f'{arm}.json'
            if path.exists():
                existing=json.loads(path.read_text())
                if existing['caption']!=text or existing['first_image']!=str(args.image.resolve()):
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
            if reasoning['first_image']!=str(args.image.resolve()) or reasoning['duration']!=args.duration:
                raise ValueError('Input image or duration changed since reasoning generation')
            evidence,words=encode_input(knowledge,image,text,args.duration,reasoning,sample=args.id,
                revision='qwen38_first_image_request_v1')
            evidence.update(words)
            atomic_torch(evidence,args.output/f'{arm}.pt')
    save(args.output/'ready.json',dict(id=args.id,caption=caption,duration=args.duration,
        image=str(args.image.resolve()),scope='first_image_and_request_only',
        text_producer='qwen3.8-flash',feature_encoder=cfg['paths']['qwen_checkpoint'],
        width=4096,no_future_video=True,native_wan_prompt_unchanged=True))


if __name__=='__main__':main()
