"""Prepare disjoint shared inputs on each model's allocated GPU; no downloads."""
from pathlib import Path
import argparse
import gc
import json
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import ROOT, BASE, B_SOURCE, activate, plan, owned_cases, shared, save, save_tensor, gpu_guard


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--stage',choices=['images','encode','evidence'],required=True)
    p.add_argument('--limit',type=int);a=p.parse_args()
    gpu_guard(a.model)
    activate(B_SOURCE)
    import torch
    import numpy as np
    import yaml
    from PIL import Image
    torch.set_num_threads(4)
    protocol=plan();s=protocol['sampling'];cases=owned_cases(a.model,a.limit)
    cfg=yaml.safe_load((BASE/'experiments/physical_content_v7/configs/physical_content.yaml').read_text())
    sys.path.insert(0,cfg['paths']['wan_code'])
    base=Path(cfg['paths']['wan_checkpoint'])
    def log(case,kind,**kwargs):
        print(json.dumps(dict(case=case['id'],stage=kind,time=time.time(),**kwargs)),flush=True)
    with torch.inference_mode():
        if a.stage=='images':
            for case in cases:
                folder=shared(case);folder.mkdir(parents=True,exist_ok=True)
                expected=dict(case=case['id'],caption=case['caption'],source_image=case['image'],
                              policy=protocol['image_policy'])
                if (folder/'image.json').exists():
                    if json.loads((folder/'image.json').read_text())!=expected:raise ValueError('Changed image provenance')
                    continue
                with Image.open(case['image']) as image:
                    image.convert('RGB').resize((s['width'],s['height']),Image.Resampling.LANCZOS).save(folder/'first.partial.png')
                (folder/'first.partial.png').replace(folder/'first.png')
                save(folder/'image.json',expected);log(case,'official_image_ready')
            return
        if a.stage=='encode':
            todo=[c for c in cases if not (shared(c)/'text.pt').exists()]
            if todo:
                from wan.modules.t5 import T5EncoderModel
                encoder=T5EncoderModel(text_len=512,dtype=torch.bfloat16,device='cuda',
                    checkpoint_path=str(base/'models_t5_umt5-xxl-enc-bf16.pth'),tokenizer_path=str(base/'google/umt5-xxl'))
                negative=encoder([s['negative_prompt']],'cuda')[0].float().cpu()[None]
                for case in todo:
                    ids=encoder.tokenizer([case['caption']],truncation=False,padding=False,add_special_tokens=True)
                    if ids.shape[-1]>512:raise ValueError('Official caption would be truncated')
                    text=encoder([case['caption']],'cuda')[0].float().cpu()[None]
                    save_tensor(shared(case)/'text.pt',dict(text=text,null_text=negative));log(case,'text_complete')
                del encoder;gc.collect();torch.cuda.empty_cache()
            todo=[c for c in cases if not (shared(c)/'conditions.pt').exists()]
            if todo:
                from wan.modules.vae2_2 import Wan2_2_VAE
                vae=Wan2_2_VAE(vae_pth=str(base/'Wan2.2_VAE.pth'),dtype=torch.float32,device='cuda')
                for case in todo:
                    folder=shared(case)
                    with Image.open(folder/'first.png') as image:
                        if image.size!=(832,480):raise ValueError('Unexpected shared image dimensions')
                        rgb=torch.from_numpy(np.array(image.convert('RGB'),copy=True)).permute(2,0,1)[:,None].float().cuda()/127.5-1
                    first=vae.encode([rgb])[0].float().cpu()[None]
                    if tuple(first.shape)!=(1,48,1,30,52):raise ValueError('Unexpected first latent dimensions')
                    value=torch.load(folder/'text.pt',map_location='cpu',weights_only=True)
                    value.update(first=first,times=torch.arange(0,81,4).float()/16,valid=torch.ones(30,52))
                    save_tensor(folder/'conditions.pt',value);log(case,'conditions_complete')
            return
        todo=[c for c in cases if not (shared(c)/'ready.json').exists()]
        if not todo:return
        from physical_scaling.teacher import Teacher
        from physical_content.evidence import recover_words,recover_request_words,recover_visual_tokens
        cfg['paths']['supervision_cache']=str(ROOT/'shared/evidence_cache')
        cfg['paths']['supervision_cache_readonly']=str(ROOT/'shared/unused_readonly')
        teacher=Teacher(cfg)
        for case in todo:
            folder=shared(case)
            with Image.open(folder/'first.png') as image:rgb=np.array(image.convert('RGB'),copy=True)
            sample=dict(id='crossbench_'+case['id'],caption=case['caption'],times=(torch.arange(0,81,4).float()/16)[None].cuda(),
                        first_rgb=torch.from_numpy(rgb).permute(2,0,1)[None].cuda().float()/127.5-1)
            for arm,drop in [('positive',False),('negative',True)]:
                path=folder/(arm+'.pt')
                if path.exists():continue
                with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):value=teacher.input(sample,drop=drop)
                value={k:(v.cpu() if torch.is_tensor(v) else v) for k,v in value.items()}
                for recover in (recover_words,recover_request_words):
                    value.update(recover(value,5.,teacher.knowledge.tokenizer))
                save_tensor(path,value)
            if not (folder/'conditions.pt').exists():raise ValueError('Missing encoded input')
            for arm in ('positive','negative'):
                check=torch.load(folder/(arm+'.pt'),map_location='cpu',weights_only=True)
                expected_caption=case['caption'] if arm=='positive' else ''
                if check['caption']!=expected_caption or check['request_word_strings']!=expected_caption.split():raise ValueError('Incomplete literal request encoding')
            save(folder/'ready.json',dict(case=case['id'],caption=case['caption'],owner=a.model,scope=protocol['evidence']))
            log(case,'shared_input_complete')


if __name__=='__main__':main()
