"""Local-only request-memory intervention: own image, another fixed request.

Normal60 and training inputs reuse the existing cache. Only16 counterfactual
request encodings are new; no API request or future-video input is involved.
"""
from pathlib import Path
import argparse,fcntl,json,os,sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap

REVISION='r11b_request_memory_swapped_v1'


def donor_map(items,controls):
    by_case={item['case']:item for item in items}
    names=sorted(controls)
    if len(set(names))!=16 or any(name not in by_case for name in names):
        raise ValueError('Use the declared fixed16 requests')
    return {name:by_case[names[(i+1)%len(names)]] for i,name in enumerate(names)}


def load_control(root,item,device='cuda'):
    import torch
    value=torch.load(Path(root)/(item['case']+'.pt'),map_location='cpu',weights_only=True)
    expected=dict(revision=REVISION,sample=item['id'],own_caption=item['caption'],
                  first_image=str(Path(item['image']).resolve()),duration=item['duration'])
    if any(value.get(k)!=v for k,v in expected.items()):
        raise ValueError('Request control does not retain the own input')
    words=value['request_words']
    if (words.ndim!=2 or words.shape[-1]!=4096 or not len(words)
            or value['donor']==item['id'] or value['donor_caption']==item['caption']
            or value['request_word_strings']!=value['donor_caption'].split()):
        raise ValueError('Invalid counterfactual literal request memory')
    return {**value,'request_words':words.to(device)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    args=p.parse_args()
    from physical_content.common import load_config,gpu_guard,atomic_json,atomic_torch
    cfg=load_config(args.config);root=Path(cfg['paths']['request_controls'])
    root.mkdir(parents=True,exist_ok=True)
    owner=(root/'encode.lock').open('a');fcntl.flock(owner,fcntl.LOCK_EX)
    shared=Path(cfg['paths']['inference_evidence'])
    inputs=json.loads((shared/'inputs.json').read_text());items=inputs['cases'];controls=inputs['controls']
    donors=donor_map(items,controls);selected=[v for v in items if v['case'] in donors]
    pending=[]
    for item in selected:
        if (root/(item['case']+'.pt')).exists():
            old=load_control(root,item,'cpu')
            if old['donor']!=donors[item['case']]['id']:raise ValueError('Fixed request donor changed')
        else:pending.append(item)
    if pending:
        import torch,numpy as np
        from PIL import Image
        from encode import encode_input
        from physgen_native.knowledge import QwenKnowledge
        gpu_guard(root/'allocations'/os.getenv('DET_EXPERIMENT_ID','local'))
        torch.set_num_threads(4)
        knowledge=QwenKnowledge(cfg).eval().requires_grad_(False)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            for item in pending:
                donor=donors[item['case']]
                own=json.loads((shared/item['id']/'positive.json').read_text())
                # Keep the canonical local template. Only prefix request words
                # are retained from this pass; physical C still uses own input.
                reasoning=dict(caption=donor['caption'],relations=own['relations'],
                    scope='first_image_and_request_only',producer='local_counterfactual',
                    trace=dict(request_id=None,prompt='Own first image; cyclic fixed16 donor request; no API call.'))
                with Image.open(item['image']) as im:array=np.array(im.convert('RGB'),copy=True)
                image=torch.from_numpy(array).permute(2,0,1)[None].cuda().float()/127.5-1
                evidence,_=encode_input(knowledge,image,donor['caption'],item['duration'],reasoning,
                    sample=item['id'],revision=REVISION)
                value=dict(revision=REVISION,sample=item['id'],own_caption=item['caption'],
                    first_image=str(Path(item['image']).resolve()),duration=item['duration'],
                    donor=donor['id'],donor_caption=donor['caption'],
                    request_words=evidence['request_words'].cpu(),
                    request_word_strings=evidence['request_word_strings'],
                    scope='Only request-memory words swapped; own first image preserved. Native T5, physical C and negative arm remain own at inference.')
                atomic_torch(value,root/(item['case']+'.pt'))
                print(json.dumps(dict(encoded=item['case'],donor=donor['case'])),flush=True)
    atomic_json(dict(complete=True,revision=REVISION,cases=16,api_calls=0,
        normal_cache=str(shared),donors={k:v['id'] for k,v in donors.items()}),root/'ready.json')


if __name__=='__main__':main()
