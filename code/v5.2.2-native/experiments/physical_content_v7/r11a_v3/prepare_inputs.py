"""Prepare shared API reasoning and local Qwen features once for all60 cases.

Only the fixed first image and original benchmark request are visible to the API.
Native T5 requests, image conditions, saved noise, scores and future videos are
never rewritten or sent as input. Loading local Qwen occurs once per encode run.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import argparse,json,os,sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
from benchmark_inputs import reference,case_input
from annotate import Client,input_reasoning,save


def reason_one(item,root):
    from PIL import Image
    folder=root/item['id']
    with Image.open(item['image']) as value:image=value.convert('RGB')
    client=Client()
    for arm,caption in [('positive',item['caption']),('negative','')]:
        path=folder/f'{arm}.json'
        expected=dict(sample=item['id'],caption=caption,first_image=item['image'],duration=item['duration'])
        if path.exists():
            saved=json.loads(path.read_text())
            if any(saved.get(k)!=v for k,v in expected.items()):raise ValueError('Existing API input differs')
            continue
        value=input_reasoning(image,caption,item['duration'],client)
        value.update(expected);save(path,value)


def donor_map(items,root,controls,training_reasoning=None):
    by_id={v['case']:v for v in items}
    positive={v['case']:json.loads((root/v['id']/'positive.json').read_text()) for v in items}
    donors={}
    for name in controls:
        own=positive[name]
        eligible=sorted(k for k,v in positive.items() if k!=name and len(v['relations'])==len(own['relations']))
        if not eligible and training_reasoning is not None:
            # A rare relation count may occur only once in60. Reuse a fixed
            # training INPUT-side reasoning text, not an evaluation outcome.
            for path in sorted(Path(training_reasoning).glob('*_text.json')):
                value=json.loads(path.read_text())
                if value['scope']=='first_image_and_request_only' and len(value['relations'])==len(own['relations']):
                    key='train:'+path.name.removesuffix('_text.json')
                    positive[key]=value;eligible=[key];break
        if not eligible:raise ValueError(f'No same-count donor for {name}')
        # Fixed ID-based cyclic choice, without scores or generation outcomes.
        donors[name]=next((k for k in eligible if k>name),eligible[0])
    return donors,positive,by_id


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--stage',choices=['reasoning','encode'],required=True)
    p.add_argument('--workers',type=int,default=8)
    args=p.parse_args()
    from physical_content.common import load_config
    cfg=load_config(args.config);root=Path(cfg['paths']['inference_evidence'])
    training_reasoning=Path(cfg['paths']['cache']).parent/'reasoning'
    controls=[v['id'] for v in json.loads((Path(__file__).parent/'subset16.json').read_text())['cases']]
    items=[case_input(case)[1] for case in reference()['cases']]
    if len(items)!=60 or len(controls)!=16:raise ValueError('Use the existing60 and fixed16')
    save(root/'inputs.json',dict(cases=items,controls=controls,scope='first_image_and_request_only',native_prompts_unchanged=True))
    if args.stage=='reasoning':
        failures={}
        for attempt in range(3):
            pending=[v for v in items if not all((root/v['id']/f'{arm}.json').exists() for arm in ('positive','negative'))]
            if not pending:break
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures={pool.submit(reason_one,v,root):v['id'] for v in pending}
                for future in as_completed(futures):
                    name=futures[future]
                    try:future.result();failures.pop(name,None)
                    except Exception as error:failures[name]=dict(error=str(error),detail=getattr(error,'detail',None))
            save(root/'reasoning_status.json',dict(attempt=attempt+1,failed=failures,total=60))
        missing=[v['id'] for v in items if not all((root/v['id']/f'{arm}.json').exists() for arm in ('positive','negative'))]
        if missing:raise RuntimeError(f'Incomplete API reasoning after three resumable passes: {missing}')
        for item in items:reason_one(item,root)  # Validate completed provenance, no repeated successful calls.
        donors,_,_=donor_map(items,root,controls,training_reasoning)
        save(root/'donors.json',donors)
        save(root/'reasoning_status.json',dict(complete=True,cases=60,api_calls_expected=120,extra_control_calls=0))
        return
    import torch,numpy as np
    from PIL import Image
    from encode import encode_input
    from condition_controls import encode_control,InferenceCache
    from physical_content.common import gpu_guard,atomic_torch
    from physgen_native.knowledge import QwenKnowledge
    gpu_guard(root/'allocations'/os.getenv('DET_EXPERIMENT_ID','local'))
    donors,positive,by_id=donor_map(items,root,controls,training_reasoning)
    if json.loads((root/'donors.json').read_text())!=donors:raise ValueError('Reasoning donor map changed')
    pending=[]
    reader=InferenceCache(root,device='cpu')
    for item in items:
        folder=root/item['id'];modes=['real','empty_reasoning','reasoning_swapped'] if item['case'] in controls else ['real']
        if (folder/'ready.json').exists():
            for mode in modes:reader.input(item,image=item['image'],duration=item['duration'],mode=mode)
            reader.input(item,image=item['image'],duration=item['duration'],drop=True)
        else:pending.append(item)
    if not pending:return
    torch.set_num_threads(4)
    knowledge=QwenKnowledge(cfg).eval().requires_grad_(False)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for item in pending:
            folder=root/item['id']
            with Image.open(item['image']) as image:array=np.array(image.convert('RGB'),copy=True)
            image=torch.from_numpy(array).permute(2,0,1)[None].cuda().float()/127.5-1
            for arm,caption in [('positive',item['caption']),('negative','')]:
                reasoning=json.loads((folder/f'{arm}.json').read_text())
                if (reasoning['sample']!=item['id'] or reasoning['first_image']!=item['image']
                    or reasoning['duration']!=item['duration']):raise ValueError('Reasoning provenance differs')
                evidence,words=encode_input(knowledge,image,caption,item['duration'],reasoning,
                    sample=item['id'],revision='qwen38_first_image_request_v1')
                evidence.update(words);evidence.update(first_image=item['image'],duration=item['duration'])
                atomic_torch(evidence,folder/f'{arm}.pt')
            modes=[]
            if item['case'] in controls:
                modes=['reasoning_swapped','empty_reasoning']
                for mode in modes:
                    evidence=encode_control(knowledge,image,item['caption'],item['duration'],positive[item['case']],
                        sample=item['id'],mode=mode,donor=positive[donors[item['case']]])
                    evidence.update(first_image=item['image'],duration=item['duration'])
                    atomic_torch(evidence,folder/'controls'/f'{mode}.pt')
            save(folder/'ready.json',dict(id=item['id'],caption=item['caption'],image=item['image'],duration=item['duration'],
                scope='first_image_and_request_only',width=4096,native_wan_prompt_unchanged=True,
                feature_encoder=cfg['paths']['qwen_checkpoint'],text_producer='qwen3.8-flash',controls=modes))
            print(json.dumps(dict(encoded=item['id'],controls=modes)),flush=True)
    save(root/'encoding_status.json',dict(complete=True,cases=60,control_cases=16))


if __name__=='__main__':main()
