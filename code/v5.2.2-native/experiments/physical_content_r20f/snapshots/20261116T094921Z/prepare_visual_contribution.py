"""Observed-video contribution to a frozen local VLM's physical description.

Whole clips only. No boxes, tracks, chunks, new models or external API.
Candidate supervision is positive first-order evidence for the actual answer,
not an attention weight declared to be physical ground truth. Pilot occlusions
check whether removing the attributed evidence actually lowers answer support.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import sys
import time
import types

import numpy as np
from PIL import Image
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
from physical_content.common import atomic_json, atomic_torch, gpu_guard, load_config
from physical_content.data import TrainingData
from localize_full_video import read_frames, panel

REVISION='FT-existing-Qwen3-whole-video-contribution-v1'
OFFSETS=list(range(0,81,4))
MODEL='/home/liuzhirui/model/Qwen3-VL/Qwen3-VL-8B-Instruct'
QUESTION=('Describe the visible physical process in30–50 words: interactions and material changes, '
    'and how they visibly unfold. Describe only evidence in this video; do not invent motion or causes. '
    'This is a cropped excerpt; its beginning and end need not show a complete process.')


def inputs_for(processor,frames,row):
    messages=[dict(role='system',content='You describe only what the supplied video visibly supports.'),
        dict(role='user',content=[dict(type='video',video='decoded_full_clip'),dict(type='text',text=QUESTION)])]
    text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    video=np.stack([np.asarray(frame.resize((384,224),Image.Resampling.BILINEAR)) for frame in frames])
    return processor(text=[text],videos=[video],return_tensors='pt',do_resize=False,
        do_sample_frames=False,video_metadata=[dict(fps=float(row['source_fps']),
        total_num_frames=81,frames_indices=OFFSETS.copy())]).to('cuda')


def with_answer(prefix,answer_ids):
    value=dict(prefix)
    value['input_ids']=torch.cat((prefix['input_ids'],answer_ids[None]),-1)
    value['attention_mask']=torch.cat((prefix['attention_mask'],
        prefix['attention_mask'].new_ones(1,len(answer_ids))),-1)
    if 'mm_token_type_ids' in value:
        value['mm_token_type_ids']=torch.cat((value['mm_token_type_ids'],
            value['mm_token_type_ids'].new_zeros(1,len(answer_ids))),-1)
    return value


def answer_score(model,inputs,answer_ids):
    output=model(**inputs,use_cache=False,logits_to_keep=len(answer_ids)+1,return_dict=True)
    logits=output.logits[0,:-1].float()
    return logits.log_softmax(-1).gather(-1,answer_ids[:,None]).mean()


def describe_and_attribute(model,processor,prefix,*,occlusion_check):
    original=model.model.get_video_features
    with torch.no_grad():
        cached=original(prefix['pixel_values_videos'],prefix['video_grid_thw'],return_dict=True)
    originals=[cached.pooler_output[0].detach(), *[v.detach() for v in cached.deepstack_features]]
    current=originals
    def features(instance,*args,**kwargs):
        result=copy.copy(cached)
        result.pooler_output=(current[0],)
        result.deepstack_features=current[1:]
        return result
    model.model.get_video_features=types.MethodType(features,model.model)
    try:
        with torch.no_grad():
            generated=model.generate(**prefix,max_new_tokens=112,do_sample=False,use_cache=True)
        answer_ids=generated[0,prefix['input_ids'].shape[1]:]
        # EOS is not part of the physical observation.
        eos=model.generation_config.eos_token_id
        eos={eos} if isinstance(eos,int) else set(eos)
        if len(answer_ids) and int(answer_ids[-1]) in eos: answer_ids=answer_ids[:-1]
        answer=processor.tokenizer.decode(answer_ids,skip_special_tokens=True).strip()
        if len(answer_ids)<3 or not answer: raise ValueError('No usable observed description')
        fixed=with_answer(prefix,answer_ids)
        current=[v.clone().requires_grad_(True) for v in originals]
        score=answer_score(model,fixed,answer_ids)
        gradients=torch.autograd.grad(score,current,allow_unused=False)
        # First-order score decrease when a token is replaced by the full-clip
        # mean visual feature. Include all actual deepstack visual inputs.
        contribution=torch.stack([(g.float()*(v.float()-v.float().mean(0,keepdim=True))).sum(-1)
                                  for g,v in zip(gradients,current)]).sum(0).detach()
        baseline=float(score.detach());del score,gradients
        current=originals
        checks={}
        if occlusion_check:
            count=max(1,math.ceil(len(contribution)*.1))
            selections=dict(top=contribution.topk(count).indices,
                random=torch.randperm(len(contribution),generator=torch.Generator().manual_seed(42))[:count].to(contribution.device))
            with torch.no_grad():
                for name,indices in selections.items():
                    current=[v.clone() for v in originals]
                    for value,reference in zip(current,originals):
                        value[indices]=reference.mean(0,keepdim=True)
                    changed=float(answer_score(model,fixed,answer_ids))
                    checks[name]=dict(tokens=count,answer_log_probability=changed,
                                      support_drop=baseline-changed)
            current=originals
        grid=prefix['video_grid_thw'][0].tolist();merge=processor.video_processor.merge_size
        geometry=(grid[0],grid[1]//merge,grid[2]//merge)
        if math.prod(geometry)!=len(contribution):raise ValueError('Visual contribution geometry differs')
        return contribution.reshape(geometry).cpu(),dict(answer=answer,answer_tokens=len(answer_ids),
            answer_log_probability=baseline,occlusion_checks=checks,
            positive_fraction=float((contribution>0).float().mean()),
            signed_range=[float(contribution.min()),float(contribution.max())])
    finally:
        model.model.get_video_features=original


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--occlusion-check',action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    cfg=load_config(args.config);gpu_guard(args.output)
    selection=json.loads(args.selection.read_text())
    if selection['scope']!='training_only_actual_full_clip' or selection.get('chunk_conditioning',False):
        raise ValueError('Actual whole training clips required')
    data=TrainingData(cfg);by_id={r['id']:i for i,r in enumerate(data.rows)}
    ids=[v['id'] for v in selection['cases']]
    if not set(ids)<=set(by_id):raise ValueError('Non-training inputs cannot enter this preparation')
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    from transformers.utils import logging
    logging.disable_progress_bar();torch.set_num_threads(4)
    model=Qwen3VLForConditionalGeneration.from_pretrained(MODEL,dtype=torch.bfloat16,
        device_map={'':'cuda'},attn_implementation='sdpa',local_files_only=True).eval().requires_grad_(False)
    processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True)
    start=time.monotonic();records=[]
    for sid in ids:
        dest=args.output/sid;dest.mkdir(exist_ok=True)
        sample=data.get(by_id[sid],device='cpu',targets=False);row=sample['row'];frames=read_frames(row)
        prefix=inputs_for(processor,frames,row)
        torch.cuda.reset_peak_memory_stats()
        field,record=describe_and_attribute(model,processor,prefix,occlusion_check=args.occlusion_check)
        times=[float(row['pts'][i]-row['pts'][0]) for i in OFFSETS]
        padded=times+([times[-1]] if len(times)%2 else [])
        centers=torch.tensor([(padded[i]+padded[i+1])/2 for i in range(0,len(padded),2)])
        if len(centers)!=len(field):raise ValueError('Observed temporal geometry differs')
        record.update(id=sid,revision=REVISION,model=MODEL,training_only=True,
            chunk_conditioning=False,bounding_boxes=False,approved_for_training=False,
            source_frame_indices=[row['indices'][i] for i in OFFSETS],
            peak_gib=torch.cuda.max_memory_allocated()/2**30)
        atomic_torch(dict(contribution=field,video_times=centers,revision=REVISION),dest/'target.pt')
        panel(frames,field,dest/'panel.jpg',sid,legend='positive physical-answer visual contribution')
        atomic_json(record,dest/'record.json');records.append(record)
        progress=dict(completed=len(records),total=len(ids),seconds=time.monotonic()-start,last=sid)
        atomic_json(progress,args.output/'progress.json');print(json.dumps(dict(**progress,**record)),flush=True)
        del prefix,field;torch.cuda.empty_cache()
    atomic_json(dict(complete=True,count=len(records),revision=REVISION,records=records,
        seconds=time.monotonic()-start,approved_for_training=False),args.output/'complete.json')


if __name__=='__main__':main()
