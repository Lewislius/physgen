"""Keep all original reasoning, and explicitly ground one C group in the request.

No new reasoning is generated. The existing frozen Qwen rereads the same first
image/request and the four original sentences, followed by the verbatim request.
All five spans remain in memory; C's four lexical anchor groups use spans 1,2,3,5.
The fourth original sentence remains available to full-context attention.
"""
from pathlib import Path
import argparse
import json
import os
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
ROOT=bootstrap.BASE/'experiments/physical_content_v7'
WORK=ROOT/'request_anchor_r1'
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import yaml
from PIL import Image
from physgen_native.knowledge import QwenKnowledge
from physical_content.evidence import find_input,recover_words
from physical_scaling.common import atomic_json,atomic_torch,gpu_guard


def select_anchors(evidence):
    """Retain four groups and the existing Q budget; keep full memory intact."""
    original_groups=evidence['word_groups']
    keep=original_groups!=3
    for name in ['words','word_groups','word_positions']:
        evidence[name]=evidence[name][keep]
    evidence['word_strings']=[w for w,selected in zip(evidence['word_strings'],keep.tolist()) if selected]
    evidence['word_groups']=torch.where(evidence['word_groups']==4,3,evidence['word_groups'])
    evidence['anchor_memory_relation_indices']=[0,1,2,4]
    if sorted(evidence['word_groups'].unique().tolist())!=[0,1,2,3]:
        raise ValueError('Expected four populated anchor groups')
    return evidence


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',nargs='+')
    args=parser.parse_args()
    job=WORK/'reports/jobs'/os.getenv('DET_EXPERIMENT_ID','local')
    gpu_guard(job)
    torch.set_num_threads(4)
    cfg=yaml.safe_load((ROOT/'configs/physical_content.yaml').read_text())
    sys.path.insert(0,str(bootstrap.BASE/'experiments/physical_scaling_q256/inference_snapshots/20261030T034046Z'))
    from protocol import manifest_for,load_inputs
    plan=manifest_for('v6_assoc','videophy40')
    selected=[c for c in plan['cases'] if not args.cases or c['id'] in args.cases]
    if args.cases and len(selected)!=len(args.cases):raise ValueError('Unknown or duplicated selected case')
    knowledge=QwenKnowledge(cfg,device='cuda').eval()
    records=[]
    for case in selected:
        case_id=case['id'];target=WORK/'cache'/f'{case_id}.pt'
        if target.exists():
            cached=torch.load(target,map_location='cpu',weights_only=True)
            if cached['caption']!=case['caption'] or cached['intervention']!='request_anchored':
                raise ValueError('Existing condition belongs to another input revision')
            records.append(dict(case=case_id,status='reused'));continue
        start=time.monotonic();inputs,canvas=load_inputs(case)
        image=np.array(Image.open(canvas).convert('RGB'),copy=True)
        first=torch.from_numpy(image).permute(2,0,1)[None].cuda().float()/127.5-1
        duration=float(inputs['times'][-1]);caption=case['caption']
        original=torch.load(find_input(f'videophy40_{case_id}_text'),map_location='cpu',weights_only=True)
        if len(original['relations'])!=4:raise ValueError('This revision expects the existing four-sentence input')
        relations=[*original['relations'],f'Requested future process to generate: {caption}']
        spans=[f'[RELATION {i+1}] {s}' for i,s in enumerate(relations)]
        body=(f'Request: {caption}\nPresentation interval: {duration:.3f} seconds.\n'
              'Conditional physical analysis:\n'+'\n'.join(spans))
        model_inputs,positions=knowledge._inputs([(first,first.new_zeros(1))],body,extract_spans=spans)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            hidden=knowledge.backbone.model(**model_inputs,use_cache=False,return_dict=True).last_hidden_state
        vectors=torch.stack([hidden[:,p].mean(1) for p in positions],1)
        evidence=dict(memory=hidden.cpu().to(torch.bfloat16),memory_mask=model_inputs['attention_mask'].bool().cpu(),
            relation_vectors=vectors.cpu().to(torch.bfloat16),relations=relations,caption=caption,
            original_relations=original['relations'],analysis='\n'.join(f'{i+1}. {s}' for i,s in enumerate(relations)),
            analysis_complete=True,scope='first_image_and_request_only',sample=f'videophy40_{case_id}',
            model=cfg['paths']['qwen_checkpoint'],intervention='request_anchored',canvas=str(canvas),
            intervention_scope='Same first image/request. Original four reasoning sentences retained in memory; fourth C anchor is the verbatim request. Negative condition unchanged.')
        evidence.update(recover_words(evidence,duration,knowledge.tokenizer))
        evidence=select_anchors(evidence)
        atomic_torch(evidence,target)
        row=dict(case=case_id,seconds=time.monotonic()-start,memory_relations=len(relations),
            c_groups=4,c_anchor_memory_relations=evidence['anchor_memory_relation_indices'],
            requested_process=relations[-1],retained_original_relations=original['relations'])
        records.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
        atomic_json(dict(status='running',cases=[c['id'] for c in selected],records=records),WORK/'reports/input_preparation.json')
        del evidence,hidden,first,original,model_inputs,vectors
    atomic_json(dict(status='completed',cases=[c['id'] for c in selected],records=records,
        new_generated_reasoning=False,new_models=False,negative_condition='original unchanged'),WORK/'reports/input_preparation.json')


if __name__=='__main__':main()
