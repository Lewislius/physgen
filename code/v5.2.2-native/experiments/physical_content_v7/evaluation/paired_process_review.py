"""Supplementary blinded process observations of saved videos, using local Qwen.

Separate from generation and training. This is a diagnostic model review,
not an independently validated physical metric or a replacement for VideoPhy2.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import copy
import json
import random
import sys
import time
import torch

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT.parents[1]
sys.path[:0]=[str(ROOT/'snapshots/20261030T141432Z'),
    str(BASE/'experiments/physical_scaling_q256/snapshots/20261029T140406Z'),
    str(BASE/'experiments/physical_variants/snapshots/20261029T064043Z')]
import bootstrap

PROMPT='''Two generated videos are shown in chronological order internally. Video1 is A; Video2 is B. Their model identities are hidden.
Requested video: {caption}
Observable process requirements: {criteria}

Evaluate only what is visibly supported. The request is not evidence that an event happened. Do not invent hidden forces, events between observed frames, or audio. If a required interaction or response is absent, a plausible stable image does not count as completion. Intended bending, fluid deformation, fracture and disappearance by absorption can be valid; distinguish these from unsupported morphing or disappearance. Describe visible evidence with approximate times. If the sampled frames are insufficient, say unclear. An initial-scene problem can be shared by both videos; do not claim it was introduced by one model.
Return JSON only with this schema:
{{"A":{{"completion":"complete|partial|absent|unclear","causal_order":"consistent|violated|not_observable|unclear","result_persistence":"consistent|violated|not_applicable|unclear","visual_defect":"none|minor|major|unclear","evidence":"at most45 words"}},"B":{{"completion":"complete|partial|absent|unclear","causal_order":"consistent|violated|not_observable|unclear","result_persistence":"consistent|violated|not_applicable|unclear","visual_defect":"none|minor|major|unclear","evidence":"at most45 words"}},"process_preference":"A|B|tie|unclear","visual_preference":"A|B|tie|unclear"}}
Use one permitted value in each categorical field, not the alternatives joined by bars.'''


def parse_result(text):
    try:
        value=json.loads(text[text.index('{'):text.rindex('}')+1])
        allowed=dict(completion={'complete','partial','absent','unclear'},
            causal_order={'consistent','violated','not_observable','unclear'},
            result_persistence={'consistent','violated','not_applicable','unclear'},
            visual_defect={'none','minor','major','unclear'})
        for label in ['A','B']:
            for field,choices in allowed.items():
                if value[label][field] not in choices:return None
        if any(value[k] not in {'A','B','tie','unclear'} for k in ['process_preference','visual_preference']):return None
        return value
    except (ValueError,KeyError,TypeError):return None


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('first',type=Path);p.add_argument('second',type=Path)
    p.add_argument('--cases',nargs='+',default=['VP004','VP010','VP015','VP034'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--max-new-tokens',type=int,default=400)
    args=p.parse_args()
    torch.set_num_threads(16)
    from decord import VideoReader,cpu
    from physgen_native.knowledge import QwenKnowledge
    from transformers.utils import logging as transformer_logging
    transformer_logging.disable_progress_bar()
    roots=[args.first.resolve(),args.second.resolve()]
    plans=[json.loads((root/'manifest.json').read_text()) for root in roots]
    indexed=[{row['id']:row for row in plan['cases']} for plan in plans]
    criteria=json.loads((Path(__file__).with_name('process_criteria16.json')).read_text())['cases']
    for case in args.cases:
        first,second=(table[case] for table in indexed)
        for field in ['caption','prepared_inputs','initial_noise','generation_seed']:
            if first.get(field)!=second.get(field):raise ValueError(f'{case}: unmatched {field}')
        if case not in criteria:raise ValueError(f'{case}: no preregistered process criteria')
    args.output.mkdir(parents=True,exist_ok=True)
    if args.device=='cuda':
        from physical_scaling.common import gpu_guard
        gpu_guard(args.output/'allocation')
    cfg=json.loads((ROOT/'runs/physical_content_66778/config.json').read_text())
    cfg=copy.deepcopy(cfg);cfg['knowledge']['freeze_backbone']=True;cfg['verifier']={'kind':'none'}
    knowledge=QwenKnowledge(cfg,device=args.device).eval()
    knowledge.requires_grad_(False)
    records=[]
    for case in args.cases:
        destination=args.output/f'{case}.json'
        if destination.exists():
            existing=json.loads(destination.read_text())
            if existing['sources']!=[str(x) for x in roots] or existing['prompt_template']!=PROMPT:
                raise ValueError('Use a new output directory for a changed review')
            records.append(existing);continue
        order=[0,1];random.Random(51603+int(case.removeprefix('VP'))).shuffle(order)
        clips=[];frame_metadata=[]
        for index in order:
            video=roots[index]/'cases'/case/'planned.mp4'
            reader=VideoReader(str(video),ctx=cpu(0),num_threads=1)
            frame_ids=torch.linspace(0,len(reader)-1,16).round().long()
            rgb=torch.from_numpy(reader.get_batch(frame_ids.tolist()).asnumpy()).permute(0,3,1,2).to(args.device).float()/127.5-1
            times=frame_ids.to(args.device).float()/reader.get_avg_fps()
            clips.append((rgb,times))
            frame_metadata.append(dict(video=str(video),indices=frame_ids.tolist(),times=times.cpu().tolist()))
        body=PROMPT.format(caption=indexed[0][case]['caption'],criteria=criteria[case])
        inputs,_=knowledge._inputs(clips,body,video=True)
        start=time.monotonic()
        with torch.autocast(args.device,dtype=torch.bfloat16):
            raw,complete=knowledge._reason(inputs,args.max_new_tokens,return_complete=True)
        result=parse_result(raw)
        record=dict(case=case,caption=indexed[0][case]['caption'],sources=[str(x) for x in roots],
            A_source=order[0],B_source=order[1],sampled_frames=frame_metadata,prompt_template=PROMPT,
            raw_response=raw,parsed=result,generation_complete=complete,seconds=time.monotonic()-start,
            observer='Existing frozen local Qwen3-VL-8B-Instruct; model identities and VideoPhy2 scores withheld',
            scope='Supplementary model observations of16 sampled frames per clip, subject to visual review; no training or generation feedback.')
        destination.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
        records.append(record)
        print(json.dumps(dict(case=case,complete=complete,parsed=result,seconds=record['seconds']),ensure_ascii=False),flush=True)
    summary=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),sources=[str(x) for x in roots],
        count=len(records),valid_responses=sum(row['parsed'] is not None and row['generation_complete'] for row in records),
        reviewed_cases=[row['case'] for row in records],human_or_assistant_visual_crosscheck='pending',
        independent_metric_validation=False,
        note='Qwen is also the conditioning/feature foundation, so this is a supplementary diagnostic observer, not independent proof. Do not optimize solely to its preferences.')
    (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
