"""Prepare this checkpoint's positive/negative mean C from cached training inputs."""
from pathlib import Path
import argparse
import json
import os
import random
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
import torch


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--count',type=int,default=64)
    parser.add_argument('--seed',type=int,default=463)
    args=parser.parse_args()
    expected=dict(checkpoint=str(args.checkpoint.resolve()),count=args.count,
        selection_seed=args.seed,split='train',kind='mean_content_condition')
    if args.output.exists():
        previous=torch.load(args.output,map_location='cpu',weights_only=True)
        if any(previous.get(k)!=v for k,v in expected.items()):
            raise ValueError('Existing mean C belongs to a different checkpoint/protocol')
        print(json.dumps(dict(status='reused_completed_mean_condition',**expected)),flush=True)
        return
    from physical_scaling.common import gpu_guard,atomic_torch,atomic_json
    gpu_guard(args.output.parent/'allocations'/os.getenv('DET_EXPERIMENT_ID','local'))
    torch.set_num_threads(4)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    cfg=state['config']
    if cfg['revision']!='physical_content_v7_r1_content_reader_r9':
        raise ValueError('Use the declared R9 checkpoint')
    sys.path.insert(0,cfg['source_snapshot'])
    from physical_content.model import Enhancement
    from physical_content.teacher import Teacher
    from physical_content.evidence import input_inventory
    model=Enhancement(cfg).eval().requires_grad_(False)
    values={name.removeprefix('enhancement.'):tensor
        for name,tensor in {**state['model'],**state['constants']}.items()
        if name.startswith('enhancement.')}
    model.load_state_dict(values,strict=True)
    model.cuda()
    teacher=Teacher(cfg)
    inventory=[row for row in input_inventory() if row['split']=='train'
        and (teacher.cache/'words'/f'{row["name"]}.pt').is_file()]
    selected={};eligible={}
    for arm,suffix in [('positive','_text'),('negative','_null')]:
        eligible[arm]=[row for row in inventory if row['name'].endswith(suffix)]
        selected[arm]=random.Random(args.seed).sample(eligible[arm],args.count)
    sums={};squared_norms={};group_ref=None
    for i in range(args.count):
        for arm,drop in [('positive',False),('negative',True)]:
            evidence=teacher.input(dict(id=selected[arm][i]['sample']),drop=drop)
            with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
                content,groups=model(evidence)
            content=content.float()
            if group_ref is None:group_ref=groups.detach().cpu()
            if not torch.equal(group_ref,groups.cpu()):
                raise ValueError('Training inputs have incompatible relation layouts')
            sums[arm]=sums.get(arm,torch.zeros_like(content))+content
            squared_norms[arm]=squared_norms.get(arm,0.)+float(content.square().mean())
        if (i+1)%8==0:print(json.dumps(dict(completed=i+1,total=args.count)),flush=True)
    means={key:(value/args.count).cpu() for key,value in sums.items()}
    metadata=dict(**expected,checkpoint_step=state['step'],revision=cfg['revision'],
        training_ids={k:[row['sample'] for row in rows] for k,rows in selected.items()},
        training_input_paths={k:[row['path'] for row in rows] for k,rows in selected.items()},
        eligible_cached_records={key:len(rows) for key,rows in eligible.items()},
        selection='Fixed-seed independent training-cache samples, availability only; no benchmark outcomes or future videos.',
        definition='Mean complete C at each slot/channel, including the input anchor and additive learned Q correction.',
        intervention='Fix positive and negative complete C separately; retain native T5, first image and gamma5.',
        rms={key:dict(mean_condition=float(value.square().mean().sqrt()),
            per_sample_rms_root_mean_square=(squared_norms[key]/args.count)**.5)
            for key,value in means.items()})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    atomic_torch(dict(**metadata,**means,groups=group_ref),args.output)
    atomic_json(metadata,args.output.with_suffix('.json'))
    print(json.dumps(metadata,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
