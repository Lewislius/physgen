"""Average raw Q updates on training inputs, retaining each test input's own anchor."""
from pathlib import Path
import argparse
import json
import os
import random
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
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
    if args.output.exists():
        previous=torch.load(args.output,map_location='cpu',weights_only=True)
        expected=dict(checkpoint=str(args.checkpoint.resolve()),count=args.count,
            selection_seed=args.seed,split='train',kind='mean_raw_query_update')
        if any(previous.get(k)!=v for k,v in expected.items()):
            raise ValueError('Existing mean update belongs to a different protocol')
        print(json.dumps(dict(status='reused_completed_mean_update',**expected)),flush=True)
        return
    from physical_scaling.common import gpu_guard
    gpu_guard(args.output.parent/'allocations'/os.getenv('DET_EXPERIMENT_ID','local'))
    torch.set_num_threads(4)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    cfg=state['config']
    if cfg['variant']!='physical_content' or cfg['model'].get('content_update_mode')!='grounded_feature_modulation':
        raise ValueError('A grounded feature-modulation checkpoint is required')
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
        if len(eligible[arm])<args.count:raise ValueError('Insufficient existing training inputs')
        selected[arm]=random.Random(args.seed).sample(eligible[arm],args.count)
    captured={}
    hook=model.content_write.register_forward_pre_hook(
        lambda module,inputs:captured.update(raw_update=inputs[1].detach().float()))
    sums={};squared_norms={};group_ref=None
    try:
        for i in range(args.count):
            for arm,drop in [('positive',False),('negative',True)]:
                row=selected[arm][i]
                evidence=teacher.input(dict(id=row['sample']),drop=drop)
                with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
                    _,groups=model(evidence)
                update=captured.pop('raw_update')
                if group_ref is None:group_ref=groups.detach().cpu()
                if not torch.equal(group_ref,groups.cpu()):
                    raise ValueError('Training evidence has incompatible relation layouts')
                sums[arm]=sums.get(arm,torch.zeros_like(update))+update
                squared_norms[arm]=squared_norms.get(arm,0.)+float(update.square().mean())
            if (i+1)%8==0:print(json.dumps(dict(completed=i+1,total=args.count)),flush=True)
    finally:
        hook.remove()
    means={key:(value/args.count).cpu() for key,value in sums.items()}
    metadata=dict(checkpoint=str(args.checkpoint.resolve()),checkpoint_step=state['step'],
        revision=cfg['revision'],kind='mean_raw_query_update',split='train',
        count=args.count,selection_seed=args.seed,
        training_ids={k:[row['sample'] for row in rows] for k,rows in selected.items()},
        training_input_paths={k:[row['path'] for row in rows] for k,rows in selected.items()},
        eligible_cached_records={key:len(rows) for key,rows in eligible.items()},
        selection='Fixed-seed independent positive/negative training-cache samples, selected by availability only; no benchmark outcomes or future videos.',
        definition='Mean raw DeltaH at each Q slot/channel, before multiplication by the input anchor and before P0.',
        intervention='Fix positive and negative DeltaH separately across test inputs; retain each input own physical-word anchor, native T5, first image and gamma5.',
        rms={key:dict(mean_update=float(value.square().mean().sqrt()),
            per_sample_rms_root_mean_square=(squared_norms[key]/args.count)**.5)
            for key,value in means.items()})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    from physical_scaling.common import atomic_torch,atomic_json
    atomic_torch(dict(**metadata,**means,groups=group_ref),args.output)
    atomic_json(metadata,args.output.with_suffix('.json'))
    print(json.dumps(metadata,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
