"""Evaluate observed-state feature distances and temporal sensitivity on real held-out clips.

This is a descriptive diagnostic, not a physical-correctness judge or training input.
Only the already installed Qwen vision weights are read, on CPU.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import importlib.util
import json
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F
import yaml

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'grounded_read_r4/snapshots/20261031T064033Z'
sys.path.insert(0,str(SOURCE))
import bootstrap
from physical_content.process_features import FrozenProcessFeatures,feature_distance

INDICES=[0,3,39,42,78,80]


def tensor(images):
    return torch.from_numpy(np.stack([np.asarray(x) for x in images])).permute(0,3,1,2).float()/127.5-1


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',nargs='+',default=['source0000278','wisa0028628'])
    parser.add_argument('--output',type=Path,default=ROOT/'reports/heldout_state_features/pilot.json')
    parser.add_argument('--reuse',type=Path,help='Reuse completed cases from this exact diagnostic protocol')
    parser.add_argument('--transition-diagnostic',action='store_true',help='Also inspect signed semantic state transitions; missing measurements require fresh encoded features')
    args=parser.parse_args()
    if args.transition_diagnostic:
        sys.path.insert(0,str(Path(__file__).resolve().parent))
        from semantic_transitions import transition_diagnostics
    torch.set_num_threads(8)
    cfg=yaml.safe_load((SOURCE/'configs/physical_content.yaml').read_text())
    destination=args.output;destination.parent.mkdir(parents=True,exist_ok=True)
    examples={x['id']:x for x in json.loads((ROOT/'heldout8/manifest.json').read_text())['cases']}
    rows={x['id']:x for x in json.loads((Path(cfg['paths']['cache'])/'base/manifest.json').read_text())['records']}
    if set(args.cases)-set(examples):raise ValueError('Use only the preselected real validation inputs')
    spec=importlib.util.spec_from_file_location('heldout_frames',bootstrap.BASE.parent/'v5.2/physgen_v52/preparation.py')
    # BASE is code/v5.2.2-native, so v5.2 is its sibling.
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    features=FrozenProcessFeatures(cfg,device='cpu').eval()
    kernel=features.encoder.patch_embed.proj.weight.detach().float()
    a,b=kernel[:,:,0],kernel[:,:,1]
    kernel_summary=dict(shape=list(kernel.shape),
        temporal_slice_cosine=float(F.cosine_similarity(a.flatten(),b.flatten(),dim=0)),
        temporal_difference_relative_rms=float((a-b).square().mean().sqrt()/kernel.square().mean().sqrt()))
    versions={
        'native400':ROOT/'inference_outputs/native_lora_66781/step0400_real_heldout8_seed42/heldout8',
        'r1_400':ROOT/'inference_outputs/physical_content_66778/step0400_real_heldout8_seed42/heldout8',
        'r1_1000':ROOT/'inference_outputs/physical_content_66856/step1000_real_heldout8_seed42/heldout8'}
    from decord import VideoReader,cpu
    previous=json.loads(args.reuse.read_text()) if args.reuse else None
    if previous and (previous['frame_indices']!=INDICES or previous['temporal_patch_kernel']!=kernel_summary):
        raise ValueError('Prior features do not use this frame/encoder protocol')
    if previous and bool(previous.get('transition_diagnostic',False))!=args.transition_diagnostic:
        raise ValueError('Reuse requires the same semantic-transition diagnostic, not scalar-only prior distances')
    report=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),status='running',cases={},
        requested_cases=args.cases,feature_source=str(SOURCE),model=cfg['paths']['qwen_checkpoint'],
        frame_indices=INDICES,temporal_patch_kernel=kernel_summary,
        transition_diagnostic=args.transition_diagnostic,
        scope='Two-case diagnostic by default, not a population rate. Same frozen feature family as the R3/R4 training auxiliary, hence not an independent physics benchmark. Ground-truth appearance is not the only valid generated outcome. No metric adoption or training feedback is automatic.')
    def publish():
        temp=destination.with_suffix('.tmp');temp.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');temp.replace(destination)
    print(json.dumps(dict(kernel=kernel_summary)),flush=True)
    for case_id in args.cases:
        row=rows[case_id];case=examples[case_id]
        if row['split']!='validation':raise ValueError('Held-out diagnostic cannot include training examples')
        earlier=previous['cases'].get(case_id) if previous else None
        if earlier and earlier['caption']==case['caption'] and set(earlier['variants'])=={
            'repeat_first','swap_within_pairs','swap_middle_final_pairs',*versions}:
            report['cases'][case_id]=dict(earlier,reused_from=str(args.reuse.resolve()))
            publish();print(json.dumps(dict(case=case_id,status='reused_completed_same_protocol')),flush=True)
            continue
        real=tensor(module.read_frames(row,INDICES))
        valid=torch.load(Path(case['prepared_inputs'])/'conditions.pt',map_location='cpu',weights_only=True)['valid']
        with torch.autocast('cpu',dtype=torch.bfloat16):target=features(real)
        value,parts=feature_distance(target,target,valid)
        result=dict(caption=case['caption'],expected_process=case['expected_process'],
            real_self=dict(distance=float(value),layers=parts),variants={})
        if earlier and earlier['caption']==case['caption']:
            result['variants']=dict(earlier['variants'])
        if args.transition_diagnostic:
            result['real_self']['semantic_transitions']=transition_diagnostics(target,target,valid)
        sources=dict(repeat_first=real[:1].expand_as(real),
            swap_within_pairs=real[[1,0,3,2,5,4]],
            swap_middle_final_pairs=real[[0,1,4,5,2,3]])
        for name,folder in versions.items():
            plan=json.loads((folder/'manifest.json').read_text())
            planned={x['id']:x for x in plan['cases']}[case_id]
            if planned['caption']!=case['caption'] or planned['prepared_inputs']!=case['prepared_inputs']:
                raise ValueError('Generation must use this same real held-out input')
            video=folder/'cases'/case_id/'planned.mp4'
            reader=VideoReader(str(video),ctx=cpu(0),num_threads=1)
            if len(reader)!=81:raise ValueError('Expected complete81-frame generated clip')
            sources[name]=tensor(reader.get_batch(INDICES).asnumpy())
        for name,frames in sources.items():
            if name in result['variants']:
                print(json.dumps(dict(case=case_id,mode=name,status='reused_completed_same_protocol')),flush=True)
                continue
            started=time.monotonic()
            with torch.autocast('cpu',dtype=torch.bfloat16):encoded=features(frames)
            distance,layers=feature_distance(encoded,target,valid)
            result['variants'][name]=dict(distance=float(distance),layers=layers,seconds=time.monotonic()-started)
            if args.transition_diagnostic:
                result['variants'][name]['semantic_transitions']=transition_diagnostics(encoded,target,valid)
            report['cases'][case_id]=result;publish()
            print(json.dumps(dict(case=case_id,mode=name,distance=float(distance),seconds=time.monotonic()-started)),flush=True)
        del sources,target,real
    report['status']='diagnostic_completed_requires_review'
    names=list(next(iter(report['cases'].values()))['variants'])
    report['mean_distances']={name:sum(c['variants'][name]['distance'] for c in report['cases'].values())/len(report['cases']) for name in names}
    report['paired_distances']={}
    for baseline,candidate in [('native400','r1_400'),('r1_400','r1_1000')]:
        differences=[c['variants'][candidate]['distance']-c['variants'][baseline]['distance'] for c in report['cases'].values()]
        report['paired_distances'][candidate+'_vs_'+baseline]=dict(mean_difference=sum(differences)/len(differences),
            lower_distance=sum(x<0 for x in differences),higher_distance=sum(x>0 for x in differences))
    publish()
    print(json.dumps(dict(output=str(destination),status=report['status'],means=report['mean_distances'])),flush=True)


if __name__=='__main__':main()
