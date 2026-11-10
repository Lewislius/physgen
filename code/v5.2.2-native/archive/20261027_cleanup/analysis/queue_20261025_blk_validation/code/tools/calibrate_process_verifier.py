"""Calibrate a retained learned verifier on within-scene temporal evidence.

Uses only the 16 existing training and 8 development clips from the earlier
study. Late-freeze views are held out from head fitting. Source classification,
process-error regression and graded process preference share the same head.
This evaluates a process-fidelity component, not universal physical correctness.
"""
import argparse
import itertools
import json
from pathlib import Path
import random
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import imageio.v2 as imageio
import numpy as np
import torch
from torch.nn import functional as F
from physgen_native.evidence_verifier import ProcessEvidenceVerifier
from physgen_native.process_encoder import VideoProcessEncoder
from physgen_native.process_distribution import ordered_process_distribution
from physgen_native.runtime import Logger,atomic_save,seed_all

TRAIN_VIEWS=('reference','generation','reference_freeze','generation_freeze',
             'reference_middle_hold','generation_middle_hold')
ALL_VIEWS=TRAIN_VIEWS+('reference_late_freeze','generation_late_freeze')


def read(path):
    reader=imageio.get_reader(path)
    try:return np.stack(list(reader.iter_data())),reader.get_meta_data()['fps']
    finally:reader.close()


def write(frames,path,fps):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.partial.mp4')
    with imageio.get_writer(temp,fps=fps,codec='libx264',quality=None,macro_block_size=None,
        ffmpeg_params=['-crf','18','-profile:v','baseline','-pix_fmt','yuv420p']) as writer:
        for frame in frames:writer.append_data(frame)
    temp.replace(path)


@torch.no_grad()
def collect(args,log):
    paths=sorted((args.source/'full/features').glob('*.pt'))
    if len(paths)!=24:raise ValueError('This controlled protocol expects the original 16 train / 8 development clips')
    encoder=None
    for path in paths:
        target=args.output/'features'/path.name
        if target.exists():
            record=torch.load(target,map_location='cpu',weights_only=True)
            if tuple(record['views'])==ALL_VIEWS:continue
            raise ValueError('Existing calibration views differ from this protocol')
        if encoder is None:
            encoder=VideoProcessEncoder('/home/liuzhirui/model/vjepa2',
                '/home/liuzhirui/model/vjepa2/checkpoint/vjepa2_1_vitG_384.pt',checkpointing=False).eval()
        original=torch.load(path,map_location='cpu',weights_only=True);name=original['id']
        real,fps=read(args.source/'reference_videos'/f'{name}.mp4')
        generated,_=read(args.generated_root/f'{original["split"]}_{name}_seed{original["seed"]}.mp4')
        views={}
        for kind,frames in [('reference',real),('generation',generated)]:
            if len(frames)!=81:raise ValueError('The temporal views require 81 RGB frames')
            middle=frames.copy();middle[25:41]=frames[24]
            late=frames.copy();late[49:]=frames[48]
            views.update({kind:frames,f'{kind}_freeze':np.repeat(frames[:1],81,axis=0),
                          f'{kind}_middle_hold':middle,f'{kind}_late_freeze':late})
        features={}
        for view in ALL_VIEWS:
            file=args.output/'views'/name/f'{view}.mp4';write(views[view],file,fps)
            observed,_=read(file)
            rgb=torch.from_numpy(observed).cuda().permute(0,3,1,2).float()/127.5-1
            with torch.autocast('cuda',dtype=torch.bfloat16):features[view]=encoder(rgb).cpu()
        valid=original['valid'].float().cuda();reference=features['reference'].cuda()
        targets={}
        for view in ALL_VIEWS:
            _,_,pairs=ordered_process_distribution(features[view].cuda(),reference,valid,return_pairs=True)
            targets[view]={h:value.cpu() for h,value in pairs.items()}
        record={k:original[k] for k in ['id','split','caption','condition','times','valid','checkpoint','optimizer_step']}
        record.update(views={k:v.half() for k,v in features.items()},targets=targets,
            training_views=list(TRAIN_VIEWS),evidence_encoder='V-JEPA 2.1 ViT-g',evidence_context='full',
            target_definition='Ordered process discrepancy to the recorded training future; not physical violation labels',
            codec='All views, including unmodified controls, receive the same H264 CRF18 pass')
        atomic_save(record,target);log.write('views_encoded',0,sample=name,split=original['split'])
    del encoder
    torch.cuda.empty_cache()
    return [torch.load(args.output/'features'/p.name,map_location='cpu',weights_only=True) for p in paths]


def inputs(record,view):
    return (record['views'][view].float().cuda(),record['condition'].float().cuda(),
            record['times'].float().cuda(),record['valid'].float().cuda())


@torch.no_grad()
def evaluate(model,records):
    result=[]
    for index,record in enumerate(records):
        scores={}
        for view in ALL_VIEWS:
            energy=model.energy(*inputs(record,view))
            scores[view]=dict(prediction=float(torch.stack([x.mean() for x in energy.values()]).mean()),
                target=float(torch.stack([x.mean() for x in record['targets'][view].values()]).mean()),
                pair_predictions={str(h):x.cpu().tolist() for h,x in energy.items()},
                pair_targets={str(h):x.tolist() for h,x in record['targets'][view].items()})
        other=next(r for r in records[index+1:]+records[:index] if r['split']==record['split'])
        f,c,t,v=inputs(record,'generation')
        swapped=model.energy(f,other['condition'].float().cuda(),t,v)
        result.append(dict(id=record['id'],split=record['split'],views=scores,
            generation_with_other_condition=float(torch.stack([x.mean() for x in swapped.values()]).mean()),
            other_condition_id=other['id']))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--generated-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=512)
    parser.add_argument('--seeds',type=int,nargs='+',default=[42,314159,271828])
    args=parser.parse_args();torch.cuda.set_device(0)
    log=Logger(args.output/'metrics.jsonl')
    records=collect(args,log);training=[r for r in records if r['split']=='train']
    if len(training)!=16:raise ValueError('Never fit on the development cases')
    results=[]
    for seed,method in itertools.product(args.seeds,['source','regression','paired']):
        seed_all(seed);rng=random.Random(seed)
        model=ProcessEvidenceVerifier().cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
        order=[]
        for step in range(args.steps):
            if step%len(training)==0:
                order=list(range(len(training)));rng.shuffle(order)
            record=training[order[step%len(training)]]
            if method=='source':
                # Same architecture and optimizer-step budget as calibrated heads.
                a,c,t,v=inputs(record,'reference');b=record['views']['generation'].float().cuda()
                loss,_=model.calibration_loss(a,b,c,t,v)
            elif method=='regression':
                view=TRAIN_VIEWS[rng.randrange(len(TRAIN_VIEWS))];a,c,t,v=inputs(record,view)
                target={h:x.cuda() for h,x in record['targets'][view].items()}
                loss,_=model.evidence_calibration_loss(a,target,c,t,v)
            else:
                va,vb=rng.sample(TRAIN_VIEWS,2);a,c,t,v=inputs(record,va);b=record['views'][vb].float().cuda()
                ta={h:x.cuda() for h,x in record['targets'][va].items()}
                tb={h:x.cuda() for h,x in record['targets'][vb].items()}
                loss=model.paired_calibration_loss(a,b,ta,tb,c,t,v)
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            if (step+1)%64==0:log.write('fit',step+1,seed=seed,method=method,loss=loss)
        model.eval();values=evaluate(model,records)
        item=dict(seed=seed,method=method,steps=args.steps,cases=values,
            train_ids=[r['id'] for r in training],held_out_edit='late_freeze',
            comparisons_are_development_only=True)
        results.append(item)
        atomic_save(dict(model={k:v.cpu() for k,v in model.state_dict().items()},method=method,
            feature_width=1664,condition_width=4096,width=128,lags=list(model.lags),
            train_ids=item['train_ids'],evidence_encoder='V-JEPA 2.1 ViT-g',evidence_context='full',
            target_definition='Calibrated ordered process evidence, not universal physics correctness'),
            args.output/f'verifier_{method}_seed{seed}.pt')
        (args.output/'results.json').write_text(json.dumps(results,indent=2))
        validation=[r for r in values if r['split']=='validation']
        summary={view:sum(r['views']['reference']['prediction']<r['views'][view]['prediction'] for r in validation)
                 for view in ['reference_freeze','reference_late_freeze','generation']}
        log.write('calibration_evaluation',args.steps,seed=seed,method=method,count=len(validation),
            preferred_reference_counts=summary,scoring_uses_reference_future=False)
    log.write('complete',args.steps);log.close()


if __name__=='__main__':main()
