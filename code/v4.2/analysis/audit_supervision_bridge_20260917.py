"""Read-only probes: feature energy and teacher-state intervention on the old EMA."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from physgen_v42.geometry import Geometry
from physgen_v42.losses import fm_loss, struct_loss
from physgen_v42.runtime import autocast

CHECKPOINT = ROOT/'checkpoints/stability_20260917T113347Z_5d0494/step0100'
OUT = ROOT/'analysis/generation_failure_20260917'
cfg = json.loads((CHECKPOINT/'config.json').read_text())
sys.path.insert(0, cfg['paths']['wan_code'])
cache = Path(cfg['paths']['cache'])
manifest = json.loads((cache/'manifest.json').read_text())
torch.set_num_threads(4)


def energy():
    stats = torch.load(cache/'stats.pt', map_location='cpu', weights_only=True)
    rows = []
    for split, count in [('train',24),('validation',8)]:
        for record in [r for r in manifest['records'] if r['split']==split][:count]:
            z = torch.load(cache/'teacher'/(record['id']+'.pt'), map_location='cpu', weights_only=True).float()
            geo = Geometry.build(record['geometry'], torch.device('cpu'))
            weight = geo.p_weight[None,None,:,None]
            def power(value):
                return float((value.square()*weight).sum()/(weight.sum()*value.shape[1]*value.shape[-1]))
            mean = z.mean(1, keepdim=True)
            raw, static, dynamic = power(z), power(mean), power(z-mean)
            rows.append(dict(id=record['id'], split=split, raw_power=raw, static_power=static,
                             dynamic_power=dynamic, dynamic_fraction=dynamic/raw,
                             constant_video_struct=dynamic/float(stats['s_z'])**2,
                             objects_reliable=record.get('objects_reliable',False)))
    report = dict(scope='first 24 train + first 8 validation; not a random population estimate', rows=rows,
                  mean_dynamic_fraction=sum(r['dynamic_fraction'] for r in rows)/len(rows),
                  min_dynamic_fraction=min(r['dynamic_fraction'] for r in rows),
                  max_dynamic_fraction=max(r['dynamic_fraction'] for r in rows))
    (OUT/'teacher_energy.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)


@torch.no_grad()
def bridge():
    from inference.infer import load_adapter
    from physgen_v42.backbone import ProcessWan, fix_first, load_wan
    from physgen_v42.data import Dataset
    device = torch.device('cuda:0')
    adapter = load_adapter(dict(config=cfg, checkpoint=str(CHECKPOINT)),device)
    model = ProcessWan(load_wan(cfg['paths']['wan_checkpoint'],device),adapter,recompute=False)
    dataset = Dataset(cfg,'train')
    index = next(i for i,r in enumerate(dataset.records) if r['id']=='source0000949')
    rows=[]
    for mode in ('i2v','t2v'):
        sample=dataset.load(index,mode,device)
        geo=Geometry.build(sample['record']['geometry'],device)
        text,first,gt=sample['text'],sample['first'],sample['latent']
        generator=torch.Generator(device=device).manual_seed(93001)
        noise=torch.randn(gt.shape,device=device,generator=generator)
        target=noise-gt
        with autocast(device): p0=adapter.initialize(text,geo,mode,sample['anchor'])
        for s in (.85,.35):
            sigma=torch.tensor(s,device=device)
            x=fix_first((1-sigma)*gt+sigma*noise,first)
            for variant in ('learned','teacher_oracle','teacher_time_mean'):
                handles=[]
                if variant!='learned':
                    state=sample['target'] if variant=='teacher_oracle' else sample['target'].mean(1,keepdim=True).expand_as(sample['target'])
                    def hook(module,args,out):
                        return (state, out[1]) if isinstance(out,tuple) else state
                    handles=[c.core.register_forward_hook(hook) for c in adapter.correctors.values()]
                try:
                    with autocast(device): result=model(x,first,text,sigma,sample['record']['duration'],geo,p0)
                finally:
                    for handle in handles: handle.remove()
                corrected=fm_loss(result['cond'],target,first is not None)
                baseline=fm_loss(result['base'],target,first is not None)
                delta=result['cond']-result['base']; residual=target-result['base']
                if first is not None: delta,residual=delta[:,:,1:],residual[:,:,1:]
                dot=(delta*residual).mean(); cost=delta.square().mean()
                norm=delta.square().mean().sqrt()*residual.square().mean().sqrt()
                row=dict(mode=mode,sigma=s,variant=variant,fm_base=float(baseline),fm=float(corrected),
                         fm_delta=float(corrected-baseline),delta_energy=float(cost),
                         twice_residual_dot_delta=float(2*dot),
                         cosine=float(dot/norm) if float(norm)>0 else None,
                         state_struct=float(struct_loss(result['p15'],sample['target'],adapter.s_z,geo.p_weight)))
                assert abs(row['fm_delta']-(row['delta_energy']-row['twice_residual_dot_delta']))<1e-5
                rows.append(row);print(json.dumps(row),flush=True)
    (OUT/'supervision_bridge.json').write_text(json.dumps(dict(
        scope='one actual training clip, full cached geometry, two modes/two sigmas; teacher intervention is diagnostic only',
        checkpoint=str(CHECKPOINT),rows=rows),indent=2)+'\n')


@torch.no_grad()
def fixed_panel():
    from inference.infer import load_adapter
    from physgen_v42.backbone import ProcessWan, fix_first, load_wan
    from physgen_v42.data import Dataset
    from physgen_v42.losses import correction_benefit
    device=torch.device('cuda:0')
    adapter=load_adapter(dict(config=cfg, checkpoint=str(CHECKPOINT)),device)
    model=ProcessWan(load_wan(cfg['paths']['wan_checkpoint'],device),adapter,recompute=False)
    rows=[]
    selected={split: Dataset(cfg,split) for split in ('train','validation')}
    for weights in ('ema.pt','adapter.pt'):
        adapter.load_state_dict(torch.load(CHECKPOINT/weights,map_location='cpu',weights_only=True,mmap=True),strict=True)
        for split,dataset in selected.items():
            for index in range(4):
                for mode in ('i2v','t2v'):
                    sample=dataset.load(index,mode,device)
                    geo=Geometry.build(sample['record']['geometry'],device)
                    gt,first,text=sample['latent'],sample['first'],sample['text']
                    g=torch.Generator(device=device).manual_seed(93001+index)
                    noise=torch.randn(gt.shape,device=device,generator=g)
                    with autocast(device): p0=adapter.initialize(text,geo,mode,sample['anchor'])
                    for s in (.10,.50,.90):
                        sigma=torch.tensor(s,device=device)
                        x=fix_first((1-sigma)*gt+sigma*noise,first)
                        with autocast(device): result=model(x,first,text,sigma,sample['record']['duration'],geo,p0)
                        fm=float(fm_loss(result['cond'],noise-gt,first is not None))
                        base=float(fm_loss(result['base'],noise-gt,first is not None))
                        rows.append(dict(weights=weights,split=split,id=sample['record']['id'],mode=mode,
                                         sigma=s,fm=fm,base=base,delta=fm-base,
                                         **correction_benefit(result['cond'],result['base'],noise-gt,first is not None)))
                print(json.dumps(dict(weights=weights,split=split,completed_videos=index+1)),flush=True)
    summaries=[]
    for weights in ('ema.pt','adapter.pt'):
        for split in selected:
            for mode in ('i2v','t2v'):
                rr=[r for r in rows if r['weights']==weights and r['split']==split and r['mode']==mode]
                mean=lambda k:sum(r[k] for r in rr)/len(rr)
                summaries.append(dict(weights=weights,split=split,mode=mode,cases=len(rr),
                                      base=mean('base'),fm=mean('fm'),delta=mean('delta'),
                                      relative_gain=1-mean('fm')/mean('base'),improved=sum(r['delta']<0 for r in rr)))
    report=dict(scope='first 4 train + first 4 validation in existing manifest; both modes; sigmas .10/.50/.90; fixed noise; 8 videos, 96 predictions, not free-generation quality',
                checkpoint=str(CHECKPOINT),summaries=summaries,rows=rows)
    (OUT/'fixed_fm_panel.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(summaries,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('energy','bridge','panel'))
    args=p.parse_args()
    if args.phase=='energy': energy()
    elif args.phase=='bridge': bridge()
    else: fixed_panel()
