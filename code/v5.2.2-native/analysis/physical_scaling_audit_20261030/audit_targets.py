"""Quantify information in the actual cached weak supervision, without GPU jobs."""
from pathlib import Path
import json
import math
from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F

BASE=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
ROOTS=[BASE/'experiments/physical_variants/supervision_cache/r1',BASE/'experiments/physical_scaling_q256/supervision_cache/r1']
torch.set_num_threads(4)


def allfiles(kind):
    files={}
    for root in ROOTS:
        files.update({p.stem:p for p in (root/kind).glob('*.pt')})
    return files


def mask(row):
    g=row['geometry'];h,w=30,52
    y=torch.arange(h).float()/h;x=torch.arange(w).float()/w
    y=((y+1/h).clamp(max=(g['top']+g['rh'])/g['h'])-y.clamp(min=g['top']/g['h'])).clamp_min(0)*h
    x=((x+1/w).clamp(max=(g['left']+g['rw'])/g['w'])-x.clamp(min=g['left']/g['w'])).clamp_min(0)*w
    return F.avg_pool2d((y[:,None]*x[None])[None,None],2)[0,0].flatten()[None].expand(20,-1)


def stats(values):
    a=np.array(values,dtype=float)
    return dict(mean=float(a.mean()),median=float(np.median(a)),p10=float(np.quantile(a,.1)),p90=float(np.quantile(a,.9)),min=float(a.min()),max=float(a.max()))


def main():
    manifest=json.loads(Path('/home/liuzhirui/Project/physGen/data_cache/wisa_4000/base/manifest.json').read_text())['records']
    rows={r['id']:r for r in manifest}
    files=allfiles('v6_assoc'); records=[]
    for i,(name,path) in enumerate(sorted(files.items())):
        d=torch.load(path,map_location='cpu',weights_only=True)
        p=d['association'][0].float().reshape(20,390,-1)
        w=mask(rows[name]);ws=w.sum();wp=w[...,None]
        def avg(x):return float((x*w).sum()/ws)
        def kl(q):return avg((p*(p.clamp_min(1e-12).log()-q.clamp_min(1e-12).log())).sum(-1))
        r=p.shape[-1]-1
        prior=(p*wp).sum((0,1),keepdim=True)/ws
        static=p.mean(0,keepdim=True)
        temporal=(p*wp).sum(1,keepdim=True)/wp.sum(1,keepdim=True)
        uniform=torch.full_like(p,1/(r+1))
        record=dict(sample=name,relations=r,uniform_kl=kl(uniform),constant_kl=kl(prior),
                    static_spatial_kl=kl(static),temporal_only_kl=kl(temporal),
                    normalized_entropy=avg(-(p*p.clamp_min(1e-12).log()).sum(-1))/math.log(r+1),
                    reference_mass=avg(p[...,0]),reference_above_half=avg((p[...,0]>.5).float()),
                    max_probability=avg(p.max(-1).values),max_above_half=avg((p.max(-1).values>.5).float()),
                    text=d['relations'])
        records.append(record)
        if i%500==0:print('targets',i,len(files),flush=True)
    keys=[k for k in records[0] if k not in ['sample','text','relations']]
    summary={k:stats([r[k] for r in records]) for k in keys}
    summary.update(count=len(records),relation_counts=dict(Counter(r['relations'] for r in records)))
    # Exact per-micro correspondence permits comparing learned KL with a static shortcut baseline.
    byid={r['sample']:r for r in records}
    logs=[json.loads(s) for s in (BASE/'experiments/physical_scaling_q256/runs/v6_assoc_66678/train.jsonl').read_text().splitlines()]
    window_comparison=[]
    for lo,hi in [(1,50),(51,128),(129,256),(257,500),(501,750),(751,950),(951,1000)]:
        active=[r for r in logs if r['kind']=='micro' and lo<=r['step']<=hi and r['aux_active']]
        entry=dict(start=lo,end=hi,n=len(active),actual_kl=float(np.mean([r['aux'] for r in active])))
        for k in ['uniform_kl','constant_kl','static_spatial_kl','temporal_only_kl']:
            entry[k]=float(np.mean([byid[r['sample']][k] for r in active]))
        window_comparison.append(entry)
    summary['same_samples_baselines']=window_comparison
    (OUT/'association_target_statistics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    (OUT/'association_target_per_sample.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
    input_records=[]
    for name,path in allfiles('input').items():
        d=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        input_records.append(dict(key=name,complete=d['analysis_complete'],relations=len(d['relations']),
                                  relation_words=[len(s.split()) for s in d['relations']],analysis=d['analysis']))
    summary_input=dict(count=len(input_records),incomplete=sum(not r['complete'] for r in input_records),
                       relation_counts=dict(Counter(r['relations'] for r in input_records)),
                       per_relation_words=stats([n for r in input_records for n in r['relation_words']]))
    (OUT/'input_analysis_statistics.json').write_text(json.dumps(summary_input,indent=2))
    (OUT/'input_analysis_per_sample.json').write_text(json.dumps(input_records,ensure_ascii=False,indent=2))
    # Real V6-write support coverage: no occurrence/strength calibration is implied by this metric.
    wr=[]
    for name,path in allfiles('v6_write').items():
        d=torch.load(path,map_location='cpu',weights_only=True)
        p=d['support'][0].float().reshape(20,390,-1);w=mask(rows[name]);den=w.sum()
        avg=lambda x:float((x*w).sum()/den)
        wr.append(dict(sample=name,relations=p.shape[-1],sum_mean=avg(p.sum(-1)),
                       any_support_fraction=avg((p.sum(-1)>0).float()),
                       saturated_sum_fraction=avg((p.sum(-1)>.99).float()),
                       tiny_support_fraction=avg((p.sum(-1)<.01).float()),
                       anchors_rms=float(d['text_anchors'].float().square().mean().sqrt()),
                       text=d['relations']))
    keys=[k for k in wr[0] if k not in ['sample','relations','text']]
    sw={k:stats([r[k] for r in wr]) for k in keys};sw['count']=len(wr)
    (OUT/'write_target_statistics.json').write_text(json.dumps(sw,indent=2))
    (OUT/'write_target_per_sample.json').write_text(json.dumps(wr,ensure_ascii=False,indent=2))
    print('ASSOCIATION',json.dumps(summary,ensure_ascii=False),flush=True)
    print('INPUT',json.dumps(summary_input),flush=True)
    print('WRITE',json.dumps(sw),flush=True)


if __name__=='__main__':main()
