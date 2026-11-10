"""CPU checkpoint probes of C and its actual learned projections; no generation claims.

All weights/evidence are immutable training/inference artifacts. Isotropic query
probes below are explicitly synthetic and are not denoising activation captures.
"""
from pathlib import Path
import sys
import json
import math
import time
import argparse
import gc
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

BASE=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
EXP=BASE/'experiments'
FOUND=EXP/'physical_variants/snapshots/20261029T064043Z'
SCALED=EXP/'physical_scaling_q256/snapshots/20261029T140406Z'
sys.path.insert(0,str(FOUND));sys.path.insert(0,str(SCALED))
from physical_scaling.model import Enhancement
torch.set_num_threads(8)

JOBS={
 'assoc1000':EXP/'physical_scaling_q256/runs/v6_assoc_66678/step1000.pt',
 'assoc500':EXP/'physical_scaling_q256/runs/v6_assoc_66678/step0500.pt',
 'assoc128':EXP/'physical_scaling_q256/runs/v6_assoc_66678/step0128.pt',
 'assoc224':EXP/'physical_variants/runs/v6_assoc_20261029T064043Z/latest.pt',
 'write1000':EXP/'physical_scaling_q256/runs/v6_write_resume860_s20_66710/step1000.pt',
 'write224':EXP/'physical_variants/runs/v6_write_20261029T064043Z/latest.pt',
 'ce1000':EXP/'physical_v5_scaling_q256/runs/v5_ce_66706/step1000.pt',
 'ce224':EXP/'physical_variants/runs/v5_ce_20261029T064043Z-retry1/latest.pt',
}
INFERENCE=EXP/'physical_variants/inference_cache/physical_r1_first_image_request/input'
ROOTS=[EXP/'physical_scaling_q256/supervision_cache/r1/input',EXP/'physical_variants/supervision_cache/r1/input']
WAN=Path('/home/liuzhirui/model/Wan2.2/Wan2.2-TI2V-5B')
INDEX=json.loads((WAN/'diffusion_pytorch_model.safetensors.index.json').read_text())['weight_map']


def base_tensor(name):
    with safe_open(WAN/INDEX[name],framework='pt',device='cpu') as f:return f.get_tensor(name).float()


def summary(x):
    a=np.asarray(x,dtype=float)
    return dict(mean=float(a.mean()),median=float(np.median(a)),min=float(a.min()),max=float(a.max()))


def representation(x,groups=None):
    x=x.float();n=x.shape[0]
    normalized=F.normalize(x,dim=-1)
    gram=x@x.T
    vals=torch.linalg.eigvalsh(gram).clamp_min(0).flip(0)
    energy=vals/vals.sum().clamp_min(1e-30)
    mean=x.mean(0,keepdim=True)
    centered=x-mean
    cosine=normalized@normalized.T
    off=~torch.eye(n,dtype=torch.bool)
    result=dict(rms=float(x.square().mean().sqrt()),pair_cosine=float(cosine[off].mean()),
                centered_energy_fraction=float(centered.square().sum()/x.square().sum().clamp_min(1e-20)),
                top1_energy=float(energy[0]),top4_energy=float(energy[:4].sum()),
                energy_effective_rank=float((-(energy*energy.clamp_min(1e-30).log()).sum()).exp()),
                stable_rank=float(vals.sum()/vals[0].clamp_min(1e-20)))
    if groups is not None:
        same=groups[:,None]==groups[None,:]
        result['within_group_cosine']=float(cosine[same&off].mean())
        if (~same).any():result['between_group_cosine']=float(cosine[~same].mean())
        gcx=torch.stack([x[groups==i].mean(0) for i in range(int(groups.max())+1)])
        within=x-gcx[groups]
        result['within_group_energy_fraction']=float(within.square().sum()/x.square().sum().clamp_min(1e-20))
        result['group_mean_energy']=representation(gcx) if len(gcx)>1 else {}
    return result


def apply_projection(c,state,site,kind,variant):
    stem=f'fusions.{site}.{kind}'
    if variant=='v5_ce':return F.linear(c,state[stem+'.weight'].float())
    basekind={'key':'k','value':'v'}[kind]
    y=F.linear(c,base_tensor(f'blocks.{site}.cross_attn.{basekind}.weight'))
    return y+F.linear(F.linear(c,state[stem+'.lora_a'].float()),state[stem+'.lora_b'].float())


def fusion_probe(c,groups,state,site,variant):
    c=c.float();stem=f'fusions.{site}.'
    k=apply_projection(c,state,site,'key',variant)
    k=k*torch.rsqrt(k.square().mean(-1,keepdim=True)+1e-6)*state[stem+'key_norm.weight'].float()
    v=apply_projection(c,state,site,'value',variant)
    o=state[stem+'output.weight'].float()
    vo=F.linear(v,o)
    result=dict(key=representation(k,groups),value=representation(v,groups),projected_value=representation(vo,groups),
                output_frobenius=float(o.norm()),time_frobenius=float(state[stem+'time.weight'].norm()))
    # One query per head/location with native norm_q scaling, synthetic by design.
    heads,hd=24,128
    q=torch.randn(128,3072,generator=torch.Generator().manual_seed(142))
    q=q*torch.rsqrt(q.square().mean(-1,keepdim=True)+1e-6)*base_tensor(f'blocks.{site}.cross_attn.norm_q.weight')
    qs=q.reshape(-1,heads,hd).permute(1,0,2)
    ks=k.reshape(-1,heads,hd).permute(1,0,2)
    vs=v.reshape(-1,heads,hd).permute(1,0,2)
    logits=qs@ks.transpose(-1,-2)/math.sqrt(hd)
    if variant=='v6_assoc':
        ref=state[stem+'reference_key'].float().reshape(1,heads,hd).permute(1,0,2)
        counts=torch.bincount(groups).float()
        logits=logits-counts[groups].log()[None,None,:]
        logits=torch.cat((logits,qs@ref.transpose(-1,-2)/math.sqrt(hd)),-1)
        vs=torch.cat((vs,torch.zeros_like(ref)),1)
    p=logits.softmax(-1)
    write=F.linear((p@vs).permute(1,0,2).reshape(-1,3072),o)
    result['synthetic_query_write']=representation(write)
    result['synthetic_query_attention_entropy']=float((-(p*p.clamp_min(1e-20).log()).sum(-1)).mean())
    if variant=='v6_assoc':result['synthetic_query_reference_mean']=float(p[...,-1].mean())
    return result


def inputs(full):
    selected=list(range(1,41)) if full else [1,6,7,14,15,25,30,32]
    result=[(f'VP{i:03d}',INFERENCE/f'videophy40_VP{i:03d}_text.pt') for i in selected]
    train=sorted(set(p.name for root in ROOTS for p in root.glob('*_text.pt')))
    for i in np.linspace(0,len(train)-1,8 if full else 4,dtype=int):
        name=train[i]
        p=next(root/name for root in ROOTS if (root/name).is_file())
        result.append((name.removesuffix('_text.pt'),p))
    return result


@torch.no_grad()
def audit(name,full):
    start=time.monotonic()
    checkpoint=torch.load(JOBS[name],map_location='cpu',weights_only=False,mmap=True)
    cfg=checkpoint['config'];state=checkpoint['model']
    with torch.device('meta'):model=Enhancement(cfg)
    model.load_state_dict({k.removeprefix('enhancement.'):v for k,v in state.items() if k.startswith('enhancement.')},assign=True)
    model.eval().requires_grad_(False)
    records=[];means=[];c_cache={}
    for i,(key,path) in enumerate(inputs(full)):
        evidence=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        with torch.autocast('cpu',dtype=torch.bfloat16):c,g=model(evidence)
        c=c[0].float();means.append(c.mean(0));c_cache[key]=c
        record=dict(sample=key,source=str(path),relations=evidence['relations'],c=representation(c,g),fusions={})
        # Actual learned V/O projections at shallow, middle, and deep injection sites.
        for site in [2,14,29]:record['fusions'][str(site)]=fusion_probe(c,g,state,site,cfg['variant'])
        if key in ['VP001','VP006','VP014','VP015','VP025','VP030','VP032','VP040']:
            nullpath=path.with_name(path.name.replace('_text.pt','_null.pt'))
            null=torch.load(nullpath,map_location='cpu',weights_only=True,mmap=True)
            with torch.autocast('cpu',dtype=torch.bfloat16):nc,ng=model(null)
            nc=nc[0].float()
            record['null_comparison']=dict(mean_cosine=float(F.cosine_similarity(c.mean(0),nc.mean(0),dim=0)),
                                           matched_token_cosine=float(F.cosine_similarity(c,nc,dim=-1).mean()),
                                           relative_difference=float((c-nc).norm()/c.norm()),
                                           null_relations=null['relations'])
        records.append(record)
        if i%4==0:print(name,i+1,'/',len(inputs(full)),'seconds',round(time.monotonic()-start,1),flush=True)
    result=dict(checkpoint=str(JOBS[name]),step=checkpoint['step'],variant=cfg['variant'],q=cfg['model']['q_tokens'],
                samples=len(records),compute='CPU bfloat16 autocast; no video-generation or actual Wan query measurements',
                cross_sample_mean=representation(torch.stack(means)),records=records,
                summary={k:summary([r['c'][k] for r in records]) for k in records[0]['c'] if isinstance(records[0]['c'][k],(int,float))})
    # Across the benchmark alone, to avoid confounding train/test mixture.
    result['benchmark_cross_sample_mean']=representation(torch.stack([c_cache[k].mean(0) for k in c_cache if k.startswith('VP')]))
    (OUT/f'representations_{name}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    torch.save(c_cache,OUT/f'c_samples_{name}.pt')
    print('COMPLETE',name,json.dumps(result['summary']), 'cross_sample',result['benchmark_cross_sample_mean'],flush=True)
    del state,checkpoint,model,c_cache;gc.collect()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--models',nargs='+',default=list(JOBS));parser.add_argument('--full',action='store_true')
    args=parser.parse_args()
    for name in args.models:audit(name,args.full)
