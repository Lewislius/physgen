"""CPU-only inspection of source values, routing and residual execution C on fixed16.

Loads the frozen enhancement checkpoint only; no Wan/Qwen model, generation,
training, API call or download. Representation sensitivity is not quality proof.
"""
import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.nn import functional as F


def cosine_rows(value):
    value = value.detach().float().reshape(-1, value.shape[-1])
    n = len(value)
    if n < 2:
        return None
    unit = F.normalize(value, dim=-1)
    return float((unit.sum(0).square().sum() - unit.square().sum()) / (n * (n - 1)))


def rms(value):
    return float(value.float().square().mean().sqrt())


def describe(value, groups):
    value = value.detach().float()[0]
    relation_means = torch.stack([value[groups == g].mean(0) for g in groups.unique()])
    return dict(token_cosine=cosine_rows(value), relation_mean_cosine=cosine_rows(relation_means),
                rms=rms(value), centered_energy=float((value-value.mean(0)).square().mean()
                / value.square().mean().clamp_min(1e-12)))


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(4);saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True);cfg=saved['config']
    if saved['step']!=400 or cfg['revision']!='physical_content_v7_source_residual_fusion_r15g':raise ValueError('Use final G own400')
    sys.path.insert(0,cfg['source_snapshot']);import bootstrap
    from physical_content.model import Enhancement
    from physical_content.evidence import resample_words
    from condition_controls import InferenceCache
    model=Enhancement(cfg).eval().requires_grad_(False)
    state={k.removeprefix('enhancement.'):v for k,v in {**saved['model'],**saved['constants']}.items() if k.startswith('enhancement.')};model.load_state_dict(state,strict=True);del state,saved
    capture={}
    model.block.register_forward_hook(lambda m,a,o:capture.update(refined=o.detach()))
    model.project.register_forward_pre_hook(lambda m,a:capture.update(values=a[0].detach()))
    model.project.register_forward_hook(lambda m,a,o:capture.update(route_update=o.detach()))
    model.execution_project.register_forward_hook(lambda m,a,o:capture.update(execution_condition=o.detach()))
    reader=InferenceCache(cfg['paths']['inference_evidence'],device='cpu');ids=[x['id'] for x in json.loads((Path(cfg['source_snapshot'])/'subset16.json').read_text())['cases']]
    rows=[];means={};start=time.monotonic()
    for case in ids:
        ready=json.loads((reader.root/f'videophy40_{case}'/'ready.json').read_text());sample=dict(id=ready['id'],caption=ready['caption'])
        evidence=reader.input(sample,image=ready['image'],duration=ready['duration'])
        content,groups=model(evidence)
        anchor,_,_=resample_words((evidence['words'].float()-model.anchor_mean)/model.anchor_scale,evidence['word_groups'],model.seed.shape[1])
        tensors=dict(native_anchor=F.linear(anchor,model.anchor_weight,model.anchor_bias),q_workspace_update=capture['refined']-anchor,
            source_values=capture['values'],route_update=capture['route_update'],execution_conditional=capture['execution_condition'],
            execution_prior=model.execution_prior.expand_as(content['execution']),routing_C=content['routing'],execution_C=content['execution'])
        row=dict(id=case,components={k:describe(v,groups) for k,v in tensors.items()},prior_to_total_RMS=rms(tensors['execution_prior'])/max(rms(content['execution']),1e-12),conditional_to_total_RMS=rms(tensors['execution_conditional'])/max(rms(content['execution']),1e-12))
        rows.append(row)
        for name,value in tensors.items():means.setdefault(name,[]).append(value[0].mean(0))
    summary={}
    for name,values in means.items():
        summary[name]={metric:statistics.mean(row['components'][name][metric] for row in rows if row['components'][name][metric] is not None) for metric in ['token_cosine','relation_mean_cosine','rms','centered_energy']}
        summary[name]['different_sample_mean_cosine']=cosine_rows(torch.stack(values))
    report=dict(checkpoint=str(args.checkpoint.resolve()),source=cfg['source_snapshot'],cases=rows,summary=summary,
        mean_prior_to_total_RMS=statistics.mean(x['prior_to_total_RMS'] for x in rows),mean_conditional_to_total_RMS=statistics.mean(x['conditional_to_total_RMS'] for x in rows),
        elapsed_seconds=time.monotonic()-start,scope='Actual frozen enhancement only, fixed16 normal input. Residual execution C has no native semantic anchor; compare learned corrections separately from total shared C in older versions. No full-Wan write-field measurement or quality claim.',quality_proven=False,training_updates=0,new_generations=0)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['summary','mean_prior_to_total_RMS','mean_conditional_to_total_RMS','elapsed_seconds']}))

if __name__=='__main__':main()
