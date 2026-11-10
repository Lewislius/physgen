"""Separate total C from its learned components on the existing fixed16 inputs."""
import json
from pathlib import Path
import sys
from datetime import datetime,timezone
import torch
from torch.nn import functional as F

ROOT=Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_v7')
SOURCE=ROOT/'execution_value_r10/snapshots/20261101T034557Z'
sys.path.insert(0,str(SOURCE))
import bootstrap
from physical_content.model import Enhancement
from physical_content.teacher import Teacher
from physical_content.evidence import resample_words


def stats(value):
    value=value.detach().float().reshape(-1,value.shape[-1])
    n=len(value);normalized=F.normalize(value,dim=-1);center=value.mean(0)
    return dict(token_cosine=float((normalized.sum(0).square().sum()-n)/(n*(n-1))),
        rms=float(value.square().mean().sqrt()),
        common_token_energy_fraction=float(center.square().sum()*n/value.square().sum().clamp_min(1e-20)))


@torch.no_grad()
def main():
    torch.set_num_threads(8)
    checkpoint=ROOT/'execution_value_r10/runs/physical_content_67003/step0400.pt'
    state=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    cfg=state['config'];model=Enhancement(cfg).eval().requires_grad_(False)
    weights={name.removeprefix('enhancement.'):v for name,v in {**state['model'],**state['constants']}.items() if name.startswith('enhancement.')}
    model.load_state_dict(weights,strict=True);model.checkpointing=False
    teacher=Teacher(cfg,device='cpu')
    captured={}
    model.execution_project.register_forward_pre_hook(lambda m,args:captured.update(update=args[0].detach()))
    model.project.register_forward_pre_hook(lambda m,args:captured.update(readout=args[0].detach()))
    rows=[];means={}
    cases=json.loads((SOURCE/'subset16.json').read_text())['cases']
    for case in cases:
        evidence=teacher.input(dict(id='videophy40_'+case['id']))
        with torch.autocast('cpu',dtype=torch.bfloat16):
            content,_=model(evidence)
            anchor_content,_=model(evidence,anchor_only=True)
        values=dict(query_update=captured['update'],final_input_readout=captured['readout'],
            routing_correction=content['routing']-anchor_content['routing'],
            execution_correction=content['execution']-anchor_content['execution'],
            routing_C=content['routing'],execution_C=content['execution'],anchor=anchor_content['routing'])
        for key,value in values.items():means.setdefault(key,[]).append(value.float().mean(1))
        rows.append(dict(id=case['id'],caption=case['caption'],reasoning=evidence.get('analysis'),
            relations=evidence.get('relations'),metrics={k:stats(v) for k,v in values.items()}))
        print(json.dumps(dict(completed=len(rows),total=len(cases),id=case['id'])),flush=True)
    result=dict(utc=datetime.now(timezone.utc).isoformat(),checkpoint=str(checkpoint),step=state['step'],
        scope='Existing fixed16, positive conditions, CPU enhancement only. Token similarity and raw magnitude do not establish actual Wan write strength, causal semantic use, or quality.',
        aggregate={key:{metric:sum(row['metrics'][key][metric] for row in rows)/len(rows) for metric in rows[0]['metrics'][key]} for key in values},
        cross_sample_means={k:stats(torch.cat(v,0)) for k,v in means.items()},cases=rows)
    out=ROOT/'r11_r12/reports';out.mkdir(parents=True,exist_ok=True)
    (out/'r10a400_fixed16_representation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(aggregate=result['aggregate'],cross_sample_means=result['cross_sample_means']),indent=2),flush=True)


if __name__=='__main__':main()
