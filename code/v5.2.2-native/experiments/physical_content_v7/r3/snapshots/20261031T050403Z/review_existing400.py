"""Test separate physical guidance on16 existing400-step cases and prepare controls."""
from pathlib import Path
import json
import os
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
from physical_scaling.common import gpu_guard,atomic_json


def subset_reference(source,destination,selected):
    plan=json.loads((source/'manifest.json').read_text())
    scores=json.loads((source/'scores.json').read_text())
    plan['cases']=[x for x in plan['cases'] if x['id'] in selected]
    scores['cases']=[x for x in scores['cases'] if x['id'] in selected]
    if len(plan['cases'])!=16 or len(scores['cases'])!=16:
        raise ValueError('Require the recorded16-case subset of the completed40')
    plan['derived_from']=str(source);plan['subset_selection']='Recorded caption/process coverage, not selected wins'
    summary=scores['summary'];summary.update(count=16,
        sa_mean=sum(x['sa'] for x in scores['cases'])/16,
        pc_mean=sum(x['pc'] for x in scores['cases'])/16,
        joint_pass_rate=sum(x['sa']>=4 and x['pc']>=4 for x in scores['cases'])/16)
    for case in plan['cases']:
        out=destination/'cases'/case['id'];out.mkdir(parents=True,exist_ok=True)
        for name in ['planned.mp4','planned.json','planned.sa.json','planned.pc.json']:
            if not (out/name).exists():os.link(source/'cases'/case['id']/name,out/name)
    atomic_json(plan,destination/'manifest.json');atomic_json(scores,destination/'scores.json')


def main():
    source=Path(__file__).resolve().parent
    root=bootstrap.BASE/'experiments/physical_content_v7'
    work=root/'inference_guidance_r2/subset16'
    job=work/'reports/jobs'/os.getenv('DET_EXPERIMENT_ID','local');gpu_guard(job)
    sys.path.insert(0,str(bootstrap.BASE/'experiments/physical_scaling_q256/inference_snapshots/20261030T034046Z'))
    from pipeline import run_stage
    cases=[x['id'] for x in json.loads((source/'subset16.json').read_text())['cases']]
    checkpoint=root/'runs/physical_content_66778/step0400.pt'
    output=work/'physical_content_66778_step0400_physical1_seed42'
    run_stage([sys.executable,'-I','-B','-u',source/'infer_pipeline.py',
        '--checkpoint',checkpoint,'--output',output,'--suite','videophy40',
        '--physical-guidance','1','--cases',*cases],job/'guidance1.log')
    references=[('native_lora_66781','native400'),('physical_content_66778','same_checkpoint_guidance5')]
    for run,label in references:
        reference=work/'references'/label
        subset_reference(root/'inference_outputs'/run/'step0400_seed42_r1/videophy40',reference,cases)
        run_stage([sys.executable,'-I','-B',root/'compare_iteration.py',reference,output/'videophy40',
            '--allow-guidance-change','--output',work/'reports'/f'guidance1_vs_{label}.json'],job/f'compare_{label}.log')
    atomic_json(dict(status='completed',checkpoint=str(checkpoint),cases=cases,
        text_guidance=5.,physical_guidance=1.,automatically_starts_training=False),job/'status.json')


if __name__=='__main__':main()
