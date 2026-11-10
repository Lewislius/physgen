"""Confirm the request-anchor change on the existing fixed16 with one new seed."""
from pathlib import Path
import json
import os
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap


def main():
    root=bootstrap.BASE/'experiments/physical_content_v7'
    work=root/'request_anchor_r1'
    source=Path(__file__).resolve().parent
    from physical_scaling.common import gpu_guard,atomic_json
    job=work/'reports/seed43_jobs'/os.getenv('DET_EXPERIMENT_ID','local')
    gpu_guard(job)
    reference=root/'module_controls16/r1_step0400_guidance5_seed42/references/real'
    cases=[x['id'] for x in json.loads((reference/'manifest.json').read_text())['cases']]
    if len(cases)!=16 or len(set(cases))!=16:
        raise ValueError('Use the previously fixed sixteen cases')
    for case in cases:
        if not (work/'cache'/f'{case}.pt').is_file():
            raise FileNotFoundError(f'Existing request-anchor input missing: {case}')
    pinned=bootstrap.BASE/'experiments/physical_scaling_q256/inference_snapshots/20261030T034046Z'
    sys.path.insert(0,str(pinned))
    from pipeline import run_stage
    checkpoint=root/'runs/physical_content_66778/step0400.pt'
    output=work/'inference_outputs/r1_step0400_subset16_gamma5_seed43'
    run_stage([sys.executable,'-I','-B','-u',source/'infer_pipeline.py',
        '--checkpoint',checkpoint,'--output',output,'--suite','videophy40',
        '--seed','43','--physical-guidance','5','--modes','real','request_anchored',
        '--request-evidence',work/'cache','--cases',*cases],job/'generate_evaluate.log')
    run_stage([sys.executable,'-I','-B',root/'compare_iteration.py',output/'videophy40',
        output/'controls/request_anchored/videophy40','--output',
        work/'reports/seed43_request_vs_original16.json'],job/'compare.log')
    atomic_json(dict(status='completed',cases=cases,seed=43,gamma=5,checkpoint=str(checkpoint),
        source=str(source),training=False,new_teacher_encoding=False,
        purpose='One paired confirmation of the completed full40 inference change; no new gamma or factorial ablation'),
        job/'status.json')


if __name__=='__main__':main()
