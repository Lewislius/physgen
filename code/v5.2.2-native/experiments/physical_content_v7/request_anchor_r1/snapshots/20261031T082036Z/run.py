"""Evaluate the explicit request anchor on the complete fixed40, gamma5."""
from pathlib import Path
import json
import os
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap


def main():
    root=bootstrap.BASE/'experiments/physical_content_v7';work=root/'request_anchor_r1'
    source=Path(__file__).resolve().parent
    from physical_scaling.common import gpu_guard,atomic_json
    job=work/'reports/jobs'/os.getenv('DET_EXPERIMENT_ID','local');gpu_guard(job)
    pinned=bootstrap.BASE/'experiments/physical_scaling_q256/inference_snapshots/20261030T034046Z'
    sys.path.insert(0,str(pinned))
    from pipeline import run_stage
    reference=root/'inference_outputs/physical_content_66778/step0400_seed42_r1/videophy40'
    cases=[c['id'] for c in json.loads((reference/'manifest.json').read_text())['cases']]
    if len(cases)!=40:raise ValueError('Use the full predefined forty cases')
    checkpoint=root/'runs/physical_content_66778/step0400.pt'
    output=work/'inference_outputs/r1_step0400_full40_gamma5_seed42'
    run_stage([sys.executable,'-I','-B','-u',source/'prepare_inputs.py','--cases',*cases],job/'prepare_inputs.log')
    run_stage([sys.executable,'-I','-B','-u',source/'infer_pipeline.py','--checkpoint',checkpoint,
        '--output',output,'--suite','videophy40','--physical-guidance','5','--modes','request_anchored',
        '--request-evidence',work/'cache','--cases',*cases],job/'generate_evaluate.log')
    candidate=output/'controls/request_anchored/videophy40'
    for label,baseline in [('r1_normal',reference),('native400',root/'inference_outputs/native_lora_66781/step0400_seed42_r1/videophy40')]:
        run_stage([sys.executable,'-I','-B',root/'compare_iteration.py',baseline,candidate,
            '--output',work/'reports'/f'full40_vs_{label}.json'],job/f'compare_{label}.log')
    atomic_json(dict(status='completed',cases=cases,checkpoint=str(checkpoint),gamma=5,seed=42,
        training=False,automatic_adoption=False,source=str(source)),job/'status.json')


if __name__=='__main__':main()
