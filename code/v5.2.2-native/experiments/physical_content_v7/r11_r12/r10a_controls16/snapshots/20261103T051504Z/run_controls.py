"""Two targeted R10A400 interventions to guide simplification; no training."""
import argparse
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--describe',action='store_true')
    args=parser.parse_args()
    root=bootstrap.BASE/'experiments/physical_content_v7'
    source=Path(__file__).resolve().parent
    checkpoint=root/'execution_value_r10/runs/physical_content_67003/step0400.pt'
    output=root/'r11_r12/r10a_controls16/inference_outputs/own400_seed42_gamma5'
    cases=[r['id'] for r in json.loads((source/'subset16.json').read_text())['cases']]
    modes=['execution_only','no_state_value']
    command=[sys.executable,'-I','-B','-u',source/'infer_pipeline.py','--checkpoint',checkpoint,
        '--output',output,'--suite','videophy40','--seed','42','--physical-guidance','5',
        '--modes',*modes,'--cases',*cases]
    plan=dict(checkpoint=str(checkpoint),source=str(source),modes=modes,cases=cases,
        generation_command=list(map(str,command)),output=str(output),new_training=False,
        baseline='Use completed R10A normal/routing_only/anchor_only/no_c on these SAME16, without rerating.',
        purpose='Isolate learned route while retaining execution, and extra state-value multiplication while retaining state-dependent attention. No gamma tests.')
    if args.describe:
        print(json.dumps(plan,ensure_ascii=False,indent=2));return
    from physical_scaling.common import gpu_guard,atomic_json
    folder=root/'r11_r12/r10a_controls16/reports/jobs'/os.getenv('DET_EXPERIMENT_ID','local')
    gpu_guard(folder);atomic_json(plan,folder/'plan.json')
    sys.path.insert(0,str(bootstrap.BASE/'experiments/physical_scaling_q256/inference_snapshots/20261030T034046Z'))
    from pipeline import run_stage
    atomic_json(dict(status='generation_and_scoring',**plan),folder/'status.json')
    run_stage(command,folder/'controls.log')
    atomic_json(dict(status='completed',**plan),folder/'status.json')


if __name__=='__main__':main()
