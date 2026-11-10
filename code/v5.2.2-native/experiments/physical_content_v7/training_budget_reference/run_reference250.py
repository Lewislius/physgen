"""One reusable R1-250 normal16 evaluation from its existing original checkpoint."""
from pathlib import Path
import argparse
import json
import os
import sys

ROOT=Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_v7')
PINNED=ROOT.parent/'physical_scaling_q256/inference_snapshots/20261030T034046Z'
INFER=ROOT/'early_execution_r3/snapshots/20261031T050403Z/infer_pipeline.py'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--describe',action='store_true')
    args=parser.parse_args()
    root=ROOT/'training_budget_reference'
    checkpoint=ROOT/'runs/physical_content_66778/step0250.pt'
    output=root/'inference_outputs/r1_step0250_subset16_physical5_seed42'
    cases=[row['id'] for row in json.loads((Path(__file__).with_name('subset16.json')).read_text())['cases']]
    command=[sys.executable,'-I','-B','-u',str(INFER),'--checkpoint',str(checkpoint),
        '--output',str(output),'--suite','videophy40','--physical-guidance','5',
        '--seed','42','--modes','real','--cases',*cases]
    plan=dict(checkpoint=str(checkpoint),checkpoint_step=250,initialization='Existing R1 own250 checkpoint; no training',
        inference_source=str(INFER),output=str(output),cases=cases,command=command,
        reuse='Single reference for the declared R9/R10 normal250 fixed16 comparisons',
        guidance=5,seed=42,additional_controls=False)
    if args.describe:
        print(json.dumps(plan,ensure_ascii=False));return
    folder=root/'reports/jobs'/os.getenv('DET_EXPERIMENT_ID','local');folder.mkdir(parents=True,exist_ok=True)
    (folder/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    sys.path.insert(0,str(PINNED))
    from pipeline import run_stage
    run_stage(command,folder/'pipeline.log')
    score=json.loads((output/'videophy40/scores.json').read_text())
    (folder/'status.json').write_text(json.dumps(dict(status='completed',summary=score['summary'],**plan),indent=2)+'\n')


if __name__=='__main__':main()
