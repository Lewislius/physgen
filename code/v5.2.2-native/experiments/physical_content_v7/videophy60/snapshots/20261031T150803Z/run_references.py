"""Prepare the fixed extra20, complete both400 references, and reuse the declared R1-250 test."""
from pathlib import Path
import argparse
import json
import os
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
from physical_scaling.common import gpu_guard,atomic_json

ROOT=bootstrap.BASE/'experiments/physical_content_v7'
HERE=ROOT/'videophy60'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--describe',action='store_true')
    args=parser.parse_args()
    source=Path(__file__).resolve().parent
    commands=[('prepare_inputs',[sys.executable,'-I','-B','-u',source/'prepare_inputs.py'])]
    for name,run in [('R1_400','physical_content_66778'),('native400','native_lora_66781')]:
        commands.append((name,[sys.executable,'-I','-B','-u',source/'run_extra20.py',
            '--checkpoint',ROOT/'runs'/run/'step0400.pt',
            '--reference40',ROOT/'inference_outputs'/run/'step0400_seed42_r1/videophy40',
            '--name',name]))
    commands.append(('R1_250',[sys.executable,'-I','-B','-u',
        ROOT/'training_budget_reference/snapshots/20261031T144712Z/run_reference250.py']))
    if args.describe:
        print(json.dumps(dict(stages={name:list(map(str,cmd)) for name,cmd in commands},
            extra20_each=['R1_400','native400'],single_step_reference16='R1_250',
            new_training=False,old40_regenerated=False,new_models=False)));return
    folder=HERE/'reports/jobs'/os.getenv('DET_EXPERIMENT_ID','local')
    gpu_guard(folder)
    pinned=bootstrap.BASE/'experiments/physical_scaling_q256/inference_snapshots/20261030T034046Z'
    sys.path.insert(0,str(pinned))
    from pipeline import run_stage
    for name,command in commands:
        atomic_json(dict(status='running',stage=name),folder/'status.json')
        run_stage(command,folder/(name+'.log'))
    atomic_json(dict(status='completed',stages=[name for name,_ in commands],
        new_videos={'R1_400':20,'native400':20,'R1_250':16},new_training=False),folder/'status.json')


if __name__=='__main__':main()
