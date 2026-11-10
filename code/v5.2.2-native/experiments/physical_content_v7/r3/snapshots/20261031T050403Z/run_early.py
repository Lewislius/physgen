"""Fresh short training and matched16 generation, without automatic extension."""
from pathlib import Path
import argparse
import os
import sys
import json
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
from physical_content.common import load_config,gpu_guard,atomic_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    args=parser.parse_args();cfg=load_config(args.config)
    root=Path(cfg['paths']['runs']).parent
    experiment=os.getenv('DET_EXPERIMENT_ID','local')
    run_id=f'{cfg["variant"]}_{experiment}'
    folder=root/'reports/jobs'/experiment;gpu_guard(folder)
    pinned=bootstrap.BASE/'experiments/physical_scaling_q256/inference_snapshots/20261030T034046Z'
    sys.path.insert(0,str(pinned))
    from pipeline import run_stage
    source=Path(cfg['source_snapshot'])
    run_stage([sys.executable,'-I','-B','-u',source/'train.py','--config',args.config,
               '--run-id',run_id,'--stop-after','64'],folder/'train.log')
    checkpoint=root/'runs'/run_id/'step0064.pt'
    output=root/'inference_outputs'/run_id/'step0064_subset16_physical1_seed42'
    cases=[item['id'] for item in json.loads((source/'subset16.json').read_text())['cases']]
    pipeline=source/'infer_pipeline.py'
    modes=['real','no_c','reasoning_swapped'] if cfg['variant']=='physical_content' else ['real']
    for mode in modes:
        run_stage([sys.executable,'-I','-B','-u',pipeline,'--checkpoint',checkpoint,
            '--output',output,'--suite','videophy40','--physical-guidance','1',
            '--modes',mode,'--cases',*cases],folder/f'{mode}.log')
    atomic_json(dict(status='short_evaluation_completed',checkpoint=str(checkpoint),
        steps=64,cases=cases,modes=modes,automatic_continuation=False,
        remaining_selected_controls=['empty_reasoning','mean','global_train_mean','anchor_only'],
        decision='Review paired SA/PC/Joint and actual generation before extending or revising'),folder/'status.json')


if __name__=='__main__':main()
