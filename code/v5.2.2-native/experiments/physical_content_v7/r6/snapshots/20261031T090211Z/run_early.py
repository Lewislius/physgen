"""Fresh short training, full40 main comparison and fixed16 module controls."""
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
    from protocol import manifest_for
    source=Path(cfg['source_snapshot'])
    stop=cfg['evaluation_budget']['first_stop']
    run_stage([sys.executable,'-I','-B','-u',source/'train.py','--config',args.config,
               '--run-id',run_id,'--stop-after',str(stop)],folder/'train.log')
    checkpoint=root/'runs'/run_id/f'step{stop:04d}.pt'
    output=root/'inference_outputs'/run_id/f'step{stop:04d}_full40_controls16_physical5_seed42'
    mean_update=root/'controls'/run_id/f'step{stop:04d}_mean_raw_query_update.pt'
    if cfg['variant']=='physical_content':
        run_stage([sys.executable,'-I','-B','-u',source/'prepare_mean_update.py',
            '--checkpoint',checkpoint,'--output',mean_update],folder/'mean_query_update_prepare.log')
    cases=[item['id'] for item in json.loads((source/'subset16.json').read_text())['cases']]
    main_cases=[item['id'] for item in manifest_for('v6_assoc','videophy40')['cases']]
    if len(main_cases)!=40:raise ValueError('The main comparison must use the complete fixed40')
    pipeline=source/'infer_pipeline.py'
    modes=['real','mean_query_update','anchor_only','reasoning_swapped','no_c'] if cfg['variant']=='physical_content' else ['real']
    for mode in modes:
        command=[sys.executable,'-I','-B','-u',pipeline,'--checkpoint',checkpoint,
            '--output',output,'--suite','videophy40','--physical-guidance','5',
            '--modes',mode,'--cases',*(main_cases if mode=='real' else cases)]
        if mode=='mean_query_update':command+=['--mean-query-update',mean_update]
        run_stage(command,folder/f'{mode}.log')
    atomic_json(dict(status='short_evaluation_completed',checkpoint=str(checkpoint),
        steps=stop,main_cases=main_cases,control_cases=cases,modes=modes,automatic_continuation=False,
        remaining_selected_controls=['empty_reasoning','mean','global_train_mean'],
        decision='Review paired SA/PC/Joint and actual generation before extending or revising'),folder/'status.json')


if __name__=='__main__':main()
