"""Fresh R1-matched400 training, one intermediate16 and final40/control16 evaluation."""
from pathlib import Path
import argparse
import json
import os
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
from physical_content.common import load_config,gpu_guard,atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
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
    budget=cfg['evaluation_budget']
    if sum(cfg['stages'].values())!=400 or budget['intermediate_step']!=250:
        raise ValueError('This runner owns only the declared250/400 comparison')
    cases=[item['id'] for item in json.loads((source/'subset16.json').read_text())['cases']]
    main_cases=[item['id'] for item in manifest_for('v6_assoc','videophy40')['cases']]
    if len(main_cases)!=40 or len(cases)!=16:raise ValueError('Use the original fixed40/fixed16')
    train=[sys.executable,'-I','-B','-u',source/'train.py','--config',args.config,'--run-id',run_id]
    atomic_json(dict(status='training_to_250',run=run_id,declared_final_step=400),folder/'status.json')
    run_stage([*train,'--stop-after','250'],folder/'train_to250.log')
    intermediate=root/'runs'/run_id/'step0250.pt'
    mid_output=root/'inference_outputs'/run_id/'step0250_subset16_physical5_seed42'
    pipeline=source/'infer_pipeline.py'
    run_stage([sys.executable,'-I','-B','-u',pipeline,'--checkpoint',intermediate,
        '--output',mid_output,'--suite','videophy40','--physical-guidance','5',
        '--modes','real','--cases',*cases],folder/'intermediate250.log')
    # Resume only this fresh run's own saved optimizer/RNG, within its predeclared400 budget.
    atomic_json(dict(status='training_250_to400',run=run_id,declared_final_step=400),folder/'status.json')
    run_stage([*train,'--resume',intermediate,'--stop-after','400'],folder/'train_to400.log')
    checkpoint=root/'runs'/run_id/'step0400.pt'
    output=root/'inference_outputs'/run_id/'step0400_full40_controls16_physical5_seed42'
    atomic_json(dict(status='final_evaluation',checkpoint=str(checkpoint)),folder/'status.json')
    run_stage([sys.executable,'-I','-B','-u',pipeline,'--checkpoint',checkpoint,
        '--output',output,'--suite','videophy40','--physical-guidance','5',
        '--modes','real','--cases',*main_cases],folder/'real.log')
    mean_content=root/'controls'/run_id/'step0400_mean_content.pt'
    run_stage([sys.executable,'-I','-B','-u',source/'prepare_mean_condition.py',
        '--checkpoint',checkpoint,'--output',mean_content],folder/'mean_content_prepare.log')
    for mode in budget['control_modes']:
        command=[sys.executable,'-I','-B','-u',pipeline,'--checkpoint',checkpoint,
            '--output',output,'--suite','videophy40','--physical-guidance','5',
            '--modes',mode,'--cases',*cases]
        if mode=='global_train_mean':command+=['--global-condition',mean_content]
        run_stage(command,folder/f'{mode}.log')
    atomic_json(dict(status='declared_evaluation_completed',checkpoint=str(checkpoint),
        steps=400,intermediate_step=250,main_cases=main_cases,control_cases=cases,
        modes=['real',*budget['control_modes']],automatic_continuation=False,
        decision='Retain R1 unless paired PC is preserved and useful conditional generation is evidenced; SA cannot offset PC loss.'),folder/'status.json')


if __name__=='__main__':main()
