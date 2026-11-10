"""Own400 training, original40+20/60 and fixed16 causal interventions; then stop."""
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
    parser.add_argument('--run-id')
    parser.add_argument('--describe',action='store_true')
    args=parser.parse_args();cfg=load_config(args.config)
    root=Path(cfg['paths']['runs']).parent
    experiment=os.getenv('DET_EXPERIMENT_ID','local')
    run_id=args.run_id or f'{cfg["variant"]}_{experiment}'
    source=Path(cfg['source_snapshot']);budget=cfg['evaluation_budget']
    extended=budget['extended_evaluation']
    prepared=json.loads(Path(extended['input_status']).read_text())
    if prepared['status']!='completed' or prepared['total']!=20:
        raise ValueError('Common extra20 inputs must already be complete')
    cases=[item['id'] for item in json.loads((source/'subset16.json').read_text())['cases']]
    pinned=bootstrap.BASE/'experiments/physical_scaling_q256/inference_snapshots/20261030T034046Z'
    sys.path.insert(0,str(pinned))
    from pipeline import run_stage
    from protocol import manifest_for
    main_cases=[item['id'] for item in manifest_for('v6_assoc','videophy40')['cases']]
    if len(main_cases)!=40 or len(cases)!=16 or budget['final_stop']!=400:
        raise ValueError('Use the declared400, original40, fixed16 protocol')
    run=root/'runs'/run_id;checkpoint=run/'step0400.pt'
    output=root/'inference_outputs'/run_id/'step0400_full40_controls16_physical5_seed42'
    plan=dict(revision=cfg['revision'],run=run_id,source=str(source),steps=400,
              checkpoint=str(checkpoint),main_cases=main_cases,control_cases=cases,
              modes=budget['control_modes'],extended_cases=60,output=str(output),
              automatic_next_candidate=False)
    if args.describe:
        print(json.dumps(plan,ensure_ascii=False,indent=2));return
    folder=root/'reports/jobs'/experiment;gpu_guard(folder)
    atomic_json(plan,folder/'plan.json')
    train=[sys.executable,'-I','-B','-u',source/'train.py','--config',args.config,'--run-id',run_id]
    if not checkpoint.exists():
        if (run/'latest.pt').exists():
            train+=['--resume',run/'latest.pt']
        atomic_json(dict(status='training_to400',run=run_id,declared_final_step=400),folder/'status.json')
        run_stage([*train,'--stop-after','400'],folder/'train_to400.log')
    pipeline=source/'infer_pipeline.py'
    atomic_json(dict(status='main40_evaluation',checkpoint=str(checkpoint)),folder/'status.json')
    run_stage([sys.executable,'-I','-B','-u',pipeline,'--checkpoint',checkpoint,
        '--output',output,'--suite','videophy40','--physical-guidance','5',
        '--modes','real','--cases',*main_cases],folder/'real.log')
    atomic_json(dict(status='extra20_and_full60_evaluation',checkpoint=str(checkpoint)),folder/'status.json')
    run_stage([sys.executable,'-I','-B','-u',extended['runner'],
        '--checkpoint',checkpoint,'--reference40',output/'videophy40',
        '--name','R11A_'+run_id],folder/'extra20_and_full60.log')
    mean_content=root/'controls'/run_id/'step0400_mean_content.pt'
    run_stage([sys.executable,'-I','-B','-u',source/'prepare_mean_condition.py',
        '--checkpoint',checkpoint,'--output',mean_content],folder/'mean_content_prepare.log')
    for mode in budget['control_modes']:
        atomic_json(dict(status='controls16_evaluation',mode=mode,checkpoint=str(checkpoint)),folder/'status.json')
        command=[sys.executable,'-I','-B','-u',pipeline,'--checkpoint',checkpoint,
            '--output',output,'--suite','videophy40','--physical-guidance','5',
            '--modes',mode,'--cases',*cases]
        if mode=='global_train_mean':command+=['--global-condition',mean_content]
        run_stage(command,folder/f'{mode}.log')
    atomic_json(dict(status='declared_evaluation_completed',**plan,
        decision='Review complete matched40/60, fixed16 attribution and actual videos before selecting R11B. Retain R10A PC and improve SA/Joint; no automatic next version.'),folder/'status.json')


if __name__=='__main__':main()
