"""Discard smoke weights, train afresh, then review complete generated processes."""
from pathlib import Path
import sys,argparse,os
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
from physical_content.common import load_config,gpu_guard,atomic_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
    cfg=load_config(args.config);root=Path(cfg['paths']['runs']).parent
    experiment=os.getenv('DET_EXPERIMENT_ID','local');run_id=f'{cfg["variant"]}_{experiment}'
    folder=root/'reports/jobs'/experiment;gpu_guard(folder)
    sys.path.insert(0,str(bootstrap.BASE/'experiments/physical_scaling_q256/inference_snapshots/20261030T034046Z'))
    from pipeline import run_stage
    source=Path(cfg['source_snapshot'])
    run_stage([sys.executable,'-I','-B','-u',source/'smoke_training.py','--config',args.config],folder/'smoke.log')
    run_stage([sys.executable,'-I','-B','-u',source/'train.py','--config',args.config,
        '--run-id',run_id,'--stop-after','400'],folder/'train.log')
    checkpoint=root/'runs'/run_id/'step0400.pt'
    output=root/'inference_outputs'/run_id
    run_stage([sys.executable,'-I','-B','-u',source/'infer_pipeline.py','--checkpoint',checkpoint,
        '--output',output/'step0400_real_heldout8_seed42','--suite','heldout8'],folder/'heldout8.log')
    # Full fixed-40 stays unchanged and is scored independently after generation.
    run_stage([sys.executable,'-I','-B','-u',source/'infer_pipeline.py','--checkpoint',checkpoint,
        '--output',output/'step0400_seed42','--suite','videophy40'],folder/'videophy40.log')
    if cfg['variant']=='physical_content':
        controls=output/'step0400_controls_seed42'
        for name in ['VP004','VP010','VP015','VP034']:
            original=output/'step0400_seed42/videophy40/cases'/name
            destination=controls/'videophy40/cases'/name;destination.mkdir(parents=True,exist_ok=True)
            for basename in ['planned.mp4','planned.json','planned.sa.json','planned.pc.json']:
                if not (destination/basename).exists():os.link(original/basename,destination/basename)
        run_stage([sys.executable,'-I','-B','-u',source/'infer_pipeline.py','--checkpoint',checkpoint,
            '--output',controls,'--suite','videophy40','--modes','real','reasoning_swapped',
            'empty_reasoning','mean','no_c','anchor_only','--cases','VP004','VP010','VP015','VP034'],folder/'controls.log')
    atomic_json(dict(status='completed',checkpoint=str(checkpoint)),folder/'status.json')


if __name__=='__main__':main()
