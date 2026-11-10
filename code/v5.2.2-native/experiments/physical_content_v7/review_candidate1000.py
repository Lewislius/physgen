"""Review the exact R1 continuation, then optionally start the R2 baseline."""
from pathlib import Path
import sys,os,argparse
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
from physical_scaling.common import gpu_guard,atomic_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--then-r2-config');args=parser.parse_args()
    root=bootstrap.BASE/'experiments/physical_content_v7'
    folder=root/'reports/candidate1000_jobs'/os.getenv('DET_EXPERIMENT_ID','local');gpu_guard(folder)
    sys.path.insert(0,str(bootstrap.BASE/'experiments/physical_scaling_q256/inference_snapshots/20261030T034046Z'))
    from pipeline import run_stage
    checkpoint=root/'runs/physical_content_66833/step1000.pt'
    if not checkpoint.is_file():raise FileNotFoundError('Submit only after the exact R1 continuation finishes')
    pipeline=root/'inference_snapshots/20261030T151158Z/infer_pipeline.py'
    output=root/'inference_outputs/physical_content_66833'
    for suite,suffix in [('heldout8','step1000_real_heldout8_seed42'),('videophy40','step1000_seed42_r1')]:
        run_stage([sys.executable,'-I','-B','-u',pipeline,'--checkpoint',checkpoint,
            '--output',output/suffix,'--suite',suite],folder/f'{suite}.log')
    controls=output/'step1000_controls_seed42_r1'
    for name in ['VP004','VP010','VP015','VP034']:
        source=output/'step1000_seed42_r1/videophy40/cases'/name
        target=controls/'videophy40/cases'/name;target.mkdir(parents=True,exist_ok=True)
        for basename in ['planned.mp4','planned.json','planned.sa.json','planned.pc.json']:
            if not (target/basename).exists():os.link(source/basename,target/basename)
    run_stage([sys.executable,'-I','-B','-u',pipeline,'--checkpoint',checkpoint,'--output',controls,
        '--modes','real','reasoning_swapped','empty_reasoning','mean','no_c','anchor_only',
        '--cases','VP004','VP010','VP015','VP034'],folder/'controls.log')
    atomic_json(dict(status='completed',checkpoint=str(checkpoint)),folder/'status.json')
    if args.then_r2_config:
        import yaml
        cfg=yaml.safe_load(Path(args.then_r2_config).read_text())
        run_stage([sys.executable,'-I','-B','-u',Path(cfg['source_snapshot'])/'train_and_review.py',
            '--config',args.then_r2_config],folder/'process_r2_baseline_and_review.log')


if __name__=='__main__':main()
