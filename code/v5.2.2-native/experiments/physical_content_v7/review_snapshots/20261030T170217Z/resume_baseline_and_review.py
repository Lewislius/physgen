"""Continue the exact baseline to1000, then generate its paired review suites."""
from pathlib import Path
import sys,os
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
from physical_scaling.common import gpu_guard,atomic_json


def main():
    root=bootstrap.BASE/'experiments/physical_content_v7'
    experiment=os.getenv('DET_EXPERIMENT_ID','local');run_id=f'native_lora_{experiment}'
    folder=root/'reports/baseline1000_jobs'/experiment;gpu_guard(folder)
    sys.path.insert(0,str(bootstrap.BASE/'experiments/physical_scaling_q256/inference_snapshots/20261030T034046Z'))
    from pipeline import run_stage
    source=root/'snapshots/20261030T141432Z'
    run_stage([sys.executable,'-I','-B','-u',source/'train.py','--config',source/'configs/native_lora.yaml',
        '--resume',root/'runs/native_lora_66781/step0400.pt','--run-id',run_id,'--stop-after','1000'],folder/'train.log')
    checkpoint=root/'runs'/run_id/'step1000.pt'
    inference=root/'inference_snapshots/20261030T151158Z/infer_pipeline.py'
    output=root/'inference_outputs'/run_id
    for suite,suffix in [('heldout8','step1000_real_heldout8_seed42'),('videophy40','step1000_seed42_r1')]:
        run_stage([sys.executable,'-I','-B','-u',inference,'--checkpoint',checkpoint,
            '--output',output/suffix,'--suite',suite],folder/f'{suite}.log')
    atomic_json(dict(status='completed',checkpoint=str(checkpoint),run=run_id),folder/'status.json')


if __name__=='__main__':main()
