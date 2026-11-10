"""Reuse one accepted GPU for the complete candidate review in useful order."""
from pathlib import Path
import sys,os,subprocess,json
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
ROOT=bootstrap.BASE/'experiments/physical_content_v7'
from physical_scaling.common import gpu_guard,atomic_json

def main():
    folder=ROOT/'reports/candidate400_jobs'/os.getenv('DET_EXPERIMENT_ID','local');folder.mkdir(parents=True,exist_ok=True)
    gpu_guard(folder)
    sys.path.insert(0,str(bootstrap.BASE/'experiments/physical_scaling_q256/inference_snapshots/20261030T034046Z'))
    from pipeline import run_stage
    checkpoint=ROOT/'runs/physical_content_66778/step0400.pt'
    try:
        run_stage([sys.executable,'-I','-B','-u',ROOT/'review_snapshots/20261030T154047Z/probe_execution_gradient.py',
            '--checkpoint',checkpoint],folder/'execution_gradient.log')
    except subprocess.CalledProcessError as error:
        # The diagnostic and generation are independent; preserve its failure
        # without withholding the actual paired videos.
        atomic_json(dict(status='failed',returncode=error.returncode,log=str(folder/'execution_gradient.log')),
                    folder/'execution_gradient_status.json')
    run_stage([sys.executable,'-I','-B','-u',ROOT/'inference_snapshots/20261030T151158Z/infer_pipeline.py',
        '--checkpoint',checkpoint,'--output',ROOT/'inference_outputs/physical_content_66778/step0400_real_heldout8_seed42',
        '--suite','heldout8'],folder/'heldout.log')
    run_stage([sys.executable,'-I','-B','-u',ROOT/'inference_snapshots/20261030T145237Z/infer_pipeline.py',
        '--checkpoint',checkpoint,'--output',ROOT/'inference_outputs/physical_content_66778/step0400_seed42_r1',
        '--modes','real','reasoning_swapped','empty_reasoning','mean','no_c','anchor_only'],folder/'videophy40.log')
    atomic_json(dict(status='completed'),folder/'status.json')

if __name__=='__main__':main()
