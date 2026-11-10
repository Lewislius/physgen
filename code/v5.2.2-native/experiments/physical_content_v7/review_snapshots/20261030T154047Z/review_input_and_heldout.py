"""Run two independent bounded stages sequentially on one accepted GPU."""
from pathlib import Path
import sys,os
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
ROOT=bootstrap.BASE/'experiments/physical_content_v7'
from physical_scaling.common import gpu_guard

def main():
    folder=ROOT/'reports/review_jobs'/os.getenv('DET_EXPERIMENT_ID','local');folder.mkdir(parents=True,exist_ok=True)
    gpu_guard(folder)
    sys.path.insert(0,str(bootstrap.BASE/'experiments/physical_scaling_q256/inference_snapshots/20261030T034046Z'))
    from pipeline import run_stage
    run_stage([sys.executable,'-I','-B','-u',ROOT/'input_snapshots/20261030T152103Z/probe_thinking_reasoning.py'],folder/'thinking.log')
    run_stage([sys.executable,'-I','-B','-u',ROOT/'inference_snapshots/20261030T151158Z/infer_pipeline.py',
        '--checkpoint',ROOT/'runs/native_lora_66781/step0400.pt',
        '--output',ROOT/'inference_outputs/native_lora_66781/step0400_real_heldout8_seed42',
        '--suite','heldout8'],folder/'heldout.log')

if __name__=='__main__':main()
