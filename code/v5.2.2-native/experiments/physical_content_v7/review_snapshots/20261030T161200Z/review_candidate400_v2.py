"""Complete paired held-out scoring and resume the unchanged fixed-40 review."""
from pathlib import Path
import sys,os
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
from physical_scaling.common import gpu_guard,atomic_json
from repair_heldout_evaluation import repair

ROOT=bootstrap.BASE/'experiments/physical_content_v7'


def main():
    folder=ROOT/'reports/candidate400_jobs'/os.getenv('DET_EXPERIMENT_ID','local')
    folder.mkdir(parents=True,exist_ok=True);gpu_guard(folder)
    sys.path.insert(0,str(bootstrap.BASE/'experiments/physical_scaling_q256/inference_snapshots/20261030T034046Z'))
    from pipeline import run_stage
    checkpoint=ROOT/'runs/physical_content_66778/step0400.pt'
    baseline=ROOT/'inference_outputs/native_lora_66781/step0400_real_heldout8_seed42/heldout8'
    candidate=ROOT/'inference_outputs/physical_content_66778/step0400_real_heldout8_seed42/heldout8'
    import json
    for directory in [baseline,candidate]:
        plan=json.loads((directory/'manifest.json').read_text())
        for case in plan['cases']:
            path=directory/'cases'/case['id']/'planned.mp4'
            if not path.is_file() or not path.with_suffix('.json').is_file():
                raise ValueError('This review resumes only after both eight-video suites are complete')
        repair(directory/'manifest.json')
    run_stage(['/home/liuzhirui/miniconda3/envs/physvid/bin/python','-I','-B','-u',
        bootstrap.BASE/'tools/evaluate_videophy2.py',baseline,candidate],folder/'heldout_paired_evaluation.log')
    run_stage([sys.executable,'-I','-B',ROOT/'compare_iteration.py',baseline,candidate,
        '--output',ROOT/'reports/heldout400_comparison.json'],folder/'heldout_comparison.log')
    run_stage([sys.executable,'-I','-B','-u',ROOT/'inference_snapshots/20261030T145237Z/infer_pipeline.py',
        '--checkpoint',checkpoint,'--output',ROOT/'inference_outputs/physical_content_66778/step0400_seed42_r1',
        '--modes','real','reasoning_swapped','empty_reasoning','mean','no_c','anchor_only'],folder/'videophy40.log')
    atomic_json(dict(status='completed'),folder/'status.json')


if __name__=='__main__':main()
