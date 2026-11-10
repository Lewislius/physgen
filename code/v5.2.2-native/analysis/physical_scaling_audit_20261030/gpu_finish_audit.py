"""Sequential independent decoder probe and video evaluation in one GPU slot."""
from pathlib import Path
import subprocess,sys,json
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1]
PINNED=BASE/'experiments/physical_scaling_q256/inference_snapshots/20261030T034046Z'
sys.path.insert(0,str(PINNED))
from pipeline import evaluator_environment,run_stage


def main():
    import os
    sys.path.insert(0,str(BASE/'experiments/physical_variants/snapshots/20261029T064043Z'))
    from physical_training.common import gpu_guard
    gpu_guard(HERE/'evaluation_allocations'/os.getenv('DET_EXPERIMENT_ID','local'))
    if not (HERE/'ce_probe/results.json').exists():
        run_stage([sys.executable,'-I','-B','-u',HERE/'gpu_ce_probe.py'],HERE/'ce_probe.log')
    folders=[HERE/'gpu_results'/m for m in ['swapped','image_only','no_c']]
    for folder in folders:
        assert len(list(folder.glob('cases/*/planned.mp4')))==2,f'Incomplete generation: {folder}'
    run_stage(['/home/liuzhirui/miniconda3/envs/physvid/bin/python','-I','-B','-u',
               BASE/'tools/evaluate_videophy2.py',*folders],HERE/'ablation_evaluation.log',evaluator_environment())
    (HERE/'gpu_finish_status.json').write_text(json.dumps(dict(status='complete')))


if __name__=='__main__':main()
