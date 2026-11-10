"""Compare a separate physical guidance on the original real validation eight."""
from pathlib import Path
import sys,os,json,argparse
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
from physical_scaling.common import gpu_guard,atomic_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--then-r2-config');args=parser.parse_args()
    root=bootstrap.BASE/'experiments/physical_content_v7'
    folder=root/'inference_guidance_r2/reports/jobs'/os.getenv('DET_EXPERIMENT_ID','local');gpu_guard(folder)
    sys.path.insert(0,str(bootstrap.BASE/'experiments/physical_scaling_q256/inference_snapshots/20261030T034046Z'))
    from pipeline import run_stage
    checkpoint=root/'runs/physical_content_66778/step0400.pt'
    baseline=root/'inference_outputs/physical_content_66778/step0400_real_heldout8_seed42/heldout8'
    native=root/'inference_outputs/native_lora_66781/step0400_real_heldout8_seed42/heldout8'
    for scale in [1.,0.]:
        output=root/'inference_guidance_r2/outputs'/f'physical_content_66778_step0400_physical{scale:g}_seed42'
        run_stage([sys.executable,'-I','-B','-u',Path(__file__).with_name('infer_pipeline.py'),
            '--checkpoint',checkpoint,'--output',output,'--suite','heldout8','--physical-guidance',str(scale)],folder/f'physical{scale:g}.log')
        for reference,label in [(baseline,'same_checkpoint_legacy5'),(native,'native_lora')]:
            run_stage([sys.executable,'-I','-B',root/'compare_iteration.py',reference,output/'heldout8',
                '--allow-guidance-change','--output',root/'inference_guidance_r2/reports'/f'physical{scale:g}_vs_{label}.json'],
                folder/f'compare_physical{scale:g}_{label}.log')
    atomic_json(dict(status='completed',checkpoint=str(checkpoint),text_scale=5.,physical_scales=[1.,0.],
        selection_suite='Original eight real validation cases; no test-specific prompt changes'),folder/'status.json')
    if args.then_r2_config:
        import yaml
        cfg=yaml.safe_load(Path(args.then_r2_config).read_text())
        run_stage([sys.executable,'-I','-B','-u',Path(cfg['source_snapshot'])/'train_and_review.py',
            '--config',args.then_r2_config],folder/'process_r2_training_and_review.log')


if __name__=='__main__':main()
