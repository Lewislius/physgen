"""Run unchanged fresh R3 training, with an explicitly chosen evaluation guidance."""
from pathlib import Path
import argparse
import json
import os
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
from physical_scaling.common import gpu_guard,atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--physical-guidance',required=True,type=float)
    args=p.parse_args()
    import yaml
    cfg=yaml.safe_load(Path(args.config).read_text())
    if cfg['revision']!='physical_content_v7_early_execution_r3' or cfg['variant']!='physical_content':
        raise ValueError('Use the reviewed fresh R3 candidate configuration')
    root=Path(cfg['paths']['runs']).parent
    experiment=os.getenv('DET_EXPERIMENT_ID','local')
    run_id=f'physical_content_{experiment}';job=root/'reports/jobs'/experiment
    gpu_guard(job)
    sys.path.insert(0,str(bootstrap.BASE/'experiments/physical_scaling_q256/inference_snapshots/20261030T034046Z'))
    from pipeline import run_stage
    source=Path(__file__).resolve().parent
    atomic_json(dict(training_config=args.config,first_stop=64,physical_guidance=args.physical_guidance,
        reason='Choose guidance from paired evidence; the original shared5 remains the PC-preservation reference',
        training_changes_from_050403_snapshot=False),job/'reviewed_execution.json')
    run_stage([sys.executable,'-I','-B','-u',Path(cfg['source_snapshot'])/'train.py',
        '--config',args.config,'--run-id',run_id,'--stop-after','64'],job/'train.log')
    checkpoint=root/'runs'/run_id/'step0064.pt'
    output=root/'inference_outputs'/run_id/f'step0064_subset16_physical{args.physical_guidance:g}_seed42'
    cases=[x['id'] for x in json.loads((source/'subset16.json').read_text())['cases']]
    modes=['real','no_c','reasoning_swapped']
    for mode in modes:
        run_stage([sys.executable,'-I','-B','-u',source/'infer_pipeline.py','--checkpoint',checkpoint,
            '--output',output,'--suite','videophy40','--physical-guidance',str(args.physical_guidance),
            '--modes',mode,'--cases',*cases],job/f'{mode}.log')
    atomic_json(dict(status='short_evaluation_completed',checkpoint=str(checkpoint),steps=64,
        cases=cases,modes=modes,physical_guidance=args.physical_guidance,automatic_continuation=False),job/'status.json')


if __name__=='__main__':main()
