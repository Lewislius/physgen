"""Matched native128 budget, continuing its own fresh64 state into a new output run."""
from pathlib import Path
import json
import os
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap


def main():
    from physical_scaling.common import gpu_guard,atomic_json
    root=bootstrap.BASE/'experiments/physical_content_v7/content_modulation_r6'
    experiment=os.getenv('DET_EXPERIMENT_ID','local')
    folder=root/'reports/native128_jobs'/experiment
    gpu_guard(folder)
    original=bootstrap.BASE/'experiments/physical_content_v7/early_execution_r3/runs/native_lora_66866'
    cfg=json.loads((original/'config.json').read_text())
    if cfg['variant']!='native_lora' or cfg['stages']!={'interface':0,'joint':224}:
        raise ValueError('Expected the existing fresh matched native training recipe')
    source=Path(cfg['source_snapshot'])
    checkpoint=original/'step0064.pt'
    run_id=f'native_lora128_{experiment}'
    new_run=Path(cfg['paths']['runs'])/run_id
    pinned=bootstrap.BASE/'experiments/physical_scaling_q256/inference_snapshots/20261030T034046Z'
    sys.path.insert(0,str(pinned))
    from pipeline import run_stage
    from protocol import manifest_for
    run_stage([sys.executable,'-I','-B','-u',source/'train.py','--config',original/'config.json',
        '--run-id',run_id,'--resume',checkpoint,'--stop-after','128'],folder/'train.log')
    checkpoint=new_run/'step0128.pt'
    output=root/'inference_outputs'/run_id/'step0128_full40_text5_seed42'
    cases=[case['id'] for case in manifest_for('v6_assoc','videophy40')['cases']]
    if len(cases)!=40:raise ValueError('The matched primary comparison must cover40')
    run_stage([sys.executable,'-I','-B','-u',source/'infer_pipeline.py','--checkpoint',checkpoint,
        '--output',output,'--suite','videophy40','--physical-guidance','5','--modes','real',
        '--cases',*cases],folder/'real.log')
    atomic_json(dict(status='native128_full40_completed',checkpoint=str(checkpoint),
        original64_run=str(original),new_run=str(new_run),total_updates=128,
        additional_updates=64,reference_preserved=True,automatic_continuation=False,
        note='Own native64 optimizer/RNG/cursor resumed in a new run. This is the128-update architecture comparator, not native400-to1000.'),folder/'status.json')


if __name__=='__main__':main()
