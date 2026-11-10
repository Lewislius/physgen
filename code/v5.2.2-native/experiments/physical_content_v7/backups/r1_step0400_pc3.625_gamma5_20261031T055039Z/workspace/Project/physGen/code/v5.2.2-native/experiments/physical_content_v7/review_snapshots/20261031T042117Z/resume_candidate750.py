"""Restore the interrupted candidate's own optimizer/RNG and review step1000."""
from pathlib import Path
import os
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap
from physical_scaling.common import gpu_guard, atomic_json


def main():
    root = bootstrap.BASE / 'experiments/physical_content_v7'
    experiment = os.environ.get('DET_EXPERIMENT_ID', 'local')
    run_id = f'physical_content_{experiment}'
    folder = root / 'reports/recovery_20261031' / experiment
    gpu_guard(folder)
    sys.path.insert(0, str(bootstrap.BASE / 'experiments/physical_scaling_q256/inference_snapshots/20261030T034046Z'))
    from pipeline import run_stage
    parent = root / 'runs/physical_content_66833/step0750.pt'
    source = root / 'snapshots/20261030T141432Z'
    atomic_json(dict(parent=str(parent),parent_experiment=66833,resume_step=750,
        exact_config_and_optimizer=True,reason='Server interruption; own-run continuation only'), folder / 'continuation.json')
    run_stage([sys.executable, '-I', '-B', '-u', source / 'train.py',
        '--config', source / 'configs/physical_content.yaml', '--resume', parent,
        '--run-id', run_id, '--stop-after', '1000'], folder / 'train.log')
    checkpoint = root / 'runs' / run_id / 'step1000.pt'
    pipeline = root / 'inference_snapshots/20261030T151158Z/infer_pipeline.py'
    output = root / 'inference_outputs' / run_id
    for suite, suffix in [('heldout8', 'step1000_real_heldout8_seed42'), ('videophy40', 'step1000_seed42_r1')]:
        run_stage([sys.executable, '-I', '-B', '-u', pipeline, '--checkpoint', checkpoint,
            '--output', output / suffix, '--suite', suite], folder / f'{suite}.log')
    controls = output / 'step1000_controls_seed42_r1'
    for name in ['VP004', 'VP010', 'VP015', 'VP034']:
        src = output / 'step1000_seed42_r1/videophy40/cases' / name
        dst = controls / 'videophy40/cases' / name
        dst.mkdir(parents=True, exist_ok=True)
        for basename in ['planned.mp4', 'planned.json', 'planned.sa.json', 'planned.pc.json']:
            if not (dst / basename).exists():
                os.link(src / basename, dst / basename)
    run_stage([sys.executable, '-I', '-B', '-u', pipeline, '--checkpoint', checkpoint,
        '--output', controls, '--modes', 'real', 'reasoning_swapped', 'empty_reasoning', 'mean', 'no_c', 'anchor_only',
        '--cases', 'VP004', 'VP010', 'VP015', 'VP034'], folder / 'controls.log')
    atomic_json(dict(status='completed',checkpoint=str(checkpoint),run=run_id), folder / 'status.json')


if __name__ == '__main__':
    main()
