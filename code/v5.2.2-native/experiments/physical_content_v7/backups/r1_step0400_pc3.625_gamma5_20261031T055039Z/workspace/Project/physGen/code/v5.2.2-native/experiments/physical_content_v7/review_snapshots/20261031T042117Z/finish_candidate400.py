"""Complete interrupted generation using the exact original checkpoint and inputs."""
from pathlib import Path
import os
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap
from physical_scaling.common import gpu_guard, atomic_json


def main():
    root = bootstrap.BASE / 'experiments/physical_content_v7'
    folder = root / 'reports/recovery_20261031' / os.environ.get('DET_EXPERIMENT_ID', 'local')
    gpu_guard(folder)
    sys.path.insert(0, str(bootstrap.BASE / 'experiments/physical_scaling_q256/inference_snapshots/20261030T034046Z'))
    from pipeline import run_stage
    checkpoint = root / 'runs/physical_content_66778/step0400.pt'
    output = root / 'inference_outputs/physical_content_66778/step0400_seed42_r1'
    # The pinned generator validates provenance and skips complete existing cases.
    run_stage([sys.executable, '-I', '-B', '-u', root / 'inference_snapshots/20261030T151158Z/infer_pipeline.py',
        '--checkpoint', checkpoint, '--output', output, '--suite', 'videophy40'], folder / 'finish400.log')
    baseline = root / 'inference_outputs/native_lora_66781/step0400_seed42_r1/videophy40'
    run_stage([sys.executable, '-I', '-B', root / 'compare_iteration.py', baseline, output / 'videophy40',
        '--output', root / 'reports/videophy400_comparison.json'], folder / 'compare400.log')
    atomic_json(dict(status='completed',checkpoint=str(checkpoint),output=str(output)), folder / 'status.json')


if __name__ == '__main__':
    main()
