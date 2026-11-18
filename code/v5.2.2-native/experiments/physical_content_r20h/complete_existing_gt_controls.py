"""Complete eight already-declared GT control videos; no new cases or scoring."""
from pathlib import Path
import os
import json
import subprocess
import sys

G = Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_r20g')
SOURCE = G/'snapshots/20261116T160340Z'
sys.path.insert(0,str(SOURCE))
import bootstrap  # noqa: F401
from physical_content.common import gpu_guard

if __name__ == '__main__':
    output = Path(__file__).parent/'reports/gt_control_jobs'/os.getenv('DET_EXPERIMENT_ID','local')
    gpu_guard(output)
    owner = json.loads((G/'reports/jobs/70452/status.json').read_text())['status']
    if owner not in ('original40_generation','original40_scoring','extra20_generation'):
        print('GT owner is entering its control stages; leave all work to the owner.',flush=True)
        raise SystemExit(0)
    subprocess.run([sys.executable,'-I','-B','-u',str(SOURCE/'infer.py'),
        '--checkpoint',str(G/'runs/r20gt_source_local_execution400/step0400.pt'),
        '--output',str(G/'inference_outputs/r20gt_source_local_execution400/step0400_seed42'),
        '--suite','videophy40','--controls','physical_features_off','no_c',
        '--cases','VP004','VP015','VP017','VP024'],check=True)
