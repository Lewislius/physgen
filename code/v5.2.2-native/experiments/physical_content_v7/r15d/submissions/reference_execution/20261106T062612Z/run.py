from pathlib import Path
import os,sys
SOURCE=Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_v7/denoising_matched_fusion_r15d/snapshots/20261106T011032Z')
sys.path.insert(0,str(SOURCE))
import bootstrap
import benchmark_inputs
from physical_content.common import gpu_guard,atomic_json
from pipeline import run_stage,evaluator_environment
from allocated_gpu import bind_allocated_gpu
ROOT=SOURCE.parent.parent;folder=ROOT/'reports/control_helpers'/os.environ['DET_EXPERIMENT_ID']
gpu_guard(folder);bind_allocated_gpu(folder,evaluator_environment)
for name in ['R1','R10A','R15B']:
 run_stage([sys.executable,'-I','-B','-u',Path(__file__).resolve().parent/'probe_reference_execution.py','--reference',name],folder/(name+'_probe.log'))
atomic_json(dict(status='completed',references=['R1','R10A','R15B'],cases=['VP004','VP038'],output_videos=0),folder/'status.json')
