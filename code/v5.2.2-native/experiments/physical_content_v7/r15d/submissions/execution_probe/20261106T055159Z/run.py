from pathlib import Path
import os,sys
SOURCE=Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_v7/denoising_matched_fusion_r15d/snapshots/20261106T011032Z')
sys.path.insert(0,str(SOURCE))
import bootstrap
import benchmark_inputs
from physical_content.common import gpu_guard
from pipeline import run_stage,evaluator_environment
from allocated_gpu import bind_allocated_gpu
ROOT=SOURCE.parent.parent
folder=ROOT/'reports/control_helpers'/os.environ['DET_EXPERIMENT_ID']
gpu_guard(folder);bind_allocated_gpu(folder,evaluator_environment)
run_stage([sys.executable,'-I','-B','-u',Path(__file__).resolve().parent/'probe_execution.py'],folder/'probe.log')
