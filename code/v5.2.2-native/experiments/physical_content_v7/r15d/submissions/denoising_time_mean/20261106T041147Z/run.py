from pathlib import Path
import os,sys,json
SOURCE=Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_v7/denoising_matched_fusion_r15d/snapshots/20261106T011032Z');HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(SOURCE))
import bootstrap
import benchmark_inputs
from physical_content.common import gpu_guard,atomic_json
from pipeline import run_stage,evaluator_environment
from allocated_gpu import bind_allocated_gpu
ROOT=Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_v7/denoising_matched_fusion_r15d');experiment=os.getenv('DET_EXPERIMENT_ID','local')
folder=ROOT/'reports/control_helpers'/experiment;gpu_guard(folder);bind_allocated_gpu(folder,evaluator_environment)
checkpoint=ROOT/'runs/physical_content_67920/step0400.pt'
if not checkpoint.is_file():raise FileNotFoundError('Own400 required')
output=ROOT/'helper_outputs/physical_content_67920'/experiment
ids=[x['id'] for x in json.loads((SOURCE/'subset16.json').read_text())['cases']]
atomic_json(dict(status='denoising_time_mean_generation',checkpoint=str(checkpoint),posthoc_mechanism_control=True),folder/'status.json')
run_stage([sys.executable,'-I','-B','-u',HERE/'infer_time_mean.py','--checkpoint',checkpoint,'--output',output,'--suite','videophy40','--modes','denoising_time_mean','--cases',*ids],folder/'generation.log')
destination=output/'controls/denoising_time_mean/videophy40'
atomic_json(dict(status='denoising_time_mean_scoring'),folder/'status.json')
run_stage(['/home/liuzhirui/miniconda3/envs/physvid/bin/python','-I','-B','-u',bootstrap.BASE/'tools/evaluate_videophy2.py',destination],folder/'scoring.log',evaluator_environment())
scores=json.loads((destination/'scores.json').read_text())
assert len(scores['cases'])==16 and {x['id'] for x in scores['cases']}==set(ids)
atomic_json(dict(status='completed',score_file=str(destination/'scores.json'),posthoc_mechanism_control=True,training_updates=0),folder/'status.json')
