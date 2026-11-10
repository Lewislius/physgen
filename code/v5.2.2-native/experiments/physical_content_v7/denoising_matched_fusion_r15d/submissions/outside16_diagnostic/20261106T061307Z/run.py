from pathlib import Path
import os,sys,json
SOURCE=Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_v7/denoising_matched_fusion_r15d/snapshots/20261106T011032Z')
sys.path.insert(0,str(SOURCE))
import bootstrap
import benchmark_inputs
from physical_content.common import gpu_guard,atomic_json
from pipeline import run_stage,evaluator_environment
from allocated_gpu import bind_allocated_gpu
ROOT=SOURCE.parent.parent;experiment=os.environ['DET_EXPERIMENT_ID']
folder=ROOT/'reports/control_helpers'/experiment;gpu_guard(folder);bind_allocated_gpu(folder,evaluator_environment)
checkpoint=ROOT/'runs/physical_content_67920/step0400.pt';output=ROOT/'helper_outputs/physical_content_67920'/experiment
ids=['VP006','VP014','VP019','VP020','VP023','VP032','VP033','VP040']
for mode in ['pretext_read','denoising_off']:
 atomic_json(dict(status=mode+'_generation',posthoc=True,cases=ids),folder/'status.json')
 run_stage([sys.executable,'-I','-B','-u',SOURCE/'infer.py','--checkpoint',checkpoint,'--output',output,'--suite','videophy40','--modes',mode,'--cases',*ids],folder/(mode+'_generation.log'))
 dest=output/'controls'/mode/'videophy40'
 atomic_json(dict(status=mode+'_scoring',posthoc=True,cases=ids),folder/'status.json')
 run_stage(['/home/liuzhirui/miniconda3/envs/physvid/bin/python','-I','-B','-u',bootstrap.BASE/'tools/evaluate_videophy2.py',dest],folder/(mode+'_scoring.log'),evaluator_environment())
 scores=json.loads((dest/'scores.json').read_text());assert len(scores['cases'])==8 and {c['id'] for c in scores['cases']}==set(ids)
atomic_json(dict(status='completed',cases=ids,posthoc=True,no_primary_score_replacement=True),folder/'status.json')
