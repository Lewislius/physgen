"""Close out the two complete EI controls, without generating more videos."""
from pathlib import Path
import json
import os
import sys

BASE=Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native')
E=BASE/'experiments/physical_content_r19ei'
SOURCE=E/'snapshots/20261114T065127Z'
sys.path.insert(0,str(SOURCE))
import bootstrap
from pipeline import evaluator_environment,run_stage
from physical_content.common import atomic_json,gpu_guard

job=E/'reports/jobs'/os.environ['DET_EXPERIMENT_ID']
job.mkdir(parents=True,exist_ok=True)
gpu_guard(job)
output=E/'inference_outputs/r19ei_scene400/step0400_seed42'
for mode in ('no_c','semantic_expert_off'):
    folder=output/'controls'/mode/'videophy40'
    summary=json.loads((folder/'summary.json').read_text())
    if summary.get('completed_cases') != 16:
        raise RuntimeError(f'Control is incomplete: {mode}')
    atomic_json(dict(status='closeout_scoring_'+mode,output=str(output),generation=False),job/'status.json')
    run_stage(['/home/liuzhirui/miniconda3/envs/physvid/bin/python','-I','-B','-u',
               str(BASE/'tools/evaluate_videophy2.py'),str(folder)],job/(mode+'.log'),evaluator_environment())
run_stage([sys.executable,'-I','-B','-u',str(SOURCE/'summarize_controls.py'),'--root',str(output)],job/'paired_summary.log')
atomic_json(dict(status='complete',scored=['no_c','semantic_expert_off'],generation=False,
                 remaining_controls='deferred for R20 priority',output=str(output)),job/'status.json')
print(json.dumps(dict(complete=True,scored=32,generated=0)))
