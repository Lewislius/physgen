"""Two inference-only20 arms on one actual RTX6000; no model training."""
from pathlib import Path
import json,os,sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import BASE,WORK,OLD,ARMS,setup
setup(OLD)
from physical_scaling.common import gpu_guard,atomic_json
from pipeline import run_stage,evaluator_environment

def main():
    source=Path(__file__).resolve().parent;experiment=os.getenv('DET_EXPERIMENT_ID','local')
    folder=WORK/'reports/jobs'/experiment;gpu_guard(folder)
    ready=json.loads((WORK/'inputs/ready.json').read_text())
    if not ready['complete'] or ready['contexts']!=80:raise ValueError('Prepare all paired cached contexts first')
    for arm in ARMS:
        atomic_json(dict(stage=arm+'_generation'),folder/'status.json')
        run_stage([sys.executable,'-I','-B','-u',source/'infer.py','--arm',arm],folder/(arm+'_generation.log'))
        atomic_json(dict(stage=arm+'_scoring'),folder/'status.json')
        run_stage(['/home/liuzhirui/miniconda3/envs/physvid/bin/python','-I','-B','-u',BASE/'tools/evaluate_videophy2.py',WORK/'outputs'/arm/'videophy40'],folder/(arm+'_scoring.log'),evaluator_environment())
        run_stage([sys.executable,'-I','-B','-u',source/'report.py'],folder/(arm+'_report.log'))
    atomic_json(dict(stage='completed',new_videos=40,training_updates=0,api_calls=0),folder/'status.json')

if __name__=='__main__':main()
