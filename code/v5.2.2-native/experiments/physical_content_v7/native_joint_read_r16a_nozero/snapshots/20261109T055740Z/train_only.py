"""Train fresh NOZERO400 on RTX; release it before separate Ada/Amp evaluation."""
from pathlib import Path
import argparse,json,os,sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
from physical_content.common import load_config,data_ready,gpu_guard,atomic_json
from benchmark_inputs import PINNED
from pipeline import run_stage


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--run-id')
    parser.add_argument('--describe',action='store_true')
    args=parser.parse_args();cfg=load_config(args.config)
    source=Path(cfg['source_snapshot']);root=Path(cfg['paths']['runs']).parent
    experiment=os.environ.get('DET_EXPERIMENT_ID','local')
    run_id=args.run_id or f'physical_content_{experiment}'
    run=root/'runs'/run_id;checkpoint=run/'step0400.pt'
    plan=dict(candidate='R16A-NOZERO',revision=cfg['revision'],source=str(source),run=run_id,
              checkpoint=str(checkpoint),steps=400,training_stages=cfg['stages'],
              initialization='official foundations only',evaluation='Separate allowed Ada/Amp job after own400; same40/60 and fixed16')
    if args.describe:print(json.dumps(plan));return
    data_ready(cfg);job=root/'reports/jobs'/experiment;gpu_guard(job);atomic_json(plan,job/'plan.json')
    def stage(name,command):
        atomic_json(dict(status=name,run=run_id,checkpoint=str(checkpoint)),job/'status.json')
        run_stage(command,job/f'{name}.log')
    python=[sys.executable,'-I','-B','-u']
    if not checkpoint.exists():
        if not (run/'latest.pt').exists():
            stage('fresh_model_preflight',[*python,source/'gpu_preflight.py','--config',args.config,'--output',job/'gpu_preflight'])
        command=[*python,source/'train.py','--config',args.config,'--run-id',run_id,'--stop-after','400']
        if (run/'latest.pt').exists():command+=['--resume',run/'latest.pt']
        stage('training_to400',command)
    atomic_json(dict(status='training_completed_evaluation_pending',**plan),job/'status.json')


if __name__=='__main__':main()
