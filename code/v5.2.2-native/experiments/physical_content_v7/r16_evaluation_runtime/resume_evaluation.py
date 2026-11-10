"""Resume a completed own400 R16 candidate on an allowed Ada/Amp GPU."""
from pathlib import Path
import argparse,importlib.util,json,os,sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from runtime import activate,stage_command,HERE


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--pool',choices=['ada-48g','amp-48g','amp-80g'],required=True)
    parser.add_argument('--describe',action='store_true')
    args=parser.parse_args()
    import yaml
    cfg=yaml.safe_load(args.config.read_text());source=Path(cfg['source_snapshot'])
    run=Path(cfg['paths']['runs'])/args.run_id;checkpoint=run/'step0400.pt'
    status=json.loads((run/'status.json').read_text())
    if status.get('status')!='complete' or status.get('step')!=400 or not checkpoint.is_file():
        raise ValueError('A completed own400 training checkpoint is required; no replacement training')
    guard=activate(source,args.pool)
    job=Path(cfg['paths']['runs']).parent/'reports/jobs'/os.getenv('DET_EXPERIMENT_ID','local')
    if not args.describe:guard(job)
    import torch
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    if saved['step']!=400 or saved['cursor']!=1600 or saved['config']!=cfg:
        raise ValueError('Checkpoint/config does not match the requested native400 model')
    del saved
    plan=dict(kind='evaluation_only_recovery',source=str(source),checkpoint=str(checkpoint),run=args.run_id,
              runtime_source=str(HERE),pool=args.pool,new_training_steps=0,main40=True,full60=True,
              paired16=cfg['evaluation_budget']['control_modes'],reuse_completed_videos=True,
              CFG_fix='Forward shared_physical_query to both velocity calls; defaultFalse preserves intended native reader',
              model_architecture_changed=False,weights_changed=False)
    if args.describe:print(json.dumps(plan));return
    (job/'evaluation_runtime.json').write_text(json.dumps(plan,indent=2)+'\n')
    spec=importlib.util.spec_from_file_location('r16_original_evaluation',source/'run_candidate.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    original=module.run_stage
    def run_stage(command,logfile,env=None):
        return original(stage_command(command,source,args.pool),logfile,env)
    module.run_stage=run_stage
    sys.argv=[str(source/'run_candidate.py'),'--config',str(args.config),'--run-id',args.run_id,'--include-extra20']
    module.main()


if __name__=='__main__':main()
