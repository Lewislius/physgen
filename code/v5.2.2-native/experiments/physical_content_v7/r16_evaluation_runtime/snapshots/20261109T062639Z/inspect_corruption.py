"""Read-only VP004 full/no-C replay to locate an observed catastrophic failure."""
from pathlib import Path
import argparse,json,os,subprocess,sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from runtime import activate,stage_command,HERE


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pool',choices=['amp-48g','amp-80g','ada-48g'],required=True)
    parser.add_argument('--describe',action='store_true')
    args=parser.parse_args()
    import yaml
    cfg=yaml.safe_load(args.config.read_text());source=Path(cfg['source_snapshot'])
    plan=dict(scope='read_only_runtime_failure_isolation',case='VP004',modes=['real','no_c'],
        checkpoint=str(args.checkpoint),source=str(source),runtime=str(HERE),pool=args.pool,
        same_inputs_seed_and_guidance=True,new_training=False,replace_primary_scores_or_videos=False,
        reason='Primary AMP80 VP002/VP004 have full-frame stripes; compare same400 real/noC on another allocated Amp/Ada GPU')
    if args.describe:print(json.dumps(plan));return
    guard=activate(source,args.pool)
    folder=args.output.parent/'reports/jobs'/os.getenv('DET_EXPERIMENT_ID','local')
    guard(folder);(folder/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    from benchmark_inputs import PINNED
    from pipeline import run_stage
    command=[sys.executable,'-I','-B','-u',str(source/'infer.py'),'--checkpoint',str(args.checkpoint),
             '--output',str(args.output),'--suite','videophy40','--modes','real','no_c','--cases','VP004']
    run_stage(stage_command(command,source,args.pool),folder/'generation.log')
    (folder/'status.json').write_text(json.dumps(dict(status='completed',**plan),indent=2)+'\n')


if __name__=='__main__':main()
