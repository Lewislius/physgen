"""Three known failure cases with this run's own128 checkpoint; visual evidence only."""
from pathlib import Path
import argparse,json,os,sys
ROOT=Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_v7')
RUNTIME=ROOT/'r16_evaluation_runtime/snapshots/20261109T060601Z'
sys.path.insert(0,str(RUNTIME))
from runtime import activate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--pool',choices=['amp-48g','amp-80g','ada-48g'],required=True)
    p.add_argument('--describe',action='store_true');args=p.parse_args()
    import yaml
    cfg=yaml.safe_load(args.config.read_text());source=Path(cfg['source_snapshot'])
    plan=dict(scope='Own R16B128 preliminary visual check; not final400 scores',cases=['VP002','VP004','VP009'],
        checkpoint=str(args.checkpoint),source=str(source),new_training_steps=0,protocol_unchanged=True,
        model_and_generation_unchanged=True,only_guard_change='Allow explicit checkpoint128 instead of400',primary_outputs_untouched=True)
    if args.describe:print(json.dumps(plan));return
    if not args.checkpoint.is_file():raise FileNotFoundError('Submit only after own128 checkpoint exists')
    guard=activate(source,args.pool);job=args.output.parent/'jobs'/os.getenv('DET_EXPERIMENT_ID','local');guard(job)
    (job/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    script=source/'infer.py';code=script.read_text();old="if saved['revision']!=REVISION or step!=400:raise ValueError('Use this candidate own400 weights')"
    if code.count(old)!=1:raise ValueError('Inference guard changed; inspect the frozen source')
    code=code.replace(old,"if saved['revision']!=REVISION or step!=128:raise ValueError('Use this candidate own128 weights')")
    sys.argv=[str(script),'--checkpoint',str(args.checkpoint),'--output',str(args.output),'--suite','videophy40',
              '--modes','real','--cases',*plan['cases']]
    exec(compile(code,str(script),'exec'),dict(__name__='__main__',__file__=str(script)))
    (job/'status.json').write_text(json.dumps(dict(status='completed',**plan),indent=2)+'\n')


if __name__=='__main__':main()
