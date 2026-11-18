"""Same R16A400 reader/weights; intervene only on the physical write formula.

This is a mechanism check, not a trained R16B evaluation.
"""
from pathlib import Path
import argparse,json,os,runpy,sys

RUNTIME=Path(__file__).resolve().parents[2]/'r16_evaluation_runtime/snapshots/20261109T060601Z'
sys.path.insert(0,str(RUNTIME))
from runtime import activate


def install(model):
    from physical_content import joint_read
    state={}
    old=joint_read.read_physical
    def read(*args,**kwargs):
        out=old(*args,**kwargs)
        state['message']=out.transpose(1,2).flatten(2)
        return out
    joint_read.read_physical=read
    for fusion in model.fusions.values():
        def save_state(module,args,out):state['interaction']=out
        fusion.state_value.register_forward_hook(save_state)
        def change_write(module,args):
            m=state.pop('message').float();s=state.pop('interaction').float()
            result=(m-m.mean(1,True))+(s*m-s.mean(1,True)*m.mean(1,True))
            return (result,)
        fusion.output.register_forward_pre_hook(change_write)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pool',choices=['ada-48g','amp-48g','amp-80g'],required=True)
    args=p.parse_args()
    import yaml
    cfg=yaml.safe_load(args.config.read_text());source=Path(cfg['source_snapshot'])
    guard=activate(source,args.pool)
    folder=args.output.parent/'jobs'/os.getenv('DET_EXPERIMENT_ID','local');guard(folder)
    plan=dict(scope='R16A400 writer-only intervention; NOT native R16B',cases=['VP002','VP004','VP009'],
              checkpoint=str(args.checkpoint),same_reader_weights_inputs_and_sampling=True,
              no_CFG_change=True,no_training=True,primary_outputs_untouched=True)
    (folder/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    import guidance
    old=guidance.guided_velocity;installed=False
    def initialize(model,*a,**k):
        nonlocal installed
        if not installed:install(model);installed=True
        return old(model,*a,**k)
    guidance.guided_velocity=initialize
    sys.argv=[str(source/'infer.py'),'--checkpoint',str(args.checkpoint),'--output',str(args.output),
              '--suite','videophy40','--modes','real','--cases',*plan['cases']]
    runpy.run_path(str(source/'infer.py'),run_name='__main__')
    (folder/'status.json').write_text(json.dumps(dict(status='completed',**plan),indent=2)+'\n')


if __name__=='__main__':main()
