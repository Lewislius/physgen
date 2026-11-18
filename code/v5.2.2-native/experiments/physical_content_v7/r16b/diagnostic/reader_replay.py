"""R16A400 full proposed joint reader, with absolute or grounded writer."""
from pathlib import Path
import argparse,importlib.util,json,os,runpy,sys
HERE=Path(__file__).resolve().parent
RUNTIME=HERE.parents[1]/'r16_evaluation_runtime/snapshots/20261109T060601Z'
sys.path[:0]=[str(HERE),str(RUNTIME)]
from runtime import activate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--writer',choices=['absolute','grounded'],required=True)
    p.add_argument('--pool',choices=['ada-48g','amp-48g','amp-80g'],required=True)
    args=p.parse_args()
    import yaml
    cfg=yaml.safe_load(args.config.read_text());source=Path(cfg['source_snapshot']);guard=activate(source,args.pool)
    folder=args.output/'job';guard(folder)
    from physical_content import joint_read
    code=HERE.parent/'snapshots/20261109T074603Z/physical_content/joint_read.py'
    spec=importlib.util.spec_from_file_location('r16b_proposed_reader',code);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    joint_read.read_physical=module.read_physical
    if args.writer=='grounded':
        from grounded_replay import install
        import guidance
        old=guidance.guided_velocity;installed=False
        def initialize(model,*a,**k):
            nonlocal installed
            if not installed:install(model);installed=True
            return old(model,*a,**k)
        guidance.guided_velocity=initialize
    plan=dict(scope='R16A400 reader intervention; not trained R16B',reader='joint_direct_and_request_C_native_request_query',
              writer=args.writer,cases=['VP002','VP004','VP009'],checkpoint=str(args.checkpoint),CFG_changed=False)
    (folder/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    sys.argv=[str(source/'infer.py'),'--checkpoint',str(args.checkpoint),'--output',str(args.output),
              '--suite','videophy40','--modes','real','--cases',*plan['cases']]
    runpy.run_path(str(source/'infer.py'),run_name='__main__')
    (folder/'status.json').write_text(json.dumps(dict(status='completed',**plan),indent=2)+'\n')


if __name__=='__main__':main()
