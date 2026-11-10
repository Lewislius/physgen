"""Generate paired videos, then run the unchanged independent evaluator."""
from pathlib import Path
import sys,argparse,json,subprocess
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--modes',nargs='+',default=['real']);parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--suite',choices=['videophy40','heldout8'],default='videophy40')
    parser.add_argument('--physical-guidance',type=float,required=True)
    parser.add_argument('--global-condition',type=Path)
    parser.add_argument('--request-evidence',type=Path)
    parser.add_argument('--cases',nargs='*');args=parser.parse_args()
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    pinned=bootstrap.BASE/'experiments/physical_scaling_q256/inference_snapshots/20261030T034046Z'
    sys.path.insert(0,str(pinned))
    from pipeline import evaluator_environment,run_stage
    command=[sys.executable,'-I','-B','-u',Path(__file__).with_name('infer.py'),'--checkpoint',args.checkpoint,
        '--output',args.output,'--seed',str(args.seed),'--suite',args.suite,
        '--physical-guidance',str(args.physical_guidance),'--modes',*args.modes]
    if args.cases:command+=['--cases',*args.cases]
    if args.global_condition:command+=['--global-condition',str(args.global_condition)]
    if args.request_evidence:command+=['--request-evidence',str(args.request_evidence)]
    run_stage(command,root/'generation.log')
    folders=[root/args.suite if mode=='real' else root/'controls'/mode/args.suite for mode in args.modes]
    run_stage(['/home/liuzhirui/miniconda3/envs/physvid/bin/python','-I','-B','-u',
        bootstrap.BASE/'tools/evaluate_videophy2.py',*folders],root/'evaluation.log',evaluator_environment())
    (root/'pipeline_status.json').write_text(json.dumps(dict(status='completed',checkpoint=args.checkpoint,
        modes=args.modes,physical_guidance=args.physical_guidance)))

if __name__=='__main__':main()
