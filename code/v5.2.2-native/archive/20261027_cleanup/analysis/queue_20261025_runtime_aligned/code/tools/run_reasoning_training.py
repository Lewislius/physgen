"""One real integration check, then actual training and a focused C comparison.

No long gradient-direction matrix or best-of-N generation. The main run uses
the final three-stage configuration and pauses at a resumable optimizer step.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from physgen_native.config import read_config, smoke_config


def check_integration(run):
    records = [json.loads(x) for x in (run/'metrics.jsonl').read_text().splitlines()]
    updates = [x for x in records if x['kind']=='train']
    if [r['phase'] for r in updates] != ['interface','calibration','joint']:
        raise RuntimeError('All three actual model stages must finish before training')
    for row in updates:
        expected = ({'critic'} if row['phase']=='calibration' else
                    {'condition','interface','knowledge'} | ({'wan_lora'} if row['phase']=='joint' else set()))
        for group,norm in row['gradients'].items():
            if (norm > 0) != (group in expected):
                raise RuntimeError(f'Actual gradient ownership differs in {row["phase"]}: {group}={norm}')
    feedback = [x for x in records if x['kind']=='completion_feedback']
    if len(feedback)!=1 or not feedback[0].get('C_gradient_rms',0)>0:
        raise RuntimeError('Actual frozen-verifier derivative must reach C')
    return dict(updates=updates,feedback=feedback,scope='execution and gradient ownership; not quality evidence')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--train-steps',type=int,default=128)
    args=parser.parse_args()
    if args.train_steps < 1: raise ValueError('A positive actual training budget is required')
    cfg=read_config(args.config)
    suffix=os.environ.get('DET_EXPERIMENT_ID')
    if not suffix: raise RuntimeError('This pipeline requires its allocated Determined experiment ID')
    smoke=smoke_config(cfg);smoke['run_name']+='-accum8'
    smoke_run=Path(cfg['paths']['runs'])/f'{smoke["run_name"]}_{suffix}'
    run=Path(cfg['paths']['runs'])/f'{cfg["run_name"]}_{suffix}'
    output=args.output;output.mkdir(parents=True,exist_ok=True)
    records=[]
    def execute(name,argv):
        item=dict(name=name,state='running',started_utc=datetime.now(timezone.utc).isoformat(),command=argv)
        records.append(item)
        (output/'pipeline.json').write_text(json.dumps(records,indent=2)+'\n')
        begin=time.monotonic()
        print(f'[{name}] starting',flush=True)
        with (output/f'{name}.stdout.log').open('w') as stream:
            child=subprocess.Popen([sys.executable,'-I','-B','-u',*argv],stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in child.stdout:
                stream.write(line);stream.flush()
                try: row=json.loads(line)
                except (ValueError,TypeError): print(line,end='',flush=True);continue
                if isinstance(row,dict) and 'kind' in row:
                    fields=('kind','step','phase','micro','sample','stage','event','allocated_gib','seconds','loss','loss/fm','loss/process','loss/use',
                            'condition/process_margin','condition/output_rms','peak_gib','C_gradient_rms','path')
                    print(json.dumps({k:row[k] for k in fields if k in row}),flush=True)
                else: print(line,end='',flush=True)
            returncode=child.wait()
        item.update(state='complete' if returncode==0 else 'failed',returncode=returncode,
                    terminating_signal=-returncode if returncode < 0 else None, seconds=time.monotonic()-begin)
        (output/'pipeline.json').write_text(json.dumps(records,indent=2)+'\n')
        print(json.dumps(item),flush=True)
        if returncode: raise SystemExit(returncode if returncode > 0 else 128-returncode)
    train=str(ROOT/'train/train.py')
    execute('real_three_stages',[train,'--config',str(args.config),'--smoke','--smoke-accumulation','8'])
    integration=check_integration(smoke_run)
    (output/'integration.json').write_text(json.dumps(integration,indent=2)+'\n')
    execute('single_trajectory_feedback',[str(ROOT/'inference/infer_heldout.py'),
        '--checkpoint',str(smoke_run/'latest.pt'),'--output',str(output/'engineering_video'),
        '--samples','1','--refine-c'])
    videos=list((output/'engineering_video').glob('*.json'))
    if len(videos)!=1: raise RuntimeError('Integration check must produce one video, not candidate selection')
    feedback=json.loads(videos[0].read_text())['latent_feedback']
    if feedback['complete_trajectories']!=1 or not all(x['gradient_rms']>0 for x in feedback['updates']):
        raise RuntimeError('Single-trajectory C feedback did not execute')
    execute('train_prefix',[train,'--config',str(args.config),'--stop-after',str(args.train_steps)])
    execute('condition_effect',[str(ROOT/'tools/evaluate_condition_use.py'),
        '--checkpoint',str(run/'latest.pt'),'--output',str(output/'condition_effect'),
        '--case-ids','source0000042','source0000086','source0000107','source0000128',
        '--seeds','42','314159','--full-samples','4','--sampling-seeds','42'])
    print(json.dumps(dict(state='prefix_and_comparison_complete',checkpoint=str(run/'latest.pt'),
        configured_stages=cfg['stages'],completed_optimizer_updates=args.train_steps,
        quality_claim='Requires paired analysis and actual video review; no automatic success label')),flush=True)


if __name__=='__main__':main()
