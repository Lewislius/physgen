"""Sequential blk-96g verifier calibration, real training check and C intervention."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--training-config',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1];project=args.project_root
    args.output.mkdir(parents=True,exist_ok=True)
    checkpoint=project/'checkpoints/v522-diagnostic-reasoned_process-8cases32updates-seed42_65836/latest.pt'
    study=args.output/'calibration';verifier=study/'verifier_regression_seed42.pt'
    jobs=[('calibration',[str(root/'tools/calibrate_process_verifier.py'),
        '--source',str(project/'analysis/pretrained_video_evidence'),
        '--generated-root',str(project/'analysis/temporal_distribution/videos'),
        '--output',str(study),'--steps','512']),
        ('real_training_check',[str(root/'train/train.py'),'--config',str(args.training_config),
         '--initialize-generator',str(checkpoint),'--initialize-critic',str(verifier),'--stop-after','4']),
        ('learned_verifier_c_feedback',[str(root/'tools/check_temporal_feedback_split.py'),
         '--checkpoint',str(checkpoint),'--verifier',str(verifier),
         '--output',str(args.output/'learned_verifier_c_feedback'),
         '--cases','0','1','2','3','4','--seeds','42','314159','--gradient-steps','50',
         '--discrete-adjoint','--single-device','--condition-controls'])]
    report=[]
    for name,arguments in jobs:
        command=[sys.executable,'-I','-B','-u',*arguments]
        item=dict(stage=name,command=command,state='running',started_unix=time.time())
        report.append(item);(args.output/'pipeline.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(dict(stage=name,state='running')),flush=True)
        with (args.output/f'{name}.stdout.log').open('w') as stream:
            result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
        item.update(returncode=result.returncode,state='complete' if not result.returncode else 'failed',
                    seconds=time.time()-item['started_unix'])
        (args.output/'pipeline.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(item),flush=True)
        if name=='calibration' and result.returncode:break
        # C feedback uses the same pre-feedback generator as the oracle controls;
        # it remains informative even if the independent training smoke fails.
    if len(report)!=len(jobs) or any(x.get('returncode') for x in report):raise SystemExit(1)


if __name__=='__main__':main()
