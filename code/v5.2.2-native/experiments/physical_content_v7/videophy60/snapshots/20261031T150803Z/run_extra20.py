"""Generate only the added20, score once, then aggregate with the completed original40."""
from pathlib import Path
import argparse
import json
import os
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap

ROOT=bootstrap.BASE/'experiments/physical_content_v7'
HERE=ROOT/'videophy60'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--reference40',type=Path,required=True)
    parser.add_argument('--name',required=True)
    parser.add_argument('--describe',action='store_true')
    args=parser.parse_args()
    args.checkpoint=args.checkpoint.resolve();args.reference40=args.reference40.resolve()
    if '/' in args.name or args.name in ['.','..']:raise ValueError('Use a single output name')
    plan=json.loads((HERE/'protocol.json').read_text())
    cases=plan['extra_ids']
    source=Path(__file__).resolve().parent
    output=HERE/'inference_outputs'/args.name/'extra20_gamma5_seed42'
    merged=HERE/'inference_outputs'/args.name/'full60_gamma5_seed42/videophy60'
    generation=[sys.executable,'-I','-B','-u',source/'infer.py','--checkpoint',args.checkpoint,
        '--output',output,'--suite','videophy60','--seed','42','--physical-guidance','5',
        '--modes','real','--input-manifest',HERE/'protocol.json','--evidence-root',HERE/'inputs',
        '--cases',*cases]
    scoring=['/home/liuzhirui/miniconda3/envs/physvid/bin/python','-I','-B','-u',
        bootstrap.BASE/'tools/evaluate_videophy2.py',output/'videophy60']
    record=dict(checkpoint=str(args.checkpoint),completed_part=str(args.reference40),
        additional_part=str(output/'videophy60'),merge_target=str(merged),extra_cases=cases,
        generation_command=list(map(str,generation)),evaluation_command=list(map(str,scoring)),
        cases_after_merge=60,new_videos=20,new_training=False,seed=42,gamma=5)
    if args.describe:
        print(json.dumps(record,ensure_ascii=False));return
    completed=json.loads((args.reference40/'scores.json').read_text())
    if completed['summary']['count']!=40:raise ValueError('Preserve the completed original40 first')
    manifest=json.loads((args.reference40/'manifest.json').read_text())
    if Path(manifest['checkpoint']).resolve()!=args.checkpoint.resolve():
        raise ValueError('The extra20 must use the same checkpoint as its original40')
    if manifest['variant']=='physical_content':
        prep=json.loads((HERE/'preparation_status.json').read_text())
        if prep['status']!='completed' or prep['completed']!=cases:
            raise ValueError('Prepare all shared input-only physical conditions first')
    folder=HERE/'reports'/args.name;folder.mkdir(parents=True,exist_ok=True)
    record_path=folder/'merge_record.json'
    record_path.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    pinned=bootstrap.BASE/'experiments/physical_scaling_q256/inference_snapshots/20261030T034046Z'
    sys.path.insert(0,str(pinned))
    from pipeline import run_stage,evaluator_environment
    run_stage(generation,folder/'generation.log')
    run_stage(scoring,folder/'evaluation.log',evaluator_environment())
    run_stage([sys.executable,'-I','-B',source/'merge_completed60.py',record_path],folder/'merge.log')
    scores=json.loads((merged/'scores.json').read_text())
    (folder/'status.json').write_text(json.dumps(dict(status='completed',summary=scores['summary'],
        original40=str(args.reference40),extra20=str(output/'videophy60'),full60=str(merged)))+'\n')


if __name__=='__main__':main()
