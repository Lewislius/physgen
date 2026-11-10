"""One paired16 no-reference intervention after formal R16A evaluations finish."""
from pathlib import Path
import argparse, json, os, sys

HERE = Path(__file__).resolve().parent
TRAINING_SOURCE = Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_v7/native_joint_read_r16a/snapshots/20261109T040426Z')
sys.path[:0] = [str(TRAINING_SOURCE), str(HERE)]
import bootstrap
from physical_content.common import atomic_json, gpu_guard
from pipeline import run_stage, evaluator_environment
from allocated_gpu import bind_allocated_gpu
from report_results import paired, summarize
from reference_completion import completion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--main-output', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--describe', action='store_true')
    args = parser.parse_args()
    ids = [v['id'] for v in json.loads((TRAINING_SOURCE/'subset16.json').read_text())['cases']]
    plan = dict(candidate='R16A', intervention='no_reference', cases=ids,
                checkpoint=str(args.checkpoint), training_source=str(TRAINING_SOURCE),
                inference_source=str(HERE), output=str(args.output),
                normal40=str(args.main_output/'videophy40/scores.json'),
                same_seed=42, gamma=5, new_training=False, new_api_calls=0,
                interpretation='Same400 weights with reference option removed in both CFG arms; not a retrained no-reference model.')
    if args.describe:
        print(json.dumps(plan)); return
    if not args.checkpoint.is_file(): raise FileNotFoundError(args.checkpoint)
    # This supplement must follow the requested complete normal40/60 evaluation.
    for suite in ['videophy40', 'full60/videophy60']:
        if not (args.main_output/suite/'scores.json').is_file():
            raise FileNotFoundError(args.main_output/suite/'scores.json')
    job = args.output.parent/'reports/jobs'/os.getenv('DET_EXPERIMENT_ID','local')
    gpu_guard(job); bind_allocated_gpu(job, evaluator_environment)
    atomic_json(plan, job/'plan.json')
    suite = args.output/'controls/no_reference/videophy40'
    generated, scored = completion(suite,args.checkpoint,TRAINING_SOURCE,ids,'no_reference',inference_source=HERE)
    if not generated:
        atomic_json(dict(status='no_reference16_generation'),job/'status.json')
        run_stage([sys.executable,'-I','-B','-u',HERE/'infer.py','--checkpoint',args.checkpoint,
                   '--output',args.output,'--suite','videophy40','--modes','no_reference','--cases',*ids],
                  job/'generation.log')
    if not scored:
        atomic_json(dict(status='no_reference16_scoring'),job/'status.json')
        run_stage(['/home/liuzhirui/miniconda3/envs/physvid/bin/python','-I','-B','-u',
                   bootstrap.BASE/'tools/evaluate_videophy2.py',suite],job/'scoring.log',evaluator_environment())
    control=json.loads((suite/'scores.json').read_text())
    normal=json.loads((args.main_output/'videophy40/scores.json').read_text())
    normal=dict(cases=[r for r in normal['cases'] if r['id'] in ids])
    if len(normal['cases'])!=16 or {r['id'] for r in control['cases']}!=set(ids):
        raise ValueError('Expected the same fixed16 cases')
    report=dict(**plan,normal=summarize(normal['cases']),no_reference=summarize(control['cases']),
                no_reference_minus_normal=paired(control,normal),
                scores=str(suite/'scores.json'),quality_proven=False,
                pending='Inspect real videos and compare with other physical controls before judging usefulness.')
    atomic_json(report,args.output.parent/'reports/comparison.json')
    atomic_json(dict(status='completed',report=str(args.output.parent/'reports/comparison.json')),job/'status.json')
    print(json.dumps({k:report[k] for k in ['normal','no_reference','no_reference_minus_normal']}))


if __name__=='__main__': main()
