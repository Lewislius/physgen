"""Same400-weight inference repair: generate/score the original fixed16 once."""
from pathlib import Path
import argparse
import json
import os
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap
import benchmark_inputs
from physical_content.common import gpu_guard, atomic_json
from pipeline import run_stage, evaluator_environment
from allocated_gpu import bind_allocated_gpu
from completed_suite import completion
from report_results import paired, summarize


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--describe',action='store_true')
    args=parser.parse_args()
    source = Path(__file__).resolve().parent
    plan = json.loads((source / 'repair_plan.json').read_text())
    root, output, checkpoint = Path(plan['root']), Path(plan['output']), Path(plan['checkpoint'])
    ids = [case['id'] for case in json.loads((source / 'subset16.json').read_text())['cases']]
    assert len(ids) == len(set(ids)) == 16 and ids == plan['cases']
    assert checkpoint.is_file() and Path(plan['training_source']).is_dir()
    if args.describe:
        print(json.dumps(plan,ensure_ascii=False));return
    folder = root / 'reports/semantic_mass_retained' / os.environ['DET_EXPERIMENT_ID']
    gpu_guard(folder); bind_allocated_gpu(folder, evaluator_environment)
    atomic_json(plan, folder / 'plan.json')
    python = [sys.executable, '-I', '-B', '-u']
    def stage(name, command, environment=None):
        atomic_json(dict(status=name, checkpoint=str(checkpoint), training_updates=0,
                         output=str(output)), folder / 'status.json')
        if environment is None:run_stage(command, folder / (name + '.log'))
        else:run_stage(command, folder / (name + '.log'), environment)
    directory = output / 'videophy40'
    generated, scored = completion(directory, checkpoint, plan['training_source'], ids, 'real', inference_source=source)
    if not generated:
        stage('repair16_generation', [*python, source / 'infer.py', '--checkpoint', checkpoint,
            '--output', output, '--suite', 'videophy40', '--modes', 'real', '--cases', *ids])
    if not scored:
        stage('repair16_scoring', ['/home/liuzhirui/miniconda3/envs/physvid/bin/python', '-I', '-B', '-u',
              bootstrap.BASE / 'tools/evaluate_videophy2.py', directory], evaluator_environment())
    actual = json.loads((directory / 'scores.json').read_text())
    assert {row['id'] for row in actual['cases']} == set(ids) and len(actual['cases']) == 16
    original_root = root / 'inference_outputs/physical_content_67955/step0400_original_fixed40_controls16_seed42'
    paths = dict(R15E_normal=original_root / 'videophy40/scores.json',
        R15E_no_c=original_root / 'controls/no_c/videophy40/scores.json',
        R15E_semantic_routing_off=original_root / 'controls/semantic_routing_off/videophy40/scores.json')
    history = json.loads((benchmark_inputs.ROOT / 'reports/r10_final_evidence.json').read_text())['source_paths']
    paths.update({name:Path(history[name + '/original40']) for name in ['R1', 'R10-A', 'native400']})
    comparisons = {}
    for name, path in paths.items():
        rows = [row for row in json.loads(path.read_text())['cases'] if row['id'] in ids]
        assert len(rows) == 16
        comparisons[name] = dict(reference=summarize(rows), repair_minus_reference=paired(actual, dict(cases=rows)), source=str(path))
    report = dict(variant='R15E posterior-C-mass-retained inference intervention',
        checkpoint=str(checkpoint), checkpoint_step=400, training_updates=0,
        source=str(source), original_training_source=plan['training_source'],
        fixed16=summarize(actual['cases']), comparisons=comparisons,
        selection='Original fixed16. Repair designed after observing R15E40 failures and paired controls; this is diagnostic, not an unseen confirmation set.',
        meaning='Uses the original physical C/reference posterior allocation, then changes only within-C semantic probabilities. All trained weights, independent physical Q, native text computation, request refinement, input images, reasoning, seed and gamma5 stay fixed.',
        quality_goal_achieved=False,
        pending='Inspect actual generated processes and artifacts. This16 result cannot replace full40/60 or cross-benchmark acceptance and is not a newly trained R15F/G model.')
    atomic_json(report, output / 'comparison16.json')
    stage('repair16_visual_panels', [*python, source / 'data_tools/prepare_visual_review.py',
          '--output', output, '--report', output / 'visual_predeclared'])
    atomic_json(dict(status='completed', checkpoint=str(checkpoint), training_updates=0, cases=16,
                     report=str(output / 'comparison16.json')), folder / 'status.json')
    print(json.dumps(dict(kind='repair16_complete', scores=report['fixed16'], report=str(output / 'comparison16.json'))), flush=True)


if __name__ == '__main__':
    main()
