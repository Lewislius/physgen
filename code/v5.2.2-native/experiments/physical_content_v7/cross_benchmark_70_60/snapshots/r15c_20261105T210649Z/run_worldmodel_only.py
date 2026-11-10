"""Score the three complete WMB70 sets on an authorized separate A100/A800.

Existing generation jobs continue their Physics-IQ videos. Helper results remain
isolated while scoring; only a complete result may be handed to the primary job.
"""
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, BENCH, plan, save, gpu_guard
from run import evaluator_environment


def primary_generation_active(model):
    status = json.loads((ROOT / 'jobs' / model / 'status.json').read_text())
    return status.get('status') == 'running' and status.get('stage') == 'generate'


def publish_complete(source, destination):
    # Publish an independent complete copy, never a hard link to the preserved
    # helper evidence. An original scorer cannot mutate that evidence.
    handle, name = tempfile.mkstemp(prefix='.complete_wmb_', dir=destination.parent)
    os.close(handle)
    temporary = Path(name)
    try:
        shutil.copyfile(source, temporary)
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    models = tuple(plan()['models'])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=models, default=list(models))
    parser.add_argument('--describe', action='store_true')
    args = parser.parse_args()
    if len(set(args.models)) != len(args.models):
        raise ValueError('Do not score the same model twice')
    cases = [c for c in plan()['cases'] if c['benchmark'] == 'worldmodelbench70']
    expected = {c['output_filename'] for c in cases}
    if len(cases) != 70 or len(expected) != 70:
        raise ValueError('Use the exact fixed70 protocol')
    spec = dict(models=args.models, cases_per_model=70, resource_pool='amp-80g',
                new_generated_videos=0, model_downloads=0, training_steps=0,
                judge=str(BENCH / 'WorldModelBench/vila-ewm-qwen2-1.5b'),
                prerequisite='All requested models have the complete fixed70 videos',
                reuse='Publish complete official results atomically; preserve any result already owned by the primary job')
    if args.describe:
        print(json.dumps(spec, ensure_ascii=False, indent=2))
        return
    folder = ROOT / 'worldmodel_helpers' / os.getenv('DET_EXPERIMENT_ID', 'local')
    save(folder / 'plan.json', spec)
    for model in args.models:
        available = {p.name for p in (ROOT / 'outputs' / model / 'seed42/worldmodelbench70').glob('*.mp4')}
        if available != expected:
            save(folder / 'status.json', dict(status='not_ready_leave_to_primary', model=model,
                                             available=len(available), expected=70))
            return
    gpu_guard(args.models[0], 'amp-80g')
    source = Path(__file__).resolve().parent
    completed = []
    handed_off = []
    for model in args.models:
        primary = ROOT / 'outputs' / model / 'seed42'
        if not primary_generation_active(model) or (primary / 'worldmodelbench_official.json').exists():
            # The official evaluator does not resume a partial result. Do not
            # overlap the original scoring owner or overwrite its partial work.
            completed.append(dict(model=model, status='leave_to_primary'))
            continue
        result_dir = folder / model
        save(folder / 'status.json', dict(status='scoring', model=model, completed=completed,
                                         time=time.time()))
        command = [str(BENCH / '.venv-vila/bin/python'), '-I', '-B', '-u',
                   str(source / 'evaluate_worldmodel.py'), '--model', model,
                   '--resource-pool', 'amp-80g', '--result-dir', str(result_dir)]
        with (folder / f'{model}.log').open('a') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                           env=evaluator_environment(), check=True)
        summary = json.loads((result_dir / 'worldmodelbench_summary.json').read_text())
        if summary['overall']['count'] != 70:
            raise ValueError('Incomplete official WMB70 summary')
        if not primary_generation_active(model):
            completed.append(dict(model=model, status='complete_isolated_primary_finished_generation'))
            continue
        try:
            publish_complete(result_dir / 'worldmodelbench_official.json',
                             primary / 'worldmodelbench_official.json')
        except FileExistsError:
            completed.append(dict(model=model, status='complete_isolated_primary_already_scoring'))
        else:
            try:
                publish_complete(result_dir / 'worldmodelbench_summary.json',
                                 primary / 'worldmodelbench_summary.json')
            except FileExistsError:
                pass
            handed_off.append(model)
            completed.append(dict(model=model, status='complete_handed_off'))
    save(folder / 'status.json', dict(status='completed', models=completed,
                                     handed_off=handed_off, time=time.time()))


if __name__ == '__main__':
    main()
