"""Generate the last16 of an existing full40 suite and publish completed cases for reuse."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

BASE = Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native')
ROOT = BASE / 'experiments/physical_content_v7'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=['native_coordinate_r7', 'content_reader_r8'], required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--describe', action='store_true')
    args = parser.parse_args()
    variant = ROOT / args.variant
    run = variant / 'runs' / args.run
    checkpoint = run / 'step0128.pt'
    primary = variant / 'inference_outputs' / args.run / 'step0128_full40_controls16_physical5_seed42/videophy40'
    reference = ROOT / 'inference_outputs/physical_content_66778/step0400_seed42_r1/videophy40/manifest.json'
    cases = json.loads(reference.read_text())['cases']
    if len(cases) != 40 or len({case['id'] for case in cases}) != 40:
        raise ValueError('Use the existing complete fixed40, with its original ordering')
    selected = [case['id'] for case in cases[24:]]
    output = variant / 'generation_dispatch/inference_outputs' / args.run / 'tail16_gamma5_seed42'
    command = [sys.executable, '-I', '-B', '-u', args.source / 'infer.py',
        '--checkpoint', checkpoint, '--output', output, '--suite', 'videophy40',
        '--physical-guidance', '5', '--seed', '42', '--modes', 'real', '--cases', *selected]
    status_file = run / 'status.json'
    status = json.loads(status_file.read_text()) if status_file.exists() else {}
    ready = checkpoint.is_file() and status.get('status') == 'paused' and status.get('step') == 128
    plan = dict(training_run=args.run, source=str(args.source), checkpoint=str(checkpoint),
        selected_cases=selected, ready=ready, generation_command=list(map(str, command)),
        primary=str(primary), isolated_output=str(output), additional_cases=0, additional_seeds=0,
        evaluation='The original full40 owner scores all40 once, after reusing these completed cases',
        mean_files='Not required: normal generation does not consume the mean controls')
    if args.describe:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    if not ready:
        raise RuntimeError('Submit after this candidate has saved and paused at128')
    sys.path.insert(0, str(args.source))
    import bootstrap
    from physical_content.common import atomic_json, gpu_guard
    job = variant / 'generation_dispatch/reports/jobs' / os.getenv('DET_EXPERIMENT_ID', 'local')
    job.mkdir(parents=True, exist_ok=True)
    atomic_json(plan, job / 'plan.json')
    selected = [case for case in selected if not (primary / 'cases' / case).exists()]
    if not selected:
        atomic_json(dict(status='left_to_original_owner', published=[]), job / 'status.json')
        return
    command = command[:command.index('--cases') + 1] + selected
    gpu_guard(job)
    pinned = BASE / 'experiments/physical_scaling_q256/inference_snapshots/20261030T034046Z'
    sys.path.insert(0, str(pinned))
    from pipeline import run_stage
    run_stage(command, job / 'generation.log')
    generated = output / 'videophy40'
    plan_generated = json.loads((generated / 'manifest.json').read_text())
    if {case['id'] for case in plan_generated['cases']} != set(selected):
        raise ValueError('Generated case list differs from the declared remaining tail')
    (primary / 'cases').mkdir(parents=True, exist_ok=True)
    published, owned = [], []
    for case in selected:
        source = generated / 'cases' / case
        metadata = json.loads((source / 'planned.json').read_text())
        expected = dict(checkpoint=str(checkpoint), step=128, mode='real', case=case, seed=42)
        if not (source / 'planned.mp4').is_file() or any(metadata.get(k) != v for k, v in expected.items()):
            raise ValueError(f'Incomplete or mismatched completed output: {case}')
        try:
            (primary / 'cases' / case).symlink_to(source, target_is_directory=True)
            published.append(case)
        except FileExistsError:
            owned.append(case)
    result = dict(status='completed', recorded_utc=datetime.now(timezone.utc).isoformat(),
        published=published, left_to_original_owner=owned, primary=str(primary),
        additional_cases=0, additional_seeds=0, scoring='Original complete40 pipeline only')
    atomic_json(result, job / 'status.json')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
