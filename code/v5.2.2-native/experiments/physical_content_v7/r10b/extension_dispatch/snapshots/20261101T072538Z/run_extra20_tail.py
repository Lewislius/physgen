"""Generate the last10 of the declared added20 and publish completed cases for its original owner."""
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
    parser.add_argument('--variant', choices=['grounded_execution_r10b'], required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--describe', action='store_true')
    args = parser.parse_args()
    variant = ROOT / args.variant
    run = variant / 'runs' / args.run
    checkpoint = run / 'step0400.pt'
    primary = ROOT / 'videophy60/inference_outputs' / ('R10B_' + args.run) / 'extra20_gamma5_seed42/videophy60'
    reference = ROOT / 'videophy60/protocol.json'
    protocol = json.loads(reference.read_text())
    extra_ids = protocol['extra_ids']
    if extra_ids != [f'VP{i:03d}' for i in range(41, 61)]:
        raise ValueError('Retain the existing declared extra20 in its original ordering')
    selected = extra_ids[10:]
    output = variant / 'extension_dispatch/inference_outputs' / args.run / 'tail10_gamma5_seed42'
    command = [sys.executable, '-I', '-B', '-u', ROOT / 'videophy60/snapshots/20261031T150803Z/infer.py',
        '--checkpoint', checkpoint, '--output', output, '--suite', 'videophy60',
        '--input-manifest', reference, '--evidence-root', ROOT / 'videophy60/inputs',
        '--physical-guidance', '5', '--seed', '42', '--modes', 'real', '--cases', *selected]
    status_file = run / 'status.json'
    status = json.loads(status_file.read_text()) if status_file.exists() else {}
    ready = checkpoint.is_file() and status.get('status') == 'complete' and status.get('step') == 400
    plan = dict(training_run=args.run, source=str(args.source), checkpoint=str(checkpoint),
        selected_cases=selected, ready=ready, generation_command=list(map(str, command)),
        input_manifest=str(reference), model_source=str(args.source), inference_source=str(ROOT / 'videophy60/snapshots/20261031T150803Z/infer.py'),
        primary=str(primary), isolated_output=str(output), additional_cases=0, additional_seeds=0,
        evaluation='The original extension owner scores the full added20 once and merges its original40 into60',
        mean_files='Not required: normal generation does not consume the mean controls')
    if args.describe:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    if not ready:
        raise RuntimeError('Submit after this candidate has completed and saved at400')
    sys.path.insert(0, str(args.source))
    import bootstrap
    from physical_content.common import atomic_json, gpu_guard
    job = variant / 'extension_dispatch/reports/jobs' / os.getenv('DET_EXPERIMENT_ID', 'local')
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
    generated = output / 'videophy60'
    plan_generated = json.loads((generated / 'manifest.json').read_text())
    if {case['id'] for case in plan_generated['cases']} != set(selected):
        raise ValueError('Generated case list differs from the declared remaining tail')
    (primary / 'cases').mkdir(parents=True, exist_ok=True)
    published, owned = [], []
    for case in selected:
        source = generated / 'cases' / case
        metadata = json.loads((source / 'planned.json').read_text())
        expected = dict(checkpoint=str(checkpoint), step=400, mode='real', case=case, seed=42)
        if not (source / 'planned.mp4').is_file() or any(metadata.get(k) != v for k, v in expected.items()):
            raise ValueError(f'Incomplete or mismatched completed extra20 output: {case}')
        try:
            (primary / 'cases' / case).symlink_to(source, target_is_directory=True)
            published.append(case)
        except FileExistsError:
            owned.append(case)
    result = dict(status='completed', recorded_utc=datetime.now(timezone.utc).isoformat(),
        published=published, left_to_original_owner=owned, primary=str(primary),
        additional_cases=0, additional_seeds=0, scoring='Original added20 evaluation and merged60 pipeline only')
    atomic_json(result, job / 'status.json')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
