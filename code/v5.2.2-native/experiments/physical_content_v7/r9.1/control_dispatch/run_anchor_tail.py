"""Generate the last six of the declared anchor16 and publish cases for reuse."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

BASE = Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native')
ROOT = BASE / 'experiments/physical_content_v7/content_value_r9_1'
SOURCE = ROOT / 'snapshots/20261031T173355Z'
RUN = 'physical_content_66970'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--describe', action='store_true')
    args = parser.parse_args()
    checkpoint = ROOT / 'runs' / RUN / 'step0400.pt'
    status_path = checkpoint.parent / 'status.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    ready = checkpoint.is_file() and status.get('step') == 400 and status.get('status') in ['complete', 'paused']
    all_cases = [c['id'] for c in json.loads((SOURCE / 'subset16.json').read_text())['cases']]
    if len(all_cases) != 16 or len(set(all_cases)) != 16:
        raise ValueError('Use the original fixed16 ordering')
    selected = all_cases[-6:]
    primary = ROOT / 'inference_outputs' / RUN / 'step0400_full40_controls16_physical5_seed42/controls/anchor_only/videophy40'
    output = ROOT / 'control_dispatch/inference_outputs' / RUN / 'anchor_tail6_seed42_gamma5'
    command = [sys.executable, '-I', '-B', '-u', SOURCE / 'infer.py',
        '--checkpoint', checkpoint, '--output', output, '--suite', 'videophy40',
        '--physical-guidance', '5', '--seed', '42', '--modes', 'anchor_only', '--cases', *selected]
    plan = dict(training_run=RUN, checkpoint=str(checkpoint), source=str(SOURCE), ready=ready,
        selected_cases=selected, original_cases=all_cases, primary=str(primary),
        isolated_output=str(output), generation_command=list(map(str, command)),
        additional_cases=0, additional_seeds=0,
        evaluation='Original owner scores the complete16 once, after reusing these completed cases')
    if args.describe:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    if not ready:
        raise RuntimeError('Own400 must be fully saved before dispatch')
    sys.path.insert(0, str(SOURCE))
    import bootstrap
    from physical_content.common import atomic_json, gpu_guard
    job = ROOT / 'control_dispatch/reports/jobs' / os.getenv('DET_EXPERIMENT_ID', 'local') / 'anchor_tail6'
    job.mkdir(parents=True, exist_ok=True)
    atomic_json(plan, job / 'plan.json')
    selected = [case for case in selected if not (primary / 'cases' / case).exists()]
    if not selected:
        atomic_json(dict(status='left_to_original_owner', published=[]), job / 'status.json')
        return
    command = command[:command.index('--cases') + 1] + selected
    gpu_guard(job)
    sys.path.insert(0, str(BASE / 'experiments/physical_scaling_q256/inference_snapshots/20261030T034046Z'))
    from pipeline import run_stage
    run_stage(command, job / 'generation.log')
    generated = output / 'controls/anchor_only/videophy40'
    manifest = json.loads((generated / 'manifest.json').read_text())
    if {c['id'] for c in manifest['cases']} != set(selected):
        raise ValueError('Generated cases differ from the declared remaining tail')
    (primary / 'cases').mkdir(parents=True, exist_ok=True)
    published, owned = [], []
    for case in selected:
        origin = generated / 'cases' / case
        meta = json.loads((origin / 'planned.json').read_text())
        expected = dict(checkpoint=str(checkpoint), step=400, mode='anchor_only', case=case, seed=42)
        if not (origin / 'planned.mp4').is_file() or any(meta.get(k) != value for k, value in expected.items()):
            raise ValueError(f'Incomplete or mismatched output: {case}')
        try:
            (primary / 'cases' / case).symlink_to(origin, target_is_directory=True)
            published.append(case)
        except FileExistsError:
            owned.append(case)
    result = dict(status='completed', recorded_utc=datetime.now(timezone.utc).isoformat(),
        published=published, left_to_original_owner=owned, primary=str(primary),
        additional_cases=0, additional_seeds=0, scoring='Original complete16 pipeline only')
    atomic_json(result, job / 'status.json')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
