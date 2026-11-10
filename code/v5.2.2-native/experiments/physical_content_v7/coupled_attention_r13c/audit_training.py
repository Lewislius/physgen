"""Audit actual R13C sample order and its deliberately changed parameter phases.

Read-only with respect to training, checkpoints and previous candidates. This is
execution evidence, never a replacement for generated-video quality evaluation.
"""
from pathlib import Path
import argparse
import importlib.util
import json


ROOT = Path(__file__).resolve().parent


def records(path, kind):
    if not path.exists():
        return []
    result = []
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # A live logger may not yet have finished its final line.
        if row.get('kind') == kind:
            result.append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    run = ROOT / 'runs' / args.run_id
    cfg = json.loads((run / 'config.json').read_text())
    reference = ROOT.parent / 'execution_value_r10/runs/physical_content_67003/train.jsonl'
    micro = records(run / 'train.jsonl', 'micro')
    steps = records(run / 'train.jsonl', 'train')
    before = records(reference, 'micro')
    expected = {(row['step'], row['micro']): row for row in before}
    fields = ('sample', 'sigma', 'text_dropped', 'aux_active', 'aux_coefficient')
    differences = []
    for row in micro:
        old = expected[(row['step'], row['micro'])]
        for key in fields:
            if row[key] != old[key]:
                differences.append(dict(step=row['step'], micro=row['micro'],
                                        field=key, actual=row[key], reference=old[key]))
    schedule = ROOT.parents[1] / 'physical_scaling_q256/snapshots/20261029T140406Z/physical_scaling/schedule.py'
    spec = importlib.util.spec_from_file_location('pinned_schedule', schedule)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    phase_errors = []
    for row in steps:
        step = row['step']
        phase = 'interface' if step <= cfg['stages']['interface'] else 'joint'
        if row['phase'] != phase or row['cursor'] != 4 * step:
            phase_errors.append(dict(step=step, issue='phase_or_cursor'))
        if row['learning_rates'] != module.learning_rates(cfg, step - 1):
            phase_errors.append(dict(step=step, issue='learning_rates'))
        if phase == 'interface' and row['gradients']['wan_lora'] != 0:
            phase_errors.append(dict(step=step, issue='frozen_lora_received_gradient'))
    complete = len(steps) == 400 and len(micro) == 1600
    report = dict(
        run=args.run_id, revision=cfg['revision'], status='complete' if complete else 'in_progress',
        optimizer_updates=len(steps), microbatches=len(micro),
        last_step=steps[-1]['step'] if steps else 0,
        original_r10a_reference=str(reference), paired_input_fields=fields,
        same_input_noise_sigma_dropout_and_auxiliary=not differences,
        input_differences=differences, declared_stages=cfg['stages'],
        observed_interface_updates=sum(row['phase'] == 'interface' for row in steps),
        observed_joint_updates=sum(row['phase'] == 'joint' for row in steps),
        phase_and_schedule_errors=phase_errors,
        joint_updates_with_lora_gradient=sum(row['phase'] == 'joint' and
            row['gradients']['wan_lora'] > 0 for row in steps),
        quality_established=False,
        scope='Sample/noise-sigma/dropout/aux records are compared with R10A. Actual noise tensors '
              'are not stored; both sources use the preserved sample-cursor-derived noise seed. '
              'R13C deliberately changes32+368 versus128+272; no claim of an identical training schedule.')
    if complete:
        if [r['step'] for r in steps] != list(range(1, 401)):
            raise ValueError('Optimizer steps are duplicated or incomplete')
        if len({(r['step'], r['micro']) for r in micro}) != 1600:
            raise ValueError('Microbatch records are duplicated')
        if report['observed_interface_updates'] != 32 or report['observed_joint_updates'] != 368:
            raise ValueError('Actual parameter phases differ from the submitted plan')
    destination = ROOT / 'reports' / ('completed400_training_audit.json' if complete else 'partial_training_audit.json')
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('status', 'optimizer_updates', 'microbatches',
          'same_input_noise_sigma_dropout_and_auxiliary', 'phase_and_schedule_errors')}, ensure_ascii=False))
    if differences or phase_errors:
        raise ValueError('Actual training differs from its declared input or phase contract')


if __name__ == '__main__':
    main()
