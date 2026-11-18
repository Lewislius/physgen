"""Resume-aware completed-run audit against B's original 400-step protocol.

A resumed run appends a second pass over the interrupted steps to train.jsonl.
This audit keeps the last occurrence of each (step, microbatch) / step and then
performs the same protocol comparison as the original audit. It does not score
quality or change weights.
"""
from pathlib import Path
import argparse
import json

ROOT = Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_v7')
REFERENCE = ROOT / 'request_bound_fusion_r15b/runs/physical_content_67829/train.jsonl'


def records(path):
    values = []
    with path.open() as stream:
        for line in stream:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            values.append(value)
    return values


def event(value):
    return value.get('kind', value.get('event'))


def latest(records, kind, fields):
    """Keep the last occurrence of each logical step/microbatch."""
    chosen = {}
    order = []
    for value in records:
        if event(value) != kind:
            continue
        key = tuple(value.get(field) for field in fields)
        if key not in chosen:
            order.append(key)
        chosen[key] = value
    return [chosen[key] for key in order]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--run-directory', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    run = args.run_directory or ROOT / 'runs' / args.run_id
    status = json.loads((run / 'status.json').read_text())
    if status.get('status') != 'complete' or status.get('step') != 400:
        print(json.dumps(dict(status='incomplete', observed=status)))
        return
    current = records(run / 'train.jsonl')
    previous = records(REFERENCE)
    micros = latest(current, 'micro', ('step', 'micro'))
    updates = latest(current, 'train', ('step',))
    ref_micro = [v for v in previous if event(v) == 'micro']
    ref_updates = [v for v in previous if event(v) == 'train']
    if len(micros) != 1600 or len(updates) != 400:
        raise ValueError(f'Expected exactly400 updates/1600 microbatches after resume dedup; got {len(updates)}/{len(micros)}')
    if len(ref_micro) != 1600 or len(ref_updates) != 400:
        raise ValueError('Reference is not the complete original400 run')
    fields = ('step', 'micro', 'phase', 'sample', 'sigma', 'text_dropped', 'aux_coefficient')
    mismatches = [dict(index=i, field=key, current=a.get(key), reference=b.get(key))
                  for i, (a, b) in enumerate(zip(micros, ref_micro)) for key in fields
                  if a.get(key) != b.get(key)]
    schedule = ('step', 'phase', 'cursor', 'learning_rates', 'aux_coefficient')
    mismatches += [dict(update=i, field=key, current=a.get(key), reference=b.get(key))
                   for i, (a, b) in enumerate(zip(updates, ref_updates)) for key in schedule
                   if a.get(key) != b.get(key)]
    interface = [v for v in updates if v['phase'] == 'interface']
    joint = [v for v in updates if v['phase'] == 'joint']
    frozen = all(v['gradients']['wan_lora'] == 0 for v in interface)
    active = all(v['gradients']['wan_lora'] > 0 for v in joint)
    ready = next(v for v in current if event(v) == 'model_ready')
    result = dict(run=args.run_id, trainable_parameters=ready['parameters'],
                  parameter_groups=ready['groups'], reference=str(REFERENCE),
                  optimizer_updates=len(updates), microbatches=len(micros),
                  unique_samples=len({v['sample'] for v in micros}),
                  interface_updates=len(interface), joint_updates=len(joint),
                  native_lora_frozen_during_interface=frozen,
                  native_lora_active_during_joint=active,
                  data_noise_dropout_and_schedule_match=not mismatches,
                  mismatches=mismatches, resume_deduplicated=True,
                  quality_proof=False,
                  scope='Training execution/protocol audit with resume dedup; not generation effectiveness')
    if len(interface) != 128 or len(joint) != 272 or not frozen or not active:
        raise ValueError('Unexpected stage/gradient ownership')
    output = args.output or ROOT / 'reports/completed_training_match.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'mismatches'}))


if __name__ == '__main__':
    main()
