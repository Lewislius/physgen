"""One completed-run audit against B's original 400-step data/noise/stage schedule.

Reads existing records only. It does not score quality, start jobs or extend a run.
"""
from pathlib import Path
import argparse
import json

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT.parent/'request_bound_fusion_r15b/runs/physical_content_67829/train.jsonl'


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    run = ROOT/'runs'/args.run_id
    status = json.loads((run/'status.json').read_text())
    if status.get('status') != 'complete' or status.get('step') != 400:
        print(json.dumps(dict(status='incomplete', observed=status)))
        return
    current = records(run/'train.jsonl')
    previous = records(REFERENCE)
    micros = [v for v in current if event(v) == 'micro']
    ref_micro = [v for v in previous if event(v) == 'micro']
    updates = [v for v in current if event(v) == 'train']
    ref_updates = [v for v in previous if event(v) == 'train']
    if len(micros) != 1600 or len(updates) != 400:
        raise ValueError('Expected exactly400 optimizer updates and1600 microbatches')
    if len(ref_micro) != 1600 or len(ref_updates) != 400:
        raise ValueError('Reference is not the complete original400 run')
    fields = ('step', 'micro', 'phase', 'sample', 'sigma', 'text_dropped', 'aux_coefficient')
    mismatches = [dict(index=i, field=key, current=a.get(key), reference=b.get(key))
                  for i,(a,b) in enumerate(zip(micros,ref_micro)) for key in fields
                  if a.get(key) != b.get(key)]
    schedule = ('step', 'phase', 'cursor', 'learning_rates', 'aux_coefficient')
    mismatches += [dict(update=i, field=key, current=a.get(key), reference=b.get(key))
                   for i,(a,b) in enumerate(zip(updates,ref_updates)) for key in schedule
                   if a.get(key) != b.get(key)]
    interface = [v for v in updates if v['phase'] == 'interface']
    joint = [v for v in updates if v['phase'] == 'joint']
    frozen = all(v['gradients']['wan_lora'] == 0 for v in interface)
    active = all(v['gradients']['wan_lora'] > 0 for v in joint)
    result = dict(run=args.run_id, reference=str(REFERENCE), optimizer_updates=len(updates),
        microbatches=len(micros), unique_samples=len({v['sample'] for v in micros}),
        interface_updates=len(interface), joint_updates=len(joint),
        native_lora_frozen_during_interface=frozen, native_lora_active_during_joint=active,
        data_noise_dropout_and_schedule_match=not mismatches, mismatches=mismatches,
        quality_proof=False, scope='Training execution/protocol audit; not generation effectiveness')
    if len(interface) != 128 or len(joint) != 272 or not frozen or not active:
        raise ValueError('Unexpected stage/gradient ownership')
    (ROOT/'reports/completed_training_match.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'mismatches'}))


if __name__ == '__main__':
    main()
