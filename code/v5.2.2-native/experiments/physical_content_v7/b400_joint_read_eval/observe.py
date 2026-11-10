"""Compact authoritative observation; call at15min intervals, no background loop."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import subprocess

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def main():
    prepared = read(ROOT / 'prepared.json')
    now = datetime.now(timezone.utc)
    observations = {}
    for name, job in prepared['jobs'].items():
        experiment = job['experiment']
        if job.get('state') in ['STATE_COMPLETED', 'STATE_ERROR', 'STATE_CANCELED']:
            actual = dict(state=job['state'], trialIds=job.get('trials', []))
        else:
            response = subprocess.run(['det', '-m', 'http://10.130.130.5:4096', 'experiment', 'describe',
                                       str(experiment), '--json'], capture_output=True, text=True, timeout=40)
            response.check_returncode()
            actual = json.loads(response.stdout)[0]['experiment']
        folder = ROOT / 'reports/jobs' / str(experiment)
        status = read(folder / 'status.json')
        gpu = read(folder / 'gpu.json')
        row = dict(experiment=experiment, state=actual['state'], trials=actual.get('trialIds', []),
                   stage=status.get('stage'), gpu={k: gpu.get(k) for k in ['actual', 'accepted', 'visible_count']})
        if status.get('error'):
            row['error'] = status['error']
        if name == 'videophy':
            output = ROOT / 'videophy_outputs'
            row['generated40'] = len(list((output / 'videophy40/cases').glob('*/planned.json')))
            row['generated_extra20'] = len(list((output / 'videophy60/cases').glob('*/planned.json')))
            row['scores40'] = read(output / 'videophy40/scores.json').get('summary')
            row['scores60'] = read(output / 'full60/videophy60/scores.json').get('summary')
        else:
            output = ROOT.parent / 'cross_benchmark_70_60/outputs/r15b_joint_i/seed42'
            row['generated_wmb'] = len(list((output / 'cases').glob('wmb_*/video.json')))
            row['generated_physics_iq'] = len(list((output / 'cases').glob('piq_*/video.json')))
            row['wmb'] = read(output / 'worldmodelbench_summary.json').get('overall')
            row['physics_iq'] = read(output / 'physics_iq_summary.json').get('scores')
        job.update(state=actual['state'], trials=actual.get('trialIds', []), last_observation=row)
        observations[name] = row
    terminal = all(r['state'] in ['STATE_COMPLETED', 'STATE_ERROR', 'STATE_CANCELED'] for r in observations.values())
    record = dict(utc=now.isoformat(), jobs=observations,
                  next_check_utc=None if terminal else (now + timedelta(minutes=15)).isoformat())
    previous = read(ROOT / 'reports/latest_observation.json')
    (ROOT / 'reports').mkdir(exist_ok=True)
    (ROOT / 'reports/latest_observation.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    prepared.update(last_observation=record, next_check_utc=record['next_check_utc'])
    (ROOT / 'prepared.json').write_text(json.dumps(prepared, ensure_ascii=False, indent=2) + '\n')
    for filename in ['plan.json', 'latest_goal_turn.json']:
        path = ROOT.parent / 'r11_r12' / filename
        state = read(path)
        state['b400_joint_read_evaluation'] = dict(prepared=str(ROOT / 'prepared.json'), runtime=record,
            scope='Explicit user requested B400 inference-only40/60,WMB70,PhysicsIQ60; formal I unchanged.')
        path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
    changed = {name: value for name, value in observations.items() if previous.get('jobs', {}).get(name) != value}
    print(json.dumps(dict(utc=record['utc'], changed=changed, next_check_utc=record['next_check_utc']), ensure_ascii=False))


if __name__ == '__main__':
    main()
