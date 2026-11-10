"""One compact live observation of the read-only design probe; no background loop."""
from pathlib import Path
from datetime import datetime, timedelta, timezone
import json
import subprocess
import statistics

ROOT = Path(__file__).resolve().parent


def read(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def det_json(arguments):
    process = subprocess.run(['det', '-m', 'http://10.130.130.5:4096', *arguments, '--json'],
                             capture_output=True, text=True, timeout=35)
    process.check_returncode()
    return json.loads(process.stdout)


def last_event(path):
    if not path.exists():
        return None
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size - 12000))
        lines = stream.read().decode(errors='replace').splitlines()
    for line in reversed(lines):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue


def main():
    prepared = read(ROOT / 'prepared.json')
    experiment = prepared['experiment']
    actual = det_json(['experiment', 'describe', str(experiment)])[0]['experiment']
    tasks = det_json(['task', 'list'])
    now = datetime.now(timezone.utc)
    terminal = actual['state'] in {'STATE_COMPLETED', 'STATE_ERROR', 'STATE_CANCELED', 'STATE_DELETED'}
    job = ROOT / 'reports/jobs' / str(experiment)
    output = ROOT / 'probe_outputs' / str(experiment)
    gpu = read(job / 'gpu.json', {})
    observation = dict(utc=now.isoformat(), experiment=experiment, state=actual['state'],
        trial_ids=actual.get('trialIds', []), active_task_ids=[x for x in tasks if x.startswith(str(experiment) + '.')],
        gpu={k: gpu.get(k) for k in ['actual', 'accepted', 'visible_count']},
        stage=read(job / 'status.json', {}).get('status'),
        generated=len(list((output / 'videophy40/cases').glob('*/planned.json'))),
        required=16, training_updates=0, not_formal_R15I=True,
        next_check_utc=None if terminal else (now + timedelta(seconds=900)).isoformat())
    event = last_event(output / 'generation.jsonl')
    if event:
        observation['last_generation'] = {k: event[k] for k in ['kind', 'case', 'step', 'time'] if k in event}
    scores = read(output / 'videophy40/scores.json', {}).get('cases')
    if scores:
        observation['scores'] = dict(count=len(scores), SA=statistics.mean(x['sa'] for x in scores),
                                     PC=statistics.mean(x['pc'] for x in scores),
                                     Joint=sum(x['sa'] >= 4 and x['pc'] >= 4 for x in scores))
    destination = ROOT / 'reports/latest_observation.json'
    previous = read(destination, {})
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(observation, ensure_ascii=False, indent=2) + '\n')
    prepared.update(last_observation=observation, state=actual['state'],
                    next_check_utc=observation['next_check_utc'],
                    status='terminal' if terminal else observation['stage'] or 'waiting_allocation_or_startup')
    (ROOT / 'prepared.json').write_text(json.dumps(prepared, ensure_ascii=False, indent=2) + '\n')
    for filename in ['plan.json', 'latest_goal_turn.json']:
        path = ROOT.parent / 'r11_r12' / filename
        state = read(path, {})
        state.update(status='R15I_design_probe_' + actual['state'], candidate='R15I-design',
                     active_experiment_id=None if terminal else experiment,
                     active_jobs=[] if terminal else [experiment],
                     active_trial_ids=[] if terminal else actual.get('trialIds', []),
                     design_probe_experiment=experiment, design_probe_runtime=observation,
                     formal_I_training_submitted=False, automatic_next_candidate=False,
                     next_check_utc=observation['next_check_utc'], next_routine_check_utc=observation['next_check_utc'])
        path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
    changed = {k: v for k, v in observation.items() if k not in ['utc', 'next_check_utc'] and previous.get(k) != v}
    print(json.dumps(dict(utc=observation['utc'], experiment=experiment, changed=changed,
                         next_check_utc=observation['next_check_utc']), ensure_ascii=False))


if __name__ == '__main__':
    main()
