"""One compact observation of the existing inference-only repair; never submits."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json
import subprocess

ROOT = Path(__file__).resolve().parent


def read(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def main():
    plan = read(ROOT / 'mass_retained_prepared.json')
    experiment = plan['experiment']
    response = subprocess.run(
        ['det', '-m', 'http://10.130.130.5:4096', 'experiment', 'describe', str(experiment), '--json'],
        capture_output=True, text=True, timeout=35, check=True)
    actual = json.loads(response.stdout)[0]['experiment']
    folder = ROOT / 'reports/semantic_mass_retained' / str(experiment)
    output = Path(plan['output'])
    now = datetime.now(timezone.utc)
    score = read(output / 'videophy40/scores.json', {})
    observation = dict(
        utc=now.isoformat(), experiment=experiment, state=actual['state'],
        trial_ids=actual.get('trialIds'), end_utc=actual.get('endTime'),
        gpu=read(folder / 'gpu.json', {}), stage=read(folder / 'status.json', {}),
        complete_videos=sum(1 for p in (output / 'videophy40/cases').glob('*/planned.json')
                            if p.with_suffix('.mp4').is_file()),
        scores=score.get('summary'), comparison_complete=(output / 'comparison16.json').is_file(),
        next_check_utc=(now + timedelta(seconds=900)).isoformat())
    target = ROOT / 'reports/latest_mass_retained_observation.json'
    previous = read(target, {})
    target.write_text(json.dumps(observation, ensure_ascii=False, indent=2) + '\n')
    changes = {k: v for k, v in observation.items()
               if k not in ['utc', 'next_check_utc', 'gpu'] and previous.get(k) != v}
    if observation['gpu'].get('actual') != previous.get('gpu', {}).get('actual'):
        changes['gpu'] = {k: observation['gpu'].get(k) for k in ['actual', 'accepted']}
    print(json.dumps(dict(utc=observation['utc'], experiment=experiment,
                         changed=changes or 'none', next_check_utc=observation['next_check_utc']),
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
