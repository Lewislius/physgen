"""One compact authoritative read; no background polling."""
from pathlib import Path
from datetime import datetime, timedelta, timezone
import json
import subprocess

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def main():
    prepared = read(ROOT / 'diagnostic_prepared.json')
    experiment = prepared['experiment']
    proc = subprocess.run(['det', '-m', 'http://10.130.130.5:4096', 'experiment', 'describe', str(experiment), '--json'], capture_output=True, text=True, timeout=40)
    proc.check_returncode()
    actual = json.loads(proc.stdout)[0]['experiment']
    now = datetime.now(timezone.utc)
    row = dict(utc=now.isoformat(), experiment=experiment, state=actual['state'], trials=actual.get('trialIds', []), models={})
    for model in ['B', 'I']:
        folder = ROOT / 'reports/diagnostics' / str(experiment) / model
        gpu = read(folder / 'gpu.json')
        result = read(folder / 'result.json')
        row['models'][model] = dict(status=read(folder / 'status.json'),
            gpu={k: gpu.get(k) for k in ['actual', 'accepted', 'visible_count']},
            rows=len(result.get('rows', [])))
    terminal = actual['state'] in ['STATE_COMPLETED', 'STATE_ERROR', 'STATE_CANCELED']
    row['next_check_utc'] = None if terminal else (now + timedelta(minutes=15)).isoformat()
    prepared.update(state=row['state'], trials=row['trials'], last_observation=row, next_check_utc=row['next_check_utc'])
    (ROOT / 'diagnostic_prepared.json').write_text(json.dumps(prepared, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(row, ensure_ascii=False))


if __name__ == '__main__': main()
