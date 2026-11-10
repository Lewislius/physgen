"""Submit the predeclared comparison only after a real checkpoint exists.

This waits on the CPU host; it does not reserve a GPU or interrupt training.
The checkpoint is hard-linked after atomic publication so continued training
cannot change the weights while the separate comparison is running.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'first_checkpoint_verification'
RUN = ROOT/'code/checkpoints/v522-native-causal-response-wisa3857_66180'
MASTER = 'http://10.130.130.5:4096'


def record(**fields):
    path = OUT/'job.json'
    state = json.loads(path.read_text())
    state.update(fields, checked_at_utc=datetime.now(timezone.utc).isoformat())
    path.write_text(json.dumps(state, indent=2)+'\n')


def main():
    if json.loads((OUT/'job.json').read_text()).get('id'):
        return
    record(watcher_pid=os.getpid(), watcher_state='waiting')
    print('Waiting for the first published trained checkpoint; no GPU reserved.', flush=True)
    deadline = time.monotonic()+4*3600
    while time.monotonic() < deadline:
        metrics = RUN/'metrics.jsonl'
        rows = []
        if metrics.is_file():
            for line in metrics.read_text().splitlines():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # Concurrent writer may not have finished its last line.
        checkpoints = [r for r in rows if r.get('kind') == 'checkpoint' and r['step'] >= 16]
        if checkpoints:
            source = Path(checkpoints[-1]['path'])
            destination = OUT/'checkpoint.pt'
            os.link(source, destination)
            record(state='checkpoint_reserved', source_checkpoint=str(source),
                   source_optimizer_step=checkpoints[-1]['step'], checkpoint=str(destination))
            result = subprocess.run(['det', '-m', MASTER, 'experiment', 'create',
                str(OUT/'experiment.yaml'), str(OUT/'context')], text=True, capture_output=True)
            (OUT/'submission.txt').write_text(result.stdout+result.stderr)
            match = re.search(r'Created experiment\s+(\d+)', result.stdout)
            if result.returncode or not match:
                record(state='submission_failed', watcher_state='finished')
                raise RuntimeError('Comparison submission did not return an experiment ID; inspect submission.txt')
            record(state='submitted', id=int(match[1]), watcher_state='finished')
            print(f'Submitted comparison {match[1]} from optimizer step {checkpoints[-1]["step"]}.', flush=True)
            return
        result = subprocess.run(['det', '-m', MASTER, 'experiment', 'describe', '66180', '--json'],
                                text=True, capture_output=True)
        if result.returncode == 0:
            response = json.loads(result.stdout)
            item = response[0] if isinstance(response, list) else response
            state = item.get('experiment', item)['state']
            if state in {'STATE_ERROR', 'STATE_CANCELED', 'STATE_DELETED', 'STATE_COMPLETED'}:
                record(state='training_ended_before_checkpoint', training_state=state, watcher_state='finished')
                return
        record(watcher_state='waiting')
        time.sleep(30)
    record(state='checkpoint_wait_expired', watcher_state='finished')


if __name__ == '__main__':
    main()
