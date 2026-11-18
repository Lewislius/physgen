"""Submit one frozen job and retry only confirmed pre-model H100 rejection.

Own lock prevents duplicate controllers. A queued/running or unobservable job is
never resubmitted. Submission ambiguity stops for reconciliation instead of
assuming failure. Uses the existing Determined CLI without changing its setup.
"""
from pathlib import Path
import argparse
import fcntl
import json
import os
import re
import subprocess
import time


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--yaml', type=Path, required=True)
    parser.add_argument('--job-root', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--master', default='http://10.130.130.5:4096')
    parser.add_argument('--det-bin', default='/home/liuzhirui/.local/bin/det')
    parser.add_argument('--starting-interval', type=float, default=5.)
    parser.add_argument('--active-interval', type=float, default=1800.)
    parser.add_argument('--rejected-delay', type=float, default=5.)
    args = parser.parse_args()
    args.state.parent.mkdir(parents=True, exist_ok=True)
    lock = args.state.with_suffix('.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cli = [args.det_bin, '-m', args.master]
    state = json.loads(args.state.read_text()) if args.state.exists() else dict(
        yaml=str(args.yaml.resolve()), job_root=str(args.job_root.resolve()), attempts=[])
    if state['yaml'] != str(args.yaml.resolve()) or state['job_root'] != str(args.job_root.resolve()):
        raise ValueError('Controller state belongs to a different frozen task')
    state['controller_pid'] = os.getpid()
    state['controller_started'] = time.time()
    if state.get('status') in ('submission_uncertain', 'submitting'):
        raise RuntimeError('Reconcile ambiguous submission before restarting controller')
    while True:
        if not state['attempts'] or state.get('status') == 'h100_rejected':
            state.update(status='submitting', observed_at=time.time())
            write(args.state, state)
            try:
                result = subprocess.run([*cli, 'experiment', 'create', str(args.yaml),
                                         str(Path(__file__).resolve().parent)],
                                        capture_output=True, text=True, timeout=180)
            except subprocess.TimeoutExpired:
                state.update(status='submission_uncertain', observed_at=time.time())
                write(args.state, state)
                return
            log = args.state.with_name(args.state.stem + f'.submit{len(state["attempts"]) + 1}.log')
            log.write_text(result.stdout + result.stderr)
            match = re.search(r'Created experiment\s+(\d+)', result.stdout, re.I)
            if match is None:
                state.update(status='submission_uncertain', returncode=result.returncode,
                             observed_at=time.time())
                write(args.state, state)
                return
            identity = int(match.group(1))
            state['attempts'].append(dict(experiment_id=identity, submitted_at=time.time()))
            state.update(status='submitted', experiment_id=identity)
            write(args.state, state)
            print(json.dumps(dict(kind='submitted', experiment_id=identity)), flush=True)
        identity = state['attempts'][-1]['experiment_id']
        try:
            result = subprocess.run([*cli, 'experiment', 'describe', str(identity), '--json'],
                                    capture_output=True, text=True, timeout=60)
            if result.returncode:
                raise RuntimeError('status query failed')
            described = json.loads(result.stdout)
            row = described[0] if isinstance(described, list) else described
            status = row['experiment']['state'].replace('STATE_', '')
        except (subprocess.TimeoutExpired, RuntimeError, ValueError, KeyError, IndexError):
            state.update(status='observation_unavailable', observed_at=time.time())
            write(args.state, state)
            time.sleep(60)
            continue
        guard_path = args.job_root / str(identity) / 'gpu.json'
        guard = json.loads(guard_path.read_text()) if guard_path.exists() else None
        state.update(status=status, observed_at=time.time(), actual_gpu=guard)
        state['attempts'][-1].update(status=status, actual_gpu=guard)
        write(args.state, state)
        if status in ('COMPLETED', 'CANCELED', 'ERROR', 'DELETED'):
            if (status == 'ERROR' and guard and not guard.get('accepted')
                    and 'H100' in guard.get('actual', '').upper()):
                state.update(status='h100_rejected', observed_at=time.time())
                write(args.state, state)
                print(json.dumps(dict(kind='retry_rejected_h100', experiment_id=identity)), flush=True)
                time.sleep(args.rejected_delay)
                continue
            print(json.dumps(dict(kind='terminal', experiment_id=identity, state=status)), flush=True)
            return
        time.sleep(args.active_interval if guard and guard.get('accepted')
                   and status == 'RUNNING' else args.starting_interval)


if __name__ == '__main__':
    main()
