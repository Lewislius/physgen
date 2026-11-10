"""Run one experiment and preserve memory / termination evidence if it is killed."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def memory():
    record = {}
    for key in ('memory.current', 'memory.max', 'memory.events', 'memory.swap.current'):
        p = Path('/sys/fs/cgroup') / key
        if p.exists():
            record[key] = p.read_text().strip()
    p = Path('/proc/meminfo')
    if p.exists():
        record['host_memory'] = {line.split(':')[0]: line.split(':')[1].strip()
                                 for line in p.read_text().splitlines()
                                 if line.startswith(('MemAvailable:', 'MemFree:', 'SwapFree:'))}
    return record


def process_tree_memory(pid):
    pending = [pid]; total = 0; processes = []
    while pending:
        current = pending.pop()
        try:
            status = (Path('/proc') / str(current) / 'status').read_text().splitlines()
            values = {line.split(':')[0]: line.split(':')[1].strip()
                      for line in status if line.startswith(('VmRSS:', 'VmHWM:', 'Name:'))}
            total += int(values.get('VmRSS', '0 kB').split()[0])
            processes.append(dict(pid=current, **values))
            children = Path(f'/proc/{current}/task/{current}/children')
            pending.extend(int(x) for x in children.read_text().split())
        except (FileNotFoundError, ProcessLookupError):
            pass
    return dict(processes=processes, tree_rss_kib=total)


def run(command, path, interval=15.):
    path.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with path.open('w', buffering=1) as stream:
        def record(kind, **values):
            stream.write(json.dumps(dict(kind=kind, unix=time.time(), seconds=time.monotonic()-start,
                                         **memory(), **values))+'\n')
        record('start', command=command)
        try:
            gpu=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used',
                                '--format=csv'],capture_output=True,text=True,timeout=10)
            record('gpu_assignment',returncode=gpu.returncode,output=gpu.stdout,stderr=gpu.stderr[-500:])
        except (OSError,subprocess.TimeoutExpired) as exc:
            record('gpu_assignment',error=str(exc))
        child = subprocess.Popen(command, start_new_session=True)
        def forward(signum, frame):
            record('parent_signal', signal=signum)
            if child.poll() is None:
                os.killpg(child.pid, signum)
        previous = {s: signal.signal(s, forward) for s in (signal.SIGTERM, signal.SIGINT)}
        try:
            while True:
                record('running', **process_tree_memory(child.pid))
                try:
                    result = child.wait(timeout=interval)
                    break
                except subprocess.TimeoutExpired:
                    continue
            record('exit', returncode=result, terminating_signal=-result if result < 0 else None)
            return result if result >= 0 else 128-result
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('A command is required')
    sys.exit(run(command, args.output))
