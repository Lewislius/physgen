"""Read worker memory limits and kernel OOM evidence without changing the node."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time


def snapshot():
    result = dict(hostname=socket.gethostname(), unix=time.time(), pid=os.getpid())
    files = ['/proc/meminfo', '/proc/self/cgroup', '/proc/self/status',
             '/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory.current',
             '/sys/fs/cgroup/memory.peak', '/sys/fs/cgroup/memory.events',
             '/sys/fs/cgroup/memory.stat', '/sys/fs/cgroup/memory.swap.max',
             '/sys/fs/cgroup/memory/memory.limit_in_bytes',
             '/sys/fs/cgroup/memory/memory.max_usage_in_bytes',
             '/sys/fs/cgroup/memory/memory.failcnt']
    for name in files:
        path = Path(name)
        if path.exists():
            result[name] = path.read_text()
    for key, command in [
        ('gpu', ['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used', '--format=csv']),
        ('kernel_oom', ['dmesg', '--ctime']),
    ]:
        try:
            p = subprocess.run(command, capture_output=True, text=True, timeout=10)
            out = p.stdout
            if key == 'kernel_oom':
                out = '\n'.join(line for line in out.splitlines()
                                if any(word in line.lower() for word in
                                       ['out of memory', 'oom-kill', 'killed process', 'memory cgroup']))[-12000:]
            result[key] = dict(returncode=p.returncode, stdout=out, stderr=p.stderr[-1000:])
        except (OSError, subprocess.TimeoutExpired) as exc:
            result[key] = dict(error=str(exc))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = snapshot()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps(data), flush=True)
