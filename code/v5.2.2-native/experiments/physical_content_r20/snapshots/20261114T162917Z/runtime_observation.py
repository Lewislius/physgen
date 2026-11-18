"""Observe an unchanged job after repeated SIGKILL, without changing its model."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def read(path):
    try:
        return Path(path).read_text().strip()
    except (OSError, PermissionError):
        return None


def descendants(pid):
    result = [pid]
    children = read(f'/proc/{pid}/task/{pid}/children') or ''
    for child in children.split():
        result.extend(descendants(int(child)))
    return result


def sample(pid, phase):
    record = dict(time=time.time(), phase=phase, processes=[])
    for child in descendants(pid):
        status = read(f'/proc/{child}/status')
        if status:
            wanted = ('Name:', 'VmRSS:', 'VmHWM:', 'VmSwap:', 'Threads:', 'NSpid:')
            process = dict(
                pid=child, status=[s for s in status.splitlines()
                                   if s.startswith(wanted)])
            limits = read(f'/proc/{child}/limits') or ''
            process['limits'] = [s for s in limits.splitlines() if
                                 s.startswith(('Max cpu time', 'Max resident set',
                                               'Max address space'))]
            stat = read(f'/proc/{child}/stat') or ''
            fields = stat[stat.rfind(')') + 2:].split()
            if len(fields) > 12:
                hz = os.sysconf('SC_CLK_TCK')
                process['cpu_seconds'] = (int(fields[11]) + int(fields[12])) / hz
            record['processes'].append(process)
    record['memory'] = {
        name: read('/sys/fs/cgroup/' + name)
        for name in ('memory.current', 'memory.peak', 'memory.max',
                     'memory.events', 'memory.events.local',
                     'memory/memory.usage_in_bytes',
                     'memory/memory.max_usage_in_bytes',
                     'memory/memory.limit_in_bytes',
                     'memory/memory.failcnt', 'memory/memory.oom_control')}
    record['memory'] = {k: v for k, v in record['memory'].items() if v is not None}
    vmstat = read('/proc/vmstat') or ''
    record['host_oom_kill'] = next(
        (s for s in vmstat.splitlines() if s.startswith('oom_kill ')), None)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--avoid-gpu-uuid', action='append', default=[])
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('A child command is required')
    job = args.output / os.getenv('DET_EXPERIMENT_ID', 'local')
    job.mkdir(parents=True, exist_ok=True)
    if args.avoid_gpu_uuid:
        query = subprocess.run(
            ['nvidia-smi', '--query-gpu=uuid,name', '--format=csv,noheader'],
            capture_output=True, text=True, check=True, timeout=10)
        assigned = [line.split(',', 1) for line in query.stdout.strip().splitlines()]
        if len(assigned) == 1 and assigned[0][0].strip() in args.avoid_gpu_uuid:
            value = dict(kind='gpu_policy', actual=assigned[0][1].strip(),
                         uuid=assigned[0][0].strip(), accepted=False,
                         reason='Same GPU repeatedly SIGKILLed training at low clocks; try another RTX before model loading',
                         experiment_id=os.getenv('DET_EXPERIMENT_ID', 'local'))
            (job/'rejected_gpu.json').write_text(json.dumps(value, indent=2)+'\n')
            print(json.dumps(value), flush=True)
            return 86
    start = time.monotonic()
    with (job / 'resources.jsonl').open('a') as stream:
        def emit(phase):
            record = sample(os.getpid(), phase)
            stream.write(json.dumps(record) + '\n')
            stream.flush()
            return record
        emit('before')
        child = subprocess.Popen(command)
        # Only the startup failure window needs frequent local observations.
        # This does not poll Determined or call the assistant.
        while child.poll() is None:
            record = emit('running')
            if time.monotonic() - start < 480:
                try:
                    gpu = subprocess.run(
                        ['nvidia-smi', '--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm,clocks.mem',
                         '--format=csv,noheader,nounits'], text=True,
                        capture_output=True, timeout=10)
                    value = dict(returncode=gpu.returncode, output=gpu.stdout,
                                 error=gpu.stderr)
                except (OSError, subprocess.SubprocessError) as error:
                    value = dict(observation_error=str(error))
                stream.write(json.dumps(dict(time=time.time(), phase='gpu',
                                             **value)) + '\n')
                stream.flush()
            try:
                child.wait(timeout=15 if time.monotonic() - start < 480 else 1800)
            except subprocess.TimeoutExpired:
                pass
        last = emit('after')
        result = dict(returncode=child.returncode, seconds=time.monotonic()-start,
                      final_memory=last['memory'], host_oom_kill=last['host_oom_kill'])
        (job / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(dict(kind='runtime_observation_complete', **result)), flush=True)
    return child.returncode if child.returncode >= 0 else 128 - child.returncode


if __name__ == '__main__':
    sys.exit(main())
