"""Minimal subprocess runner used by run_candidate.py."""
import os
import subprocess
from pathlib import Path


def evaluator_environment():
    env = dict(os.environ)
    for name in tuple(env):
        if name.startswith(('PYTHON', 'CONDA')) or name == 'VIRTUAL_ENV':
            env.pop(name)
    prefix = '/home/liuzhirui/miniconda3/envs/physvid'
    for name in ('PATH', 'LD_LIBRARY_PATH'):
        env[name] = ':'.join(p for p in env.get(name, '').split(':')
                             if p and '/envs/' not in p)
    env['PATH'] = prefix + '/bin:' + env['PATH']
    env['CONDA_PREFIX'] = prefix
    return env


def run_stage(command, log, environment=None):
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a') as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                env=environment)
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)
