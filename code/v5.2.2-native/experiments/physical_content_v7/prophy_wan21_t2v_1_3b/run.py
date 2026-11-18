"""Run official Wan2.1-T2V-1.3B on the ProPhy 600/180 VideoPhy2 prompt list."""
from pathlib import Path
import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, gpu_guard, save


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int)
    parser.add_argument('--skip-diagnostic', action='store_true')
    parser.add_argument('--score-only', action='store_true',
                        help='Score published videos as they become available')
    args = parser.parse_args()
    job = ROOT / 'jobs' / os.getenv('DET_EXPERIMENT_ID', 'local')
    job.mkdir(parents=True, exist_ok=True)
    gpu_guard(job)
    source = Path(__file__).resolve().parent
    stage = 'starting'

    def run(script, extra=(), evaluate=False):
        nonlocal stage
        stage = script.removesuffix('.py')
        save(job / 'status.json', dict(status='running', stage=stage,
                                        model='wan21_t2v_1_3b',
                                        experiment_id=os.getenv('DET_EXPERIMENT_ID'),
                                        time=time.time()))
        interpreter = ('/home/liuzhirui/miniconda3/envs/physvid/bin/python'
                       if evaluate else sys.executable)
        cmd = [interpreter, '-I', '-B', '-u', str(source / script), *extra]
        with (job / (stage + '.log')).open('a') as log:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                    env=evaluator_environment() if evaluate else None)
        if result.returncode:
            raise subprocess.CalledProcessError(result.returncode, cmd)

    try:
        if args.score_only:
            run('evaluate.py', ('--wait-for-videos',), evaluate=True)
        else:
            if not args.skip_diagnostic:
                run('generate.py', ('--limit', '1'))
                run('evaluate.py', ('--partial',), evaluate=True)
            run('generate.py', () if args.limit is None else ('--limit', str(args.limit)))
            run('evaluate.py', evaluate=True)
        save(job / 'status.json', dict(status='completed', model='wan21_t2v_1_3b',
                                        time=time.time(), source=str(source)))
    except BaseException as exc:
        save(job / 'status.json', dict(status='failed', stage=stage,
                                        error=str(exc), time=time.time(),
                                        model='wan21_t2v_1_3b'))
        raise


if __name__ == '__main__':
    main()
