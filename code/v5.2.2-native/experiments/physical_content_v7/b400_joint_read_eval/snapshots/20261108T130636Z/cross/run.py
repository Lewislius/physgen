"""Reuse the fixed WMB70/PhysicsIQ60 inputs and official scorers for B400 joint-read."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys

SOURCE = Path(__file__).resolve().parent
sys.path[:0] = [str(SOURCE), str(SOURCE.parent)]
from common import ROOT, BENCH, plan, save, shared, gpu_guard
from allocated_gpu import bind_allocated_gpu


def evaluator_environment():
    env = dict(os.environ)
    for key in tuple(env):
        if key.startswith(('PYTHON', 'CONDA')) or key == 'VIRTUAL_ENV':
            env.pop(key)
    for key in ('PATH', 'LD_LIBRARY_PATH'):
        env[key] = ':'.join(p for p in env.get(key, '').split(':') if p and '/envs/' not in p)
    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
               MPLBACKEND='Agg', OMP_NUM_THREADS='4')
    return env


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='r15b_joint_i', choices=['r15b_joint_i'])
    p.add_argument('--describe', action='store_true')
    args = p.parse_args()
    protocol = plan()
    spec = protocol['models'][args.model]
    details = dict(model=args.model, checkpoint=spec, cases=len(protocol['cases']),
                   wmb=70, physics_iq=60, new_training_updates=0, not_trained_R15I=True,
                   resource_pool=spec['resource_pool'], source=str(SOURCE), reuse_shared_inputs=True)
    if args.describe:
        print(json.dumps(details, ensure_ascii=False, indent=2))
        return
    for case in protocol['cases']:
        for filename in ('ready.json', 'first.png', 'conditions.pt', 'positive.pt', 'negative.pt'):
            if not (shared(case) / filename).is_file():
                raise FileNotFoundError(shared(case) / filename)
    gpu_guard(args.model)
    root = ROOT.parent / 'b400_joint_read_eval'
    job = root / 'reports/jobs' / os.getenv('DET_EXPERIMENT_ID', 'local')
    job.mkdir(parents=True, exist_ok=True)
    bind_allocated_gpu(job, evaluator_environment)
    save(job / 'plan.json', details)
    stage = 'starting'
    try:
        for filename, interpreter in [('generate.py', sys.executable),
                                      ('evaluate_worldmodel.py', str(BENCH / '.venv-vila/bin/python')),
                                      ('evaluate_physics_iq.py', str(BENCH / '.venv-physicsiq/bin/python'))]:
            stage = filename.removesuffix('.py')
            save(job / 'status.json', dict(status='running', stage=stage, **details))
            command = [interpreter, '-I', '-B', '-u', str(SOURCE / filename), '--model', args.model]
            with (job / (stage + '.log')).open('a') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                               env=evaluator_environment() if interpreter != sys.executable else None)
        subprocess.run([sys.executable, '-I', '-B', str(SOURCE.parent / 'report.py')], check=True)
        save(job / 'status.json', dict(status='completed', stage='completed', **details))
    except BaseException as error:
        save(job / 'status.json', dict(status='failed', stage=stage, error=str(error), **details))
        raise


if __name__ == '__main__':
    main()
