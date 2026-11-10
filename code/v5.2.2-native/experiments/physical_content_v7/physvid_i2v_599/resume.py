"""Resume the frozen benchmark after interruption, reusing completed inputs/videos."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_v7/physvid_i2v_599')
SOURCE = ROOT / 'snapshots/20261101T141534Z'
sys.path.insert(0, str(SOURCE))
from common import plan, save, gpu_guard, shared
from run import evaluator_environment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=('r9', 'r10a', 'r10b'))
    parser.add_argument('--describe', action='store_true')
    args = parser.parse_args()
    protocol = plan()
    missing = [case['id'] for case in protocol['cases'] if not (shared(case) / 'ready.json').exists()]
    if missing:
        raise RuntimeError('Shared preparation incomplete: ' + ','.join(missing))
    completed = {}
    for seed in protocol['generation_seeds']:
        folder = ROOT / 'outputs' / args.model / f'seed{seed}' / 'cases'
        records = [folder / case['id'] / 'video.json' for case in protocol['cases']]
        finished = [path for path in records if path.exists()]
        for path in finished:
            video = path.with_name('video.mp4')
            if not video.exists() or video.stat().st_size == 0:
                raise RuntimeError('Completed marker without video: ' + str(path))
        completed[str(seed)] = len(finished)
    info = dict(model=args.model, frozen_source=str(SOURCE), checkpoint=protocol['models'][args.model],
                shared_ready=len(protocol['cases']), completed_before=completed,
                remaining={seed: len(protocol['cases']) - n for seed, n in completed.items()})
    print(json.dumps(info), flush=True)
    if args.describe:
        return
    gpu_guard(args.model)
    job = ROOT / 'jobs' / args.model
    experiment = os.environ.get('DET_EXPERIMENT_ID')
    stage = 'starting'
    try:
        for stage in ('generate', 'evaluate'):
            save(job / 'status.json', dict(model=args.model, status='running', stage=stage,
                 experiment_id=experiment, resumed=True, source=str(SOURCE), time=time.time()))
            evaluator = stage == 'evaluate'
            interpreter = '/home/liuzhirui/miniconda3/envs/physvid/bin/python' if evaluator else sys.executable
            command = [interpreter, '-I', '-B', '-u', str(SOURCE / (stage + '.py')), '--model', args.model]
            with (job / (stage + '.log')).open('a') as log:
                log.write('\n' + json.dumps(dict(event='resume', experiment_id=experiment, time=time.time())) + '\n')
                log.flush()
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                               env=evaluator_environment() if evaluator else None)
        save(job / 'status.json', dict(model=args.model, status='completed', experiment_id=experiment,
             resumed=True, source=str(SOURCE), time=time.time()))
    except BaseException as exc:
        save(job / 'status.json', dict(model=args.model, status='failed', stage=stage,
             experiment_id=experiment, error=str(exc), resumed=True, time=time.time()))
        raise


if __name__ == '__main__':
    main()
