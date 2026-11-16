"""Eight-GPU ada-48g preparation for R20FT-Wan2.1.

Stages:
  1. base links once.
  2. N independent shard workers, one GPU each, run vae/native/inference.
  3. finalize global readiness records.
  4. fit_interface + prepare_inputs on CPU.
  5. one full-size check on GPU 0 and write the design-bound marker.

The formal blk-96g train-eval job consumes the marker.  This wrapper never
starts training.
"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap
from physical_content.common import atomic_json, data_ready, design_hash, load_config

ROOT = Path(__file__).resolve().parent


def _counts(cfg):
    cache = Path(cfg['paths']['cache'])
    return (len(list((cache / 'base/vae').glob('*.pt'))),
            len(list(Path(cfg['paths']['native_words_cache']).glob('*.pt'))),
            len(list(Path(cfg['paths']['inference_evidence']).glob('*/positive.pt'))))


def _run(command, log, env=None):
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a') as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                env=env)
    if result.returncode:
        raise RuntimeError(f'Command failed ({result.returncode}): {command}; log={log}')
    return result


def _gpu_env(rank):
    env = dict(os.environ)
    env['CUDA_VISIBLE_DEVICES'] = str(rank)
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--gpus', type=int, default=8)
    parser.add_argument('--stage', choices=('prepare', 'prepare-check'),
                        default='prepare-check')
    args = parser.parse_args()
    if args.gpus < 1:
        raise ValueError('--gpus must be positive')
    cfg = load_config(args.config, require_paths=True)
    if cfg['task'] != 't2v':
        raise ValueError('R20FT-Wan2.1 is text-to-video only')
    workspace = Path(cfg['paths']['run_dir']).parent.parent
    job = workspace / 'reports/jobs' / os.getenv('DET_EXPERIMENT_ID', 'local')
    job.mkdir(parents=True, exist_ok=True)
    marker = workspace / 'preparation/full_model_check.json'
    python = [sys.executable, '-I', '-B', '-u']
    started = time.monotonic()
    atomic_json(dict(status='prepare_sharded', gpus=args.gpus,
                     stage=args.stage, config=str(args.config)), job / 'status.json')

    # Base links once, on a process that does not need a GPU.
    _run([*python, str(ROOT / 'prepare_wan21_cache.py'),
          '--config', str(args.config), '--stage', 'base'],
         job / 'base.log', _gpu_env(0))

    if _counts(cfg) != (1600, 3320, 60):
        workers = []
        for rank in range(args.gpus):
            command = [*python, str(ROOT / 'prepare_wan21_cache.py'),
                       '--config', str(args.config), '--stage', 'shard',
                       '--shard-index', str(rank), '--shard-count', str(args.gpus),
                       '--device', 'cuda']
            log = job / f'shard{rank}.log'
            stream = log.open('a')
            process = subprocess.Popen(command, stdout=stream,
                                       stderr=subprocess.STDOUT,
                                       env=_gpu_env(rank))
            workers.append((rank, process, stream, log))
        failures = []
        for rank, process, stream, log in workers:
            code = process.wait()
            stream.close()
            if code:
                failures.append((rank, code, str(log)))
        if failures:
            raise RuntimeError(f'Preparation shards failed: {failures}')

    counts = _counts(cfg)
    if counts != (1600, 3320, 60):
        raise RuntimeError(f'Wan2.1 cache incomplete after shards: {counts}')
    atomic_json(dict(status='completed', stage='vae', records=1600,
                     files=counts[0], shards=args.gpus),
                Path(cfg['paths']['train_cache']) / 'base/vae/ready.json')
    atomic_json(dict(status='completed', stage='native', records=3320,
                     files=counts[1], shards=args.gpus, width=1536),
                Path(cfg['paths']['native_words_cache']) / 'ready.json')
    atomic_json(dict(status='completed', stage='inference', records=60,
                     files=counts[2], shards=args.gpus),
                Path(cfg['paths']['inference_evidence']) / 'ready.json')

    if not Path(cfg['paths']['interface']).is_file():
        _run([*python, str(ROOT / 'fit_interface.py'), '--config', str(args.config)],
             job / 'fit_interface.log', _gpu_env(0))
    _run([*python, str(ROOT / 'prepare_inputs.py'), '--config', str(args.config)],
         job / 'prepare_inputs.log', _gpu_env(0))
    data_ready(cfg)

    if args.stage == 'prepare-check':
        _run([*python, str(ROOT / 'check_full_model.py'),
              '--config', str(args.config), '--output', str(marker),
              '--gpu-policy', 'ada48'],
             job / 'full_model_check.log', _gpu_env(0))
        if not marker.is_file():
            raise RuntimeError('Full-size check did not write its marker')
        record = json.loads(marker.read_text())
        if (record.get('passed') is not True
                or record.get('design_hash') != design_hash(cfg)
                or record.get('model') != cfg['base']['model']
                or list(record.get('latent_shape', [])) != list(cfg['data']['latent_shape'])):
            raise RuntimeError(f'Full-size check marker does not match design: {record}')
    atomic_json(dict(status='completed', gpus=args.gpus, stage=args.stage,
                     cache_counts=list(counts), check_marker=str(marker),
                     check_passed=marker.is_file(),
                     seconds=time.monotonic() - started), job / 'status.json')
    print(json.dumps(dict(status='completed', gpus=args.gpus,
                          stage=args.stage, cache_counts=counts,
                          seconds=time.monotonic() - started)), flush=True)


if __name__ == '__main__':
    main()
