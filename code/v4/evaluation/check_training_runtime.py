"""Explicit GPU-node diagnostics; never called automatically by the training launcher.

Run under torchrun in the same allocation/runtime as training. This checks real
NCCL transfers and optionally spawns real cache readers AFTER CUDA/NCCL startup.
It does not load Wan, update weights, modify caches, or write checkpoints.
"""
import argparse
from contextlib import ExitStack
from datetime import timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from physgen_v4.environments import require_environment
require_environment('runtime')

import torch
import torch.distributed as dist

from physgen_v4.data import CachedWISA, TrainingOrder, move_sample, training_loader
from physgen_v4.diagnostics import RankDiagnostics
from physgen_v4.runtime import read_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(Path(__file__).resolve().parents[1] / 'configs/wisa_native_p.yaml'))
    parser.add_argument('--mode', choices=('collectives', 'loader'), default='collectives')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--pin-memory', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--numel', type=int, default=16 * 1024 * 1024,
                        help='BF16 elements per rank: default 32 MiB, plus gathered outputs')
    parser.add_argument('--timeout-seconds', type=float, default=120)
    args = parser.parse_args()
    if min(args.iterations, args.numel, args.timeout_seconds) <= 0 or args.workers < 0:
        parser.error('iterations, numel and timeout must be positive; workers must be nonnegative')
    config = read_config(args.config)
    config['run_name'] = f'runtime_check_{args.mode}'
    config['train'].update(workers=args.workers, pin_memory=args.pin_memory,
                           loader_timeout_seconds=args.timeout_seconds)
    with ExitStack() as cleanup:
        trace = cleanup.enter_context(RankDiagnostics(config))
        with trace.stage('init_distributed'):
            import os
            device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
            torch.cuda.set_device(device)
            dist.init_process_group('nccl', device_id=device, timeout=timedelta(seconds=args.timeout_seconds))
        def destroy_group():
            with trace.stage('destroy_process_group'):
                dist.destroy_process_group()
        cleanup.callback(destroy_group)
        rank, world = dist.get_rank(), dist.get_world_size()
        if world < 2:
            raise ValueError('Use at least two ranks to check inter-GPU communication')
        trace.emit('runtime_check_start', mode=args.mode, gpu=torch.cuda.get_device_name(device),
                   torch_version=torch.__version__, cuda_version=torch.version.cuda,
                   nccl_version=list(torch.cuda.nccl.version()), world_size=world)
        with trace.stage('initial_collective'):
            initial = torch.tensor([rank + 1.], device=device)
            dist.all_reduce(initial)
            torch.cuda.synchronize(device)
            if initial.item() != world * (world + 1) / 2:
                raise RuntimeError('Initial all_reduce returned incorrect values')
        if args.mode == 'loader':
            with trace.stage('dataset_init'):
                dataset = CachedWISA(config['paths']['cache_root'], 'train',
                                     limit=config['train'].get('sample_limit', 0))
                keys = [(r['view']['frames'], r['view']['height'], r['view']['width']) for r in dataset.records]
                order = TrainingOrder(keys, config['train']['seed'], rank, world, 0)
                loader = training_loader(dataset, order, config['train'], rank)
            with trace.stage('dataloader_iter', workers=args.workers,
                             start_method='spawn' if args.workers else 'single_process', pin_memory=args.pin_memory):
                iterator = iter(loader)
            for iteration in range(args.iterations):
                with trace.stage('data_next', iteration=iteration):
                    sample = next(iterator)
                with trace.stage('data_to_device_and_sync', iteration=iteration):
                    sample = move_sample(sample, device)
                    torch.cuda.synchronize(device)
                with trace.stage('sample_collective', iteration=iteration, index=sample['record']['index']):
                    value = torch.tensor([rank + 1.], device=device)
                    dist.all_reduce(value)
                    if value.item() != world * (world + 1) / 2:
                        raise RuntimeError('all_reduce after loading a sample returned incorrect values')
                del sample
            with trace.stage('dataloader_release'):
                del iterator, loader
        else:
            with trace.stage('allocate_collective_buffers', numel=args.numel):
                source = torch.empty(args.numel, dtype=torch.bfloat16, device=device)
                gathered = [torch.empty_like(source) for _ in range(world)]
            for iteration in range(args.iterations):
                with trace.stage('all_gather', iteration=iteration):
                    source.fill_(rank + 1)
                    dist.all_gather(gathered, source)
                    torch.cuda.synchronize(device)
                with trace.stage('verify_all_gather', iteration=iteration):
                    for peer, values in enumerate(gathered):
                        if not torch.all(values == peer + 1).item():
                            raise RuntimeError(f'all_gather data mismatch from rank {peer}')
                with trace.stage('all_reduce', iteration=iteration):
                    dist.all_reduce(source)
                    torch.cuda.synchronize(device)
                with trace.stage('verify_all_reduce', iteration=iteration):
                    if not torch.all(source == world * (world + 1) / 2).item():
                        raise RuntimeError('all_reduce data mismatch')
        trace.emit('runtime_check_complete', mode=args.mode, iterations=args.iterations,
                   peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30)


if __name__ == '__main__':
    main()
