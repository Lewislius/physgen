"""V-JEPA-only worker; no Wan/VAE imports or training optimizer in this process."""
import argparse
from contextlib import nullcontext
import ctypes
from datetime import timedelta
import json
import os
from pathlib import Path
import signal
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from physgen_v4.environments import require_environment


def serve(channel, teacher, device, runtime):
    import torch
    pending = {}
    channel.send(dict(runtime=runtime))
    while True:
        try:
            request = channel.receive()
        except EOFError:
            return
        operation = request['operation']
        token = request['token']
        if operation == 'forward':
            if token in pending:
                raise ValueError(f'Duplicate teacher graph: {token}')
            video = request['video'].to(device).requires_grad_(request['needs_grad'])
            autocast = torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()
            with torch.set_grad_enabled(request['needs_grad']), autocast:
                features = teacher(video, request['frame_count'])
            if request['needs_grad']:
                pending[token] = (video, features)
            response = dict(token=token, features=features.detach().cpu())
            del video, features
        elif operation == 'backward':
            video, features = pending.pop(token)
            # Frozen FSDP still needs backward hooks for input gradients.
            torch.autograd.backward(features, request['gradient'].to(device))
            if video.grad is None:
                raise RuntimeError('V-JEPA did not return an input gradient')
            response = dict(gradient=video.grad.detach().cpu())
            del video, features
        else:
            raise ValueError(f'Unknown teacher operation: {operation}')
        del request
        if device.type == 'cuda':
            # Both interpreters share each GPU. Release unused cached allocations to Wan.
            torch.cuda.empty_cache()
        channel.send(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--read-fd', type=int, required=True)
    parser.add_argument('--write-fd', type=int, required=True)
    parser.add_argument('--parent-pid', type=int, required=True)
    args = parser.parse_args()
    # Avoid orphaned GPU workers when torchrun terminates a training rank.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'Could not set worker parent-death signal')
    if os.getppid() != args.parent_pid:
        return
    runtime = require_environment('teacher')
    import torch
    import torch.distributed as dist
    from physgen_v4.teacher_transport import TensorChannel
    from physgen_v4.encoders import VideoTeacher
    channel = TensorChannel(args.read_fd, args.write_fd, timeout=None)
    try:
        config = channel.receive()
        sys.path.insert(0, config['paths']['teacher_code'])
        device = torch.device(config['device'])
        if device.type == 'cuda':
            torch.cuda.set_device(device)
        if config['sharded']:
            # Separate rendezvous and communicator from the moviestory training ranks.
            dist.init_process_group('nccl', init_method=Path(config['rendezvous']).as_uri(),
                                    rank=config['rank'], world_size=config['world'], device_id=device,
                                    timeout=timedelta(minutes=10))
        teacher = VideoTeacher(config['paths'], device, config['sharded'], config['recompute'])
        runtime.update(torch=str(torch.__version__), rank=config['rank'], world_size=config['world'])
        print(json.dumps(dict(event='online_teacher', **runtime)), flush=True)
        serve(channel, teacher, device, runtime)
    except BaseException:
        try:
            channel.send(dict(error=traceback.format_exc()))
        except (BrokenPipeError, OSError):
            pass
        raise
    finally:
        channel.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
