"""Differentiable V-JEPA calls from moviestory into the fixed teacher runtime."""
import os
from pathlib import Path
import subprocess
import tempfile

import torch
import torch.distributed as dist
from torch.autograd.function import once_differentiable

from .environments import python_for, require_environment, subprocess_environment
from .teacher_transport import TensorChannel


class _TeacherAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, video, frame_count, worker):
        result = worker._forward(video, frame_count, True)
        ctx.worker, ctx.token = worker, result['token']
        ctx.device, ctx.dtype = video.device, video.dtype
        return result['features'].to(video.device)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        if ctx.device.type == 'cuda':
            torch.cuda.empty_cache()
        result = ctx.worker._request(dict(operation='backward', token=ctx.token,
                                          gradient=gradient.detach().cpu()))
        return result['gradient'].to(device=ctx.device, dtype=ctx.dtype), None, None


class RemoteVideoTeacher:
    def __init__(self, paths, device, sharded=False, recompute=False):
        require_environment('runtime')
        self.process = self.channel = self.rendezvous = None
        self.counter = 0
        rank, world = (dist.get_rank(), dist.get_world_size()) if sharded else (0, 1)
        rendezvous = [None]
        if sharded:
            if rank == 0:
                self.rendezvous = tempfile.TemporaryDirectory(prefix='v4-teacher-')
                rendezvous[0] = str(Path(self.rendezvous.name) / 'store')
            dist.broadcast_object_list(rendezvous, src=0)
        initialization = dict(paths=paths, device=str(device), sharded=sharded, recompute=recompute,
                              rank=rank, world=world, rendezvous=rendezvous[0])
        worker_read, parent_write = os.pipe()
        parent_read, worker_write = os.pipe()
        self.channel = TensorChannel(parent_read, parent_write)
        try:
            script = Path(__file__).resolve().parents[1] / 'tools/teacher_worker.py'
            self.process = subprocess.Popen(
                [python_for('teacher'), '-I', '-B', str(script), '--read-fd', str(worker_read),
                 '--write-fd', str(worker_write), '--parent-pid', str(os.getpid())],
                env=subprocess_environment('teacher'), pass_fds=(worker_read, worker_write))
        except BaseException:
            self.close()
            raise
        finally:
            os.close(worker_read)
            os.close(worker_write)
        try:
            self.runtime = self._request(initialization)['runtime']
        except BaseException:
            self.close()
            raise

    def _request(self, value):
        if self.channel is None:
            raise RuntimeError('V-JEPA worker is closed')
        try:
            self.channel.send(value)
            result = self.channel.receive()
        except (EOFError, BrokenPipeError, TimeoutError) as error:
            status = self.process.poll() if self.process is not None else None
            raise RuntimeError(f'V-JEPA worker failed (exit={status}): {error}') from error
        if 'error' in result:
            raise RuntimeError(f"V-JEPA worker failed: {result['error']}")
        return result

    def _forward(self, video, frame_count, needs_grad):
        self.counter += 1
        if video.device.type == 'cuda':
            torch.cuda.empty_cache()
        return self._request(dict(operation='forward', token=self.counter, video=video.detach().cpu(),
                                  frame_count=frame_count, needs_grad=needs_grad))

    def __call__(self, video, frame_count):
        if torch.is_grad_enabled() and video.requires_grad:
            return _TeacherAutograd.apply(video, frame_count, self)
        return self._forward(video, frame_count, False)['features'].to(video.device)

    def close(self):
        if self.channel is not None:
            # EOF also releases pending graphs, including after a failed training step.
            self.channel.close()
            self.channel = None
        if self.process is not None:
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            self.process = None
        if self.rendezvous is not None:
            self.rendezvous.cleanup()
            self.rendezvous = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
