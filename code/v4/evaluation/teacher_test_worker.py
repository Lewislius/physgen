"""CPU fixture for the real cross-environment worker protocol; never a training entrypoint."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.checkpoint import checkpoint

from physgen_v4 import encoders
from physgen_v4.environments import require_environment
from tools.teacher_worker import main


class TinyVideoTeacher(torch.nn.Module):
    def __init__(self, paths, device, sharded=False, recompute=False):
        super().__init__()
        require_environment('teacher')
        self.recompute = recompute

    def forward(self, video, frame_count):
        if frame_count < 1:
            raise ValueError('frame_count must be positive')
        def encode(value):
            return (torch.sin(value * 0.7) + 0.2 * value.square()).flatten(2).transpose(1, 2)
        if self.recompute and torch.is_grad_enabled():
            return checkpoint(encode, video, use_reentrant=False)
        return encode(video)


if __name__ == '__main__':
    encoders.VideoTeacher = TinyVideoTeacher
    main()
