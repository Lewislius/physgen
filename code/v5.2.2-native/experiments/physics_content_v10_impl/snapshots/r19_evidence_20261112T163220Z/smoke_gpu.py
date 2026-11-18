"""One real-GPU R18H forward/backward smoke test.

This deliberately does not call the formal RTX-only guard: it is a short
integration check, not a training run.  It is used only to verify the new
package on a free GPU while the formal job is queued.
"""
from pathlib import Path
import argparse
import json
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap  # noqa: F401

import torch

from physical_content.common import load_config
from physical_content.data import TrainingData
from physical_content.model import PhysicalModel
from physical_content.objectives import backward_micro
from physical_content.teacher import Teacher


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args(argv)
    cfg = load_config(args.config, require_paths=True)
    cfg['train']['gradient_checkpointing'] = True
    torch.set_num_threads(4)
    started = time.monotonic()
    model = PhysicalModel(cfg).train()
    model.wan.eval()
    data = TrainingData(cfg)
    sample = data.get(0)
    teacher = Teacher(cfg)
    evidence = teacher.input(sample, drop=False)
    noise = torch.randn_like(sample['latent'])
    sigma = torch.tensor(0.5, device='cuda')
    with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
        values = backward_micro(model, evidence, sample, sample['text'], sigma,
                                noise, 0.0, 1)
    torch.cuda.synchronize()
    report = dict(status='ok', task=cfg['task'], plan=cfg['plan'],
                  seconds=time.monotonic() - started,
                  peak_gib=torch.cuda.max_memory_allocated() / 2**30,
                  prediction='finite', values=values,
                  sample=sample['id'])
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
