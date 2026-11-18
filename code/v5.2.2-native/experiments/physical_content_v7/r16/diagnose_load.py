"""Print staged full-size model construction timings; no training or saves."""
from pathlib import Path
import argparse, json, os, sys, time
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap
import torch
from physical_content.common import load_config, atomic_json

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--config', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
args = p.parse_args()
print('stage: start', flush=True)
cfg = load_config(args.config)
print('stage: config', flush=True)
print('gpu:', torch.cuda.get_device_name(0), 'count', torch.cuda.device_count(), flush=True)
from physical_content.model import PhysicalModel
print('stage: import model', flush=True)
t = time.monotonic()
model = PhysicalModel(cfg)
print(f'stage: model constructed {time.monotonic()-t:.1f}s', flush=True)
print('allocated_gib', torch.cuda.memory_allocated() / 2**30,
      'reserved_gib', torch.cuda.memory_reserved() / 2**30, flush=True)
atomic_json(dict(status='constructed', seconds=time.monotonic()-t,
                 parameters=sum(p.numel() for p in model.parameters()),
                 trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
                 allocated_gib=torch.cuda.memory_allocated() / 2**30,
                 reserved_gib=torch.cuda.memory_reserved() / 2**30), args.output)
print('stage: wrote result', flush=True)
