import json
import math
import os
from pathlib import Path
import random
import shutil
import sys

import numpy as np
import torch
import torch.distributed as dist
import yaml

from .monitoring import Logger, checkpoint_source_files, normalize_logging, public_config


ROOT = Path(__file__).resolve().parents[1]


def read_config(path):
    config = yaml.safe_load(Path(path).read_text())
    config["paths"] = {key: os.path.expandvars(value.replace("${V4_ROOT}", str(ROOT)).replace("${V42_ROOT}", str(ROOT)))
                       for key, value in config["paths"].items()}
    return normalize_logging(config)


def configure_batch(config, world_size):
    """Derive per-rank accumulation from the actual torchrun worker count."""
    cfg = config['train']
    if 'global_batch_size' not in cfg:
        # Recover the target batch from checkpoints written before dynamic allocation.
        previous = config.get('hardware', {}).get(config.get('hardware_name', '4x48g'), {})
        cfg['global_batch_size'] = previous.get('gpus', 4) * cfg.get('accumulation_steps', 2)
    target = int(cfg['global_batch_size'])
    if world_size < 1 or target < 1:
        raise ValueError('Worker count and global_batch_size must be positive')
    cfg['accumulation_steps'] = max(1, (target + world_size - 1) // world_size)
    cfg['effective_global_batch_size'] = world_size * cfg['accumulation_steps']
    config['world_size'] = world_size
    config.pop('hardware', None)
    return config


def resume_rng(checkpoint, rank, world_size, previous_world_size, step, seed):
    if world_size == previous_world_size:
        restore_rng(torch.load(Path(checkpoint) / f'rng_rank{rank}.pt', map_location='cpu', weights_only=False))
    else:
        # Worker identities change with the allocation; optimizer state and data cursor do not.
        seed_all(seed + 100003 * step + rank)


def add_external_paths(config):
    sys.path.insert(0, config["paths"]["wan_code"])
    sys.path.insert(0, config["paths"]["teacher_code"])


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state())


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda"])


def init_distributed():
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    return dist.get_rank(), dist.get_world_size(), device


def reduce_metrics(metrics, device):
    names = sorted(metrics)
    values = torch.stack([torch.as_tensor(metrics[name], device=device).float().reshape(()) for name in names])
    dist.all_reduce(values)
    values /= dist.get_world_size()
    return dict(zip(names, values.cpu().tolist()))


def synchronize_gradients(groups):
    # Each active group participates in this step's graph on every rank.
    # The same global schedule selects auxiliary forwards on all ranks.
    for parameters in groups.values():
        for parameter in parameters:
            dist.all_reduce(parameter.grad)
            parameter.grad.div_(dist.get_world_size())


def group_metrics(groups):
    values = {}
    for name, parameters in groups.items():
        count = sum(p.numel() for p in parameters)
        grad_square = sum(p.grad.detach().float().square().sum() for p in parameters)
        weight_square = sum(p.detach().float().square().sum() for p in parameters)
        values[f"grad/{name}_rms"] = (grad_square / count).sqrt()
        values[f"weight/{name}_rms"] = (weight_square / count).sqrt()
        values[f"grad/{name}_norm"] = grad_square.sqrt()
    return values


def memory_metrics(device):
    local = torch.tensor([torch.cuda.memory_allocated(device), torch.cuda.memory_reserved(device),
                          torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)],
                         device=device, dtype=torch.float64) / 2**30
    buffers = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(buffers, local)
    names = ("allocated_gib", "reserved_gib", "peak_allocated_gib", "peak_reserved_gib")
    return {f"memory/rank_{rank}/{name}": value for rank, buffer in enumerate(buffers)
            for name, value in zip(names, buffer.cpu().tolist())}


def make_scheduler(optimizer, config):
    cfg = config["train"]
    def multiplier(step):
        if step < cfg["warmup_steps"]:
            return (step + 1) / cfg["warmup_steps"]
        progress = (step - cfg["warmup_steps"]) / (cfg["steps"] - cfg["warmup_steps"])
        return cfg["minimum_lr_ratio"] + (1 - cfg["minimum_lr_ratio"]) * 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def save_checkpoint(config, corrector, optimizer, scheduler, step, cursor, metrics, final=False):
    rank = dist.get_rank()
    root = Path(config["paths"]["checkpoint_root"]) / config["run_name"]
    destination = root / ("checkpoint-final" if final else f"checkpoint-{step:07d}")
    # Only rank zero reserves the directory. Reject an existing save (including a
    # partial one) before any weights/RNG files can be overwritten by any rank.
    error = [None]
    if rank == 0:
        try:
            destination.mkdir(parents=True, exist_ok=False)
        except OSError as exc:
            error[0] = f'Cannot create checkpoint {destination}; existing saves are never overwritten: {exc}'
    if dist.is_initialized():
        dist.broadcast_object_list(error, src=0)
    if error[0] is not None:
        raise RuntimeError(error[0])
    torch.save(rng_state(), destination / f"rng_rank{rank}.pt")
    if rank == 0:
        torch.save({name: value.detach().cpu() for name, value in corrector.state_dict().items()}, destination / "corrector.pt")
        torch.save(dict(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), next_step=step,
                        data_cursor=cursor, world_size=dist.get_world_size()), destination / "training.pt")
        (destination / "config.yaml").write_text(yaml.safe_dump(public_config(config), allow_unicode=True, sort_keys=False))
        (destination / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
        shutil.copy2(Path(config["paths"]["cache_root"]) / "manifest.json", destination / "data_manifest.json")
        for source in checkpoint_source_files(ROOT):
            target = destination / "source" / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        for source in ROOT.glob("*.md"):
            shutil.copy2(source, destination / "source" / source.name)
        (destination / "runtime.json").write_text(json.dumps(dict(torch_version=torch.__version__,
            python_version=sys.version, world_size=dist.get_world_size(), process_dtype="float32",
            trainable_parameter_dtype="float32", autocast_dtype="bfloat16"), indent=2))
    dist.barrier()
    if rank == 0:
        (root / "latest.json").write_text(json.dumps(dict(checkpoint=str(destination), next_step=step, stage=config["stage"]), indent=2))
        if config['corrector'].get('architecture') == 'v4_fullwidth3_write_v1':
            variant = 'with_prior' if config['loss']['prior_weight'] > 0 else 'without_prior'
            pointer = Path(config['paths']['checkpoint_root']) / f'v42_v4_fullwidth3_write_{variant}_latest.json'
            temporary = pointer.with_suffix('.tmp')
            temporary.write_text(json.dumps(dict(checkpoint=str(destination), next_step=step,
                architecture='v4_fullwidth3_write_v1', loss_variant=variant, run_name=config['run_name']), indent=2))
            temporary.replace(pointer)
    return destination
