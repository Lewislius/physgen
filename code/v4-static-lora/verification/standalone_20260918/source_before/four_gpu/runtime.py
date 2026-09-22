"""Single process, four CUDA devices; report actual per-device placement/memory."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import torch

from static_lora.runtime import log as base_log, save_json as base_save_json


def require_four_gpus():
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("Use one Python process for four-GPU model parallelism; do not use torchrun/DDP")
    if "DET_SLOT_IDS" in os.environ and len(json.loads(os.environ["DET_SLOT_IDS"])) != 4:
        raise RuntimeError("Determined slots_per_trial must be 4 on a single node")
    torch.cuda.init()
    if torch.cuda.device_count() != 4:
        raise RuntimeError(f"Expected exactly four visible CUDA GPUs, got {torch.cuda.device_count()}")
    devices = []
    for index in range(4):
        device = torch.device("cuda", index)
        properties = torch.cuda.get_device_properties(device)
        if properties.total_memory < 40 * 2**30:
            raise RuntimeError(f"GPU {index} has {properties.total_memory / 2**30:.1f} GiB; allocate four 48GB-class GPUs")
        with torch.cuda.device(device):
            probe = torch.zeros(1, device=device)
            torch.cuda.synchronize(device)
            del probe
        devices.append(dict(index=index, name=properties.name, total_gib=properties.total_memory / 2**30))
    torch.cuda.set_device(0)
    base_log(event="four_gpus_ready", python=sys.executable, torch_version=torch.__version__,
             torch_cuda=torch.version.cuda, devices=devices, processes=1, global_batch=8)
    return torch.device("cuda", 0)


def memory_metrics():
    return {str(index): dict(allocated_gib=torch.cuda.memory_allocated(index) / 2**30,
                             peak_allocated_gib=torch.cuda.max_memory_allocated(index) / 2**30,
                             peak_reserved_gib=torch.cuda.max_memory_reserved(index) / 2**30)
            for index in range(4)}


def log(log_path=None, /, *, console=True, **values):
    if values.get("event") in ("update", "case_complete"):
        started = time.perf_counter()
        for index in range(4):
            torch.cuda.synchronize(index)
        if "seconds" in values:
            values["seconds"] += time.perf_counter() - started
        values["cuda_memory"] = memory_metrics()
        values["model_parallel_devices"] = 4
        for index in range(4):
            torch.cuda.reset_peak_memory_stats(index)
    base_log(log_path, console=console, **values)


def save_json(value, path):
    if Path(path).name == "run.json" and "trainable_names" in value:
        value = dict(value, execution=value["config"]["execution"],
                     gpu_devices=[dict(index=i, name=torch.cuda.get_device_name(i)) for i in range(4)])
    base_save_json(value, path)


def isolated_module(name, path):
    # Reuse the unchanged loop in a separate namespace; never patch shared globals.
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
