"""Write only under this experiment; reuse read-only helpers without patching v4.2."""
import contextlib
import json
import os
from pathlib import Path
import tempfile

from . import ROOT, V42_ROOT
from physgen_v42.runtime import (autocast, checkpoint_code_compatible, checkpoint_identity_compatible,
                                digest, file_identity, lock_assets, native_negative,
                                read_json, require_environment, require_single_gpu, seed_all,
                                sha256, verify_files)


def output_path(path):
    path = Path(path).expanduser().resolve()
    if ROOT not in path.parents:
        raise ValueError(f"LoRA outputs must stay below {ROOT}: {path}")
    return path


def atomic_write(path, writer):
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    os.close(fd)
    try:
        writer(name)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def save_json(value, path):
    atomic_write(path, lambda name: Path(name).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"))


def save_tensor(value, path):
    import torch
    atomic_write(path, lambda name: torch.save(value, name))


def log(log_path=None, /, *, console=True, **values):
    """The event's ``path`` is payload, never the JSONL destination."""
    line = json.dumps(values, ensure_ascii=False, allow_nan=False)
    if console:
        print(line, flush=True)
    if log_path is not None:
        path = output_path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as stream:
            stream.write(line + "\n")


@contextlib.contextmanager
def job_lock(path):
    import fcntl
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def code_fingerprint():
    files = {str(p.relative_to(ROOT)): sha256(p)
             for directory in ("static_lora", "train", "inference")
             for p in sorted((ROOT / directory).glob("*.py"))}
    for name in ("__init__", "runtime", "data", "geometry", "backbone", "encoders", "losses", "optim"):
        p = V42_ROOT / "physgen_v42" / (name + ".py")
        files["v4.2/" + str(p.relative_to(V42_ROOT))] = sha256(p)
    return files
