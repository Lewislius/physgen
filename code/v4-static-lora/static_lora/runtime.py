"""Local environment, integrity and I/O helpers; no sibling experiment dependency."""
import contextlib
import json
import hashlib
import random
import socket
import subprocess
import sys
import os
from pathlib import Path
import tempfile

from . import ROOT

ENVS = {"runtime": "moviestory"}


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
    return files


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def file_identity(path):
    path = Path(path)
    stat = path.stat()
    return dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=sha256(path))


def verify_files(root, identities, full=False):
    root = Path(root).resolve()
    for relative, identity in identities.items():
        path = (root / relative).resolve()
        if root not in path.parents:
            raise ValueError(f"Artifact escaped its root: {relative}")
        stat = path.stat()
        if full:
            valid = stat.st_size == identity["bytes"] and sha256(path) == identity["sha256"]
        else:
            valid = (stat.st_size, stat.st_mtime_ns) == (identity["bytes"], identity["mtime_ns"])
        if not valid:
            raise ValueError(f"Artifact changed after finalization: {path}")


def python_for(role):
    return f"/home/liuzhirui/miniconda3/envs/{ENVS[role]}/bin/python"


def require_environment(role):
    expected = Path(python_for(role)).parent.parent
    if Path(sys.prefix).resolve() != expected.resolve():
        raise RuntimeError(f"{role} requires {expected}/bin/python; got {sys.executable}")


def require_single_gpu():
    import torch
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("Single-GPU LoRA requires one process; do not launch torchrun with multiple workers")
    if "DET_SLOT_IDS" in os.environ and len(json.loads(os.environ["DET_SLOT_IDS"])) != 1:
        raise RuntimeError("Determined slots_per_trial must be 1")
    try:
        # Initialize CUDA directly so failures retain the driver's actual error.
        # is_available() alone reduces that error to an uninformative False.
        torch.cuda.init()
        torch.cuda.set_device(0)
        probe = torch.zeros(1, device="cuda:0")
        torch.cuda.synchronize(0)
        del probe
    except (AssertionError, RuntimeError) as error:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,uuid,driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10, check=False)
            driver = dict(returncode=result.returncode, stdout=result.stdout[-4000:], stderr=result.stderr[-4000:])
        except (OSError, subprocess.TimeoutExpired) as driver_error:
            driver = dict(error=f"{type(driver_error).__name__}: {driver_error}")
        log(event="gpu_unavailable", hostname=socket.gethostname(), python=sys.executable,
            torch_version=torch.__version__, torch_cuda=torch.version.cuda, torch_file=torch.__file__,
            environment={key: os.environ.get(key) for key in (
                "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES",
                "LD_LIBRARY_PATH", "CONDA_PREFIX", "DET_EXPERIMENT_ID", "DET_SLOT_IDS", "WORLD_SIZE")},
            cuda_error=f"{type(error).__name__}: {error}", nvidia_smi=driver)
        raise RuntimeError(
            f"CUDA initialization failed: {error}. See gpu_unavailable diagnostics above; "
            "check the allocated container's GPU visibility and NVIDIA driver.") from error
    return torch.device("cuda:0")


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast(device):
    import torch
    return torch.autocast(device.type, dtype=torch.bfloat16, cache_enabled=False)


def native_negative():
    # Avoid wan.__init__, which initializes CUDA even when only a config string is needed.
    import ast
    source = next((Path(p) / "wan/configs/shared_config.py" for p in sys.path
                   if (Path(p) / "wan/configs/shared_config.py").is_file()), None)
    if source is None:
        raise FileNotFoundError("Native Wan shared_config.py is not on the configured source path")
    tree = ast.parse(source.read_text())
    value = next((ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                  and any(isinstance(t, ast.Attribute) and t.attr == "sample_neg_prompt" for t in node.targets)), "")
    if not value.strip():
        raise ValueError("Native Wan negative prompt is empty")
    return value


def checkpoint_identity_compatible(saved, current):
    return (saved.keys() == current.keys()
            and all(saved[key] == current[key] for key in saved if key != "code")
            and checkpoint_code_compatible(saved.get("code"), current.get("code")))


def checkpoint_code_compatible(saved, current):
    # Training resume requires identical local source. Historical inference
    # migrations are separately audited in inference/checkpoint_compatibility.json.
    return saved == current


def wan_asset_files(cfg):
    paths = cfg["paths"]
    wan = Path(paths["wan_checkpoint"])
    index = read_json(wan / "diffusion_pytorch_model.safetensors.index.json")
    files = [wan / name for name in sorted(set(index["weight_map"].values()))]
    files += [wan / name for name in ("config.json", "diffusion_pytorch_model.safetensors.index.json",
                                      "Wan2.2_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth")]
    files += [p for p in (wan / "google/umt5-xxl").rglob("*") if p.is_file()]
    source = Path(paths["wan_code"]) / "wan"
    files += list(source.rglob("*.py")) + list(source.rglob("*.yaml"))
    return sorted(set(files))


def lock_assets(cfg, create=False, *, inference=False):
    """Validate the locally pinned training inventory; inference checks only Wan assets.

    Retain historical unused teacher entries in the returned identity so existing
    checkpoints remain comparable, without opening teacher files during inference.
    """
    if create:
        raise ValueError("The asset inventory is frozen; create a new experiment for different base assets")
    inventory = read_json(ROOT / "assets/training_asset_inventory.json")
    required = {str(p) for p in wan_asset_files(cfg)}
    roots = (Path(cfg["paths"]["wan_checkpoint"]), Path(cfg["paths"]["wan_code"]) / "wan")
    recorded = {name for name in inventory if any(Path(name).is_relative_to(root) for root in roots)}
    if required != recorded:
        raise ValueError("Wan base/tokenizer/source files differ from the frozen asset inventory")
    for name in sorted(required if inference else inventory):
        expected = inventory[name]
        stat = Path(name).stat()
        if (stat.st_size, stat.st_mtime_ns) != (expected["bytes"], expected["mtime_ns"]):
            raise ValueError(f"Model file metadata changed since training: {name}")
        if expected.get("sha256") and sha256(name) != expected["sha256"]:
            raise ValueError(f"Model file content changed since training: {name}")
    return inventory
