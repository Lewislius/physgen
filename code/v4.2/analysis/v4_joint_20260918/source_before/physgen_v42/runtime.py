import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import tempfile

from . import VERSION

ROOT = Path(__file__).resolve().parents[1]
ENVS = {"runtime": "moviestory", "teacher": "vjepa2-312"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def output_path(path):
    path = Path(path).expanduser().resolve()
    if ROOT not in path.parents:
        raise ValueError(f"All v4.2 outputs must be below {ROOT}: {path}")
    return path


def atomic_write(path, writer):
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    os.close(fd)
    try:
        writer(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def save_json(value, path):
    atomic_write(path, lambda p: Path(p).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n"))


def save_tensor(value, path):
    import torch
    atomic_write(path, lambda p: torch.save(value, p))


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def log(log_path=None, /, *, console=True, **values):
    """Keep the destination separate from event fields such as ``path``."""
    line = json.dumps(values, ensure_ascii=False, allow_nan=False)
    if console:
        print(line, flush=True)
    if log_path is not None:
        path = output_path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as stream:
            stream.write(line + "\n")


def read_config(path):
    import yaml
    source = Path(path).read_text().replace("${V42_ROOT}", str(ROOT))
    # Checkpoints use JSON scientific notation (5e-05). YAML 1.1 treats that
    # spelling as a string; preserve the exact saved numeric configuration.
    cfg = json.loads(source) if Path(path).suffix == ".json" else yaml.safe_load(source)
    expected = {
        "model_version": VERSION,
        "data.frames": 121, "data.teacher_frames": 32, "data.teacher_size": 384,
        "data.limit": 2400, "data.validation_stride": 20,
        "data.max_long_side": 512, "data.max_area": 147456,
        "state.sites": [5, 15], "state.width": 1664, "state.inner": 512,
        "state.parameter_sharing": "per_block",
        "state.heads": 8, "state.depth": 2,
        "train.steps": 1200, "train.accumulation": 8, "train.world_size": 1,
        "loss.fm": 1.0,
        "train.save_every": 100, "train.ema_decay": .995, "train.grad_clip": 1.0,
        "train.peak_lr": {"core5": .00005, "core15": .00005, "text_init": .00002, "writer5": .00001, "writer15": .00002},
        "inference.steps": 50, "inference.shift": 5.0, "inference.guidance": 5.0,
        "inference.seed": 42,
    }
    from .optim import TRAINING_RECIPE, GATED_RECIPE, REPAIR_RECIPE
    monitoring = cfg.get('monitoring', {})
    for name, minimum in (('train_every', 1), ('health_ablation_cases', 0)):
        value = monitoring.get(name, 25 if name == 'train_every' else 2)
        if type(value) is not int or value < minimum:
            raise ValueError(f"monitoring.{name} must be an integer >= {minimum}")
    initializer = cfg['state'].get('initializer', 'static_v1')
    if initializer not in ('static_v1', 'condition_trajectory_v1'):
        raise ValueError('Unknown deployment-condition initializer')
    if initializer != 'static_v1' and cfg['train']['recipe'] not in (GATED_RECIPE,REPAIR_RECIPE):
        raise ValueError('The trajectory initializer requires the corrected training recipe')
    if cfg["train"]["recipe"] in (GATED_RECIPE,REPAIR_RECIPE):
        expected.update({"state.write_gate": True, "loss.temp": 0.,
                         "loss.struct_warmup_steps": 200, "loss.struct_noise_width": .2,
                         "loss.struct_initialization_gradient": True,
                         "train.sampling": "uniform_modes_v1", "train.validate_every": 100})
        if "aux_ratio" in cfg["train"] or "aux_reference" in cfg["train"]:
            raise ValueError("The corrected recipe has no additional adaptive STRUCT scaling")
        if cfg["train"].get("auxiliary_gradient_policy") not in ("none", "project_conflicts"):
            raise ValueError("Explicit auxiliary_gradient_policy must be none or project_conflicts")
        if not math.isfinite(cfg["loss"]["struct"]) or not 0 <= cfg["loss"]["struct"] <= .1:
            raise ValueError("Corrected STRUCT coefficient must be in [0, 0.1]; zero is the FM-only control")
        if cfg['train']['recipe'] == REPAIR_RECIPE:
            expected['state.initializer'] = 'condition_trajectory_v1'
            if not math.isfinite(cfg['loss']['write']) or not 0 <= cfg['loss']['write'] <= .05:
                raise ValueError('Paired WRITE coefficient must be in [0, .05]; zero disables WRITE only')
            for key,limit in (('clean_probability',1.),('max_relative_strength',.4),('warp_latent_pixels',1.)):
                value = cfg['repair'][key]
                if not math.isfinite(value) or not 0 <= value <= limit:
                    raise ValueError(f'repair.{key} must be finite and in [0,{limit}]')
    elif cfg["train"]["recipe"] == TRAINING_RECIPE:
        expected.update({"loss.temp": .05, "train.aux_ratio": .20,
                         "train.aux_reference": "all_trainable_sum_of_norms"})
        if cfg["state"].get("write_gate", False):
            raise ValueError("Legacy checkpoints have no write gates")
    else:
        raise ValueError("Unknown v4.2 training recipe")
    for key, value in expected.items():
        actual = cfg
        for name in key.split("."):
            actual = actual[name]
        if actual != value:
            raise ValueError(f"Fixed v4.2 contract: {key}={value}, got {actual}")
    if cfg["train"]["recipe"] == TRAINING_RECIPE and (not math.isfinite(cfg["loss"]["struct"]) or cfg["loss"]["struct"] <= 0):
        raise ValueError("loss.struct must be a finite positive coefficient")
    for name in ("cache", "annotations", "checkpoints", "logs", "outputs", "asset_lock"):
        output_path(cfg["paths"][name])
    sys.path.insert(0, cfg["paths"]["wan_code"])
    return cfg


def python_for(role):
    return f"/home/liuzhirui/miniconda3/envs/{ENVS[role]}/bin/python"


def subprocess_env(role):
    env = dict(os.environ)
    for key in list(env):
        if key.startswith(("PYTHON", "CONDA_")) or key == "VIRTUAL_ENV":
            env.pop(key)
    prefix = str(Path(python_for(role)).parent.parent)
    for key in ("PATH", "LD_LIBRARY_PATH"):
        entries = [p for p in env.get(key, "").split(":") if p and "/miniconda3/envs/" not in p]
        env[key] = ":".join(([prefix + "/bin"] if key == "PATH" else []) + entries)
    env.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
    return env


def run_role(role, script, *args):
    subprocess.run([python_for(role), "-I", "-B", str(ROOT / script), *map(str, args)],
                   env=subprocess_env(role), check=True)


def require_environment(role):
    expected = Path(python_for(role)).parent.parent
    if Path(sys.prefix).resolve() != expected.resolve():
        raise RuntimeError(f"{role} requires {expected}/bin/python; got {sys.executable}")


def require_single_gpu():
    import torch
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("v4.2 is single GPU; do not launch torchrun with multiple workers")
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


def code_fingerprint():
    files = [p for directory in ("physgen_v42", "train", "tools", "inference")
             for p in (ROOT / directory).glob("*.py")]
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(files)}


def checkpoint_code_compatible(saved, current):
    """Allow only explicitly audited old -> new source digests for resume/inference.

    The JSON ledger is outside the Python fingerprint to avoid a self-referential
    hash. Unknown edits still fail closed; this is not an ignore-code switch.
    """
    if saved == current:
        return True
    ledger = read_json(ROOT / "physgen_v42/checkpoint_compatibility.json")
    return any(saved == entry["before"] and current == entry["after"]
               for entry in ledger["transitions"])


def checkpoint_identity_compatible(saved, current):
    return (saved.keys() == current.keys()
            and all(saved[key] == current[key] for key in saved if key != "code")
            and checkpoint_code_compatible(saved.get("code"), current.get("code")))


def preparation_fingerprint():
    # Cache identity describes tensor numerics, not training architecture or job
    # orchestration. Changing reuse/retry logic must not invalidate VAE outputs.
    import ast
    source = (ROOT / "physgen_v42/data.py").read_text()
    tree = ast.parse(source)
    reader = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "read_window")
    return {"recipe": "nativefps_f121_fp32vae_bf16t5_jepa32_v1",
            "physgen_v42/geometry.py": sha256(ROOT / "physgen_v42/geometry.py"),
            "physgen_v42/encoders.py": sha256(ROOT / "physgen_v42/encoders.py"),
            "read_window": hashlib.sha256(ast.get_source_segment(source, reader).encode()).hexdigest()}


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


def asset_files(cfg):
    paths = cfg["paths"]
    wan = Path(paths["wan_checkpoint"])
    index = read_json(wan / "diffusion_pytorch_model.safetensors.index.json")
    files = [wan / s for s in sorted(set(index["weight_map"].values()))]
    files += [wan / p for p in ("config.json", "diffusion_pytorch_model.safetensors.index.json",
                                "Wan2.2_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth")]
    files += [p for p in (wan / "google/umt5-xxl").rglob("*") if p.is_file()]
    files += [Path(paths["teacher_checkpoint"])]
    for directory in (Path(paths["wan_code"]) / "wan", Path(paths["teacher_code"]) / "app/vjepa_2_1"):
        files += list(directory.rglob("*.py")) + list(directory.rglob("*.yaml"))
    return sorted(set(files))


def lock_assets(cfg, create=False):
    path = Path(cfg["paths"]["asset_lock"])
    previous = read_json(path) if path.exists() else {}
    mode = cfg.get("preparation", {}).get("asset_verification", "metadata")
    if mode not in ("metadata", "sha256"):
        raise ValueError("asset_verification must be metadata or sha256")
    if mode == "metadata":
        result = {}
        for file in asset_files(cfg):
            stat = file.stat()
            result[str(file)] = dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                                     sha256=None, verification="metadata_only")
        if previous and previous != result:
            raise ValueError("Model file metadata changed; use a fresh dataset/cache")
        if create:
            save_json(result, path)
            log(event="asset_inventory", files=len(result), verification="metadata_only", weight_content_hashing=False)
        elif not previous:
            raise FileNotFoundError("Model asset inventory has not been prepared")
        return result
    if not previous and not create:
        raise FileNotFoundError("Run tools/prepare.py --pass assets before encoding/training")
    result = {}
    for file in asset_files(cfg):
        stat = file.stat()
        old = previous.get(str(file), {})
        fingerprint = old.get("sha256") if (old.get("bytes"), old.get("mtime_ns")) == (stat.st_size, stat.st_mtime_ns) else sha256(file)
        result[str(file)] = dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=fingerprint)
    if previous and {p: v["sha256"] for p, v in result.items()} != {p: v["sha256"] for p, v in previous.items()}:
        raise ValueError("Model/code assets changed; use a new version/cache and explicit new lock")
    if create:
        save_json(result, path)
    return result


@contextlib.contextmanager
def job_lock(path):
    import fcntl
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
