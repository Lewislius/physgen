"""Validated, resumable preprocessing caches; all checks run on CPU."""
import json
import os
from pathlib import Path
import tempfile

import torch


def atomic_write(path, writer):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_json(value, path):
    atomic_write(path, lambda temporary: Path(temporary).write_text(
        json.dumps(value, ensure_ascii=False, indent=2)))


def save_tensor(value, path):
    atomic_write(path, lambda temporary: torch.save(value, temporary))


def manifest_identity(manifest):
    # Output locations and decoded views do not change the encoder inputs.
    return dict(data=manifest["data"], teacher=manifest["teacher"],
                paths={k: v for k, v in manifest["paths"].items()
                       if k not in ("cache_root", "checkpoint_root", "log_root")},
                records=[{k: v for k, v in record.items() if k not in ("view", "excluded_reason")}
                         for record in manifest["records"]])


def preserve_manifest(root, manifest):
    path = Path(root) / "manifest.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if manifest_identity(previous) != manifest_identity(manifest):
            raise ValueError(f"Cache inputs differ from {path}. Choose a new cache_root; "
                             "existing caches were not overwritten.")
        return previous
    if any(next((Path(root) / stage).glob("*.pt"), None) is not None
           for stage in ("vae", "text", "teacher")) or (Path(root) / "null_text.pt").exists():
        raise ValueError(f"Cache files exist without {path}; choose a new cache_root.")
    save_json(manifest, path)
    return manifest


def check_manifest_config(manifest, config):
    if manifest["data"] != config["data"] or any(
            manifest["paths"].get(key) != value for key, value in config["paths"].items()
            if key not in ("cache_root", "checkpoint_root", "log_root")):
        raise ValueError("Configuration differs from the cached manifest; choose a new cache_root.")


def _tensor(value, shape):
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape):
        raise ValueError(f"Expected tensor shape {tuple(shape)}, got {getattr(value, 'shape', None)}")
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise ValueError("Cache tensor is not floating point or contains NaN/Inf")


def _text(value):
    if not isinstance(value, torch.Tensor) or value.ndim != 2 or not 1 <= value.shape[0] <= 512:
        raise ValueError("Invalid T5 sequence")
    _tensor(value, (value.shape[0], 4096))


def read_view(root, record):
    return json.loads((Path(root) / "views" / f"{record['index']:07d}.json").read_text())


def exclude_short_views(root, manifest, require_all_views=False):
    """Keep the original selection and splits, explicitly marking unusable clips."""
    for record in manifest["records"]:
        path = Path(root) / "views" / f"{record['index']:07d}.json"
        if not path.exists() and not require_all_views:
            continue
        view = read_view(root, record)
        if view["frames"] < 5:
            record["excluded_reason"] = dict(reason="fewer_than_5_contiguous_frames",
                                             source_frames=view["source_frames"], frames=view["frames"])
    save_json(manifest, Path(root) / "manifest.json")
    return [dict(index=r["index"], **r["excluded_reason"]) for r in manifest["records"]
            if r.get("excluded_reason")]


def validate_value(stage, value, view, cfg):
    if stage == "text":
        _text(value)
        return
    frames, height, width = view["frames"], view["height"], view["width"]
    teacher_frames = 2 * (min(frames, cfg["teacher_frames"]) // 2)
    if (frames < 5 or (frames - 1) % 4 or frames > cfg["max_frames"]
            or height % 32 or width % 32 or min(height, width) <= 0
            or teacher_frames != view["teacher_frames"]):
        raise ValueError("Invalid cached video geometry")
    tokens = teacher_frames // 2 * (cfg["teacher_size"] // 16) ** 2
    if stage == "teacher":
        _tensor(value, (1, tokens, 1664))
        return
    if value["video_metadata"] != view:
        raise ValueError("VAE metadata differs from views JSON")
    _tensor(value["latent"], (1, 48, (frames - 1) // 4 + 1, height // 16, width // 16))
    _tensor(value["first"], (1, 48, 1, height // 16, width // 16))
    _tensor(value["times"], (frames,))
    _tensor(value["target_weight"], (1, tokens))
    if not bool((value["times"].diff() > 0).all()):
        raise ValueError("Non-increasing timestamps")
    if not torch.equal(value["times"], torch.tensor(view["actual_times"], dtype=value["times"].dtype)):
        raise ValueError("VAE timestamps differ from views JSON")


def cache_error(root, stage, record=None, cfg=None):
    """Return None only for readable tensors with the expected shape and finite values."""
    root = Path(root)
    path = root / "null_text.pt" if stage == "null_text" else root / stage / f"{record['index']:07d}.pt"
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
        if stage == "null_text":
            _text(value)
        else:
            view = None if stage == "text" else read_view(root, record)
            validate_value(stage, value, view, cfg)
    except Exception as error:
        return f"{type(error).__name__}: {error}"
    return None


def audit_cache(root, manifest):
    records = [r for r in manifest["records"] if not r.get("excluded_reason")]
    excluded = [dict(index=r["index"], **r["excluded_reason"]) for r in manifest["records"]
                if r.get("excluded_reason")]
    summary = {stage: {"valid": 0, "invalid": 0} for stage in ("vae", "text", "teacher")}
    examples = []
    for stage, counts in summary.items():
        for record in records:
            error = cache_error(root, stage, record, manifest["data"])
            counts["invalid" if error else "valid"] += 1
            if error and len(examples) < 8:
                examples.append(dict(stage=stage, index=record["index"], error=error))
    null_error = cache_error(root, "null_text")
    return dict(selected_samples=len(manifest["records"]), samples=len(records), excluded=excluded,
                stages=summary, null_text_error=null_error,
                complete=bool(records) and not null_error
                         and all(counts["invalid"] == 0 for counts in summary.values()),
                examples=examples)


def finalize_cache(root, manifest):
    summary = audit_cache(root, manifest)
    if not summary["complete"]:
        raise ValueError("Cache is incomplete; rerun PREPARE=1. " + json.dumps(summary))
    for record in manifest["records"]:
        record["view"] = read_view(root, record)
    save_json(manifest, Path(root) / "manifest.json")
    return summary
