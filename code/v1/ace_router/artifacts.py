from __future__ import annotations

import json
import os
import platform
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .paths import require_write_path
from .upstream_guard import sha256_file, sha256_text


def utc_run_id(prefix: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{prefix.lower()}-{timestamp}"


def jsonable_path_hashes(paths: dict[str, Path | None]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, path in paths.items():
        if path is None:
            result[name] = None
        else:
            result[name] = {
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
    return result


def context_record(
    semantic_text: str,
    positive_text: str,
    counterfactual_text: str,
    cfg_negative_prompt: str,
    token_lengths: dict[str, int] | None = None,
) -> dict[str, Any]:
    texts = {
        "semantic": semantic_text,
        "positive": positive_text,
        "counterfactual": counterfactual_text,
        "cfg_negative": cfg_negative_prompt,
    }
    return {
        "semantic_text": semantic_text,
        "positive_text": positive_text,
        "counterfactual_text": counterfactual_text,
        "cfg_negative_prompt": cfg_negative_prompt,
        "utf8_sha256": {name: sha256_text(text) for name, text in texts.items()},
        "token_lengths": token_lengths or {},
    }


def atomic_write_json(path: str | Path, data: Any) -> Path:
    target = require_write_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return target


def atomic_write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> Path:
    target = require_write_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return target


def runtime_versions(torch_module: Any, diffusers_module: Any) -> dict[str, Any]:
    gpu_name = None
    cuda_capability = None
    if torch_module.cuda.is_available():
        gpu_name = torch_module.cuda.get_device_name(torch_module.cuda.current_device())
        cuda_capability = list(
            torch_module.cuda.get_device_capability(torch_module.cuda.current_device())
        )
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch_module.__version__,
        "diffusers": diffusers_module.__version__,
        "cuda_runtime": torch_module.version.cuda,
        "cudnn": torch_module.backends.cudnn.version(),
        "gpu": gpu_name,
        "cuda_capability": cuda_capability,
    }
