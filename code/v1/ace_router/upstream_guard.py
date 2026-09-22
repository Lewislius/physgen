from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


EXPECTED_SOURCE_HASHES = {
    "wan/modules/model.py": "8b39115298ca7322806c19b3165b3f435a94fe4a58f0624aec24f8e7f4997432",
    "wan/modules/attention.py": "c79158e1cc9b7a17bc934e6acb34f42e588a41adb89801146957cda2d6d4379d",
    "wan/modules/t5.py": "8b0cebf3192c542f92a344255a06c203df3ba24160715899a055cc8de0cd930f",
    "wan/textimage2video.py": "228f2fabf23014ed41ec6b8cd713d5be7371bb558501c71a62a0258b491a877b",
    "wan/configs/wan_ti2v_5B.py": "26d9e7b9c555eb0900b13751267a596556f869e7aee2d1572afc2a0a4a76f4c7",
}
EXPECTED_CHECKPOINT_HASHES = {
    "config.json": "d1fea36899d00c2501b836c13ad65af56e2f9529ba622e50886d3f5c3e6c02bc",
    "diffusion_pytorch_model.safetensors.index.json": "bfa2337f1163e195d24151a72298daf34a620543898109be47e414c8daa5b3fe",
}
WEIGHT_GLOB = "diffusion_pytorch_model-*.safetensors"


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git_head(repo: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def capture_upstream(
    wan_repo: str | Path,
    checkpoint_dir: str | Path,
) -> dict[str, Any]:
    repo = Path(wan_repo).expanduser().resolve()
    checkpoint = Path(checkpoint_dir).expanduser().resolve()
    if not repo.is_dir():
        raise ValueError(f"Wan repo does not exist: {repo}")
    if not checkpoint.is_dir():
        raise ValueError(f"checkpoint directory does not exist: {checkpoint}")

    source_hashes: dict[str, str] = {}
    for relative_path in EXPECTED_SOURCE_HASHES:
        path = repo / relative_path
        if not path.is_file():
            raise ValueError(f"missing required Wan source: {path}")
        source_hashes[relative_path] = sha256_file(path)

    checkpoint_hashes: dict[str, str] = {}
    for relative_path in EXPECTED_CHECKPOINT_HASHES:
        path = checkpoint / relative_path
        if not path.is_file():
            raise ValueError(f"missing checkpoint metadata: {path}")
        checkpoint_hashes[relative_path] = sha256_file(path)

    weight_shards = []
    for path in sorted(checkpoint.glob(WEIGHT_GLOB)):
        stat = path.stat()
        weight_shards.append(
            {
                "name": path.name,
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    if not weight_shards:
        raise ValueError(f"no Wan weight shards found under {checkpoint}")

    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    return {
        "wan_repo": str(repo),
        "wan_git_head": _git_head(repo),
        "source_hashes": source_hashes,
        "checkpoint": str(checkpoint),
        "checkpoint_hashes": checkpoint_hashes,
        "weight_shards": weight_shards,
        "checkpoint_model_config": {
            key: config.get(key)
            for key in (
                "_class_name",
                "model_type",
                "num_layers",
                "dim",
                "num_heads",
                "text_len",
                "in_dim",
                "out_dim",
            )
        },
    }


def verify_expected_upstream(
    snapshot: dict[str, Any], *, allow_drift: bool = False
) -> list[str]:
    mismatches: list[str] = []
    for path, expected in EXPECTED_SOURCE_HASHES.items():
        actual = snapshot["source_hashes"].get(path)
        if actual != expected:
            mismatches.append(f"{path}: expected {expected}, got {actual}")
    for path, expected in EXPECTED_CHECKPOINT_HASHES.items():
        actual = snapshot["checkpoint_hashes"].get(path)
        if actual != expected:
            mismatches.append(f"checkpoint/{path}: expected {expected}, got {actual}")

    model_config = snapshot["checkpoint_model_config"]
    expected_config = {
        "model_type": "ti2v",
        "num_layers": 30,
        "dim": 3072,
        "num_heads": 24,
        "text_len": 512,
    }
    for key, expected in expected_config.items():
        if model_config.get(key) != expected:
            mismatches.append(
                f"checkpoint config {key}: expected {expected!r}, "
                f"got {model_config.get(key)!r}"
            )
    if mismatches and not allow_drift:
        joined = "\n".join(f"- {item}" for item in mismatches)
        raise RuntimeError(
            "Wan source/checkpoint drift detected. Re-audit the ACE adapter or pass "
            f"--allow_upstream_drift for an explicit diagnostic run:\n{joined}"
        )
    return mismatches


def assert_upstream_unchanged(
    before: dict[str, Any], after: dict[str, Any]
) -> None:
    keys = ("source_hashes", "checkpoint_hashes", "weight_shards")
    changed = [key for key in keys if before.get(key) != after.get(key)]
    if changed:
        raise RuntimeError(f"read-only upstream changed during inference: {changed}")
