from __future__ import annotations

from pathlib import Path


V1_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ROOT = Path("/home/liuzhirui/model").resolve()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def require_write_path(path: str | Path, *, v1_root: str | Path = V1_ROOT) -> Path:
    """Return a resolved writable target, rejecting paths outside code/v1.

    ``Path.resolve`` also resolves existing symlink components, preventing a link
    inside code/v1 from escaping into the read-only model tree.
    """

    target = resolved(path)
    allowed_root = resolved(v1_root)
    if not _is_relative_to(target, allowed_root):
        raise ValueError(
            f"write target must stay under {allowed_root}, got {target}"
        )
    if _is_relative_to(target, DEFAULT_MODEL_ROOT):
        raise ValueError(f"refusing to write inside model tree: {target}")
    return target


def ensure_runtime_dirs(
    output_root: str | Path,
    *,
    v1_root: str | Path = V1_ROOT,
) -> dict[str, Path]:
    roots = {
        "output": require_write_path(output_root, v1_root=v1_root),
        "logs": require_write_path(resolved(v1_root) / "logs" / "ace_router"),
        "cache": require_write_path(resolved(v1_root) / ".cache"),
        "tmp": require_write_path(resolved(v1_root) / "tmp"),
    }
    for directory in roots.values():
        directory.mkdir(parents=True, exist_ok=True)
    return roots
