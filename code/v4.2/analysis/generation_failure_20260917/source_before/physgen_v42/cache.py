"""Validate and reuse individual encodings, independently of training/checkpoint versions."""
import errno
import os
from pathlib import Path
import pickle
import shutil

from .runtime import atomic_write, digest, log, preparation_fingerprint, read_json, save_json


# Audited original v4.2 f121 recipe. Never accept arbitrary legacy fingerprints.
LEGACY_PREPARATION = {
    "physgen_v42/geometry.py": "7bffb5719539c594d73c3c6d9bdebac44628a4bc8bd9b73d0fb1511c018814e2",
    "physgen_v42/encoders.py": "53f23662005593af81d1ad015716010140c51c6038248e677eef4395e90d56e9",
    "tools/prepare.py": "9762266d061bfbdd33b6a527a0a52ae2e0a1740f5260721f7c894c5505adc24e",
    "tools/dataset_index.py": "f9d961b27662e1aafc1f982d9a653718acd5e370a20b32baeaeddb4665d7a7fa",
    "tools/bootstrap.py": "4c9ee22261d9b94af9763d5ec05a7da79320b0acf7e7dd538f8bb5b1945fc31f",
}
LEGACY_READER = "8f844375b00667382b2b8e9f1f06a331a1cf27ace8cc6bf2daf45c737fe5ddca"
ENCODING_FIELDS = ("frames", "max_long_side", "max_area", "decode_chunk_frames", "teacher_frames", "teacher_size")
CACHE_ERRORS = (OSError, ValueError, RuntimeError, EOFError, KeyError, TypeError, AttributeError,
                IndexError, pickle.UnpicklingError)


def compatible_preparation(saved):
    current = preparation_fingerprint()
    return saved == current or (
        saved == LEGACY_PREPARATION and current["read_window"] == LEGACY_READER and
        all(current[k] == LEGACY_PREPARATION[k] for k in ("physgen_v42/geometry.py", "physgen_v42/encoders.py")))


def encoding_config(data):
    return {k: data.get(k) for k in ENCODING_FIELDS}


def sample_inputs(stage, record):
    if stage == "text":
        return {"caption": record["caption"]}
    keys = ("source_id", "video", "source_file", "geometry", "indices", "pts")
    return {k: record[k] for k in keys}


def validate(stage, value, record):
    import torch
    h, w = record["geometry"]["h"], record["geometry"]["w"]

    def tensor(x, shape, dtype=None):
        if (not isinstance(x, torch.Tensor) or tuple(x.shape) != tuple(shape) or
                (dtype is not None and x.dtype != dtype) or not bool(torch.isfinite(x).all())):
            raise ValueError(f"Invalid {stage} cache for {record['id']}")

    if stage == "vae":
        tensor(value["latent"], (1, 48, 1 + (len(record["indices"]) - 1) // 4, h // 16, w // 16), torch.float32)
        tensor(value["first"], (1, 48, 1, h // 16, w // 16), torch.float32)
    elif stage == "text":
        if not isinstance(value, torch.Tensor) or value.ndim != 2 or not 1 <= value.shape[0] <= 512:
            raise ValueError("Invalid T5 shape")
        tensor(value, (value.shape[0], 4096), torch.bfloat16)
    elif stage == "teacher":
        tensor(value, (1, 16, 576, 1664), torch.bfloat16)
    elif stage == "anchors":
        tensor(value, (1, 576, 1664), torch.bfloat16)
    elif stage == "instances":
        tensor(value, (sum(o["reliable"] for o in record["objects"]), 16, 576))
    else:
        raise ValueError(f"Unknown cache stage: {stage}")


def link_or_copy(source, target):
    """Hardlinks save disk; atomic replacement on repairs leaves the source intact."""
    def write(temporary):
        os.unlink(temporary)
        try:
            os.link(source, temporary)
        except OSError as error:
            if error.errno not in (errno.EXDEV, errno.EPERM, errno.EACCES, errno.ENOTSUP):
                raise
            shutil.copy2(source, temporary)
    atomic_write(target, write)


class CacheStore:
    def __init__(self, cfg, manifest):
        self.root = Path(cfg["paths"]["cache"])
        self.manifest = manifest
        self.receipts_path = self.root / "validated_cache.json"
        try:
            self.receipts = read_json(self.receipts_path)
            if not isinstance(self.receipts, dict):
                self.receipts = {}
        except (OSError, ValueError):
            self.receipts = {}
        self.contract = digest(dict(code=preparation_fingerprint(), assets=manifest["assets"],
                                    data=encoding_config(manifest["data_config"])))
        self.sources = []
        for source in cfg.get("preparation", {}).get("reuse_sources", []):
            source = Path(source)
            if source.resolve() == self.root.resolve() or not (source / "manifest.json").is_file():
                continue
            try:
                other = read_json(source / "manifest.json")
                if (not compatible_preparation(other.get("preparation_code")) or
                        other["assets"] != manifest["assets"] or
                        encoding_config(other["data_config"]) != encoding_config(manifest["data_config"])):
                    log(event="cache_source_skipped", source=str(source), reason="incompatible encoding inputs")
                    continue
                self.sources.append((source, other, {r["id"]: r for r in other["records"]}))
            except (OSError, ValueError, KeyError, TypeError) as error:
                log(event="cache_source_skipped", source=str(source), reason=str(error))

    def _key(self, stage, record, negative):
        return digest(dict(contract=self.contract, stage=stage,
                           inputs={"caption": self.manifest["negative_prompt"]} if negative else sample_inputs(stage, record)))

    @staticmethod
    def _stat(path):
        stat = path.stat()
        return dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns,
                    inode=stat.st_ino, device=stat.st_dev)

    def remember(self, stage, record, negative=False):
        relative = "negative.pt" if negative else f"{stage}/{record['id']}.pt"
        self.receipts[relative] = dict(key=self._key(stage, record, negative), **self._stat(self.root / relative))

    def ensure(self, stage, record, negative=False):
        """Return local/reused/missing; malformed files become missing work, not fatal errors."""
        import torch
        relative = "negative.pt" if negative else f"{stage}/{record['id']}.pt"
        local = self.root / relative
        key = self._key(stage, record, negative)
        candidates = [(local, True)]
        for root, manifest, records in self.sources:
            other = records.get(record["id"])
            same = manifest["negative_prompt"] == self.manifest["negative_prompt"] if negative else (
                other is not None and sample_inputs(stage, other) == sample_inputs(stage, record))
            if same:
                candidates.append((root / relative, False))
        for path, own in candidates:
            if not path.is_file():
                continue
            try:
                before = self._stat(path)
                receipt = self.receipts.get(relative, {})
                if own and receipt == dict(key=key, **before):
                    return "local"
                # A receipt with different inputs means a stale tensor, even if its shape matches.
                if own and receipt and receipt.get("key") != key:
                    continue
                validate(stage, torch.load(path, map_location="cpu", weights_only=True), record)
                if before != self._stat(path):
                    raise ValueError("Cache changed while being validated")
                if not own:
                    link_or_copy(path, local)
                    linked = self._stat(local)
                    if (linked["bytes"], linked["mtime_ns"]) != (before["bytes"], before["mtime_ns"]):
                        raise ValueError("Cache source changed while being reused")
                self.remember(stage, record, negative)
                return "local" if own else "reused"
            except CACHE_ERRORS as error:
                log(event="cache_invalid", stage=stage, path=str(path), reason=str(error))
        return "missing"

    def save(self):
        save_json(self.receipts, self.receipts_path)


def ready_valid(cfg):
    """Fast startup check; missing/changed files trigger stage-wise validation and repair."""
    from . import VERSION
    from .runtime import sha256, verify_files
    root = Path(cfg["paths"]["cache"])
    try:
        manifest, ready = read_json(root / "manifest.json"), read_json(root / "ready.json")
        if (manifest["version"] != VERSION or manifest["data_config"] != cfg["data"] or
                manifest["preparation_code"] != preparation_fingerprint() or
                ready["manifest_hash"] != digest(manifest) or ready["statistics_sha256"] != sha256(root / "stats.pt")):
            return False
        expected = {f"{stage}/{r['id']}.pt" for r in manifest["records"] for stage in ("vae", "text", "teacher", "anchors")}
        if set(ready["files"]) != expected | {"negative.pt"}:
            return False
        verify_files(root, ready["files"])
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False
