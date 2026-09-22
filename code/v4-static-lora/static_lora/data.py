"""Read existing VAE/T5 data with locally frozen validation and exposure queues."""
from collections import defaultdict
import math
import re
from pathlib import Path

import torch

from . import ROOT
from .sampling import ExposurePlan as ReferencePlan
from .runtime import digest, log, read_json, verify_files


class Dataset:
    def __init__(self, cfg, split, require_complete=True):
        self.root = Path(cfg["paths"]["cache"])
        if not (self.root / "manifest.json").is_file():
            raise FileNotFoundError(f"Prepared VAE/T5 training cache is missing: {self.root}")
        self.manifest = read_json(self.root / "manifest.json")
        contract = read_json(ROOT / "assets/cache_recipe.json")
        if (self.manifest["version"] != contract["version"] or
                self.manifest["data_config"] != cfg["data"] or
                self.manifest.get("preparation_code") != contract["preparation_code"]):
            raise ValueError("Training cache version/config/encoding recipe differs from the frozen LoRA data")
        validate_records(self.manifest["records"], cfg)
        self.records = [r for r in self.manifest["records"] if r["split"] == split]
        if len(self.records) != self.manifest["source_counts"][split]:
            raise ValueError(f"Unexpected record count in {split}")
        if require_complete:
            ready = read_json(self.root / "ready.json")
            if ready["manifest_hash"] != digest(self.manifest):
                raise ValueError("Shared manifest differs from the finalized v4.2 cache")
            # LoRA needs no teacher, anchor, instance targets, or calibration tensors.
            wanted = [f"{stage}/{r['id']}.pt" for r in self.records for stage in ("vae", "text")]
            verify_files(self.root, {name: ready["files"][name] for name in wanted})

    def __len__(self):
        return len(self.records)

    def load(self, index, mode, device):
        if mode not in ("i2v", "t2v"):
            raise ValueError(f"Unknown mode: {mode}")
        record = self.records[index]
        name = record["id"] + ".pt"
        vae = torch.load(self.root / "vae" / name, map_location="cpu", weights_only=True)
        text = torch.load(self.root / "text" / name, map_location="cpu", weights_only=True)
        return dict(record=record, latent=vae["latent"].to(device), text=text[None].to(device),
                    first=vae["first"].to(device) if mode == "i2v" else None)


def check_data(cfg):
    train, validation = Dataset(cfg, "train"), Dataset(cfg, "validation")
    log(event="training_cache_reused", cache=cfg["paths"]["cache"],
        train=len(train), validation=len(validation), tensor_stages=["vae", "text"])
    return train, validation


class ExposurePlan:
    def __init__(self, records, seed):
        if not records:
            raise ValueError("Training records cannot be empty")
        self.records, self.seed = records, seed
        self.reference = ReferencePlan(records, seed)
        self.completed_step = 0

    def batch(self, step):
        if step != self.completed_step + 1 or not 1 <= step <= 1200:
            raise ValueError("LoRA exposure steps must be sequential in 1..1200")
        source = self.reference.batch(step)
        self.completed_step = step
        # Preserve source, mode, exposure order and noise mixture. The original
        # temporal/repair labels do NOT enable any extra objective or repair here.
        return [dict(index=e["index"], mode=e["mode"], exposure=e["exposure"],
                     low_noise=e["temporal"]) for e in source]

    def state_dict(self):
        state = self.reference.state_dict()
        return dict(seed=self.seed, completed_step=self.completed_step,
                    ordinary=state["ordinary"], low_noise=state["temporal"],
                    counts=[dict(i2v=c["i2v"], t2v=c["t2v"], low_noise=c["temp"]) for c in state["counts"]])

    def validate_state(self, state, step):
        if (state.get("seed") != self.seed or state.get("completed_step") != step or
                state.get("ordinary") != 7 * step or state.get("low_noise") != step or
                len(state.get("counts", [])) != len(self.records)):
            raise ValueError("LoRA queue identity/cursors differ from completed step")
        for row in state["counts"]:
            if (set(row) != {"i2v", "t2v", "low_noise"} or
                    any(type(v) is not int or v < 0 for v in row.values()) or
                    row["low_noise"] > row["i2v"] + row["t2v"]):
                raise ValueError("Invalid per-source LoRA exposure counts")
        if tuple(sum(row[k] for row in state["counts"]) for k in ("i2v", "t2v", "low_noise")) != (6 * step, 2 * step, step):
            raise ValueError("Expected eight FM samples, six I2V/two T2V, and one low-noise source per step")

    def load_state_dict(self, state):
        step = state.get("completed_step")
        if type(step) is not int or not 0 <= step <= 1200:
            raise ValueError("Invalid completed step")
        self.validate_state(state, step)
        self.reference = ReferencePlan(self.records, self.seed)
        self.reference.load_state_dict(dict(ordinary=state["ordinary"], temporal=state["low_noise"],
            counts=[dict(i2v=c["i2v"], t2v=c["t2v"], temp=c["low_noise"]) for c in state["counts"]]))
        self.completed_step = step


def validate_records(records, cfg):
    """Validate v4-style source windows and train/validation source isolation."""
    ids, sources = set(), set()
    counts = defaultdict(int)
    for record in records:
        key, source = record["id"], record["source_id"]
        if not re.fullmatch(r"source[0-9]+", key) or not re.fullmatch(r"[0-9a-f]{64}", source):
            raise ValueError(f"Invalid source identity: {key}")
        if key in ids:
            raise ValueError(f"Duplicate dataset index: {key}")
        ids.add(key)
        sources.add(source)
        if record["split"] not in ("train", "validation") or record["category"] not in cfg["data"]["categories"]:
            raise ValueError(f"Invalid split/category: {key}")
        counts[record["split"]] += 1
        indices, pts = record["indices"], record["pts"]
        frames = len(indices)
        if (frames < 5 or frames > cfg["data"]["frames"] or (frames - 1) % 4 or
                any(type(i) is not int for i in indices) or
                indices != list(range(indices[0], indices[0] + frames)) or indices[0] < 0 or
                indices[-1] >= record["source_frames"]):
            raise ValueError(f"Expected a consecutive 4k+1 source window of 5..121 frames: {key}")
        if len(pts) != frames or not all(math.isfinite(t) for t in pts):
            raise ValueError(f"Invalid PTS: {key}")
        gaps = [b - a for a, b in zip(pts, pts[1:])]
        if min(gaps) <= 0:
            raise ValueError(f"Noncontinuous PTS: {key}")
        if not math.isclose(record["duration"], pts[-1] - pts[0], rel_tol=0., abs_tol=1e-6):
            raise ValueError(f"Duration differs from actual source PTS: {key}")
        if not isinstance(record["caption"], str) or not record["caption"].strip():
            raise ValueError(f"Empty source caption: {key}")
        from .geometry import canvas, fit_geometry
        geo = record["geometry"]
        h, w = canvas(geo["source_h"], geo["source_w"], cfg["data"]["max_long_side"], cfg["data"]["max_area"])
        expected_geo = dict(fit_geometry(geo["source_h"], geo["source_w"], h, w, upscale=False), frames=frames)
        if geo != expected_geo:
            raise ValueError(f"Noncanonical letterbox geometry: {key}")
        if any(type(i) is not int or not 0 <= i < frames for i in record["interactions"]):
            raise ValueError(f"Invalid relative interaction index: {key}")
        for obj in record.get("objects", []):
            if len(obj["boxes"]) != frames:
                raise ValueError(f"Object tracking must cover the full clip: {key}")
            for box in obj["boxes"]:
                if box is not None and not (len(box) == 4 and all(math.isfinite(v) for v in box)
                        and 0 <= box[0] < box[2] <= w and 0 <= box[1] < box[3] <= h):
                    raise ValueError(f"Object box is outside the shared canvas: {key}")
    train_sources = {r["source_id"] for r in records if r["split"] == "train"}
    validation_sources = {r["source_id"] for r in records if r["split"] == "validation"}
    if train_sources & validation_sources or not counts["train"] or not counts["validation"]:
        raise ValueError("Train/validation must be nonempty and source-disjoint")
    if cfg.get("purpose") == "quick16_pipeline_only" and (counts["train"], counts["validation"]) != (12, 4):
        raise ValueError("Quick16 requires 12 training and 4 validation clips")
