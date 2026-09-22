from collections import defaultdict, deque
import random
from pathlib import Path
import math
import re
from statistics import median

import torch

from . import VERSION
from .runtime import digest, preparation_fingerprint, read_json, verify_files


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


def stratified_order(records, seed):
    rng = random.Random(seed)
    groups = defaultdict(list)
    for i, record in enumerate(records):
        groups[record["category"]].append(i)
    for group in groups.values():
        rng.shuffle(group)
    names = sorted(groups)
    rng.shuffle(names)
    queues = {name: deque(groups[name]) for name in names}
    result = []
    while any(queues.values()):
        for name in names:
            if queues[name]:
                result.append(queues[name].popleft())
    return result


class Queue:
    def __init__(self, records, seed):
        self.records, self.seed, self.cursor = records, seed, 0
        self._epoch, self._order = -1, None

    def next(self):
        epoch, index = divmod(self.cursor, len(self.records))
        if self._epoch != epoch:
            self._epoch = epoch
            self._order = stratified_order(self.records, self.seed + epoch)
        self.cursor += 1
        return self._order[index]


class ExposurePlan:
    """Persistent ordinary/temporal queues and mode coverage; exactly 6:2 each update."""
    def __init__(self, records, seed):
        self.records, self.seed = records, seed
        self.ordinary, self.temporal = Queue(records, seed), Queue(records, seed + 7919)
        self.counts = [dict(i2v=0, t2v=0, temp=0) for _ in records]

    def batch(self, step):
        rng = random.Random(self.seed + step * 31)
        samples = [(self.ordinary.next(), False) for _ in range(7)]
        samples.append((self.temporal.next(), True))
        # Fix the temporal 3:1 ratio in every four consecutive JOINT batches.
        offset = (step - 1) % 4
        cycle = (step - 1) // 4
        temporal_t2v = random.Random(self.seed + 100003 + cycle).randrange(4) == offset
        modes = {7: "t2v" if temporal_t2v else "i2v"}
        candidates = list(range(7))
        rng.shuffle(candidates)
        # Higher deficit means a video has received too few T2V exposures; rotate eligible videos.
        candidates.sort(key=lambda i: (self.counts[samples[i][0]]["i2v"] -
                                       3 * self.counts[samples[i][0]]["t2v"]), reverse=True)
        n_t2v = 2 - sum(mode == "t2v" for mode in modes.values())
        for rank, i in enumerate(candidates):
            modes[i] = "t2v" if rank < n_t2v else "i2v"
        results = []
        for i, (index, temporal) in enumerate(samples):
            counts = self.counts[index]
            exposure = counts["i2v"] + counts["t2v"]
            results.append(dict(index=index, mode=modes[i], temporal=temporal,
                                repair=temporal and step >= 801, exposure=exposure))
            counts[modes[i]] += 1
            counts["temp"] += int(temporal)
        rng.shuffle(results)
        return results

    def state_dict(self):
        return dict(ordinary=self.ordinary.cursor, temporal=self.temporal.cursor, counts=self.counts)

    def load_state_dict(self, state):
        self.ordinary.cursor, self.temporal.cursor = state["ordinary"], state["temporal"]
        self.counts = state["counts"]


class Dataset:
    def __init__(self, cfg, split, require_complete=True):
        self.root = Path(cfg["paths"]["cache"])
        if not (self.root / "manifest.json").is_file() or (require_complete and not (self.root / "ready.json").is_file()):
            raise RuntimeError("The new v4.2 source-derived dataset is not ready. Launch train/train_stability_1x96g.sh "
                               "with PREPARE=1 (the default); it prepares the data before starting training.")
        self.manifest = read_json(self.root / "manifest.json")
        if self.manifest["version"] != VERSION or self.manifest["data_config"] != cfg["data"]:
            raise ValueError("v4.2 cache version/config mismatch; old v4 cache is incompatible")
        if self.manifest.get("preparation_code") != preparation_fingerprint():
            raise ValueError("Preprocessing implementation changed; prepare a new cache")
        validate_records(self.manifest["records"], cfg)
        if require_complete:
            ready = read_json(self.root / "ready.json")
            if ready["manifest_hash"] != digest(self.manifest):
                raise ValueError("Cache manifest changed after finalization")
            verify_files(self.root, ready["files"])
        self.records = [r for r in self.manifest["records"] if r["split"] == split]
        expected = self.manifest["source_counts"][split]
        if len(self.records) != expected:
            raise ValueError(f"Expected {expected} {split} records, got {len(self.records)}")

    def __len__(self):
        return len(self.records)

    def load(self, index, mode, device):
        record = self.records[index]
        name = record["id"] + ".pt"
        def load(stage):
            return torch.load(self.root / stage / name, map_location="cpu", weights_only=True)
        vae = load("vae")
        text = load("text")
        result = dict(record=record, latent=vae["latent"].to(device), text=text[None].to(device),
                      target=load("teacher").float().to(device), first=None, anchor=None, instances=None)
        # No per-sample A_image read on T2V exposures, even though the teacher target exists.
        if mode == "i2v":
            result["first"] = vae["first"].to(device)
            result["anchor"] = load("anchors").float().to(device)
        if record.get("objects_reliable"):
            result["instances"] = load("instances").to(device)
        return result


def read_window(record, device="cpu"):
    from decord import VideoReader, cpu
    from .geometry import letterbox
    import numpy as np
    reader = VideoReader(record["video"], ctx=cpu(0), num_threads=1)
    indices = record["indices"]
    actual_pts = reader.get_frame_timestamp(indices)[:, 0].astype(np.float64)
    if not np.allclose(actual_pts, record["pts"], atol=1e-6, rtol=0):
        raise ValueError("Source PTS changed since source indexing")
    parts = []
    for start in range(0, len(indices), 8):
        raw = torch.from_numpy(reader.get_batch(indices[start:start + 8]).asnumpy()).permute(3, 0, 1, 2)
        parts.append(letterbox(raw.float() / 127.5 - 1, record["geometry"]))
    return torch.cat(parts, dim=1)[None].to(device)
