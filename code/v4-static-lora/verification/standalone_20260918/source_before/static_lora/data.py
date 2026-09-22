"""Read only the existing VAE/T5 caches and reproduce formal v4.2 source/mode queues."""
from copy import deepcopy
from pathlib import Path

import torch

from physgen_v42.data import Dataset as ReferenceDataset, ExposurePlan as ReferencePlan
from .runtime import digest, log, read_json, verify_files


class Dataset(ReferenceDataset):
    def __init__(self, cfg, split, require_complete=True):
        super().__init__(cfg, split, require_complete=False)
        if require_complete:
            ready = read_json(self.root / "ready.json")
            if ready["manifest_hash"] != digest(self.manifest):
                raise ValueError("Shared manifest differs from the finalized v4.2 cache")
            # LoRA needs no teacher, anchor, instance targets, or calibration tensors.
            wanted = [f"{stage}/{r['id']}.pt" for r in self.records for stage in ("vae", "text")]
            verify_files(self.root, {name: ready["files"][name] for name in wanted})

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
    log(event="existing_v42_cache_reused", cache=cfg["paths"]["cache"],
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
