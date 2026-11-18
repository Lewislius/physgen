"""Build the deterministic 20k-candidate pool for R20FT WISA preprocessing.

The pool excludes every source video already present in the 3,857 approved
cache, so the new 18,400 training clips cannot duplicate an existing one.
No video is decoded here; the index stage verifies geometry/frame count.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import atomic_json, load_config

PREFILTER_MIN_FRAMES = 81


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--total", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = Path(cfg["paths"]["output_root"])
    out.mkdir(parents=True, exist_ok=True)

    catalog = json.loads(Path(cfg["paths"]["source_catalog"]).read_text())
    existing = json.loads(Path(cfg["paths"]["existing_manifest"]).read_text())["records"]
    existing_names = {row["video_name"] for row in existing}
    existing_ids = {row["id"] for row in existing}
    video_root = Path(cfg["paths"]["video_root"])

    selection = cfg["selection"]
    total = args.total or int(selection["candidates_total"])
    rng = random.Random(int(selection["seed"]))

    candidates = []
    seen = set()
    skipped = dict(existing=0, no_caption=0, too_small=0, too_short=0, bad_fps=0,
                   path_escape=0)
    for catalog_index, item in enumerate(catalog):
        name = item.get("video_name")
        if not name or name in seen:
            continue
        seen.add(name)
        if name in existing_names:
            skipped["existing"] += 1
            continue
        caption = item.get("captions")
        if not isinstance(caption, str) or not caption.strip():
            skipped["no_caption"] += 1
            continue
        width, height = item.get("width"), item.get("height")
        if not isinstance(width, int) or not isinstance(height, int) or width < 832 or height < 480:
            skipped["too_small"] += 1
            continue
        fps, duration = item.get("fps"), item.get("duration")
        if not isinstance(fps, (int, float)) or not isinstance(duration, (int, float)) or fps <= 0:
            skipped["bad_fps"] += 1
            continue
        if float(fps) * float(duration) + 1e-6 < PREFILTER_MIN_FRAMES:
            skipped["too_short"] += 1
            continue
        path = (video_root / name).resolve()
        try:
            path.relative_to(video_root.resolve())
        except ValueError:
            skipped["path_escape"] += 1
            continue
        sample_id = f"r20ft{catalog_index:07d}"
        if sample_id in existing_ids:
            raise ValueError(f"Generated ID collides with existing cache: {sample_id}")
        candidates.append(dict(
            id=sample_id, catalog_index=catalog_index, video_name=name,
            video=str(path), original_caption=caption.strip(),
            catalog_width=width, catalog_height=height,
            catalog_fps=float(fps), catalog_duration=float(duration),
            label=item.get("label"),
            source_annotation="wisa_catalog",
            split="train",
        ))

    rng.shuffle(candidates)
    # Keep the manifest order deterministic and independent of the shuffle.
    candidates = candidates[:total]
    candidates.sort(key=lambda row: row["catalog_index"])
    shard_count = int(selection["shard_count"])
    shard_size = int(selection["shard_size"])
    if len(candidates) < shard_count * shard_size:
        raise RuntimeError(f"Only {len(candidates)} candidates, need {shard_count * shard_size}")
    for index, row in enumerate(candidates):
        row["pool_index"] = index
        row["shard"] = index // shard_size
    # The last shard gets the remaining candidates; the shard sizes are equal
    # for the two-task configuration.
    for shard in range(shard_count):
        shard_rows = [row for row in candidates if row["shard"] == shard]
        atomic_json(dict(version=cfg["run_name"], shard=shard, count=len(shard_rows),
                         candidates=shard_rows),
                    out / "work" / f"candidates.shard{shard}.json")
    atomic_json(dict(version=cfg["run_name"], total=len(candidates),
                     shard_count=shard_count, shard_size=shard_size,
                     seed=int(selection["seed"]), skipped=skipped,
                     excluded_existing_sources=len(existing_names),
                     candidates=candidates),
                out / "work" / "candidates.json")
    atomic_json(dict(version=cfg["run_name"], target_new=int(selection["new_records"]),
                     existing_training=int(selection["existing_training_records"]),
                     target_total=int(selection["target_total"]),
                     candidates=len(candidates), shard_count=shard_count,
                     shard_size=shard_size, seed=int(selection["seed"]),
                     skipped=skipped),
                out / "work" / "selection.json")
    print(json.dumps(dict(candidates=len(candidates), shard_count=shard_count,
                          shard_size=shard_size, skipped=skipped,
                          excluded_existing_sources=len(existing_names)),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
