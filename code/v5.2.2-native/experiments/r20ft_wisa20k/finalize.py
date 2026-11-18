"""Merge the two completed shards into the final 20k R20FT cache.

The final cache keeps the existing 1,600 training clips by symlink and adds
18,400 newly processed clips. Only R20FT-readable features are kept; JEPA,
first-image VAE, requirements/tasks/change/video and I2V controls are absent.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import atomic_json, load_config, sha1_text

STAGES = ("caption", "reasoning", "text_native", "vae")


def word_count(text):
    return len(re.findall(r"\S+", text or ""))


def link_file(src, dst):
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.link(os.path.realpath(src), dst)
    except OSError:
        os.symlink(os.path.realpath(src), dst)


def row_files(cfg, row, shard):
    root = Path(cfg["paths"]["output_root"])
    base = root / f"shard{shard}"
    return dict(
        vae=base / "base" / "vae" / f"{row['id']}.pt",
        text=base / "base" / "text" / f"{row['id']}.pt",
        input_text=base / "reasoning" / "input" / f"{row['id']}_text.pt",
        input_null=base / "reasoning" / "input" / f"{row['id']}_null.pt",
        words_text=base / "reasoning" / "words" / f"{row['id']}_text.pt",
        words_null=base / "reasoning" / "words" / f"{row['id']}_null.pt",
        native_text=base / "reasoning" / "native_words" / f"{row['id']}_text.pt",
        native_null=base / "reasoning" / "native_words" / f"{row['id']}_null.pt",
    )


def row_complete(cfg, row, shard):
    if row.get("status") != "active":
        return False
    completed = row.get("completed", {})
    if any(not completed.get(stage) for stage in STAGES):
        return False
    return all(path.is_file() for path in row_files(cfg, row, shard).values())


def load_new_rows(cfg):
    rows = []
    for shard in range(int(cfg["selection"]["shard_count"])):
        candidates = json.loads(
            (Path(cfg["paths"]["output_root"]) / "work" /
             f"candidates.shard{shard}.json").read_text())["candidates"]
        for candidate in candidates:
            path = Path(cfg["paths"]["output_root"]) / "work" / "rows" / f"{candidate['id']}.json"
            if path.is_file():
                row = json.loads(path.read_text())
                if row_complete(cfg, row, shard):
                    row["_shard"] = shard
                    rows.append(row)
    return rows


def select_new(rows, per_shard, target):
    groups = {}
    for row in rows:
        groups.setdefault(row["_shard"], []).append(row)
    for group in groups.values():
        group.sort(key=lambda item: item["catalog_index"])
    selected = []
    for shard in sorted(groups):
        selected.extend(groups[shard][:per_shard])
    if len(selected) < target:
        selected = [row for shard in sorted(groups) for row in groups[shard]]
        selected.sort(key=lambda item: (item["_shard"], item["catalog_index"]))
        selected = selected[:target]
    if len(selected) < target:
        raise RuntimeError(f"Only {len(selected)} complete new clips, need {target}")
    return selected


def existing_rows(cfg):
    sequence = json.loads(Path(cfg["paths"]["existing_sequence"]).read_text())
    ids = sequence["ids"]
    manifest = json.loads(Path(cfg["paths"]["existing_manifest"]).read_text())["records"]
    by_id = {row["id"]: row for row in manifest}
    rows = []
    for sample_id in ids:
        row = by_id[sample_id]
        paths = [Path(cfg["paths"]["existing_vae"]) / f"{sample_id}.pt",
                 Path(cfg["paths"]["existing_text"]) / f"{sample_id}.pt",
                 Path(cfg["paths"]["existing_input"]) / f"{sample_id}_text.pt",
                 Path(cfg["paths"]["existing_input"]) / f"{sample_id}_null.pt",
                 Path(cfg["paths"]["existing_words"]) / f"{sample_id}_text.pt",
                 Path(cfg["paths"]["existing_words"]) / f"{sample_id}_null.pt",
                 Path(cfg["paths"]["existing_native_words"]) / f"{sample_id}_text.pt",
                 Path(cfg["paths"]["existing_native_words"]) / f"{sample_id}_null.pt"]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Existing cache incomplete for {sample_id}: {missing[:3]}")
        rows.append(row)
    if len(rows) != int(cfg["selection"]["existing_training_records"]):
        raise ValueError(f"Existing sequence has {len(rows)} rows")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Existing sequence contains duplicates")
    return rows


def clean_row(row):
    skip = {"completed", "status", "failure", "_shard", "shard", "pool_index",
            "catalog_width", "catalog_height", "catalog_fps", "catalog_duration"}
    value = {key: item for key, item in row.items() if key not in skip}
    value["split"] = "train"
    value.setdefault("source_id", sha1_text(row["video_name"]))
    return value


def build_links(cfg, existing, selected):
    root = Path(cfg["paths"]["output_root"])
    final_vae = root / "base" / "vae"
    final_text = root / "base" / "text"
    final_input = root / "reasoning" / "input"
    final_words = root / "reasoning" / "words"
    final_native = root / "reasoning" / "native_words"
    for row in existing:
        sample_id = row["id"]
        link_file(Path(cfg["paths"]["existing_vae"]) / f"{sample_id}.pt", final_vae / f"{sample_id}.pt")
        link_file(Path(cfg["paths"]["existing_text"]) / f"{sample_id}.pt", final_text / f"{sample_id}.pt")
        link_file(Path(cfg["paths"]["existing_input"]) / f"{sample_id}_text.pt", final_input / f"{sample_id}_text.pt")
        link_file(Path(cfg["paths"]["existing_input"]) / f"{sample_id}_null.pt", final_input / f"{sample_id}_null.pt")
        link_file(Path(cfg["paths"]["existing_words"]) / f"{sample_id}_text.pt", final_words / f"{sample_id}_text.pt")
        link_file(Path(cfg["paths"]["existing_words"]) / f"{sample_id}_null.pt", final_words / f"{sample_id}_null.pt")
        link_file(Path(cfg["paths"]["existing_native_words"]) / f"{sample_id}_text.pt", final_native / f"{sample_id}_text.pt")
        link_file(Path(cfg["paths"]["existing_native_words"]) / f"{sample_id}_null.pt", final_native / f"{sample_id}_null.pt")
    for row in selected:
        sample_id = row["id"]
        files = row_files(cfg, row, row["_shard"])
        link_file(files["vae"], final_vae / f"{sample_id}.pt")
        link_file(files["text"], final_text / f"{sample_id}.pt")
        link_file(files["input_text"], final_input / f"{sample_id}_text.pt")
        link_file(files["input_null"], final_input / f"{sample_id}_null.pt")
        link_file(files["words_text"], final_words / f"{sample_id}_text.pt")
        link_file(files["words_null"], final_words / f"{sample_id}_null.pt")
        link_file(files["native_text"], final_native / f"{sample_id}_text.pt")
        link_file(files["native_null"], final_native / f"{sample_id}_null.pt")
    null_path = root / "base" / "null_text.pt"
    if not null_path.exists():
        os.symlink(os.path.realpath(cfg["paths"]["existing_null_text"]), null_path)


def audit(cfg, rows):
    root = Path(cfg["paths"]["output_root"])
    errors = []
    stats = dict(records=len(rows), vae=0, text=0, input_text=0, input_null=0,
                 native_text=0, native_null=0)
    for index, row in enumerate(rows, 1):
        sample_id = row["id"]
        try:
            vae = torch.load(root / "base" / "vae" / f"{sample_id}.pt",
                             map_location="cpu", weights_only=True)
            if tuple(vae["latent"].shape) != (1, 48, 21, 30, 52):
                raise ValueError(f"latent {tuple(vae['latent'].shape)}")
            stats["vae"] += 1
            text = torch.load(root / "base" / "text" / f"{sample_id}.pt",
                              map_location="cpu", weights_only=True)
            if text.ndim != 2 or text.shape[1] != 4096:
                raise ValueError(f"text {tuple(text.shape)}")
            stats["text"] += 1
            input_text = torch.load(root / "reasoning" / "input" / f"{sample_id}_text.pt",
                                    map_location="cpu", weights_only=True)
            if (input_text["sample"] != sample_id or input_text["scope"] != "request_only"
                    or input_text["caption"] != row["caption"]):
                raise ValueError("input text provenance")
            stats["input_text"] += 1
            words_text = torch.load(root / "reasoning" / "words" / f"{sample_id}_text.pt",
                                    map_location="cpu", weights_only=True)
            if words_text["word_strings"] != input_text["word_strings"]:
                raise ValueError("words/input word strings differ")
            native_text = torch.load(root / "reasoning" / "native_words" / f"{sample_id}_text.pt",
                                     map_location="cpu", weights_only=True)
            if (native_text["sample"] != sample_id or native_text["scope"] != "request_only"
                    or native_text["word_strings"] != words_text["word_strings"]
                    or tuple(native_text["native_words"].shape) != (len(words_text["word_strings"]), 3072)):
                raise ValueError("native text shape/provenance")
            stats["native_text"] += 1
            input_null = torch.load(root / "reasoning" / "input" / f"{sample_id}_null.pt",
                                    map_location="cpu", weights_only=True)
            words_null = torch.load(root / "reasoning" / "words" / f"{sample_id}_null.pt",
                                    map_location="cpu", weights_only=True)
            native_null = torch.load(root / "reasoning" / "native_words" / f"{sample_id}_null.pt",
                                     map_location="cpu", weights_only=True)
            if (input_null["sample"] != sample_id or input_null["caption"] != ""
                    or input_null["scope"] != "request_only"
                    or words_null["word_strings"] != input_null["word_strings"]
                    or native_null["word_strings"] != words_null["word_strings"]
                    or tuple(native_null["native_words"].shape) != (4, 3072)):
                raise ValueError("null arm shape/provenance")
            stats["input_null"] += 1
            stats["native_null"] += 1
        except Exception as exc:
            errors.append(dict(id=sample_id, error=f"{type(exc).__name__}: {exc}"))
            if len(errors) >= 20:
                break
        if index % 500 == 0 or index == len(rows):
            print(json.dumps(dict(stage="audit", done=index, total=len(rows),
                                  errors=len(errors))), flush=True)
    if errors:
        raise RuntimeError(f"Audit failed for {len(errors)} clips: {errors[:5]}")
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg["paths"]["output_root"])
    if (root / "READY.json").exists():
        print("READY.json already exists; refusing to rebuild")
        return
    existing = existing_rows(cfg)
    new_rows = load_new_rows(cfg)
    per_shard = int(cfg["selection"]["per_shard_target"])
    target_new = int(cfg["selection"]["new_records"])
    selected = select_new(new_rows, per_shard, target_new)
    if len(selected) != target_new:
        raise RuntimeError(f"Selected {len(selected)} new clips, need {target_new}")
    print(json.dumps(dict(existing=len(existing), complete_new=len(new_rows),
                          selected_new=len(selected),
                          shards=dict((s, sum(row['_shard'] == s for row in selected))
                                      for s in sorted({row['_shard'] for row in selected}))),
                     ensure_ascii=False), flush=True)
    build_links(cfg, existing, selected)

    records = [clean_row(row) for row in existing]
    for row in selected:
        value = clean_row(row)
        value["refresh_origin"] = "new_qwen3vl_thinking"
        value["caption_version"] = "r20ft_qwen3vl_thinking_v1"
        records.append(value)
    if len(records) != int(cfg["selection"]["target_total"]):
        raise RuntimeError(f"Final record count {len(records)}")
    manifest = dict(
        schema="r20ft_wisa20k_v1",
        revision="R20FT-Wisa20k-Qwen3VL8BThinking-20261117",
        selection="existing_1600_plus_18400_new_qwen3vl_thinking",
        source_catalog=cfg["paths"]["source_catalog"],
        caption_version="r20ft_qwen3vl_thinking_v1",
        caption_review_version="r20ft_thinking_caption_v1",
        reasoning_revision="R20FT-Wisa20k-Qwen3VL8BThinking-v1",
        jepa_mode="not_used",
        existing_training=len(existing), new_records=len(selected),
        splits=dict(train=len(records)), records=records)
    atomic_json(manifest, root / "base" / "manifest.json")

    rng = random.Random(int(cfg["selection"]["seed"]))
    ids = [row["id"] for row in records]
    rng.shuffle(ids)
    drop = [rng.random() < 0.10 for _ in ids]
    sigma = [rng.random() for _ in ids]
    atomic_json(dict(source="r20ft_wisa20k_v1", ids=ids, sigma=sigma, drop=drop),
                root / "work" / "sample_sequence_20000.json")

    caption_words_existing = [word_count(row["caption"]) for row in existing]
    caption_words_new = [word_count(row["caption"]) for row in selected]
    ratio_new = [word_count(row["caption"]) / max(1, word_count(row.get("original_caption", "")))
                 for row in selected]
    relation_counts, relation_words = [], []
    for row in selected[:2000]:
        value = torch.load(root / "reasoning" / "input" / f"{row['id']}_text.pt",
                           map_location="cpu", weights_only=True)
        relation_counts.append(len(value["relations"]))
        relation_words.append(sum(word_count(item) for item in value["relations"]))
    stats = dict(
        existing_caption_words=dict(n=len(caption_words_existing),
            median=statistics.median(caption_words_existing),
            p10=sorted(caption_words_existing)[len(caption_words_existing)//10],
            p90=sorted(caption_words_existing)[9*len(caption_words_existing)//10]),
        new_caption_words=dict(n=len(caption_words_new),
            median=statistics.median(caption_words_new),
            p10=sorted(caption_words_new)[len(caption_words_new)//10],
            p90=sorted(caption_words_new)[9*len(caption_words_new)//10]),
        new_caption_ratio=dict(median=statistics.median(ratio_new),
            p10=sorted(ratio_new)[len(ratio_new)//10],
            p90=sorted(ratio_new)[9*len(ratio_new)//10]),
        new_relation_counts=dict((str(value), relation_counts.count(value))
                                 for value in sorted(set(relation_counts))),
        new_relation_words=dict(median=statistics.median(relation_words),
            p10=sorted(relation_words)[len(relation_words)//10],
            p90=sorted(relation_words)[9*len(relation_words)//10]))
    atomic_json(stats, root / "work" / "caption_stats.json")

    report = audit(cfg, records)
    report["caption_stats"] = stats
    report["manifest_records"] = len(records)
    atomic_json(report, root / "work" / "audit.json")

    training_paths = dict(
        train_cache=str(root),
        input_cache=str(root / "reasoning" / "input"),
        words_cache=str(root / "reasoning" / "words"),
        native_words_cache=str(root / "reasoning" / "native_words"),
        manifest=str(root / "base" / "manifest.json"),
        sample_sequence=str(root / "work" / "sample_sequence_20000.json"),
        records=len(records), existing=len(existing), new=len(selected))
    atomic_json(training_paths, root / "training_paths.json")
    ready = dict(status="ready", training_ready=True, records=len(records),
                 splits=dict(train=len(records)), version="r20ft_wisa20k_v1",
                 completed_stages=["index", "caption", "reasoning", "text_native", "vae"],
                 jepa_mode="not_used", deferred_features=[],
                 existing_training=len(existing), new_records=len(selected),
                 manifest=str(root / "base" / "manifest.json"),
                 manifest_bytes=(root / "base" / "manifest.json").stat().st_size,
                 audit=str(root / "work" / "audit.json"),
                 caption_stats=str(root / "work" / "caption_stats.json"),
                 sample_sequence=str(root / "work" / "sample_sequence_20000.json"),
                 training_paths=str(root / "training_paths.json"))
    atomic_json(ready, root / "READY.json")
    print(json.dumps(ready, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
