"""Compare Thinking caption variants on an existing 1,600 clip with a known qwen38 correction."""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (ThinkingClient, atomic_json, caption_edit, load_config,
                    read_frames)

SAMPLE_ID = "source0000035"


def words(text):
    return len(re.findall(r"\S+", text or ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    manifest = json.loads(Path(cfg["paths"]["existing_manifest"]).read_text())["records"]
    row = next(item for item in manifest if item["id"] == SAMPLE_ID)
    original = row["original_caption"]
    target = row["caption"]
    client = ThinkingClient(cfg, "cuda:0")
    out = Path(cfg["paths"]["output_root"]) / "work" / "pilot_compare"
    out.mkdir(parents=True, exist_ok=True)
    report = dict(id=SAMPLE_ID, original_words=words(original),
                  target_words=words(target), target=target, variants=[])
    variants = [
        ("16frames_no_thinking", 16, True, True),
        ("16frames_thinking", 16, True, False),
        ("40frames_no_thinking", 40, True, True),
    ]
    for name, frames, strong, no_thinking in variants:
        offsets = np.linspace(0, 80, frames, dtype=int).tolist()
        try:
            images = read_frames(row, offsets)
            old_max = cfg["prepare"]["caption_max_new_tokens"]
            if not no_thinking:
                cfg["prepare"]["caption_max_new_tokens"] = 1024
            started = time.monotonic()
            caption, evidence = caption_edit(
                client, images, original, offsets, cfg,
                strong=strong, no_thinking=no_thinking)
            elapsed = time.monotonic() - started
            cfg["prepare"]["caption_max_new_tokens"] = old_max
            item = dict(name=name, seconds=round(elapsed, 2),
                        caption=caption, caption_words=words(caption),
                        ratio_to_original=round(words(caption) / max(1, words(original)), 3),
                        ratio_to_target=round(words(caption) / max(1, words(target)), 3),
                        similarity_to_target=round(difflib.SequenceMatcher(
                            None, caption, target).ratio(), 4),
                        identical_to_original=caption.strip() == original.strip(),
                        trace_kinds=[entry.get("kind") for entry in evidence["trace"]],
                        length_ok=evidence.get("length_ok"))
            atomic_json(item, out / f"{name}.json")
            report["variants"].append(item)
            print(json.dumps(item, ensure_ascii=False)[:1600], flush=True)
        except Exception as exc:
            report["variants"].append(dict(name=name, error=f"{type(exc).__name__}: {exc}"))
            print(json.dumps(report["variants"][-1], ensure_ascii=False), flush=True)
    atomic_json(report, out / "report.json")


if __name__ == "__main__":
    main()
