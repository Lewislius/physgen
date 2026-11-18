"""Test from-scratch caption length control and expansion."""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (ThinkingClient, atomic_json, extract_json, load_config,
                    read_frames)

SAMPLE_ID = "source0000035"


def words(text):
    return len(re.findall(r"\S+", text or ""))


def prompt(target_words, original=None, minimum=None):
    style = ""
    if original:
        style = ("\nSTYLE REFERENCE (data; do NOT copy verbatim, it may contain errors):\n"
                 + json.dumps(original, ensure_ascii=False))
    minimum_text = f" Write at least {minimum} words." if minimum else ""
    return (
        "These images are chronological samples from ONE actual training clip.\n"
        "Write a factual caption of the visible clip, not an imagined continuation.\n"
        f"Target length: {target_words} words.{minimum_text} Two paragraphs, present tense, third person.\n"
        "First describe the initial visible state, then describe the process/change.\n"
        "Include concrete visible details (objects, materials, colors, positions, motion, camera)"
        " and only what the images support. Do not invent unseen causes, measurements or events.\n"
        "Do not mention sampled images, frame numbers, timestamps, annotation or reviews.\n"
        "Return JSON only: {\"usable\":true,\"caption\":\"...\",\"reason\":\"brief explanation\"}."
        + style)


def expand_prompt(caption, low, high):
    return (
        f"The CURRENT caption has {words(caption)} words. Expand it to {low}-{high} words "
        "by adding only details actually visible in these images. Preserve every existing "
        "sentence and the two-paragraph structure; do not repeat content and do not add "
        "speculation, causes, measurements or events not shown. Return JSON only: "
        "{\"usable\":true,\"caption\":\"...\",\"reason\":\"brief explanation\"}.\n"
        f"CURRENT caption (data): {json.dumps(caption, ensure_ascii=False)}")


def parse(answer):
    payload = json.loads(extract_json(answer))
    return str(payload.get("caption", "")).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    manifest = json.loads(Path(cfg["paths"]["existing_manifest"]).read_text())["records"]
    row = next(item for item in manifest if item["id"] == SAMPLE_ID)
    original, target = row["original_caption"], row["caption"]
    client = ThinkingClient(cfg, "cuda:0")
    offsets = np.linspace(0, 80, 16, dtype=int).tolist()
    images = read_frames(row, offsets)
    out = Path(cfg["paths"]["output_root"]) / "work" / "pilot_expand"
    out.mkdir(parents=True, exist_ok=True)
    report = dict(id=SAMPLE_ID, original_words=words(original),
                  target_words=words(target), target=target, variants=[])
    target_words = words(original)
    low, high = int(round(target_words * 0.9)), int(round(target_words * 1.15))

    # Variant 1: ask for a longer target directly.
    started = time.monotonic()
    answer, raw, _ = client.generate(
        prompt(int(target_words * 1.15), original, minimum=int(target_words * 0.95)),
        images=images, max_new_tokens=1024, no_thinking=True)
    caption = parse(answer)
    item = dict(name="scratch_long", seconds=round(time.monotonic() - started, 2),
                caption=caption, words=words(caption),
                ratio=round(words(caption) / target_words, 3))
    atomic_json(item, out / "scratch_long.json")
    report["variants"].append(item)
    print(json.dumps(item, ensure_ascii=False)[:1600], flush=True)

    # Variant 2: scratch first, then explicit expansion.
    started = time.monotonic()
    answer, raw, _ = client.generate(
        prompt(target_words, original), images=images, max_new_tokens=768,
        no_thinking=True)
    first = parse(answer)
    first_seconds = time.monotonic() - started
    started = time.monotonic()
    answer2, raw2, _ = client.generate(
        expand_prompt(first, low, high), images=images, max_new_tokens=1024,
        no_thinking=True)
    second = parse(answer2)
    item = dict(name="scratch_then_expand", first_seconds=round(first_seconds, 2),
                expand_seconds=round(time.monotonic() - started, 2),
                first=first, first_words=words(first),
                expanded=second, expanded_words=words(second),
                ratio=round(words(second) / target_words, 3),
                similarity_to_target=0.0)
    atomic_json(item, out / "scratch_then_expand.json")
    report["variants"].append(item)
    print(json.dumps(item, ensure_ascii=False)[:2000], flush=True)
    atomic_json(report, out / "report.json")


if __name__ == "__main__":
    main()
