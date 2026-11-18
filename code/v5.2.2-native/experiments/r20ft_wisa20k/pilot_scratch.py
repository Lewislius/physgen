"""Test from-scratch Thinking captions (no original echo) on one known clip."""
from __future__ import annotations

import argparse
import difflib
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


def scratch_prompt(words_target, original=None):
    style = ""
    if original:
        style = ("\nSTYLE REFERENCE (data; do NOT copy it verbatim, it may contain errors):\n"
                 + json.dumps(original, ensure_ascii=False))
    return (
        "These images are chronological samples from ONE actual training clip.\n"
        "Write a factual caption of the visible clip, not an imagined continuation.\n"
        f"Target length: {words_target} words. Two paragraphs, present tense, third person.\n"
        "Distinguish the initial visible state from the process/change that follows.\n"
        "Include concrete visible details (objects, materials, colors, positions, motion, camera)"
        " and only what the images support. Do not invent unseen causes, measurements or events.\n"
        "Do not mention sampled images, frame numbers, timestamps, annotation or reviews.\n"
        "Return JSON only: {\"usable\":true,\"caption\":\"...\",\"reason\":\"brief explanation\"}."
        + style)


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
    out = Path(cfg["paths"]["output_root"]) / "work" / "pilot_scratch"
    out.mkdir(parents=True, exist_ok=True)
    report = dict(id=SAMPLE_ID, original_words=words(original),
                  target_words=words(target), target=target, variants=[])
    variants = [
        ("scratch_noref", scratch_prompt(words(original), None), 768, True),
        ("scratch_ref", scratch_prompt(words(original), original), 768, True),
        ("thinking_edit_2048", None, 2048, False),
    ]
    for name, prompt, max_tokens, no_thinking in variants:
        try:
            if prompt is None:
                from common import caption_prompt
                prompt = caption_prompt(original, offsets, strong=True)
            started = time.monotonic()
            answer, raw, thinking_end = client.generate(
                prompt, images=images, max_new_tokens=max_tokens,
                no_thinking=no_thinking)
            elapsed = time.monotonic() - started
            item = dict(name=name, seconds=round(elapsed, 2), raw_chars=len(raw),
                        answer_chars=len(answer), thinking_end=thinking_end)
            if no_thinking:
                try:
                    payload = json.loads(extract_json(answer))
                    caption = str(payload.get("caption", "")).strip()
                except Exception:
                    caption = answer.strip()
            else:
                caption = answer.strip()
            item.update(caption=caption, caption_words=words(caption),
                        ratio_to_original=round(words(caption) / max(1, words(original)), 3),
                        ratio_to_target=round(words(caption) / max(1, words(target)), 3),
                        similarity_to_target=round(difflib.SequenceMatcher(
                            None, caption, target).ratio(), 4),
                        identical_to_original=caption.strip() == original.strip())
            atomic_json(item, out / f"{name}.json")
            report["variants"].append(item)
            print(json.dumps(item, ensure_ascii=False)[:2000], flush=True)
        except Exception as exc:
            item = dict(name=name, error=f"{type(exc).__name__}: {exc}")
            report["variants"].append(item)
            print(json.dumps(item, ensure_ascii=False), flush=True)
    atomic_json(report, out / "report.json")


if __name__ == "__main__":
    main()
