"""Small Qwen3-VL-8B-Thinking pilot: caption + T2V reasoning on a few clips."""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/liuzhirui/Project/physGen/code/v5.2")

from common import (TASK, ThinkingClient, atomic_json, atomic_torch, build_evidence,
                    caption_scratch, caption_offsets, index_source, load_config,
                    parse_relations, read_frames)


def word_count(text):
    return len(re.findall(r"\S+", text or ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = Path(cfg["paths"]["output_root"]) / "work" / "pilot"
    out.mkdir(parents=True, exist_ok=True)
    candidates = json.loads(
        (Path(cfg["paths"]["output_root"]) / "work" / "candidates.shard0.json").read_text()
    )["candidates"][:args.limit]
    client = ThinkingClient(cfg, args.device)
    offsets = caption_offsets(cfg)
    report = []
    for candidate in candidates:
        item = dict(id=candidate["id"], catalog_index=candidate["catalog_index"])
        try:
            t0 = time.monotonic()
            row = index_source(candidate, cfg)
            item["index_seconds"] = round(time.monotonic() - t0, 2)
            item["geometry"] = row["geometry"]
            item["source_frames"] = row["source_frames"]
            item["source_fps"] = row["source_fps"]
            t0 = time.monotonic()
            images = read_frames(row, offsets)
            item["decode_seconds"] = round(time.monotonic() - t0, 2)
            t0 = time.monotonic()
            caption, annotation = caption_scratch(
                client, images, row["original_caption"], offsets, cfg)
            item["caption_seconds"] = round(time.monotonic() - t0, 2)
            item["caption"] = caption
            item["original_words"] = word_count(row["original_caption"])
            item["caption_words"] = word_count(caption)
            item["caption_ratio"] = round(item["caption_words"] / max(1, item["original_words"]), 3)
            item["caption_approval_method"] = annotation.get("approval_method")
            item["caption_length_ok"] = annotation.get("length_ok")
            item["caption_trace_kinds"] = [entry.get("kind") for entry in annotation.get("trace", [])]
            atomic_json(annotation, out / f"{row['id']}.caption.json")

            prompt = (f"Request: {caption}\n"
                      f"Presentation interval: {row['duration']:.3f} seconds.\n" + TASK)
            t0 = time.monotonic()
            answer, raw, thinking_end = client.generate(
                prompt, max_new_tokens=int(cfg["prepare"]["reasoning_max_new_tokens"]))
            item["reasoning_seconds"] = round(time.monotonic() - t0, 2)
            relations = parse_relations(answer)
            item["relations"] = relations
            item["relation_count"] = len(relations)
            item["relation_words"] = [word_count(value) for value in relations]
            item["relation_total_words"] = sum(item["relation_words"])
            item["reasoning_raw_chars"] = len(raw)
            item["reasoning_answer_chars"] = len(answer)
            item["thinking_end"] = thinking_end
            if any(count > 24 for count in item["relation_words"]):
                raise ValueError(f"relation exceeds 24 words: {item['relation_words']}")
            evidence, words = build_evidence(
                client, caption, row["duration"], relations, row["id"],
                analysis=answer, thinking_end=thinking_end)
            atomic_torch(evidence, out / f"{row['id']}_input_text.pt")
            atomic_torch(words, out / f"{row['id']}_words_text.pt")
            atomic_json(item, out / f"{row['id']}.json")
        except Exception as exc:  # Keep the pilot report even when a clip fails.
            item["error"] = f"{type(exc).__name__}: {exc}"
            atomic_json(item, out / f"{item['id']}.json")
        report.append(item)
        print(json.dumps(item, ensure_ascii=False)[:1200], flush=True)
    summary = dict(records=len(report),
                   errors=sum("error" in item for item in report),
                   caption_ratios=[item.get("caption_ratio") for item in report if "caption_ratio" in item],
                   relation_counts=[item.get("relation_count") for item in report if "relation_count" in item],
                   relation_total_words=[item.get("relation_total_words") for item in report if "relation_total_words" in item],
                   caption_seconds=[item.get("caption_seconds") for item in report if "caption_seconds" in item],
                   reasoning_seconds=[item.get("reasoning_seconds") for item in report if "reasoning_seconds" in item])
    atomic_json(dict(summary=summary, items=report), out / "report.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
