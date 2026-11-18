"""Qwen3-VL-8B-Thinking worker: caption repair + T2V reasoning for one GPU rank."""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from common import (TASK, ThinkingClient, annotation_path, atomic_json, atomic_torch,
                    build_evidence, caption_scratch, caption_offsets, load_config,
                    load_row, parse_relations, read_frames, save_row)

CAPTION_REVIEW_VERSION = "r20ft_thinking_caption_v1"

REVISION = "R20FT-Wisa20k-Qwen3VL8BThinking-v1"


def word_count(text):
    return len(re.findall(r"\S+", text or ""))


def relation_problem(relations):
    if len(relations) != 4:
        return f"expected exactly 4 sentences, got {len(relations)}"
    for index, value in enumerate(relations):
        if word_count(value) > 24:
            return f"sentence {index + 1} has {word_count(value)} words"
    return None


def process(row, client, cfg, offsets):
    if row.get("status") == "rejected":
        return False, False
    caption_done = bool(row.get("completed", {}).get("caption"))
    reasoning_done = bool(row.get("completed", {}).get("reasoning"))
    if caption_done and reasoning_done:
        return False, False
    if not caption_done:
        images = read_frames(row, offsets)
        caption, annotation = caption_scratch(
            client, images, row["original_caption"], offsets, cfg)
        atomic_json(annotation, annotation_path(cfg, row["id"]))
        row.update(caption=caption, caption_scope="clip",
                   caption_version="r20ft_qwen3vl_thinking_v1",
                   caption_review_version=CAPTION_REVIEW_VERSION)
        row.setdefault("completed", {})["caption"] = dict(
            caption=caption, annotation=str(annotation_path(cfg, row["id"])),
            model=cfg["paths"]["qwen_checkpoint"],
            approval_method=annotation.get("approval_method"),
            captured_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        save_row(cfg, row)
        caption_done = True
    if not reasoning_done:
        prompt = (f"Request: {row['caption']}\n"
                  f"Presentation interval: {row['duration']:.3f} seconds.\n" + TASK
                  + "\nReturn exactly four numbered sentences, each at most 24 words.")
        answer, raw, thinking_end = client.generate(
            prompt, max_new_tokens=int(cfg["prepare"]["reasoning_max_new_tokens"]))
        if thinking_end is None:
            problem = "missing </think>; response likely truncated"
        else:
            relations = parse_relations(answer)
            problem = relation_problem(relations)
        if problem is not None:
            stricter = (prompt + "\nReturn exactly four numbered sentences. "
                        "Each sentence must be at most 24 words.")
            answer, raw, thinking_end = client.generate(
                stricter, max_new_tokens=int(cfg["prepare"]["reasoning_max_new_tokens"]))
            if thinking_end is None:
                raise ValueError("Qwen reasoning was truncated before </think> after retry")
            relations = parse_relations(answer)
            problem = relation_problem(relations)
        if problem is not None:
            raise ValueError(f"Qwen reasoning style mismatch: {problem}")
        evidence, words = build_evidence(
            client, row["caption"], row["duration"], relations, row["id"],
            analysis=answer, thinking_end=thinking_end)
        null_evidence, null_words = build_evidence(
            client, "", row["duration"], ["", "", "", ""], row["id"],
            analysis="", thinking_end=None, null=True)
        shard = shard_dir_for_row(cfg, row)
        atomic_torch(evidence, shard / "reasoning" / "input" / f"{row['id']}_text.pt")
        atomic_torch(words, shard / "reasoning" / "words" / f"{row['id']}_text.pt")
        atomic_torch(null_evidence, shard / "reasoning" / "input" / f"{row['id']}_null.pt")
        atomic_torch(null_words, shard / "reasoning" / "words" / f"{row['id']}_null.pt")
        row.setdefault("completed", {})["reasoning"] = dict(
            relations=relations, revision=REVISION,
            model=cfg["paths"]["qwen_checkpoint"],
            thinking_end=thinking_end,
            captured_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        save_row(cfg, row)
        reasoning_done = True
    return caption_done, reasoning_done


def shard_dir_for_row(cfg, row):
    # A row belongs to exactly one shard; the shard is stored in its work row.
    shard = row.get("shard", 0)
    return Path(cfg["paths"]["output_root"]) / f"shard{shard}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world", type=int, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    candidate_file = (Path(cfg["paths"]["output_root"]) / "work" /
                      f"candidates.shard{args.shard}.json")
    candidates = json.loads(candidate_file.read_text())["candidates"]
    torch.cuda.set_device(args.rank)
    device = f"cuda:{args.rank}"
    pending = []
    for candidate in candidates[args.rank::args.world]:
        path = Path(cfg["paths"]["output_root"]) / "work" / "rows" / f"{candidate['id']}.json"
        if not path.is_file():
            continue
        row = load_row(cfg, candidate["id"])
        if row.get("status") != "rejected" and not (
                row.get("completed", {}).get("caption") and
                row.get("completed", {}).get("reasoning")):
            pending.append(row)
    print(json.dumps(dict(stage="qwen", shard=args.shard, rank=args.rank,
                          world=args.world, device=device,
                          gpu=torch.cuda.get_device_name(args.rank),
                          pending=len(pending))), flush=True)
    if not pending:
        return
    client = ThinkingClient(cfg, device)
    offsets = caption_offsets(cfg)
    started = time.monotonic()
    for index, row in enumerate(pending, 1):
        try:
            process(row, client, cfg, offsets)
        except Exception as exc:
            row["status"] = "rejected"
            row["failure"] = dict(stage="qwen", type=type(exc).__name__,
                                  error=str(exc),
                                  traceback=traceback.format_exc()[-4000:])
            save_row(cfg, row)
            print(json.dumps(dict(stage="qwen", rank=args.rank, id=row["id"],
                                  status="rejected", error=str(exc))), flush=True)
            continue
        if index % 10 == 0 or index == len(pending):
            elapsed = time.monotonic() - started
            print(json.dumps(dict(stage="qwen", shard=args.shard, rank=args.rank,
                                  done=index, total=len(pending),
                                  seconds=round(elapsed, 1),
                                  per_record=round(elapsed / index, 2))), flush=True)


if __name__ == "__main__":
    main()
