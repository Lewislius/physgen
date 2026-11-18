"""Wan2.2 worker: UMT5 main caption + 3072 native words, or VAE latents."""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from common import (atomic_torch, load_config, load_row, native_embedding,
                    read_frames, save_row, shard_dir)


def load_umt5(cfg, device):
    sys.path.insert(0, cfg["paths"]["wan_code"])
    from wan.modules.t5 import T5EncoderModel
    checkpoint = Path(cfg["paths"]["wan_checkpoint"])
    return T5EncoderModel(
        text_len=512, dtype=torch.bfloat16, device=device,
        checkpoint_path=str(checkpoint / "models_t5_umt5-xxl-enc-bf16.pth"),
        tokenizer_path=str(checkpoint / "google/umt5-xxl"))


def encode_main_text(encoder, caption, device):
    _, mask = encoder.tokenizer([caption], return_mask=True, add_special_tokens=True,
                                truncation=False, padding=True)
    if bool((mask.sum(dim=1) > 512).any()):
        raise ValueError("Caption exceeds 512 T5 tokens")
    with torch.inference_mode():
        values = [value.detach().cpu().half() for value in encoder([caption], device)]
    value = values[0]
    if value.ndim != 2 or value.shape[1] != 4096 or not len(value):
        raise ValueError(f"Unexpected T5 output shape {tuple(value.shape)}")
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError("T5 produced nonfinite values")
    return value


def encode_native(encoder, native, relations, strings, groups, device):
    features, tokens = [], []
    for group, relation in enumerate(relations):
        count = sum(1 for item in groups if int(item) == group)
        if not relation:
            if count != 1:
                raise ValueError("Empty relation must have one delimiter slot")
            features.append(torch.zeros(3072))
            tokens.append("")
            continue
        clean = encoder.tokenizer._clean(relation)
        encoded = encoder.tokenizer.tokenizer(
            clean, return_tensors="pt", return_offsets_mapping=True,
            add_special_tokens=True, padding=False, truncation=True, max_length=128)
        offsets = encoded.pop("offset_mapping")[0]
        with torch.inference_mode():
            hidden = encoder.model(encoded["input_ids"].to(device),
                                   encoded["attention_mask"].to(device))
            target = native(hidden).float()[0]
        matches = list(re.finditer(r"\S+", clean))
        group_strings = [text for text, item in zip(strings, groups) if int(item) == group]
        if len(matches) != len(group_strings):
            raise ValueError("Native tokenization changed a relation word count")
        for match, word in zip(matches, group_strings):
            selected = ((offsets[:, 1] > match.start()) &
                        (offsets[:, 0] < match.end())).nonzero().flatten()
            if not len(selected):
                raise ValueError("Native physical word has no token")
            features.append(target[selected.to(target.device)].mean(0))
            tokens.append(word)
    return torch.stack(features).cpu().to(torch.bfloat16), tokens


def text_native(cfg, rows, rank, world):
    device = f"cuda:{rank}"
    encoder = load_umt5(cfg, device)
    native = native_embedding(device, cfg["paths"]["wan_checkpoint"])
    pending = []
    for row in rows[rank::world]:
        if (row.get("status") == "rejected" or
                not row.get("completed", {}).get("reasoning") or
                row.get("completed", {}).get("text_native")):
            continue
        pending.append(row)
    print(json.dumps(dict(stage="text_native", rank=rank, world=world,
                          device=device, pending=len(pending))), flush=True)
    started = time.monotonic()
    for index, row in enumerate(pending, 1):
        shard = shard_dir(cfg, row["shard"])
        text_path = shard / "base" / "text" / f"{row['id']}.pt"
        native_text = shard / "reasoning" / "native_words" / f"{row['id']}_text.pt"
        native_null = shard / "reasoning" / "native_words" / f"{row['id']}_null.pt"
        if not (text_path.is_file() and native_text.is_file() and native_null.is_file()):
            text_value = encode_main_text(encoder, row["caption"], device)
            atomic_torch(text_value, text_path)
            input_text = torch.load(
                shard / "reasoning" / "input" / f"{row['id']}_text.pt",
                map_location="cpu", weights_only=True)
            words = torch.load(
                shard / "reasoning" / "words" / f"{row['id']}_text.pt",
                map_location="cpu", weights_only=True)
            features, tokens = encode_native(
                encoder, native, input_text["relations"], words["word_strings"],
                words["word_groups"].tolist(), device)
            if tokens != list(words["word_strings"]):
                raise ValueError("Native words disagree with Qwen word strings")
            atomic_torch(dict(
                native_words=features, word_strings=tokens,
                scope="request_only", sample=row["id"],
                reasoning_revision="R20FT-Wisa20k-Qwen3VL8BThinking-v1",
                teacher="official frozen T5 then official Wan text_embedding"),
                native_text)
            atomic_torch(dict(
                native_words=torch.zeros(4, 3072, dtype=torch.bfloat16),
                word_strings=["", "", "", ""], scope="request_only",
                sample=row["id"],
                teacher="official frozen T5 then official Wan text_embedding"),
                native_null)
        row.setdefault("completed", {})["text_native"] = dict(
            text=str(text_path), native_text=str(native_text),
            native_null=str(native_null),
            captured_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        save_row(cfg, row)
        if index % 25 == 0 or index == len(pending):
            elapsed = time.monotonic() - started
            print(json.dumps(dict(stage="text_native", rank=rank, done=index,
                                  total=len(pending), seconds=round(elapsed, 1),
                                  per_record=round(elapsed / index, 2))), flush=True)


def vae(cfg, rows, rank, world):
    device = f"cuda:{rank}"
    sys.path.insert(0, cfg["paths"]["wan_code"])
    from wan.modules.vae2_2 import Wan2_2_VAE
    checkpoint = Path(cfg["paths"]["wan_checkpoint"])
    encoder = Wan2_2_VAE(vae_pth=str(checkpoint / "Wan2.2_VAE.pth"),
                         dtype=torch.float32, device=device)
    pending = []
    for row in rows[rank::world]:
        if (row.get("status") == "rejected" or
                not row.get("completed", {}).get("reasoning") or
                row.get("completed", {}).get("vae")):
            continue
        pending.append(row)
    print(json.dumps(dict(stage="vae", rank=rank, world=world,
                          device=device, pending=len(pending))), flush=True)
    started = time.monotonic()
    for index, row in enumerate(pending, 1):
        shard = shard_dir(cfg, row["shard"])
        target = shard / "base" / "vae" / f"{row['id']}.pt"
        if not target.is_file():
            frames = read_frames(row)
            x = torch.from_numpy(np.stack([np.asarray(frame) for frame in frames]))
            x = x.permute(3, 0, 1, 2).float().to(device) / 127.5 - 1.0
            with torch.inference_mode():
                latent = encoder.encode([x])[0].unsqueeze(0).cpu()
                first = encoder.encode([x[:, :1]])[0].unsqueeze(0).cpu()
            if tuple(latent.shape) != (1, 48, 21, 30, 52):
                raise ValueError(f"{row['id']}: VAE latent {tuple(latent.shape)}")
            if tuple(first.shape) != (1, 48, 1, 30, 52):
                raise ValueError(f"{row['id']}: VAE first {tuple(first.shape)}")
            atomic_torch(dict(latent=latent, first=first,
                              cache_signature=row["cache_signature"]), target)
        row.setdefault("completed", {})["vae"] = dict(
            path=str(target),
            captured_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        save_row(cfg, row)
        if index % 10 == 0 or index == len(pending):
            elapsed = time.monotonic() - started
            print(json.dumps(dict(stage="vae", rank=rank, done=index,
                                  total=len(pending), seconds=round(elapsed, 1),
                                  per_record=round(elapsed / index, 2))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--stage", choices=("text_native", "vae"), required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world", type=int, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    torch.cuda.set_device(args.rank)
    candidates = json.loads(
        (Path(cfg["paths"]["output_root"]) / "work" /
         f"candidates.shard{args.shard}.json").read_text())["candidates"]
    rows = []
    for candidate in candidates:
        path = Path(cfg["paths"]["output_root"]) / "work" / "rows" / f"{candidate['id']}.json"
        if path.is_file():
            rows.append(load_row(cfg, candidate["id"]))
    if args.stage == "text_native":
        text_native(cfg, rows, args.rank, args.world)
    else:
        vae(cfg, rows, args.rank, args.world)


if __name__ == "__main__":
    main()
