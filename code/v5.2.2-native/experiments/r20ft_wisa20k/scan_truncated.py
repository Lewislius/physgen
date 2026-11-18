"""Scan (and optionally reset) reasoning rows without a closed Thinking block.

The guard in qwen_worker.py is only loaded by workers started after it was
added.  Shard0's Qwen workers started earlier, so this scan is the safety net:
a row whose input evidence has thinking_end=None may have used the CoT text as
the final answer under the old fallback.

Usage:
  python scan_truncated.py --config configs/r20ft_wisa20k.yaml --shard 0
  python scan_truncated.py --config configs/r20ft_wisa20k.yaml --shard 0 --reset
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import atomic_json, load_config, load_row, save_row, shard_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg["paths"]["output_root"])
    base = shard_dir(cfg, args.shard)
    bad = []
    for path in sorted(glob.glob(str(base / "reasoning" / "input" / "*_text.pt"))):
        value = torch.load(path, map_location="cpu", weights_only=True)
        sample_id = os.path.basename(path)[:-len("_text.pt")]
        relations = value.get("relations") or []
        analysis = value.get("analysis") or ""
        problem = None
        if value.get("thinking_end") is None:
            problem = "missing_think_end"
        elif len(relations) != 4:
            problem = f"relations={len(relations)}"
        elif not analysis.strip():
            problem = "empty_analysis"
        if problem:
            bad.append(dict(id=sample_id, problem=problem,
                            relations=len(relations),
                            analysis_words=len(re.findall(r"\S+", analysis))))
    report = dict(shard=args.shard, scanned=len(
        glob.glob(str(base / "reasoning" / "input" / "*_text.pt"))),
        bad=bad)
    atomic_json(report, root / "work" / f"truncated_shard{args.shard}.json")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not args.reset or not bad:
        return
    for item in bad:
        sample_id = item["id"]
        row = load_row(cfg, sample_id)
        row.setdefault("completed", {}).pop("reasoning", None)
        row["completed"].pop("text_native", None)
        row["status"] = "active"
        save_row(cfg, row)
        for sub, suffix in (("input", "_text.pt"), ("input", "_null.pt"),
                            ("words", "_text.pt"), ("words", "_null.pt"),
                            ("native_words", "_text.pt"), ("native_words", "_null.pt")):
            path = base / "reasoning" / sub / f"{sample_id}{suffix}"
            path.unlink(missing_ok=True)
    print(f"reset {len(bad)} rows; rerun this shard to regenerate reasoning/text_native",
          flush=True)


if __name__ == "__main__":
    main()
