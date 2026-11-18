"""Run one 4-GPU R20FT WISA shard: index -> Qwen caption/reasoning -> T5/native -> VAE."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import atomic_json, index_source, load_config, row_path, save_row

QWEN_PYTHON = "/home/liuzhirui/miniconda3/envs/qwen3vl/bin/python"
WAN_PYTHON = "/home/liuzhirui/miniconda3/envs/moviestory/bin/python"


def child_environment():
    env = os.environ.copy()
    env.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
               PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
    return env


def index_one(arguments):
    cfg, candidate = arguments
    path = row_path(cfg, candidate["id"])
    if path.is_file():
        row = json.loads(path.read_text())
        return row["id"], row.get("status", "unknown")
    try:
        row = index_source(candidate, cfg)
        save_row(cfg, row)
        return row["id"], "active"
    except Exception as exc:
        row = dict(candidate, caption=candidate["original_caption"],
                   caption_scope="unreviewed_clip", completed={}, status="rejected",
                   failure=dict(stage="index", type=type(exc).__name__, error=str(exc),
                                traceback=traceback.format_exc()[-4000:]))
        save_row(cfg, row)
        return row["id"], "rejected"


def run_stage(config, shard, stage, gpus):
    python = QWEN_PYTHON if stage == "qwen" else WAN_PYTHON
    processes = []
    try:
        for rank in range(gpus):
            command = [python, "-I", "-B", "-u",
                       str(Path(__file__).resolve().parent /
                           ("qwen_worker.py" if stage == "qwen" else "wan_worker.py")),
                       "--config", str(config), "--shard", str(shard),
                       "--rank", str(rank), "--world", str(gpus)]
            if stage != "qwen":
                command += ["--stage", stage]
            print(f"[launch] {stage} rank={rank}: {' '.join(command)}", flush=True)
            processes.append(subprocess.Popen(command, env=child_environment()))
        while True:
            codes = [process.poll() for process in processes]
            if any(code is not None and code != 0 for code in codes):
                raise RuntimeError(f"{stage} worker failed: exit codes {codes}")
            if all(code == 0 for code in codes):
                break
            time.sleep(2)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def stage_counts(cfg, shard):
    candidates = json.loads(
        (Path(cfg["paths"]["output_root"]) / "work" /
         f"candidates.shard{shard}.json").read_text())["candidates"]
    rows = []
    for candidate in candidates:
        path = row_path(cfg, candidate["id"])
        if path.is_file():
            rows.append(json.loads(path.read_text()))
    return dict(
        rows=len(rows),
        active=sum(row.get("status") == "active" for row in rows),
        rejected=sum(row.get("status") == "rejected" for row in rows),
        caption=sum(bool(row.get("completed", {}).get("caption")) for row in rows),
        reasoning=sum(bool(row.get("completed", {}).get("reasoning")) for row in rows),
        text_native=sum(bool(row.get("completed", {}).get("text_native")) for row in rows),
        vae=sum(bool(row.get("completed", {}).get("vae")) for row in rows),
    )


def progress(cfg, shard, stage, extra=None):
    value = dict(shard=shard, stage=stage, updated_utc=time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()), counts=stage_counts(cfg, shard))
    if extra:
        value.update(extra)
    atomic_json(value, Path(cfg["paths"]["output_root"]) / "work" /
                f"progress.shard{shard}.json")
    print(json.dumps(value, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--index-workers", type=int, default=16)
    parser.add_argument("--target-active", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    candidates = json.loads(
        (Path(cfg["paths"]["output_root"]) / "work" /
         f"candidates.shard{args.shard}.json").read_text())["candidates"]
    target = args.target_active or int(cfg["selection"]["per_shard_target"])
    print(json.dumps(dict(shard=args.shard, candidates=len(candidates),
                          target_active=target, gpus=args.gpus)), flush=True)

    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.index_workers) as pool:
        for done, (sample_id, status) in enumerate(
                pool.map(index_one, ((cfg, candidate) for candidate in candidates),
                         chunksize=8), 1):
            if done % 500 == 0 or done == len(candidates):
                print(json.dumps(dict(stage="index", shard=args.shard, done=done,
                                      total=len(candidates),
                                      seconds=round(time.monotonic() - started, 1),
                                      sample_id=sample_id, status=status)), flush=True)
    progress(cfg, args.shard, "index")
    counts = stage_counts(cfg, args.shard)
    if counts["active"] < target:
        raise RuntimeError(f"Shard {args.shard}: only {counts['active']} active clips, "
                           f"need {target}; extend the candidate pool before GPU stages")

    run_stage(args.config, args.shard, "qwen", args.gpus)
    progress(cfg, args.shard, "qwen")

    run_stage(args.config, args.shard, "text_native", args.gpus)
    progress(cfg, args.shard, "text_native")

    run_stage(args.config, args.shard, "vae", args.gpus)
    progress(cfg, args.shard, "vae")

    counts = stage_counts(cfg, args.shard)
    ready = dict(status="completed", shard=args.shard, counts=counts,
                 target_active=target,
                 seconds=round(time.monotonic() - started, 1),
                 updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    atomic_json(ready, Path(cfg["paths"]["output_root"]) / "work" /
                f"SHARD_READY{args.shard}.json")
    print(json.dumps(ready, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
