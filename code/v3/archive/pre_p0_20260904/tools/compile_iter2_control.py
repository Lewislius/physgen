#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


CODE_ROOT = Path("/home/liuzhirui/Project/physGen/code")
V3_ROOT = Path("/home/liuzhirui/Project/physGen/code/v3")
for root in (CODE_ROOT, V3_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from v2.ace_router.trace_schema import TracePlan
from trace_writer_v3.compile import compile_trace_plan_v3
from trace_writer_v3.control_schema import merge_overlay


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile one v1 plan into a TRACE v3 control bundle")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--mode", choices=("i2v", "t2v"), required=True)
    parser.add_argument("--overlay", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--minimal_pair_mode", choices=("compat", "strict"), default="strict")
    args = parser.parse_args()
    plan_path = Path(args.plan).expanduser().resolve()
    raw = json.loads(plan_path.read_text(encoding="utf-8"))
    overlay = None
    if args.overlay:
        overlay = json.loads(Path(args.overlay).expanduser().resolve().read_text(encoding="utf-8"))
    merged = merge_overlay(raw, overlay)
    compiled = compile_trace_plan_v3(
        TracePlan.from_dict(merged),
        merged,
        mode=args.mode,
        minimal_pair_mode=args.minimal_pair_mode,
    )
    content = json.dumps(compiled.manifest(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content, encoding="utf-8")
    else:
        print(content, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
