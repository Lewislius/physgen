#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


V2_ROOT = Path(__file__).resolve().parents[1]
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

from ace_router.trace_validation import validate_trace_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose a TRACE plan. Issues are printed; no fallback is selected."
    )
    parser.add_argument("plan", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan_path = args.plan.expanduser().resolve()
    try:
        raw = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[TRACE VALIDATION][ERROR] input {plan_path}: {exc}", file=sys.stderr)
        return 2
    if not isinstance(raw, dict):
        print("[TRACE VALIDATION][ERROR] $: JSON root must be an object", file=sys.stderr)
        return 2
    validate_trace_plan(raw, emit=True)
    # Diagnostics deliberately do not become a fallback or a non-zero policy exit.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
