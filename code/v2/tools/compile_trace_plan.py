#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


V2_ROOT = Path(__file__).resolve().parents[1]
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

from ace_router.trace_compile import TraceCompileError, compile_trace_plan
from ace_router.trace_schema import TraceSchemaError, load_trace_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile a fixed-five TRACE plan without automatic fallback."
    )
    parser.add_argument("plan", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--crossfade-tokens", type=int, choices=(1, 2), default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan_path = args.plan.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    try:
        plan = load_trace_plan(plan_path)
        compiled = compile_trace_plan(
            plan,
            crossfade_tokens=args.crossfade_tokens,
            emit_validation=True,
        )
        compiled.write_manifest(output_path)
    except (TraceSchemaError, TraceCompileError, ValueError) as exc:
        print(f"[TRACE COMPILE][ERROR] {exc}", file=sys.stderr)
        print("[TRACE COMPILE] stopped explicitly; no fallback was selected", file=sys.stderr)
        return 2
    print(f"[TRACE COMPILE] wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
