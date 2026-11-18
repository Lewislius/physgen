"""Generate only the fixed original40 suffix VP033-040 on spare RTX; primary scores all40."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import yaml


def publish(generated, destination, cases, expected):
    """Expose complete immutable cases without replacing an existing primary owner."""
    published, owned = [], []
    destination.mkdir(parents=True, exist_ok=True)
    for case in cases:
        source = generated / case
        metadata = json.loads((source / "planned.json").read_text())
        if not (source / "planned.mp4").is_file():
            raise ValueError(f"Missing completed video: {case}")
        if any(metadata.get(key) != value for key, value in (expected | {"case": case}).items()):
            raise ValueError(f"Different inference provenance: {case}")
        try:
            (destination / case).symlink_to(source.resolve(), target_is_directory=True)
            published.append(case)
        except FileExistsError:
            owned.append(case)
    return published, owned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--parent-experiment", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--describe", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    source = Path(cfg["source_snapshot"])
    root = Path(cfg["paths"]["runs"]).parent
    checkpoint = root / "runs" / args.run_id / "step0400.pt"
    parent = root / "reports/jobs" / str(args.parent_experiment)
    parent_plan = json.loads((parent / "plan.json").read_text())
    target = Path(parent_plan["output"])
    assert parent_plan["source"] == str(source) and parent_plan["checkpoint"] == str(checkpoint)
    suites = {"videophy40": [f"VP{i:03d}" for i in range(33, 41)]}
    assert set(suites["videophy40"]) <= set(parent_plan["main_cases"])
    plan = dict(parent_experiment=args.parent_experiment, run_id=args.run_id, source=str(source),
                checkpoint=str(checkpoint), primary_output=str(target), suites=suites, videos=8,
                additional_cases=0, additional_seeds=0, api_calls=0, training_updates=0,
                scoring="Primary full40 only; publication follows ownership, never quality selection")
    if args.describe:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    run_state = json.loads((checkpoint.parent / "status.json").read_text())
    if not checkpoint.is_file() or run_state.get("status") != "complete" or run_state.get("step") != 400:
        raise ValueError("Own400 must be complete before dispatch")
    sys.path.insert(0, str(source))
    import bootstrap
    import benchmark_inputs
    from physical_content.common import gpu_guard, atomic_json
    from pipeline import run_stage
    experiment = os.getenv("DET_EXPERIMENT_ID", "local")
    folder = root / "reports/control_helpers" / experiment
    gpu_guard(folder)
    atomic_json(plan, folder / "plan.json")
    output = root / "helper_outputs" / args.run_id / experiment
    results = []
    for suite, proposed in suites.items():
        stage = json.loads((parent / "status.json").read_text())["status"]
        completed40 = len(list((target / "videophy40/cases").glob("*/planned.json")))
        # Fixed eight-case suffix, dispatched early enough to finish before
        # primary reaches VP033. No supplementary20 cases are generated.
        early = stage == "original40_generation" and completed40 <= 8
        cases = [case for case in proposed if not (target / suite / "cases" / case).exists()]
        if not early or not cases:
            results.append(dict(suite=suite, status="left_to_primary", stage=stage, completed40=completed40))
            continue
        atomic_json(dict(status=suite + "_tail_generation", cases=cases, results=results), folder / "status.json")
        run_stage([sys.executable, "-I", "-B", "-u", source / "infer.py", "--checkpoint", checkpoint,
                   "--output", output, "--suite", suite, "--modes", "real", "--cases", *cases],
                  folder / (suite + "_tail_generation.log"))
        generated = output / suite
        manifest = json.loads((generated / "manifest.json").read_text())
        if {case["id"] for case in manifest["cases"]} != set(cases):
            raise ValueError("Generated cases differ from the declared suffix")
        published, owned = publish(generated / "cases", target / suite / "cases", cases,
            dict(checkpoint=str(checkpoint), step=400, mode="real", seed=42,
                 inference_source=str(source), inference_evidence=cfg["paths"]["inference_evidence"]))
        result = dict(suite=suite, published=published, left_to_primary=owned,
                      isolated_output=str(generated), primary_output=str(target / suite))
        results.append(result)
        atomic_json(result, folder / (suite + "_handoff.json"))
    atomic_json(dict(status="completed", recorded_utc=datetime.now(timezone.utc).isoformat(),
                     results=results, **plan), folder / "status.json")


if __name__ == "__main__":
    main()
