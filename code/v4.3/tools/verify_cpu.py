"""Run semantic CPU checks, inspect all real-width modules, and preserve an audit report."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from physgen_v43.environments import require_environment
require_environment("runtime")
import torch
from physgen_v43.corrector import Corrector
from physgen_v43.runtime import ROOT, read_config, configure_batch
from physgen_v43.config import validate_config
from train.train import check_config


def main():
    started = time.perf_counter()
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    config = configure_batch(validate_config(read_config(ROOT / "configs/flow_oracle.yaml")), 1)
    with contextlib.redirect_stdout(io.StringIO()):
        native = check_config(config)
    with torch.device("meta"):
        model = Corrector(config["corrector"])
    width_report = {name: dict(input=m.in_features, output=m.out_features)
                    for name, m in model.named_modules() if isinstance(m, torch.nn.Linear)}
    checks = ROOT / "analysis/checks"
    checks.mkdir(parents=True, exist_ok=True)
    reference = json.loads((checks / "v42_source_sha256.json").read_text())
    source = ROOT.parent / "v4.2"
    changed = [name for name, identity in reference.items()
               if not (source / name).is_file() or hashlib.sha256((source / name).read_bytes()).hexdigest() != identity]
    report = dict(status="passed" if result.wasSuccessful() and not changed else "failed",
                  tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                  skipped=len(result.skipped), seconds=time.perf_counter() - started,
                  torch_version=torch.__version__, native_config=native,
                  v42_source_files_checked=len(reference), v42_source_changes=changed,
                  gpu_test_run=False, full_training_run=False, video_quality_evaluated=False,
                  note="CPU fixtures use explicit small test widths. Production dimensions were independently inspected on meta tensors.")
    (checks / "native_width_manifest.json").write_text(json.dumps(width_report, indent=2) + "\n")
    (checks / "cpu_verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
