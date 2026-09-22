"""Reproduce the real step0800 CPU loading/41-case audit without allocating a GPU."""
from collections import Counter
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
import test_static_lora as fixtures
import torch
from torch import nn

from static_lora.backbone import WanLoRA
from static_lora.checkpoint import checkpoint_path
from static_lora.config import configuration
from static_lora.lora import LoRALinear, load_lora_state_dict, lora_named_parameters
from static_lora.runtime import digest, read_json, save_json


def main():
    destination = ROOT / "verification/step0800_demo41_20260918"
    checkpoint = ROOT / "checkpoints/lora_20260917T140727Z_dae2a0/step0800"
    cfg = configuration(checkpoint / "config.json")
    assert checkpoint_path(cfg, checkpoint) == checkpoint
    args = SimpleNamespace(checkpoint=str(checkpoint), suite="demo", duration=5., portrait=False)
    plan = fixtures.inference.build_plan(cfg, args)
    weights = fixtures.inference.validate_ema(cfg, checkpoint)

    native = read_json(Path(cfg["paths"]["wan_checkpoint"]) / "config.json")
    with torch.device("meta"):
        wan = fixtures.NativeWan(**{k: v for k, v in native.items() if not k.startswith("_")})
        model = WanLoRA(wan, cfg["lora"], recompute=False)
    # Materialize just the real 180 MiB adapter; frozen 5B weights remain on meta.
    for module in model.modules():
        if isinstance(module, LoRALinear):
            for name in ("lora_A", "lora_B"):
                param = getattr(module, name)
                setattr(module, name, nn.Parameter(torch.empty(param.shape, dtype=torch.float32)))
    values = torch.load(checkpoint / "ema.pt", map_location="cpu", weights_only=True)
    load_lora_state_dict(model, values)
    assert all(torch.equal(param, values[name]) for name, param in lora_named_parameters(model))

    # Exercise one actual trained projection against its original base weights.
    from safetensors import safe_open
    base = Path(cfg["paths"]["wan_checkpoint"])
    index = read_json(base / "diffusion_pytorch_model.safetensors.index.json")["weight_map"]
    layer = model.wan.blocks[0].self_attn.q
    linear = nn.Linear(layer.in_features, layer.out_features, bias=True, dtype=torch.bfloat16)
    with torch.no_grad():
        for name in ("weight", "bias"):
            key = "blocks.0.self_attn.q." + name
            with safe_open(str(base / index[key]), framework="pt", device="cpu") as reader:
                getattr(linear, name).copy_(reader.get_tensor(key))
    layer.base = linear.requires_grad_(False)
    generator = torch.Generator().manual_seed(42)
    x = torch.randn(1, 4, layer.in_features, generator=generator)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        baseline, adapted = linear(x), layer(x)
    effect = float((adapted.float() - baseline.float()).square().mean().sqrt())
    assert effect > 0 and bool(torch.isfinite(adapted).all())

    v4 = fixtures.import_file("lora_audit_v4_cases", ROOT.parent / "v4/inference/cases.py")
    previous = v4.collect_cases(SimpleNamespace(
        suite="both", image=cfg["paths"]["reference_image"], prompt=cfg["paths"]["reference_prompt"],
        mode="i2v", demo_root=cfg["paths"]["demo_root"], sample_ids="all", demo_modes="both"))
    assert [(c["id"], c["mode"], c["prompt"], c.get("image")) for c in plan["cases"]] == [
        (c["case_id"], c["mode"], c["prompt"], c.get("image")) for c in previous]
    assert all(c.get("image_sha256") == p.get("image_sha256") for c, p in zip(plan["cases"], previous))
    v42 = fixtures.import_file("lora_audit_v42_inference", ROOT.parent / "v4.2/inference/infer.py")
    previous42 = v42.collect_cases(cfg, args)
    assert [(c["id"], c["mode"], c["prompt"], c.get("image")) for c in plan["cases"]] == [
        (c["id"], c["mode"], c["prompt"], c.get("image")) for c in previous42]
    assert len(plan["cases"]) == 41 and all("gt_record" not in c for c in plan["cases"])
    test_log = destination / "unit_tests.log"
    tests = dict(status="not_run")
    if test_log.exists():
        log = test_log.read_text()
        total = re.search(r"Ran (\d+) tests", log)
        outcome = re.search(r"^OK(?: \(skipped=(\d+)\))?$", log, re.MULTILINE)
        tests = dict(status="passed" if total and outcome else "failed_or_incomplete", log="unit_tests.log")
        if total and outcome:
            count, skipped = int(total[1]), int(outcome[1] or 0)
            tests.update(run=count, passed=count-skipped, skipped=skipped, failed=0)
    result = dict(status="passed", checkpoint=str(checkpoint), checkpoint_step=plan["checkpoint_step"],
                  ema_updates=plan["ema_updates"], ema_sha256=plan["ema_sha256"], lora=cfg["lora"],
                  weights=weights, strict_load_into_native_model=True, all_loaded_tensors_equal=True,
                  actual_trained_projection_rms_delta=effect,
                  cases=len(plan["cases"]), modes=dict(Counter(c["mode"] for c in plan["cases"])),
                  ordered_conditions_match_v4_and_v42=True, source_image_hashes_match_v4=True,
                  canvases=dict(Counter(f'{c["geometry"]["w"]}x{c["geometry"]["h"]}' for c in plan["cases"])),
                  frame_counts=sorted(set(c["geometry"]["frames"] for c in plan["cases"])),
                  durations=sorted(set(c["duration"] for c in plan["cases"])), recipe=plan["recipe"],
                  gpu_sampling_run=False, cpu_base_weights="meta, except one original q projection",
                  unit_tests=tests)
    save_json(plan, destination / "cases.json")
    save_json(result, destination / "result.json")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
