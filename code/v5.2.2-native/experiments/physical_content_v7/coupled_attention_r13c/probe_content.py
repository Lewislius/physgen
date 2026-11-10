"""Own400 fixed16 cached inputs: distinguish Q, C residuals and sample means on CPU."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "snapshots/20261105T032330Z"
sys.path.insert(0, str(SOURCE))
import bootstrap
from physical_content.model import Enhancement
from physical_content.teacher import Teacher


def stats(value):
    value = value.detach().float().reshape(-1, value.shape[-1])
    n = len(value)
    unit = F.normalize(value, dim=-1)
    return dict(
        token_cosine=float((unit.sum(0).square().sum() - n) / (n * (n - 1))),
        rms=float(value.square().mean().sqrt()),
        common_token_energy_fraction=float(value.mean(0).square().sum() * n /
                                          value.square().sum().clamp_min(1e-20)),
    )


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    checkpoint = ROOT / "runs/physical_content_67682/step0400.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    assert saved["step"] == 400 and saved["revision"] == "physical_content_v7_coupled_attention_r13c"
    cfg = saved["config"]
    assert Path(cfg["source_snapshot"]) == SOURCE
    model = Enhancement(cfg).eval().requires_grad_(False)
    weights = {name.removeprefix("enhancement."): value
               for name, value in {**saved["model"], **saved["constants"]}.items()
               if name.startswith("enhancement.")}
    model.load_state_dict(weights, strict=True)
    model.checkpointing = False
    teacher = Teacher(cfg, device="cpu")
    captured = {}
    model.execution_project.register_forward_pre_hook(lambda _, args: captured.update(update=args[0]))
    model.project.register_forward_pre_hook(lambda _, args: captured.update(readout=args[0]))
    rows, means = [], {}
    cases = json.loads((SOURCE / "subset16.json").read_text())["cases"]
    assert len(cases) == 16
    for case in cases:
        evidence = teacher.input(dict(id="videophy40_" + case["id"]))
        with torch.autocast("cpu", dtype=torch.bfloat16):
            content, _ = model(evidence)
            anchor, _ = model(evidence, anchor_only=True)
        values = dict(query_update=captured["update"], final_input_readout=captured["readout"],
                      routing_correction=content["routing"] - anchor["routing"],
                      execution_correction=content["execution"] - anchor["execution"],
                      routing_C=content["routing"], execution_C=content["execution"], anchor=anchor["routing"])
        for key, value in values.items():
            means.setdefault(key, []).append(value.float().mean(1))
        rows.append(dict(id=case["id"], metrics={key: stats(value) for key, value in values.items()}))
        print(json.dumps(dict(completed=len(rows), total=16, case=case["id"])), flush=True)
    result = dict(
        utc=datetime.now(timezone.utc).isoformat(), checkpoint=str(checkpoint), source=str(SOURCE), step=400,
        scope="Same fixed16 positive cached conditions and CPU enhancement-only protocol as the existing R10A probe. No Wan/GPU generation, API, download or new data.",
        limitations="These statistics distinguish within-sample tokens from cross-sample means. They do not establish useful physics, effective Wan writes, correct conditioning use, or video quality.",
        aggregate={key: {metric: sum(row["metrics"][key][metric] for row in rows) / len(rows)
                         for metric in rows[0]["metrics"][key]} for key in values},
        cross_sample_means={key: stats(torch.cat(value, 0)) for key, value in means.items()}, cases=rows,
    )
    out = ROOT / "reports/own400_fixed16_representation.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("aggregate", "cross_sample_means")}), flush=True)


if __name__ == "__main__":
    main()
