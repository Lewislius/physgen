"""Read-only audit of saved generation/training evidence; writes separate reports."""
import json
from pathlib import Path
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/generation_failure_20260917"
RUN = "stability_20260917T113347Z_5d0494"
GENERATED = ROOT / "inference_outputs" / (RUN + "_step0100_ema_demo_seed42")
OUT.mkdir(exist_ok=True)
torch.set_num_threads(4)


def stats(x):
    x = x.float()
    return dict(shape=list(x.shape), finite=bool(x.isfinite().all()), mean=float(x.mean()),
                std=float(x.std()), rms=float(x.square().mean().sqrt()), min=float(x.min()), max=float(x.max()))


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summary(xs):
    return dict(min=float(min(xs)), mean=float(np.mean(xs)), max=float(max(xs))) if xs else None


report = dict(videos={}, training={})
for folder in sorted(GENERATED.iterdir()):
    if not (folder / "metadata.json").exists():
        continue
    meta = json.loads((folder / "metadata.json").read_text())
    if meta.get("status") != "completed":
        continue
    latent = torch.load(folder / "final_latent.pt", map_location="cpu", weights_only=True)
    condition = torch.load(folder / "condition.pt", map_location="cpu", weights_only=True)
    trajectory = rows(folder / "trajectory.jsonl")
    report["videos"][folder.name] = dict(latent=stats(latent), text=stats(condition["text"]),
        sampling_seconds=meta["sampling_seconds"], geometry=meta["geometry"],
        first_latent_max_abs=float((latent[:, :, :1] - condition["first"]).abs().max()) if condition["first"] is not None else None,
        corrections={name: {key: summary([r["metrics"][name][key] for r in trajectory]) for key in
                      ("over_base", "actual_rms", "candidate_rms", "compression")} for name in ("writer5", "writer15", "velocity")})
    if folder.name not in ("reference_i2v", "P01_i2v", "P01_t2v", "P02_t2v"):
        continue
    with imageio.get_reader(folder / "video.mp4") as reader:
        frames = [frame for frame in reader]
    means = np.array([f.mean((0, 1)) / 255 for f in frames])
    report["videos"][folder.name]["video"] = dict(frames=len(frames),
        mean_rgb_jump=summary(np.abs(np.diff(means, axis=0)).mean(-1).tolist()))
    sheet = Image.new("RGB", (5 * 256, 2 * 180), "#222222")
    draw = ImageDraw.Draw(sheet)
    for n, i in enumerate((0, 1, 4, 11, 22, 33, 55, 76, 98, 120)):
        im = Image.fromarray(frames[i]); im.thumbnail((256, 154))
        x, y = (n % 5) * 256, (n // 5) * 180
        sheet.paste(im, (x, y + 24)); draw.text((x + 5, y + 5), f"{folder.name} frame {i}", fill="white")
    sheet.save(OUT / (folder.name + "_actual_mp4.jpg"))

logs = ROOT / "train/train_log" / RUN
micro, updates = rows(logs / "micro.jsonl"), rows(logs / "updates.jsonl")
report["training"] = dict(micro_count=len(micro), updates=len(updates),
    unique_sources=len({r["id"] for r in micro}), modes={m:sum(r["mode"]==m for r in micro) for m in ("i2v", "t2v")},
    first_update=updates[0], last_update=updates[-1], fm=summary([r["fm"] for r in micro]),
    temporal=summary([r["temp"] for r in micro if r["temporal_active"]]),
    out_of_range=summary([r["out_of_range"] for r in micro if r["temporal_active"]]),
    sigma_bins={f"{lo}-{hi}":summary([r["fm"] for r in micro if lo<=r["sigma"]<hi]) for lo,hi in
                ((0,.15),(.15,.35),(.35,.6),(.6,.85),(.85,1.))},
    correction_velocity=summary([r["interventions"]["velocity"]["over_base"] for r in micro]))
cache = ROOT / "cache/wisa2400_nativefps_f121_a147456_fp32_jepa32"
manifest=json.loads((cache / "manifest.json").read_text())
report["data"] = dict(records=len(manifest["records"]), splits={s:sum(r["split"]==s for r in manifest["records"]) for s in ("train","validation")})
report["training_samples"] = []
for key in list(dict.fromkeys(r["id"] for r in micro))[:4]:
    rec=next(r for r in manifest["records"] if r["id"]==key)
    value=torch.load(cache / "vae" / (key+".pt"),map_location="cpu",weights_only=True)
    report["training_samples"].append(dict(id=key,caption=rec["caption"],geometry=rec["geometry"],duration=rec["duration"],
                                          latent=stats(value["latent"]),first=stats(value["first"])))
(OUT / "saved_evidence.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({"report":str(OUT / "saved_evidence.json"),"videos":{k:{"latent_rms":v["latent"]["rms"],"velocity":v["corrections"]["velocity"]["over_base"]} for k,v in report["videos"].items()},"training":{k:v for k,v in report["training"].items() if k not in ("first_update","last_update")}},indent=2))
