"""Decode the requested run and produce reproducible visual-audit evidence.

Only reads model outputs; saves annotated, downsampled contact sheets, selected
unaltered decoded frames, and descriptive image statistics next to this script.
Statistics are texture/motion proxies, not perceptual or physics quality scores.
"""
from pathlib import Path
import csv
import hashlib
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
RUN = ROOT.parents[1] / "outputs/trace_writer/trace-v3-p01-p20-strong-stage-json-20260904-144957"
BASE = RUN.parent / "trace-v3-p02-baseline-20260904-141716/P02/i2v/seed_000042"
FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
cv2.setNumThreads(2)


def make_sheet(frames, out, title, width=320, cols=5):
    height = round(width * 704 / 1280)
    cell_h = height + 26
    sheet = Image.new("RGB", (cols * width, 30 + ((len(frames)+cols-1)//cols)*cell_h), "#141923")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 5), title, fill="white", font=FONT)
    for k, (index, frame) in enumerate(frames):
        x, y = (k % cols)*width, 30+(k//cols)*cell_h
        draw.text((x+5, y+3), f"f{index:02d} | {index/24:.3f}s", fill="white", font=FONT)
        sheet.paste(Image.fromarray(frame).resize((width, height), Image.Resampling.LANCZOS), (x, y+26))
    sheet.save(out)


def decode_sample(label, directory):
    target = ROOT / label
    target.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(directory / "video.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames, metrics, prev = [], [], None
    index = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if index % 4 == 0:
            frames.append((index, rgb))
        if index in (0, 24, 48, 72, 96):
            Image.fromarray(rgb).save(target / f"f{index:03d}.png")
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        lowpass = cv2.GaussianBlur(gray, (0, 0), 1.0)
        small = cv2.resize(gray, (320, 176), interpolation=cv2.INTER_AREA)
        row = dict(sample=label, frame=index, time_s=index/fps,
                   mean_luma=float(gray.mean()),
                   highpass_abs_native=float(np.abs(gray-lowpass).mean()),
                   laplacian_var_native=float(cv2.Laplacian(gray, cv2.CV_32F).var()),
                   clip_rgb_fraction=float(((rgb <= 2) | (rgb >= 253)).mean()),
                   frame_mad_320=float(np.abs(small-prev).mean()) if prev is not None else 0.0)
        if label in ("P02", "P02_baseline"):
            # Fixed felt patches outside the intended two-ball route. They can
            # still contain artifacts/objects; interpret as image-change proxies.
            for name, (x0,y0,x1,y1) in {"felt_far":(450,300,750,390), "felt_near":(80,590,360,680), "wall":(160,60,1100,190)}.items():
                roi=gray[y0:y1,x0:x1]
                row[name+"_hp"] = float(np.abs(roi-cv2.GaussianBlur(roi,(0,0),1.0)).mean())
                row[name+"_std"] = float(roi.std())
        metrics.append(row)
        prev = small
        index += 1
    cap.release()
    make_sheet(frames, ROOT / f"{label}_dense.png", f"{label} | every 4 frames | decoded MP4 | no enhancement")
    with (ROOT / f"{label}_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    return {"sample": label, "video": str(directory / "video.mp4"),
            "sha256": hashlib.sha256((directory / "video.mp4").read_bytes()).hexdigest(),
            "decoded_frames": index, "fps": fps, "inspected_sheet_frames": [x[0] for x in frames],
            "mean_highpass_f4_f96": float(np.mean([r["highpass_abs_native"] for r in metrics[4:]])),
            "mean_mad_f4_f96": float(np.mean([r["frame_mad_320"] for r in metrics[4:]]))}, metrics


def main():
    inventory, all_metrics = [], []
    for number in range(1, 21):
        label = f"P{number:02d}"
        summary, rows = decode_sample(label, RUN / label / "i2v/seed_000042")
        inventory.append(summary)
        all_metrics.extend(rows)
        print(label, summary["decoded_frames"], flush=True)
    summary, rows = decode_sample("P02_baseline", BASE)
    inventory.append(summary)
    all_metrics.extend(rows)
    keys = sorted({k for row in all_metrics for k in row})
    with (ROOT / "frame_metrics.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_metrics)
    (ROOT / "inventory.json").write_text(json.dumps(inventory, indent=2)+"\n")
    print("Saved", len(inventory), "videos and", len(all_metrics), "frame records", flush=True)


if __name__ == "__main__":
    main()
