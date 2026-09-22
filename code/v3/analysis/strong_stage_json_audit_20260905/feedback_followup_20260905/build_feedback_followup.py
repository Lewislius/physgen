"""Supplement the visual diagnosis with reproducible decoded-frame evidence.

CPU only. Does not modify videos, plans, inference code, or older audit files.
Native crops preserve decoded pixels; contact sheets are labeled thumbnails.
"""
from pathlib import Path
import hashlib
import json
import subprocess

import cv2
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
V3 = HERE.parents[2]
RUNS = V3 / "outputs/trace_writer"
STRONG = RUNS / "trace-v3-p01-p20-strong-stage-json-20260904-144957"
BASE = RUNS / "trace-v3-p02-baseline-20260904-141716/P02/i2v/seed_000042"
FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
cv2.setNumThreads(2)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sheet(items, path, title, width=None, cols=3):
    w, h = items[0][1].size
    dest_w = width or w
    dest_h = round(h * dest_w / w)
    result = Image.new("RGB", (cols * dest_w, 32 + ((len(items) + cols - 1) // cols) * (dest_h + 24)), "#151922")
    draw = ImageDraw.Draw(result)
    draw.text((6, 6), title, fill="white", font=FONT)
    for index, (frame_id, img) in enumerate(items):
        x, y = (index % cols) * dest_w, 32 + (index // cols) * (dest_h + 24)
        draw.text((x + 6, y + 3), f"f{frame_id:03d} | {frame_id / 24:.3f}s", fill="white", font=FONT)
        result.paste(img if width is None else img.resize((dest_w, dest_h), Image.Resampling.LANCZOS), (x, y + 24))
    result.save(path)


def main():
    metadata = {"source": "original MP4 decode; no enhancement", "videos": [], "pair": {}, "code": [], "windows": {}}
    old_inventory = json.loads((HERE.parent / "inventory.json").read_text())
    old_hashes = {item["sample"]: item["sha256"] for item in old_inventory}
    for label in [f"P{i:02d}" for i in range(1, 21)] + ["P02_baseline"]:
        directory = BASE if label == "P02_baseline" else STRONG / label / "i2v/seed_000042"
        video = directory / "video.mp4"
        probe = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
            "stream=codec_name,profile,width,height,pix_fmt,r_frame_rate,avg_frame_rate,duration,nb_frames,bit_rate:format=size,duration",
            "-of", "json", str(video)
        ]))
        video_hash = digest(video)
        metadata["videos"].append({"sample": label, "path": str(video), "sha256": video_hash,
                                   "matches_previous_inventory": video_hash == old_hashes[label], "ffprobe": probe})
    paired = STRONG / "P02/i2v/seed_000042"
    for name in ["prompt.used.txt", "negative_prompt.used.txt", "stage_prompts.used.json", "temporal_route.used.json", "plan.input.json"]:
        a, b = BASE / name, paired / name
        metadata["pair"][name] = {"baseline_sha256": digest(a), "strong_sha256": digest(b), "same_bytes": a.read_bytes() == b.read_bytes(),
                                  "same_parsed": json.loads(a.read_text()) == json.loads(b.read_text()) if name.endswith(".json") else a.read_bytes() == b.read_bytes()}
    old_code = {item["path"]: item["sha256"] for item in json.loads((HERE.parent / "code_provenance.json").read_text())}
    for path in [V3 / "inference/infer_trace_writer.py", Path("/home/liuzhirui/model/Wan2.2/wan/configs/wan_ti2v_5B.py"),
                 Path("/home/liuzhirui/model/Wan2.2/wan/modules/vae2_2.py"), Path("/home/liuzhirui/model/Wan2.2/wan/utils/utils.py")]:
        value = digest(path)
        metadata["code"].append({"path": str(path), "sha256": value,
                                 "same_as_previous_audit": value == old_code[str(path)] if str(path) in old_code else None})
    # Windows target the user's additional observations, including fine texture.
    specs = {
        "P09": (range(56, 81), [56, 60, 64, 68, 72, 76], (360, 240, 920, 590)),
        "P10": (range(20, 33), [20, 24, 25, 26, 27, 28], (450, 190, 930, 570)),
        "P11": (range(44, 65), [44, 48, 52, 56, 60, 64], (460, 230, 1020, 650)),
        "P12": (range(60, 81), [60, 64, 68, 72, 76, 80], (700, 0, 1220, 300)),
        "P16": (range(0, 13), [0, 4, 8, 16, 32, 48], (360, 70, 920, 390)),
        "P17": (range(68, 89), [68, 72, 76, 80, 82, 84], (480, 0, 1080, 350)),
    }
    for label, (window, crop_frames, crop) in specs.items():
        cap = cv2.VideoCapture(str(STRONG / label / "i2v/seed_000042/video.mp4"))
        window = list(window)
        needed = set(window) | set(crop_frames)
        selected = {}
        for frame_id in range(max(needed) + 1):
            ok, bgr = cap.read()
            if not ok:
                raise RuntimeError(f"{label}: decode stopped at frame {frame_id}")
            if frame_id in needed:
                selected[frame_id] = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        cap.release()
        sheet([(f, selected[f]) for f in window], HERE / f"{label}_consecutive.png", f"{label} | consecutive decoded MP4 frames", width=320, cols=5)
        sheet([(f, selected[f].crop(crop)) for f in crop_frames], HERE / f"{label}_native_crops.png", f"{label} | native crops, 1 source pixel = 1 crop pixel")
        metadata["windows"][label] = {"consecutive_frames": window, "crop_frames": crop_frames, "crop_xyxy": crop}
        print(label, "saved", flush=True)
    (HERE / "review_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print("Paired inputs, 21 video hashes/ffprobe records, and six additional windows saved.")


if __name__ == "__main__":
    main()
