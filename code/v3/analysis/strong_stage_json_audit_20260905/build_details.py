"""Add dense consecutive-frame evidence and P02 paired diagnostics."""
import json
import csv
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
from build_audit import ROOT, RUN, BASE, FONT, make_sheet, decode_sample

windows = {"P02":(16,35), "P03":(64,87), "P04":(60,83), "P06":(64,87),
           "P10":(68,91), "P13":(76,96), "P18":(24,43), "P19":(16,39)}
extra_native = {"P02":(12,20,25,26,27,28), "P04":(68,84), "P05":(8,),
                "P10":(80,), "P11":(52,), "P16":(8,32), "P18":(30,32,34,36),
                "P19":(28,), "P20":(84,)}
for sid in sorted(set(windows)|set(extra_native)):
    cap=cv2.VideoCapture(str(RUN/sid/"i2v/seed_000042/video.mp4"))
    frames=[]
    for i in range(97):
        ok,bgr=cap.read()
        if not ok: break
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        if sid in windows and windows[sid][0] <= i <= windows[sid][1]:frames.append((i,rgb))
        if i in extra_native.get(sid,()):Image.fromarray(rgb).save(ROOT/sid/f"f{i:03d}.png")
    cap.release()
    if frames:make_sheet(frames,ROOT/f"{sid}_transition.png",f"{sid} | consecutive frames | no enhancement",cols=5)

for label,directory in [("P02",RUN/"P02/i2v/seed_000042"),("P02_baseline",BASE)]:
    decode_sample(label,directory)

# Native pixel scale: the same 720x370 crop, no magnification/sharpening.
sheet=Image.new("RGB",(1440,4*400),(20,25,35))
draw=ImageDraw.Draw(sheet)
for col,label in enumerate(["P02_baseline","P02"]):
    for row,i in enumerate([0,12,20,24]):
        p=ROOT/label/f"f{i:03d}.png"
        if not p.exists():
            cap=cv2.VideoCapture(str(BASE/"video.mp4"));cap.set(cv2.CAP_PROP_POS_FRAMES,i)
            ok,bgr=cap.read();cap.release();assert ok
            Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).save(p)
        frame=Image.open(p)
        # Moving subjects span the table, so center crop is a fixed image region.
        crop=frame.crop((500,280,1220,650))
        sheet.paste(crop,(col*720,row*400+30))
        draw.text((col*720+8,row*400+6),f"{label} | f{i:02d} | 1 image pixel = 1 crop pixel",font=FONT,fill="white")
sheet.save(ROOT/"P02_native_pair.png")

report={}
for label in ["P02_baseline","P02"]:
    rows=json.loads((ROOT/f"{label}_metrics.json").read_text())
    report[label]={}
    for a,b in [(4,24),(32,60),(4,96)]:
        subset=[r for r in rows if a<=r["frame"]<=b]
        report[label][f"f{a}_f{b}"]={k:float(np.mean([r[k] for r in subset])) for k in
                    ["highpass_abs_native","clip_rgb_fraction","felt_far_hp","felt_near_hp","wall_hp","frame_mad_320"]}
(ROOT/"P02_paired_metrics.json").write_text(json.dumps(report,indent=2)+"\n")

all_rows=[]
for sid in [f"P{i:02d}" for i in range(1,21)]+["P02_baseline"]:
    all_rows.extend(json.loads((ROOT/f"{sid}_metrics.json").read_text()))
with (ROOT/"frame_metrics.csv").open("w") as f:
    writer=csv.DictWriter(f,fieldnames=sorted({k for r in all_rows for k in r}))
    writer.writeheader();writer.writerows(all_rows)
print(json.dumps(report,indent=2))
print("Consecutive sheet frames:",sum(b-a+1 for a,b in windows.values()))
