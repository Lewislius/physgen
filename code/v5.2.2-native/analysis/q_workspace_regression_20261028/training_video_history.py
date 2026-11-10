"""Render already saved training samples at the same case and generation seed."""
from pathlib import Path
import cv2
from PIL import Image, ImageDraw
from review_existing import ROOT, OUT, RUNS

indices = [0, 16, 32, 48, 64, 80]
width, height, label = 300, 173, 24
for step in [32, 64, 128, 224]:
    canvas = Image.new('RGB', (width*len(indices), (height+label)*len(RUNS)), '#151922')
    draw = ImageDraw.Draw(canvas)
    for row, (name, (_, video_root)) in enumerate(RUNS.items()):
        parent = ROOT/video_root
        parent = parent.parent if name == 'Q66443' else parent.parent.parent
        path = parent/f'step{step:04d}_source0000042_seed42.mp4'
        cap = cv2.VideoCapture(str(path))
        for frame_index in range(81):
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index in indices:
                col = indices.index(frame_index)
                x, y = col*width, row*(height+label)
                draw.text((x+4, y+5), f'{name} step{step} source42 | f{frame_index}', fill='white')
                rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                canvas.paste(rgb.resize((width, height), Image.Resampling.LANCZOS), (x, y+label))
        cap.release()
    canvas.save(OUT/f'training_source42_step{step:04d}.jpg', quality=93)
print('Rendered existing source42/seed42 histories at steps 32, 64, 128, 224.')
