"""Read-only four-cell video/score comparison for the two input diagnostics."""
import csv
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
OLD = ROOT / 'inference_outputs/native_physics_prompt_baseline12_seed42_20261027_v2'
NEW = ROOT / 'inference_outputs/generated_first_frame_diagnostic_seed42_20261027'
CASES = ['VP001', 'VP006']


def grids():
    indices = [0, 12, 24, 36, 48, 60, 80]
    w, h = 256, 148
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
    for cid in CASES:
        rows = [(label + ' + ' + text, base / arm / 'cases' / cid / 'planned.mp4')
                for base, label in [(OLD, 'Old first'), (NEW, 'Generated first')]
                for arm, text in [('original', 'original text'), ('physics_text', 'expanded text')]]
        if not all(path.with_suffix('.complete.json').is_file() for _, path in rows):
            raise RuntimeError('All four videos must finish before making the final comparison')
        canvas = Image.new('RGB', (w * len(indices), len(rows) * (h + 30) + 30), 'white')
        draw = ImageDraw.Draw(canvas)
        for i, frame in enumerate(indices):
            draw.text((i*w+5, 5), f'Frame {frame}', font=font, fill='black')
        for row, (label, path) in enumerate(rows):
            select = '+'.join(f'eq(n\\,{frame})' for frame in indices)
            result = subprocess.run(['ffmpeg', '-v', 'error', '-threads', '1', '-i', str(path),
                '-vf', f'select={select},scale={w}:{h}', '-vsync', '0', '-f', 'rawvideo',
                '-pix_fmt', 'rgb24', 'pipe:1'], check=True, capture_output=True)
            frames = np.frombuffer(result.stdout, dtype=np.uint8).reshape(-1, h, w, 3)
            if len(frames) != len(indices):
                raise ValueError(f'Missing decoded frames: {path}')
            y = 30 + row*(h+30)
            draw.text((5, y+5), cid + ' | ' + label, font=font, fill='black')
            for i, frame in enumerate(frames):
                canvas.paste(Image.fromarray(frame), (i*w, y+30))
        canvas.save(OUT / f'{cid}_first_frame_diagnostic.png')


def scores():
    rows = []
    for base, name in [(OLD, 'old_first'), (NEW, 'generated_first')]:
        for arm in ['original', 'physics_text']:
            for cid in CASES:
                folder = base / arm / 'cases' / cid
                paths = [folder/f'planned.{task}.json' for task in ['sa', 'pc']]
                if not all(path.exists() for path in paths):
                    print('Scores pending; frame grids are available.'); return
                sa, pc = (json.loads(path.read_text())['score'] for path in paths)
                rows.append(dict(id=cid, first_image=name, text=arm, sa=sa, pc=pc,
                                 joint_pass=sa>=4 and pc>=4, video=str(folder/'planned.mp4')))
    with (OUT/'first_frame_scores.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    paired = []
    for cid in CASES:
        for arm in ['original', 'physics_text']:
            a, b = [r for r in rows if r['id'] == cid and r['text'] == arm]
            paired.append(dict(id=cid, text=arm, old_sa=a['sa'], new_sa=b['sa'],
                old_pc=a['pc'], new_pc=b['pc'], sa_delta=b['sa']-a['sa'], pc_delta=b['pc']-a['pc'],
                joint_delta=int(b['joint_pass'])-int(a['joint_pass'])))
    result = dict(scope='Two targeted input-mismatch examples, chosen before video scoring; not an aggregate benchmark estimate',
        frozen_generator=True, video_seed=42, noise_and_text_unchanged=True,
        changed_condition='first image only within each text arm', cases=rows, input_effects=paired)
    (OUT/'first_frame_comparison.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(paired, indent=2))


if __name__ == '__main__':
    grids()
    scores()
