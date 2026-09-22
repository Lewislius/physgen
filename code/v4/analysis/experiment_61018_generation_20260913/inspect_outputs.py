"""Read completed videos and export frame strips and descriptive temporal statistics."""
from pathlib import Path
from datetime import datetime, timezone
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FONT = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
SMALL = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 12)
folders = {'smoke2': ROOT/'inference_outputs/20260912T151800Z-e189c27c_smoke_step2',
           'formal50': ROOT/'inference_outputs/wisa_native_p_1xada48g'}
rows = []
for group, folder in folders.items():
    for path in sorted(folder.rglob('*.mp4')):
        meta = path.with_suffix('.json')
        if not meta.is_file():
            continue
        try:
            m = json.loads(meta.read_text())
        except (OSError, ValueError):
            continue
        if m.get('status') != 'completed':
            continue
        a = m['arguments']
        case = m.get('case') or a.get('case') or {}
        cid = case.get('case_id', 'reference' if 'reference' in str(path) or path.name == 'smoke_before.mp4' else path.parent.name)
        mode = m.get('generation_mode', a.get('mode', 'i2v'))
        name = f'{group}_{cid}_{mode}' + ('_persistent' if path.name == 'smoke_before.mp4' else '')
        with imageio.get_reader(path) as reader:
            frames = np.stack([frame for frame in reader]).astype(np.float32) / 255
        count, height, width, _ = frames.shape
        means = frames.mean(axis=(1, 2))
        luma = means @ np.array([.2126, .7152, .0722])
        delta = np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2, 3))
        global_delta = np.linalg.norm(np.diff(means, axis=0), axis=1)
        boundary = np.array([i % 4 == 1 for i in range(1, count)])
        indices = [i for i in [0, 1, 4, 12, 25, 37, 50, 75, 100] if i < count]
        tw, th = 256, round(height * 256 / width)
        sheet = Image.new('RGB', (tw * len(indices), th + 70), (20, 20, 20))
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 3), f'{name} | {a["steps"]} steps | {count} frames | {width}x{height}', fill='white', font=FONT)
        draw.text((8, 24), a['prompt'], fill='white', font=SMALL)
        for j, i in enumerate(indices):
            img = Image.fromarray((frames[i] * 255).round().astype('uint8')).resize((tw, th))
            sheet.paste(img, (j * tw, 70))
            draw.text((j * tw + 5, 49), f'f{i:03d} {i / float(a["fps"]):.2f}s', fill='white', font=SMALL)
        sheet_path = OUT / f'{name}.jpg'
        sheet.save(sheet_path, quality=92)
        row = dict(name=name, group=group, case_id=cid, mode=mode, path=str(path), metadata=str(meta),
                   contact_sheet=str(sheet_path), prompt=a['prompt'], steps=a['steps'], frames=count,
                   width=width, height=height, state_policy=m.get('state_policy', dict(reset_state=a.get('reset_state'))),
                   rgb_mean_range=(means.max(0) - means.min(0)).tolist(), luma_range=float(np.ptp(luma)),
                   luma_mean_jump=float(np.abs(np.diff(luma)).mean()), luma_max_jump=float(np.abs(np.diff(luma)).max()),
                   pixel_mean_frame_difference=float(delta.mean()), global_rgb_delta_mean=float(global_delta.mean()),
                   global_rgb_delta_max=float(global_delta.max()), vae_boundary_pixel_diff=float(delta[boundary].mean()),
                   vae_inner_pixel_diff=float(delta[~boundary].mean()), rgb_by_frame=means.tolist(),
                   luma_by_frame=luma.tolist(), pixel_diff_by_frame=delta.tolist())
        rows.append(row)
        print(name, count, 'luma_range', round(row['luma_range'], 4), 'pixel_diff', round(row['pixel_mean_frame_difference'], 4), flush=True)
report = dict(snapshot_utc=datetime.now(timezone.utc).isoformat(), completed_video_count=len(rows),
              groups={k: sum(r['group'] == k for r in rows) for k in folders}, videos=rows)
(OUT / 'video_inventory_metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print('SNAPSHOT', report['snapshot_utc'], report['groups'])
