"""Read-only video extraction for scientific inspection; no training-data changes."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT.parent.parent / 'data_cache/wisa_4000'
inventory = json.loads((HERE / 'inventory.json').read_text())
manifest = {r['id']: r for r in json.loads((CACHE / 'base/manifest.json').read_text())['records']}
(HERE / 'montages').mkdir(exist_ok=True)
(HERE / 'references').mkdir(exist_ok=True)


def frames(path, selected):
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened(), path
    result = []
    for i in selected:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        assert ok, (path, i)
        result.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return result


def montage(rows, path):
    width, height, title = 276, 159, 28
    canvas = Image.new('RGB', (5 * width, len(rows) * (height + title)), '#171c22')
    draw = ImageDraw.Draw(canvas)
    for ri, (label, images) in enumerate(rows):
        y = ri * (height + title)
        draw.text((6, y + 6), label[:170], fill='white')
        for ci, rgb in enumerate(images):
            canvas.paste(Image.fromarray(rgb).resize((width, height)), (ci * width, y + title))
    canvas.save(path, quality=90)


reference_rows = []
for cid in ('source0000042', 'source0000086', 'source0000104', 'source0000107'):
    row = manifest[cid]
    raw = frames(row['video'], row['indices'])
    geo = row['geometry']
    rgb = [cv2.resize(x, (geo['resize_w'], geo['resize_h']), interpolation=cv2.INTER_AREA)[
        geo['crop_top']:geo['crop_top'] + geo['h'], geo['crop_left']:geo['crop_left'] + geo['w']] for x in raw]
    path = HERE / 'references' / f'{cid}.mp4'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 30, (832, 480))
    for x in rgb:
        writer.write(cv2.cvtColor(x, cv2.COLOR_RGB2BGR))
    writer.release()
    reference_rows.append((f'REFERENCE {cid} | actual cached source interval | frames 0,20,40,60,80', [rgb[i] for i in (0,20,40,60,80)]))
    for label in ('H', 'Q'):
        record = next((r for r in inventory if r['run'] == label and r['case_id'] == cid and r['step'] == 224 and r['seed'] in (42,43)), None)
        if record:
            reference_rows.append((f'{label} step224 {cid} seed{record["seed"]}', frames(record['video'], (0,20,40,60,80))))
montage(reference_rows, HERE / 'montages/reference_vs_final.jpg')
cases = sorted({r['case_id'] for r in inventory if r['prepared_inputs']})
for start in range(0, len(cases), 3):
    rows = []
    for cid in cases[start:start + 3]:
        for label in ('H', 'Q'):
            record = next((r for r in inventory if r['run'] == label and r['case_id'] == cid), None)
            if record:
                rows.append((f'{label} {cid}: {record["caption"]}', frames(record['video'], (0,20,40,60,80))))
    montage(rows, HERE / 'montages' / f'demo_{start:02d}.jpg')
print(json.dumps(dict(reference_cases=4, demo_cases=cases, output=str(HERE / 'montages'))))
