"""Paired existing-video inspection; no generation, API or judge calls."""
from pathlib import Path
import json
import decord
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent / 'r11b_visual_review'
OUT.mkdir(exist_ok=True)
BASE = ROOT / 'request_memory_r11b/inference_outputs/physical_content_67308/step0400_api_fixed60_controls16_seed42'
folders = {
    'R11B full': BASE / 'videophy40',
    'R11B request off': BASE / 'controls/request_memory_off/videophy40',
    'R11B all injection off': BASE / 'controls/no_c/videophy40',
}
scores = {name: {x['id']: x for x in json.loads((folder/'scores.json').read_text())['cases']}
          for name, folder in folders.items()}
font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
records = []
for case in ('VP004', 'VP010', 'VP038'):
    panel = Image.new('RGB', (1664, 1650), 'white')
    draw = ImageDraw.Draw(panel)
    record = dict(case=case, videos={}, frames={}, scores={}, caption=scores['R11B full'][case]['caption'],
                  selection='VP004/010 were prespecified material/absorption cases; VP038 is a score-selected lost-Joint case. Not a representative full-suite quality estimate.')
    for row, (name, folder) in enumerate(folders.items()):
        path = folder / 'cases' / case / 'planned.mp4'
        reader = decord.VideoReader(str(path), ctx=decord.cpu(0))
        ids = np.linspace(0, len(reader)-1, 8, dtype=int).tolist()
        frames = reader.get_batch(ids).asnumpy()
        score = scores[name][case]
        draw.text((8, row*550+4), f'{case} | {name} | SA {score["sa"]} PC {score["pc"]}', fill='black', font=font)
        for i, (index, frame) in enumerate(zip(ids, frames)):
            x = (i%4)*416
            y = row*550+30+(i//4)*260
            panel.paste(Image.fromarray(frame).resize((416, 240)), (x, y))
            draw.text((x+6, y+240), f'frame {index}', fill='black', font=font)
        record['videos'][name] = str(path)
        record['frames'][name] = ids
        record['scores'][name] = {k: score[k] for k in ('sa', 'pc')}
    path = OUT / f'{case}.png'
    panel.save(path)
    record['panel'] = str(path)
    records.append(record)
(OUT/'selection.json').write_text(json.dumps(records, ensure_ascii=False, indent=2)+'\n')
print(json.dumps([x['panel'] for x in records]))
