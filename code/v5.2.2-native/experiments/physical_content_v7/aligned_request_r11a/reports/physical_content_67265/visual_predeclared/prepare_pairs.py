"""Decode the three already-declared video cases; no evaluator or model calls."""
import json
from pathlib import Path

import decord
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CASES = ('VP004', 'VP008', 'VP010')


def main():
    refs = json.loads((ROOT / 'reports/r10_final_evidence.json').read_text())['source_paths']
    folders = {
        'native400': Path(refs['native400/original40']).parent,
        'R10-A': Path(refs['R10-A/original40']).parent,
        'new-R11A-67265': ROOT / 'aligned_request_r11a/inference_outputs/physical_content_67265/step0400_api_fixed60_controls16_seed42/videophy40',
    }
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    records = []
    for case in CASES:
        panel = Image.new('RGB', (1664, 1650), 'white')
        draw = ImageDraw.Draw(panel)
        record = dict(case=case, videos={}, frames={}, predeclared_before_new_scores=True,
                      declaration=str(ROOT / 'aligned_request_r11a/DESIGN.zh.txt'))
        for arm_index, (name, folder) in enumerate(folders.items()):
            path = folder / 'cases' / case / 'planned.mp4'
            reader = decord.VideoReader(str(path), ctx=decord.cpu(0))
            ids = np.linspace(0, len(reader) - 1, 8, dtype=int).tolist()
            images = reader.get_batch(ids).asnumpy()
            ybase = arm_index * 550
            draw.text((8, ybase + 4), f'{case} | {name} | {len(reader)} frames', fill='black', font=font)
            for i, (index, array) in enumerate(zip(ids, images)):
                x, y = (i % 4) * 416, ybase + 30 + (i // 4) * 260
                panel.paste(Image.fromarray(array).resize((416, 240)), (x, y))
                draw.text((x + 6, y + 240), f'frame {index}', fill='black', font=font)
            record['videos'][name] = str(path)
            record['frames'][name] = ids
        target = HERE / f'{case}_matched400.png'
        panel.save(target)
        record['panel'] = str(target)
        records.append(record)
    manifest = dict(cases=records, scope='Eight sampled frames of three prespecified cases; not a full-suite quality estimate.',
                    comparison='Same original cases, input frames, seed42/gamma5/sampler. New R11A changes revised training data/API inputs and architecture versus old references; same-data R10A400 is pending.')
    (HERE / 'selection.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps([v['panel'] for v in records]))


if __name__ == '__main__':
    main()
