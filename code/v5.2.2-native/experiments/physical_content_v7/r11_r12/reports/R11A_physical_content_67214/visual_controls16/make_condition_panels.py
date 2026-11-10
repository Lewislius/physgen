"""Two explicitly score-selected normal/control pairs; decode existing videos only."""
import argparse
import json
from pathlib import Path

import decord
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUTPUT = ROOT / 'request_adapt_r11a/inference_outputs/physical_content_67214/step0400_full40_controls16_physical5_seed42'
SELECTIONS = {
    'empty_reasoning': {
        'VP034': 'Largest PC increase under empty reasoning (+2).',
        'VP010': 'Only Joint loss under empty reasoning.',
    },
    'global_train_mean': {
        'VP034': 'One of the two new Joint passes under training-mean C; largest PC increase (+2).',
        'VP038': 'The other new Joint pass under training-mean C.',
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', choices=tuple(SELECTIONS), default='empty_reasoning')
    control = parser.parse_args().control
    suffix = 'empty' if control == 'empty_reasoning' else 'mean'
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    selection = []
    for case, reason in SELECTIONS[control].items():
        panel = Image.new('RGB', (1664, 1100), 'white')
        draw = ImageDraw.Draw(panel)
        record = dict(case=case, selection=reason, selected_after_scores=True,
                      scope='Eight sampled frames per arm; targeted review, not a full-suite quality estimate.',
                      videos={}, frames={})
        for arm_index, (arm, base) in enumerate([
            ('normal', OUTPUT / 'videophy40'),
            (control, OUTPUT / 'controls' / control / 'videophy40'),
        ]):
            path = base / 'cases' / case / 'planned.mp4'
            reader = decord.VideoReader(str(path), ctx=decord.cpu(0))
            ids = np.linspace(0, len(reader) - 1, 8, dtype=int).tolist()
            images = reader.get_batch(ids).asnumpy()
            ybase = arm_index * 550
            draw.text((8, ybase + 4), f'{case} | {arm} | {len(reader)} frames', fill='black', font=font)
            for i, (index, array) in enumerate(zip(ids, images)):
                x, y = (i % 4) * 416, ybase + 30 + (i // 4) * 260
                panel.paste(Image.fromarray(array).resize((416, 240)), (x, y))
                draw.text((x + 6, y + 240), f'frame {index}', fill='black', font=font)
            record['videos'][arm] = str(path)
            record['frames'][arm] = ids
        target = HERE / f'{case}_normal_vs_{suffix}.png'
        panel.save(target)
        record['panel'] = str(target)
        selection.append(record)
    (HERE / f'{suffix}_selection.json').write_text(json.dumps(selection, indent=2) + '\n')
    print(json.dumps([v['panel'] for v in selection]))


if __name__ == '__main__':
    main()
