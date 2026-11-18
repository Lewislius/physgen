"""Read existing matched A videos into paired panels; no model/score calls."""
import argparse
import json
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['at', 'ai'], required=True)
    parser.add_argument('--cases', nargs='+', required=True)
    args = parser.parse_args()
    source = ROOT/'inference_outputs'/f'r20{args.task}_native_physical400/step0400_seed42'
    output = ROOT/f'review_{args.task}_paired'
    output.mkdir(exist_ok=True)
    records = []
    for case in args.cases:
        modes = ['normal', 'no_c', 'direct_only', 'source_queries_off']
        panel = Image.new('RGB', (1280, 170*len(modes)), '#222222')
        draw = ImageDraw.Draw(panel)
        items = []
        for row, mode in enumerate(modes):
            base = source if mode == 'normal' else source/'controls'/mode
            video = base/'videophy40/cases'/case/'planned.mp4'
            cap = cv2.VideoCapture(str(video))
            indices = [0, 20, 40, 60, 80]
            for column, index in enumerate(indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = cap.read()
                if not ok:
                    raise ValueError(f'Cannot decode {video}: {index}')
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                image = ImageOps.contain(image, (256, 148))
                panel.paste(image, (column*256, row*170))
                draw.text((column*256+3, row*170+149), f'{case} {mode} f{index}', fill='white')
            cap.release()
            items.append(dict(mode=mode, path=str(video), frames=indices))
        path = output/f'{case}.jpg'
        panel.save(path, quality=94)
        records.append(dict(case=case, panel=str(path), sources=items))
    (output/'manifest.json').write_text(json.dumps(dict(
        task=args.task, same_checkpoint=True, same_seed=True, same_caption=True,
        same_first_image=args.task == 'ai', cases=records,
        scope='Selected sentinel and score-changing paired cases; sampled frames, not all-video acceptance.'), indent=2)+'\n')
    print(json.dumps(dict(task=args.task, panels=len(records), output=str(output))))


if __name__ == '__main__':
    main()
