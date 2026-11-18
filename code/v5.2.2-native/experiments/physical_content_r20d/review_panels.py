"""CPU-only contact sheets of existing videos; never generates or scores videos."""
import argparse
import json
import textwrap
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['bt', 'bi'], required=True)
    parser.add_argument('--cases', nargs='+', required=True)
    parser.add_argument('--modes', nargs='+', default=['normal'])
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source = ROOT/'inference_outputs'/f'r20{args.task}_video_physics400/step0400_seed42'
    out = ROOT/args.output
    out.mkdir(parents=True, exist_ok=True)
    captions = {}
    for suite in ['videophy40', 'videophy60']:
        p = source/suite/'scores.json'
        if p.is_file():
            captions.update({x['id']: x['caption'] for x in json.loads(p.read_text())['cases']})
    rows = []
    for case in args.cases:
        for mode in args.modes:
            base = source if mode == 'normal' else source/'controls'/mode
            video = base/'videophy40/cases'/case/'planned.mp4'
            if not video.is_file() and mode == 'normal':
                video = base/'videophy60/cases'/case/'planned.mp4'
            if not video.is_file():
                raise FileNotFoundError(video)
            rows.append(dict(case=case, mode=mode, video=str(video), caption=captions.get(case, '')))
    panels = []
    for start in range(0, len(rows), 5):
        part = rows[start:start+5]
        sheet = Image.new('RGB', (1280, 195*len(part)), '#222222')
        draw = ImageDraw.Draw(sheet)
        for row, item in enumerate(part):
            cap = cv2.VideoCapture(item['video'])
            item['frames'] = [0, 20, 40, 60, 80]
            for col, index in enumerate(item['frames']):
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = cap.read()
                if not ok:
                    raise ValueError(f"Cannot decode {item['video']} frame {index}")
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                sheet.paste(ImageOps.contain(img, (256, 148)), (col*256, row*195))
            cap.release()
            label = f"{item['case']} {item['mode']}: {item['caption']}"
            draw.text((4, row*195+149), '\n'.join(textwrap.wrap(label, 180)), fill='white')
        dest = out/f'panel_{start//5+1:02d}.jpg'
        sheet.save(dest, quality=94)
        panels.append(dict(path=str(dest), rows=part))
    (out/'manifest.json').write_text(json.dumps(dict(task=args.task, panels=panels,
        sampled_frames_only=True, actual_visual_review_pending=True), indent=2)+'\n')
    print(json.dumps(dict(output=str(out), panels=len(panels), videos=len(rows))))


if __name__ == '__main__':
    main()
