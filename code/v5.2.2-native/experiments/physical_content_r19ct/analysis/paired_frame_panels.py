"""Make fixed-frame paired panels from existing videos, without model execution."""
import argparse
import json
from pathlib import Path

import cv2
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', required=True)
    parser.add_argument('--modes', nargs='+', default=['normal', 'reasoning_swapped'])
    parser.add_argument('--reference-root', type=Path)
    parser.add_argument('--reference-label', default='reference')
    parser.add_argument('--suite', default='videophy40', choices=['videophy40', 'videophy60'])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for start in range(0, len(args.cases), 2):
        cases = args.cases[start:start + 2]
        canvas = Image.new('RGB', (1440, 190 * len(cases) * len(args.modes)), '#202020')
        draw = ImageDraw.Draw(canvas)
        for row, (case, mode) in enumerate((c, m) for c in cases for m in args.modes):
            if args.reference_root is not None and mode == args.reference_label:
                folder = args.reference_root
            else:
                folder = args.root if mode == 'normal' else args.root/'controls'/mode
            video = folder/args.suite/'cases'/case/'planned.mp4'
            capture = cv2.VideoCapture(str(video))
            for column, frame_index in enumerate([0, 20, 40, 60, 80]):
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(f'Cannot decode {video} frame {frame_index}')
                frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((288, 166))
                x, y = column * 288, row * 190
                canvas.paste(frame, (x, y))
                draw.text((x + 3, y + 168), f'{case} {mode} f{frame_index}', fill='white')
            capture.release()
        output = args.output/('_'.join(cases) + '.jpg')
        canvas.save(output, quality=92)
        records.append(dict(cases=cases, modes=args.modes, path=str(output)))
    (args.output/'panels.json').write_text(json.dumps(records, indent=2) + '\n')
    print(json.dumps(dict(panels=len(records), output=str(args.output))))


if __name__ == '__main__':
    main()
