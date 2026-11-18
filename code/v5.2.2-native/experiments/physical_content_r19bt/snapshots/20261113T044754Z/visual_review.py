"""Prepare all primary videos for actual visual review; never auto-pass quality."""
import argparse
import json
from pathlib import Path

import cv2
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    destination = args.root/'visual_review'
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for suite in ['videophy40', 'videophy60']:
        for video in sorted((args.root/suite/'cases').glob('*/planned.mp4')):
            case = video.parent.name
            target = destination/f'{case}.jpg'
            cap = cv2.VideoCapture(str(video))
            indices = list(range(0, 81, 4))
            canvas = Image.new('RGB', (256*7, 166*3), '#202020')
            draw = ImageDraw.Draw(canvas)
            for i, index in enumerate(indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = cap.read()
                if not ok:
                    raise ValueError(f'Cannot decode {case} frame {index}')
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((256,148))
                x,y = (i%7)*256, (i//7)*166
                canvas.paste(image,(x,y));draw.text((x+4,y+148),f'{case} f{index}',fill='white')
            cap.release();canvas.save(target, quality=90)
            records.append(dict(case=case, video=str(video), panel=str(target),
                                reviewed=False, catastrophic_failure=None,
                                physical_process_complete=None, request_adherence=None,
                                notes=''))
    output = destination/'review.json'
    if output.exists():
        # Keep previously entered review conclusions when rerunning a recovery.
        previous={r['case']:r for r in json.loads(output.read_text())['cases']}
        records=[previous.get(r['case'],r) for r in records]
    output.write_text(json.dumps(dict(status='requires_actual_review', cases=records,
                                     warning='Contact sheets do not automatically certify temporal quality'),
                                 ensure_ascii=False, indent=2))
    print(json.dumps(dict(panels=len(records), review=str(output), automatic_pass=False)))


if __name__ == '__main__':
    main()
