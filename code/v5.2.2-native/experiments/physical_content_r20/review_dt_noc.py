"""Compare existing matched-caption videos; no generation, ranking, or API use."""
import json
from pathlib import Path

import cv2
from PIL import Image, ImageDraw


BASE = Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native')
CT = BASE/'experiments/physical_content_r19ct'
OUTPUT = BASE/'experiments/physical_content_r20/dt_noc6_content'
CASE_IDS = ['VP002', 'VP004', 'VP009', 'VP015', 'VP026', 'VP038']


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((BASE/'experiments/physical_content_v7/videophy60/protocol.json').read_text())
    cases = {row['id']: row for row in protocol['cases']}
    ei = BASE/'experiments/physical_content_r19dt/inference_outputs/r19dt_interaction400/step0400_seed42'
    records = []
    for case in CASE_IDS:
        caption = cases[case]['caption']
        suffix = Path('videophy40')/'cases'/case/'planned.mp4'
        videos = [('DT_normal', ei/suffix), ('DT_noC', ei/'controls/no_c'/suffix)]
        canvas = Image.new('RGB', (1440, 190 * len(videos)), '#202020')
        draw = ImageDraw.Draw(canvas)
        sources = []
        for row, (label, video) in enumerate(videos):
            capture = cv2.VideoCapture(str(video))
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = capture.get(cv2.CAP_PROP_FPS)
            frames = [round((count - 1)*fraction/4) for fraction in range(5)]
            for column, index in enumerate(frames):
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(f'Cannot decode {video} frame {index}')
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                image = image.resize((288, 166))
                canvas.paste(image, (288*column, 190*row))
                draw.text((288*column+3, 190*row+169), f'{case} {label} f{index}', fill='white')
            capture.release()
            sources.append(dict(label=label, video=str(video), frames=count, fps=fps, sampled_frames=frames))
        image_path = OUTPUT/f'{case}.jpg'
        canvas.save(image_path, quality=94)
        records.append(dict(case=case, caption=caption, panel=str(image_path), sources=sources))
    (OUTPUT/'manifest.json').write_text(json.dumps(dict(
        scope='Six fixed paired diagnostic cases, same DT400 weights, T2V prompt and noise; no first image',
        selection='Existing content/structure sentinel cases; subset review, not all16 or main40',
        caveat='Sampled frames cannot certify all intermediate video frames; evaluator scores remain separate',
        cases=records), indent=2)+'\n')
    print(json.dumps(dict(panels=len(records), output=str(OUTPUT))))


if __name__ == '__main__':
    main()
