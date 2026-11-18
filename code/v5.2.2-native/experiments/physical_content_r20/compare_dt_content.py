"""Compare existing matched-caption videos; no generation, ranking, or API use."""
import json
from pathlib import Path

import cv2
from PIL import Image, ImageDraw


BASE = Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native')
CT = BASE/'experiments/physical_content_r19ct'
OUTPUT = BASE/'experiments/physical_content_r20/dt_first6_content'
CASE_IDS = ['VP002', 'VP003', 'VP004', 'VP009', 'VP014', 'VP015']


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((BASE/'experiments/physical_content_v7/videophy60/protocol.json').read_text())
    cases = {row['id']: row for row in protocol['cases']}
    reference = json.loads(Path('/home/liuzhirui/model/PhysVid/res/videophy2/scores.json').read_text())
    by_caption = {row['caption'].strip(): row for row in reference.values()}
    r18e = BASE/'experiments/physics_content_v10_impl/inference_outputs/r18e_nchunks400/step0400_fixed40_extra20_seed42'
    r19ct = CT/'inference_outputs/r19ct_grounded400/step0400_seed42'
    records = []
    for case in CASE_IDS:
        caption = cases[case]['caption']
        ref = by_caption[caption.strip()]
        suffix = Path('videophy40')/'cases'/case/'planned.mp4'
        videos = [('PhysVid', Path(ref['video_path'])), ('R18E', r18e/suffix), ('R19CT', r19ct/suffix)]
        if (r19ct/'controls/no_c'/suffix).is_file():
            videos.append(('R19CT_noC', r19ct/'controls/no_c'/suffix))
        dt = BASE/'experiments/physical_content_r19dt/inference_outputs/r19dt_interaction400/step0400_seed42'
        videos.append(('R19DT', dt/suffix))
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
        scope='Six existing preselected content/geometry failures, matched exact caption; not a controlled model ablation',
        selection='Fixed before viewing PhysVid or R18E results; not selected by reference score',
        caveat='Different model weights, random noise and generation settings; sampled frames cannot certify all intermediate frames',
        cases=records), indent=2)+'\n')
    print(json.dumps(dict(panels=len(records), output=str(OUTPUT))))


if __name__ == '__main__':
    main()
