"""Render existing R20A/reference videos for manual review; no model calls."""
import argparse
import json
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageOps


BASE = Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native')
EXPERIMENTS = BASE / 'experiments'
WORKSPACE = EXPERIMENTS / 'physical_content_r20'
V7 = EXPERIMENTS / 'physical_content_v7'


def sources(task, case, physvid):
    suffix = Path('videophy40' if int(case['id'][2:]) <= 40 else 'videophy60')
    suffix = suffix / 'cases' / case['id'] / 'planned.mp4'
    if task == 't2v':
        own = WORKSPACE / 'inference_outputs/r20at_native_physical400'
        ref = physvid.get(case['caption'].strip())
        videos = [] if ref is None else [('PhysVid', Path(ref['video_path']))]
        for label, run in [('R18E', 'r18e'), ('R18G', 'r18g')]:
            videos.append((label, EXPERIMENTS / 'physics_content_v10_impl/inference_outputs'
                           / f'{run}_nchunks400/step0400_fixed40_extra20_seed42' / suffix))
        videos.append(('R19DT', EXPERIMENTS / 'physical_content_r19dt/inference_outputs'
                       / 'r19dt_interaction400/step0400_seed42' / suffix))
        label = 'R20AT'
    else:
        own = WORKSPACE / 'inference_outputs/r20ai_native_physical400'
        videos = [
            ('R15B', V7 / 'request_bound_fusion_r15b/inference_outputs/physical_content_67829'
             / 'step0400_original_fixed40_controls16_seed42' / suffix),
            ('B400+I', V7 / 'b400_joint_read_eval/videophy_outputs' / suffix),
            ('R19EI', EXPERIMENTS / 'physical_content_r19ei/inference_outputs'
             / 'r19ei_scene400/step0400_seed42' / suffix),
        ]
        label = 'R20AI'
    return videos + [(label, own / 'step0400_seed42' / suffix)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=('t2v', 'i2v'), required=True)
    parser.add_argument('--cases', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((V7 / 'videophy60/protocol.json').read_text())
    cases = {case['id']: case for case in protocol['cases']}
    physvid = {}
    if args.task == 't2v':
        raw = json.loads(Path('/home/liuzhirui/model/PhysVid/res/videophy2/scores.json').read_text())
        physvid = {row['caption'].strip(): row for row in raw.values()}
    args.output.mkdir(parents=True, exist_ok=True)
    records, missing = [], []
    for case_id in args.cases:
        case = cases[case_id]
        videos = sources(args.task, case, physvid)
        # Own metadata is written after video encoding and verification.
        if not videos[-1][1].with_suffix('.json').is_file():
            missing.append(case_id)
            continue
        if not videos[-1][1].is_file():
            raise FileNotFoundError(videos[-1][1])
        absent_references = [dict(label=label, video=str(video))
                             for label, video in videos[:-1] if not video.is_file()]
        videos = [(label, video) for label, video in videos[:-1] if video.is_file()] + videos[-1:]
        panel = Image.new('RGB', (1440, 190 * len(videos)), '#202020')
        draw = ImageDraw.Draw(panel)
        used = []
        for row, (label, video) in enumerate(videos):
            capture = cv2.VideoCapture(str(video))
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if count <= 0:
                raise ValueError(f'No decodable frames: {video}')
            indices = [round((count - 1) * index / 4) for index in range(5)]
            fps = capture.get(cv2.CAP_PROP_FPS)
            geometry = None
            for column, index in enumerate(indices):
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(f'Cannot decode {video}, frame {index}')
                frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                geometry = frame.size
                tile = ImageOps.contain(frame, (288, 166))
                panel.paste(tile, (column * 288 + (288 - tile.width) // 2,
                                   row * 190 + (166 - tile.height) // 2))
                draw.text((column * 288 + 3, row * 190 + 169),
                          f'{case_id} {label} f{index}', fill='white')
            capture.release()
            used.append(dict(label=label, video=str(video), frame_count=count,
                             fps=fps, geometry=geometry, sampled_frames=indices))
        image = args.output / f'{case_id}.jpg'
        panel.save(image, quality=94)
        records.append(dict(case=case_id, caption=case['caption'],
                            panel=str(image), sources=used,
                            missing_references=absent_references))
    manifest = dict(task=args.task, requested_cases=args.cases, rendered_cases=records,
                    own_videos_not_ready=missing,
                    scope='Explicitly selected diagnostic cases; existing videos only.',
                    comparison='R-series matched historical prompt/noise protocol; PhysVid matches caption only and has different model/settings.',
                    caveat='Five sampled frames expose major defects but do not certify every intermediate frame or the full40/60.')
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(rendered=len(records), missing=missing, output=str(args.output))))


if __name__ == '__main__':
    main()
