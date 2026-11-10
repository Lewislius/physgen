"""Inspect two post-hoc SA improvements, alongside the predeclared failure review."""
from pathlib import Path
import json
from decord import VideoReader, cpu
from PIL import Image, ImageDraw

ROOT = Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_v7')
CURRENT = ROOT / 'semantic_addressing_r15e'
FRAMES = [0, 11, 22, 34, 45, 57, 68, 80]


def main():
    refs = json.loads((ROOT / 'reports/r10_final_evidence.json').read_text())['source_paths']
    paths = {name:Path(refs[name + '/original40']) for name in ['R1', 'R10-A']}
    paths['R15E'] = CURRENT / 'inference_outputs/physical_content_67955/step0400_original_fixed40_controls16_seed42/videophy40/scores.json'
    scores = {name:{row['id']:row for row in json.loads(path.read_text())['cases']} for name,path in paths.items()}
    destination = CURRENT / 'reports/posthoc_score_movers'
    destination.mkdir(parents=True, exist_ok=True)
    for case in ['VP008', 'VP021']:
        panel = Image.new('RGB', (1536, 1500), 'white')
        draw = ImageDraw.Draw(panel)
        record = dict(case=case, selection='Post-hoc: SA improvement versus R10A (VP008) or R1 (VP021). Not predeclared, not an aggregate-quality sample.',
                      caption=scores['R15E'][case]['caption'], frame_indices=FRAMES, videos={}, scores={})
        for row_index, (name, path) in enumerate(paths.items()):
            video = path.parent / 'cases' / case / 'planned.mp4'
            meta = json.loads(video.with_suffix('.json').read_text())
            assert meta['seed'] == 42 and meta['step'] == 400
            assert meta['caption'] == record['caption']
            frames = VideoReader(str(video), ctx=cpu(0), num_threads=1).get_batch(FRAMES).asnumpy()
            record['videos'][name] = str(video)
            record['scores'][name] = {key:scores[name][case][key] for key in ['sa','pc']}
            for i, (index, frame) in enumerate(zip(FRAMES, frames)):
                x, y = i % 4 * 384, row_index * 500 + i // 4 * 250
                draw.text((x + 4, y + 4), f'{name} {case} frame {index}', fill='black')
                panel.paste(Image.fromarray(frame).resize((384, 222)), (x, y + 24))
        panel.save(destination / f'{case}.png')
        (destination / f'{case}.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps(dict(case=case, panel=str(destination / f'{case}.png'))), flush=True)


if __name__ == '__main__':
    main()
