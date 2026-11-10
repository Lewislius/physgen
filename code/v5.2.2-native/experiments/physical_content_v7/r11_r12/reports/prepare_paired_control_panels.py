"""Display existing matched videos for a declared control; never generate frames."""
from pathlib import Path
import argparse
import json
import decord
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--generation-root', type=Path, required=True)
    p.add_argument('--control', required=True)
    p.add_argument('--cases', nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--selection', required=True)
    args = p.parse_args()
    folders = {'normal': args.generation_root / 'videophy40',
               args.control: args.generation_root / 'controls' / args.control / 'videophy40'}
    scores = {name: {v['id']: v for v in json.loads((folder / 'scores.json').read_text())['cases']}
              for name, folder in folders.items()}
    args.output.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    records = []
    for case in args.cases:
        panel = Image.new('RGB', (1664, 1100), 'white')
        draw = ImageDraw.Draw(panel)
        record = dict(case=case, caption=scores['normal'][case]['caption'],
                      selection=args.selection, videos={}, frames={}, scores={})
        inputs = []
        for row, (name, folder) in enumerate(folders.items()):
            path = folder / 'cases' / case / 'planned.mp4'
            meta = json.loads(path.with_suffix('.json').read_text())
            inputs.append((meta['caption'], meta['canvas'], meta['seed'], meta['checkpoint']))
            reader = decord.VideoReader(str(path), ctx=decord.cpu(0))
            indices = np.linspace(0, len(reader) - 1, 8, dtype=int).tolist()
            frames = reader.get_batch(indices).asnumpy()
            score = scores[name][case]
            draw.text((8, row * 550 + 4), f'{case} | {name} | SA {score["sa"]} PC {score["pc"]}',
                      fill='black', font=font)
            for index, (frame_id, frame) in enumerate(zip(indices, frames)):
                x, y = (index % 4) * 416, row * 550 + 30 + (index // 4) * 260
                panel.paste(Image.fromarray(frame).resize((416, 240)), (x, y))
                draw.text((x + 6, y + 240), f'frame {frame_id}', fill='black', font=font)
            record['videos'][name] = str(path)
            record['frames'][name] = indices
            record['scores'][name] = {k: score[k] for k in ('sa', 'pc')}
        if inputs[0] != inputs[1]:
            raise ValueError('This control panel requires the same checkpoint, prompt, image and seed')
        destination = args.output / (case + '.png')
        panel.save(destination)
        record['panel'] = str(destination)
        records.append(record)
    (args.output / 'selection.json').write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps([v['panel'] for v in records]))


if __name__ == '__main__':
    main()
