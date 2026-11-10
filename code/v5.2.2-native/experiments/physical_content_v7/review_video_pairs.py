"""Make equal-frame evidence sheets from existing paired videos, without scoring."""
from pathlib import Path
import argparse
import json
import textwrap
from PIL import Image, ImageDraw, ImageFont
from decord import VideoReader, cpu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', nargs='*')
    parser.add_argument('--frames', nargs='+', type=int, default=[0, 16, 32, 48, 64, 80])
    parser.add_argument('--labels', nargs=2, default=['native LoRA400', 'physical C400'])
    args = parser.parse_args()
    plans = [json.loads((path / 'manifest.json').read_text()) for path in [args.baseline, args.candidate]]
    indexed = [{case['id']: case for case in plan['cases']} for plan in plans]
    chosen = args.cases or list(indexed[0])
    args.output.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 17)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    records = []
    for name in chosen:
        if name not in indexed[0] or name not in indexed[1]:
            raise ValueError(f'{name}: not in both manifests')
        a, b = indexed[0][name], indexed[1][name]
        for key in ['caption', 'prepared_inputs', 'initial_noise', 'generation_seed']:
            if a.get(key) != b.get(key):
                raise ValueError(f'{name}: unmatched {key}')
        files = [path / 'cases' / name / 'planned.mp4' for path in [args.baseline, args.candidate]]
        if not all(path.exists() and path.with_suffix('.json').exists() for path in files):
            continue
        tile_w, tile_h, left, header, gap = 288, 166, 154, 80, 28
        width = left + tile_w * len(args.frames)
        sheet = Image.new('RGB', (width, header + 2 * (tile_h + gap)), '#f8f8f8')
        draw = ImageDraw.Draw(sheet)
        draw.text((10, 7), f'{name} | Same initial image, request, seed and frame indices', fill='black', font=font)
        for line, text in enumerate(textwrap.wrap(a['caption'], width=170)):
            draw.text((10, 31 + 19 * line), text, fill='black', font=small)
        for row, (video, label) in enumerate(zip(files, args.labels)):
            reader = VideoReader(str(video), ctx=cpu(0), num_threads=1)
            if len(reader) != 81 or max(args.frames) >= len(reader):
                raise ValueError(f'{video}: requires a complete81-frame clip')
            frames = reader.get_batch(args.frames).asnumpy()
            y = header + row * (tile_h + gap)
            for i, frame in enumerate(frames):
                image = Image.fromarray(frame).resize((tile_w, tile_h), Image.Resampling.LANCZOS)
                sheet.paste(image, (left + i * tile_w, y + gap))
                draw.text((left + i * tile_w + 5, y + 5), f'frame {args.frames[i]}', fill='black', font=small)
            for line, text in enumerate(textwrap.wrap(label, width=16)):
                draw.text((10, y + 65 + 20 * line), text, fill='black', font=font)
        destination = args.output / f'{name}.png'
        sheet.save(destination)
        records.append(dict(id=name, caption=a['caption'], frame_indices=args.frames,
            baseline=str(files[0]), candidate=str(files[1]), labels=args.labels, panel=str(destination)))
    (args.output / 'manifest.json').write_text(json.dumps(dict(
        scope='Assistant visual review support; selected equal-time frames are not a full-motion or independent-human evaluation.',
        cases=records), ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(panels=len(records), output=str(args.output))))


if __name__ == '__main__':
    main()
