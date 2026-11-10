"""Review the three already-generated cases whose SA improves without shift."""
from pathlib import Path
import json
import decord
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent / 'native_operator_r13a'
OUTPUT = ROOT / 'inference_outputs/physical_content_67598/step0400_original_fixed40_controls16_seed42'
DEST = ROOT / 'reports/physical_content_67598/visual_score_selected'
FRAMES = [0, 16, 32, 48, 64, 80]
MODES = ['normal', 'no_c', 'shift_off']
CASES = ['VP017', 'VP025', 'VP026']


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    record = dict(selected_after_scores=True, cases=CASES, frames=FRAMES, modes=MODES,
                  basis='All three SA losses from adding shift to the scale-only fixed16 control.',
                  new_generation=0, new_scoring=0, evidence={})
    for case in CASES:
        paths = {mode: (OUTPUT / 'videophy40' if mode == 'normal' else
                       OUTPUT / 'controls' / mode / 'videophy40') / 'cases' / case / 'planned.mp4'
                 for mode in MODES}
        data = {mode: json.loads(path.with_suffix('.json').read_text()) for mode, path in paths.items()}
        normal = data['normal']
        for mode in MODES:
            assert all(data[mode][key] == normal[key] for key in ['caption', 'seed', 'sampler', 'checkpoint'])
            assert np.array_equal(np.asarray(Image.open(data[mode]['canvas'])),
                                  np.asarray(Image.open(normal['canvas'])))
        panel = Image.new('RGB', (1920, 675), 'white')
        draw = ImageDraw.Draw(panel)
        draw.text((8, 2), case + ' | matched own400 | selected after SA scores', fill='black', font=font)
        for row, mode in enumerate(MODES):
            draw.text((8, 25 + row * 215), mode, fill='black', font=font)
            reader = decord.VideoReader(str(paths[mode]), ctx=decord.cpu(0), num_threads=1)
            assert len(reader) == 81
            for col, (frame, pixels) in enumerate(zip(FRAMES, reader.get_batch(FRAMES).asnumpy())):
                x, y = col * 320, 48 + row * 215
                panel.paste(Image.fromarray(pixels).resize((320, 180)), (x, y))
                draw.text((x + 4, y + 180), str(frame), fill='black', font=font)
        path = DEST / (case + '_normal_off_shift.png')
        panel.save(path)
        record['evidence'][case] = dict(caption=normal['caption'], panel=str(path),
                                       videos={k: str(v) for k, v in paths.items()})
    (DEST / 'selection.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'panels': [str(DEST / (c + '_normal_off_shift.png')) for c in CASES]}))


if __name__ == '__main__':
    main()
