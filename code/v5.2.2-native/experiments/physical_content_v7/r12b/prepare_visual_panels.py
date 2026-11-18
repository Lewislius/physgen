"""Render eight existing video frames for the predeclared R12B review cases."""
import argparse
import json
from pathlib import Path

import decord
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', nargs='+', required=True,
                        choices=['VP004', 'VP010', 'VP027', 'VP034'])
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    parent = root.parent
    primary = root / 'inference_outputs/physical_content_67456/step0400_original_fixed40_controls16_seed42'
    folders = {
        'R10A': parent / 'execution_value_r10/inference_outputs/physical_content_67003/step0400_full40_controls16_physical5_seed42/videophy40',
        'R12A': parent / 'reciprocal_fusion_r12a/inference_outputs/physical_content_67346/step0400_original_fixed60_controls16_seed42/videophy40',
        'R12B': primary / 'videophy40',
    }
    controls = ['native_text_value_off', 'pretext_read', 'no_c']
    helpers = json.loads((root / 'core_controls_helper_prepared.json').read_text())['helpers']
    out = root / 'reports/physical_content_67456/visual_predeclared'
    out.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    results = []
    for case in args.cases:
        if not (folders['R12B'] / 'cases' / case / 'planned.json').exists():
            results.append(dict(case=case, rendered=[], reason='R12B normal video is not ready'))
            continue
        case_folders = dict(folders)
        for mode in controls:
            destination = primary / 'controls' / mode / 'videophy40'
            if not (destination / 'cases' / case / 'planned.json').exists():
                helper = next((h for h in helpers if h['mode'] == mode and h.get('experiment_id')), None)
                if helper:
                    destination = root / 'helper_outputs/physical_content_67456' / str(helper['experiment_id']) / 'controls' / mode / 'videophy40'
            case_folders['R12B_' + mode] = destination
        record = dict(case=case, selected_before_scores=True, scores_used_for_selection=False,
                      videos={}, panels={}, frames={}, source_metadata={}, unavailable=[])
        reference = None
        for name, folder in case_folders.items():
            path = folder / 'cases' / case / 'planned.mp4'
            if not path.exists() or not path.with_suffix('.json').exists():
                record['unavailable'].append(name)
                continue
            metadata = json.loads(path.with_suffix('.json').read_text())
            canvas = np.asarray(Image.open(metadata['canvas']).convert('RGB'))
            signature = (metadata['caption'], metadata['seed'])
            if reference is None:
                reference = (signature, canvas)
            if signature != reference[0] or not np.array_equal(canvas, reference[1]):
                raise ValueError(f'{case}/{name}: different caption, image or seed')
            if name.startswith('R12B_'):
                normal = record['source_metadata']['R12B']
                if metadata['checkpoint'] != normal['checkpoint']:
                    raise ValueError('Control uses a different checkpoint')
            reader = decord.VideoReader(str(path), ctx=decord.cpu(0))
            indices = np.linspace(0, len(reader) - 1, 8, dtype=int).tolist()
            frames = reader.get_batch(indices).asnumpy()
            panel = Image.new('RGB', (1664, 550), 'white')
            draw = ImageDraw.Draw(panel)
            draw.text((8, 4), f'{case} | {name} | own400 | fixed input and seed42', fill='black', font=font)
            for k, (index, frame) in enumerate(zip(indices, frames)):
                x, y = k % 4 * 416, 30 + k // 4 * 260
                panel.paste(Image.fromarray(frame).resize((416, 240)), (x, y))
                draw.text((x + 5, y + 240), f'frame {index}', fill='black', font=font)
            destination = out / f'{case}_{name}.png'
            panel.save(destination)
            record['videos'][name] = str(path)
            record['panels'][name] = str(destination)
            record['frames'][name] = indices
            record['source_metadata'][name] = {key: metadata.get(key) for key in (
                'caption', 'canvas', 'seed', 'checkpoint', 'step', 'mode')}
        record['same_prompt_first_image_pixels_seed'] = True
        (out / f'{case}_selection.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
        results.append(dict(case=case, rendered=list(record['panels']), unavailable=record['unavailable']))
    print(json.dumps(results))


if __name__ == '__main__':
    main()
