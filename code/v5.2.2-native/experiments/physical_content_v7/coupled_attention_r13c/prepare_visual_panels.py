"""Render existing matched videos for the R13C process/appearance review."""
import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CASES = {
    'VP004': 'Ruler bending/cracking: material persistence, person and background artifacts.',
    'VP010': 'Mop/water absorption: requested process versus bright water-like artifacts.',
    'VP027': 'Bread boule shaping: useful content effects and hand/dough consistency.',
    'VP034': 'Balloon inflation: actual expansion versus a stable drifting balloon.',
    'VP038': 'Paint roller: paint deposition, roller persistence and prior Joint loss.',
}
MODES = ('no_c', 'query_modulation_off', 'key_modulation_off', 'reasoning_swapped',
         'empty_reasoning', 'global_train_mean')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', nargs='+', choices=list(CASES), default=list(CASES))
    parser.add_argument('--controls', nargs='*', choices=MODES,
                        default=['no_c', 'query_modulation_off', 'key_modulation_off'])
    parser.add_argument('--describe', action='store_true')
    args = parser.parse_args()
    prepared = json.loads((ROOT / 'prepared.json').read_text())
    run = 'physical_content_' + str(prepared['experiment_id'])
    output = ROOT / 'inference_outputs' / run / 'step0400_original_fixed40_controls16_seed42'
    destination = ROOT / 'reports' / run / 'visual_predeclared'
    refs = json.loads((ROOT.parent / 'reports/r10_final_evidence.json').read_text())['source_paths']
    folders = {'R13C': output / 'videophy40'}
    for name in ('native400', 'R1', 'R10-A'):
        folders[name] = Path(refs[name + '/original40']).parent
    folders['R13A'] = ROOT.parent / 'native_operator_r13a/inference_outputs/physical_content_67598/step0400_original_fixed40_controls16_seed42/videophy40'
    folders['R13B'] = ROOT.parent / 'interaction_query_r13b/inference_outputs/physical_content_67610/step0400_original_fixed40_controls16_seed42/videophy40'
    for mode in args.controls:
        folders['R13C_' + mode] = output / 'controls' / mode / 'videophy40'
    helper_folders = {}
    for kind, modes in (('core', ('query_modulation_off', 'key_modulation_off')),
                        ('reasoning', ('reasoning_swapped', 'empty_reasoning')),
                        ('no_c', ('no_c',)),
                        ('mean', ('global_train_mean',))):
        path = ROOT / (kind + '_controls_helper_prepared.json')
        if not path.exists():
            continue
        helper = json.loads(path.read_text())
        if not helper.get('actual_sampling_verified'):
            continue
        for mode in modes:
            helper_folders['R13C_' + mode] = (ROOT / 'helper_outputs' / run /
                str(helper['experiment_id']) / 'controls' / mode / 'videophy40')
    plan = dict(run=run, selected_before_R13C_generation_and_scores=True,
                basis='Prior R10A/R13A/R13B process failures and local benefits; all five are in the fixed16.',
                cases={case: CASES[case] for case in args.cases},
                folders={name: str(path) for name, path in folders.items()},
                frames=[0, 11, 22, 34, 45, 57, 68, 80], output=str(destination),
                new_generation=0, new_scoring=0,
                scope='Diagnostic sampled-frame review, not an aggregate visual score. '
                      'Additional score-selected cases must be labeled separately.')
    if args.describe:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return

    import decord
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    destination.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    results = []
    for case in args.cases:
        normal = folders['R13C'] / 'cases' / case / 'planned.json'
        if not normal.exists():
            results.append(dict(case=case, status='awaiting_R13C_video'))
            continue
        normal_data = json.loads(normal.read_text())
        normal_pixels = np.asarray(Image.open(normal_data['canvas']).convert('RGB'))
        record = dict(case=case, reason=CASES[case], selected_before_scores=True,
                      videos={}, panels={}, metadata={}, unavailable=[])
        for name, folder in folders.items():
            video = folder / 'cases' / case / 'planned.mp4'
            metadata = video.with_suffix('.json')
            if (not video.exists() or not metadata.exists()) and name in helper_folders:
                video = helper_folders[name] / 'cases' / case / 'planned.mp4'
                metadata = video.with_suffix('.json')
            if not video.exists() or not metadata.exists():
                record['unavailable'].append(name)
                continue
            data = json.loads(metadata.read_text())
            for key in ('caption', 'seed', 'sampler'):
                if data[key] != normal_data[key]:
                    raise ValueError(f'{case}/{name}: mismatched {key}')
            pixels = np.asarray(Image.open(data['canvas']).convert('RGB'))
            if not np.array_equal(pixels, normal_pixels):
                raise ValueError(f'{case}/{name}: different first image')
            if name.startswith('R13C_') and data['checkpoint'] != normal_data['checkpoint']:
                raise ValueError('Control and normal must use the same own400 checkpoint')
            reader = decord.VideoReader(str(video), ctx=decord.cpu(0), num_threads=1)
            if len(reader) != 81:
                raise ValueError(f'Expected81 frames, found{len(reader)}: {video}')
            frames = reader.get_batch(plan['frames']).asnumpy()
            panel = Image.new('RGB', (1664, 550), 'white')
            draw = ImageDraw.Draw(panel)
            draw.text((8, 4), f'{case} | {name} | own400 | matched image/request/seed42',
                      fill='black', font=font)
            for k, (index, frame) in enumerate(zip(plan['frames'], frames)):
                x, y = k % 4 * 416, 30 + k // 4 * 260
                panel.paste(Image.fromarray(frame).resize((416, 240)), (x, y))
                draw.text((x + 5, y + 240), f'frame {index}', fill='black', font=font)
            path = destination / f'{case}_{name}.png'
            panel.save(path)
            record['videos'][name] = str(video)
            record['panels'][name] = str(path)
            record['metadata'][name] = {key: data.get(key) for key in
                ('caption', 'canvas', 'seed', 'sampler', 'checkpoint', 'step', 'mode')}
        (destination / f'{case}_selection.json').write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + '\n')
        results.append(dict(case=case, rendered=list(record['panels']),
                            unavailable=record['unavailable']))
    print(json.dumps(results, ensure_ascii=False))


if __name__ == '__main__':
    main()
