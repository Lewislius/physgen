"""CPU-only sampled-frame review of five predeclared cases; no scoring or generation."""
from pathlib import Path
import argparse,json
from decord import VideoReader, cpu
from PIL import Image, ImageDraw

ROOT = Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_v7')
PRIOR = ROOT / 'local_state_operator_r15c/reports/physical_content_67904/visual_predeclared'
CASES = ['VP004', 'VP010', 'VP027', 'VP034', 'VP038']
FRAMES = [0, 11, 22, 34, 45, 57, 68, 80]
CONTROLS = ['no_c', 'semantic_response_off', 'native_physical_query',
            'reasoning_swapped', 'empty_reasoning', 'global_train_mean', 'direct_write_off']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    output=args.output;report=args.report
    report.mkdir(parents=True, exist_ok=True)
    created = []
    for case in CASES:
        prior = json.loads((PRIOR / f'{case}_selection.json').read_text())
        record = dict(case=case, predeclared_cases=CASES, selected_before_R15H_scores=True,
                      reason=prior['reason'], frame_indices=FRAMES, videos={}, panels={}, metadata={},
                      scope='Sampled-frame diagnostic; not a complete video assessment or aggregate metric.')
        for label in ['R1', 'R10-A', 'R13C', 'R15A', 'R15B']:
            for field in ['videos', 'panels', 'metadata']:
                record[field][label] = prior[field][label]
        candidates = {'R15H': output / 'videophy40/cases' / case / 'planned.mp4',
            'R15D': ROOT / 'denoising_matched_fusion_r15d/inference_outputs/physical_content_67920/step0400_original_fixed40_controls16_seed42/videophy40/cases' / case / 'planned.mp4'}
        candidates.update({f'R15H_{mode}': output / 'controls' / mode / 'videophy40/cases' / case / 'planned.mp4'
                           for mode in CONTROLS})
        for label, video in candidates.items():
            if not video.is_file() or not video.with_suffix('.json').is_file():
                continue
            meta = json.loads(video.with_suffix('.json').read_text())
            assert all(meta[k] == prior['metadata']['R13C'][k]
                       for k in ['caption', 'canvas', 'seed', 'step']), (case, label)
            assert all(meta['sampler'][k] == prior['metadata']['R13C']['sampler'][k]
                       for k in ['steps', 'shift', 'guidance', 'sampler']), (case, label)
            panel_path = report / f'{case}_{label}.png'
            if not panel_path.is_file():
                reader = VideoReader(str(video), ctx=cpu(0), num_threads=1)
                frames = reader.get_batch(FRAMES).asnumpy()
                panel = Image.new('RGB', (1536, 500), (245, 245, 245))
                draw = ImageDraw.Draw(panel)
                for i, (index, frame) in enumerate(zip(FRAMES, frames)):
                    x, y = (i % 4) * 384, (i // 4) * 250
                    panel.paste(Image.fromarray(frame).resize((384, 222)), (x, y + 24))
                    draw.text((x + 6, y + 5), f'{label} {case} | frame {index}', fill=(0, 0, 0))
                panel.save(panel_path)
                created.append(f'{case}/{label}')
            record['videos'][label] = str(video)
            record['panels'][label] = str(panel_path)
            record['metadata'][label] = {k: meta[k] for k in ['caption', 'canvas', 'seed', 'sampler', 'checkpoint', 'step', 'mode']}
            record['metadata'][label]['guidance_composition']=meta.get('guidance')
        if 'R15H' in record['videos']:
            (report / f'{case}_selection.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'created_panels': created}))


if __name__ == '__main__':
    main()
