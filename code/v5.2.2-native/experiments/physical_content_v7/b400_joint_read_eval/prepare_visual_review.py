"""Compare declared new cases outside the reused16 before expanded-suite scores."""
from pathlib import Path
import argparse
import json
from decord import VideoReader, cpu
from PIL import Image, ImageDraw

ROOT = Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_v7')
B = ROOT / 'request_bound_fusion_r15b/inference_outputs/physical_content_67829/step0400_original_fixed40_controls16_seed42'
CASES = ['VP001', 'VP006', 'VP011']
FRAMES = [0, 11, 22, 34, 45, 57, 68, 80]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--cases', nargs='+', default=CASES)
    p.add_argument('--postscore', action='store_true')
    args = p.parse_args()
    args.report.mkdir(parents=True, exist_ok=True)
    for case in args.cases:
        sources = {'B_request_bound': B / 'videophy40',
                   'B_joint_extended': args.output / 'videophy40'}
        original = json.loads((sources['B_request_bound'] / f'cases/{case}/planned.json').read_text())
        record = dict(case=case, frame_indices=FRAMES, selected_before_expanded_scores=not args.postscore, outside_reused16=True, selection_reason="changed_PC/SA_case_after_scoring" if args.postscore else "declared_new_cases_before_scores", panels={},
                      scope='Sampled-frame comparison, not a complete visual score')
        for label, folder in sources.items():
            video = folder / f'cases/{case}/planned.mp4'
            meta = json.loads(video.with_suffix('.json').read_text())
            assert all(meta[k] == original[k] for k in ['caption', 'canvas', 'seed', 'step'])
            assert all(meta['sampler'][k] == original['sampler'][k] for k in ['steps', 'shift', 'guidance', 'sampler'])
            path = args.report / f'{case}_{label}.png'
            if not path.exists():
                frames = VideoReader(str(video), ctx=cpu(0), num_threads=1).get_batch(FRAMES).asnumpy()
                panel = Image.new('RGB', (1536, 500), (245, 245, 245))
                draw = ImageDraw.Draw(panel)
                for i, (index, frame) in enumerate(zip(FRAMES, frames)):
                    x, y = (i % 4) * 384, (i // 4) * 250
                    panel.paste(Image.fromarray(frame).resize((384, 222)), (x, y + 24))
                    draw.text((x + 6, y + 5), f'{label} {case} | frame {index}', fill=(0, 0, 0))
                panel.save(path)
            record['panels'][label] = str(path)
        (args.report / f'{case}_selection.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(cases=args.cases, status='panels_ready')))


if __name__ == '__main__':
    main()
