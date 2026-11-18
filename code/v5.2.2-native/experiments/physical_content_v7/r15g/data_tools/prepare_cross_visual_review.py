"""Reuse six historical diagnostic selections for CPU-only R15G frame review."""
import json
from pathlib import Path

from decord import VideoReader, cpu
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT.parent / 'cross_benchmark_70_60'
REPORT = ROOT / 'reports/cross_visual'


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    records = []
    for group in ['visual_score_selected', 'physics_iq_visual_predeclared']:
        for source in sorted((BENCH / 'reports' / group).glob('*_selection.json')):
            previous = json.loads(source.read_text())
            is_world = group == 'visual_score_selected'
            case_id = 'wmb_' + previous['case'] if is_world else previous['case']['id']
            indices = previous['models']['r1']['frames'] if is_world else previous['frames']
            folder = BENCH / 'outputs/r15g/seed42/cases' / case_id
            record = dict(case=case_id, original_selection=str(source),
                baseline_panel=previous['panel'], frame_indices=indices,
                selection_precedes_R15G_scores=True, generated_metadata=str(folder / 'video.json'))
            records.append(record)
            if not (folder / 'video.json').is_file():
                record['status'] = 'pending_video'
                continue
            metadata = json.loads((folder / 'video.json').read_text())
            reference = json.loads((BENCH / 'outputs/r1/seed42/cases' / case_id / 'video.json').read_text())
            for key in ['case', 'caption', 'seed', 'noise_seed', 'first_image', 'output_frames']:
                if metadata[key] != reference[key]:
                    raise ValueError(f'Shared-input protocol mismatch: {case_id}/{key}')
            # R15G records the historical joint CFG formula explicitly; R1 has
            # the same sampler fields but predates these extra metadata keys.
            for key, value in reference['sampling'].items():
                if metadata['sampling'].get(key) != value:
                    raise ValueError(f'Shared sampler mismatch: {case_id}/{key}')
            if metadata['sampling'].get('guidance_composition') != 'joint_two_branch_cfg':
                raise ValueError(f'Unexpected R15G main inference policy: {case_id}')
            panel_path = REPORT / (case_id + '_R15G.png')
            if not panel_path.is_file():
                frames = VideoReader(str(folder / 'video.mp4'), ctx=cpu(0), num_threads=1).get_batch(indices).asnumpy()
                columns = len(indices) // 2
                panel = Image.new('RGB', (columns * 384, 500), (245, 245, 245))
                draw = ImageDraw.Draw(panel)
                for i, (index, frame) in enumerate(zip(indices, frames)):
                    x, y = (i % columns) * 384, (i // columns) * 250
                    panel.paste(Image.fromarray(frame).resize((384, 222)), (x, y + 24))
                    draw.text((x + 6, y + 5), f'R15G | frame {index}', fill=(0, 0, 0))
                panel.save(panel_path)
            record.update(status='ready_for_review', caption=metadata['caption'],
                panel=str(panel_path), checkpoint=metadata['checkpoint'], noise_seed=metadata['noise_seed'])
    result = dict(scope='Six historical diagnostic selections, reused before R15G cross scores. Sampled frames are not exhaustive temporal ratings or a representative population metric.',
        new_generation=0, new_scoring=0, cases=records)
    (REPORT / 'selection.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({r['case']: r['status'] for r in records}))


if __name__ == '__main__':
    main()
