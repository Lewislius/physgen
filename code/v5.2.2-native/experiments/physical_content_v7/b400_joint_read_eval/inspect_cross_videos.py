"""Export paired sampled frames for score-selected cross-benchmark examples, on CPU."""
import json
from pathlib import Path
from decord import VideoReader, cpu
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
CROSS = ROOT.parent / 'cross_benchmark_70_60/outputs'
FRAMES = [0, 11, 22, 34, 45, 57, 68, 80]
SELECTED = {
    'wmb_1m6h59wI_yI_33_35to192': 'Industrial: instruction 3 to 1, physical 4 to 5 versus R1.',
    'wmb_SwIB-QlxNCo_65_0to103': 'Natural: instruction 1 to 3, physical remains 5 versus R1.',
    'piq_0044': 'Center view of largest scene-mean Verified gain versus R1: cut-paper.',
    'piq_0077': 'Center view of largest scene-mean Verified loss versus R1: light-on-mug.',
}


def main():
    out = ROOT / 'reports/cross_visual'
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for case_id, reason in SELECTED.items():
        selected_frames = FRAMES[:-1] + [79] if case_id.startswith('piq_') else FRAMES
        record = dict(case=case_id, selection='Post-score illustrative paired cases, not a representative quality estimate.',
                      reason=reason, frames=selected_frames, videos={}, panels={})
        for model in ['r1', 'r15b_joint_i']:
            folder = CROSS / model / 'seed42/cases' / case_id
            meta = json.loads((folder / 'video.json').read_text())
            if 'caption' in record and meta['caption'] != record['caption']:
                raise ValueError('Different paired captions')
            record['caption'] = meta['caption']
            video = folder / 'video.mp4'
            panel_path = out / f'{case_id}_{model}.png'
            if not panel_path.exists():
                frames = VideoReader(str(video), ctx=cpu(0), num_threads=1).get_batch(selected_frames).asnumpy()
                panel = Image.new('RGB', (1536, 500), (245, 245, 245))
                draw = ImageDraw.Draw(panel)
                for i, (index, frame) in enumerate(zip(selected_frames, frames)):
                    x, y = i % 4 * 384, i // 4 * 250
                    panel.paste(Image.fromarray(frame).resize((384, 222)), (x, y + 24))
                    draw.text((x + 6, y + 5), f'{model} frame {index}', fill=(0, 0, 0))
                panel.save(panel_path)
            record['videos'][model] = str(video)
            record['panels'][model] = str(panel_path)
        records.append(record)
    (out / 'selection.json').write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps([{'case': r['case'], 'caption': r['caption']} for r in records], ensure_ascii=False))


if __name__ == '__main__':
    main()
