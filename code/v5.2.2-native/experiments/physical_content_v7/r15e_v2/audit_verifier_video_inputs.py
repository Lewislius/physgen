"""Inspect the exact local AutoEval processor inputs; no model inference or scoring changes."""
from pathlib import Path
import json
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'inference_outputs/physical_content_67955/step0400_original_fixed40_controls16_seed42/videophy40'


def main():
    manifest = json.loads((OUTPUT / 'manifest.json').read_text())
    evaluation = manifest['evaluation']
    sys.path.insert(0, evaluation['code'])
    import torch
    from PIL import Image, ImageDraw
    from decord import VideoReader
    from transformers.models.llama.tokenization_llama import LlamaTokenizer
    from mplug_owl_video.processing_mplug_owl import (
        MplugOwlProcessor, MplugOwlImageProcessor, load_video, get_index)
    from template import PROMPT_PHYSICS

    tokenizer = LlamaTokenizer.from_pretrained(evaluation['checkpoint'], local_files_only=True)
    processor = MplugOwlProcessor(MplugOwlImageProcessor.from_pretrained(
        evaluation['checkpoint'], local_files_only=True), tokenizer)
    destination = ROOT / 'reports/verifier_input_audit'
    destination.mkdir(parents=True, exist_ok=True)
    report = dict(utc=datetime.now(timezone.utc).isoformat(),
                  model_inference=False, scores_modified=False,
                  evaluator=evaluation, cases={})
    for case in ['VP010', 'VP034']:
        folder = OUTPUT / 'cases' / case
        video = folder / 'planned.mp4'
        inputs = processor(text=[PROMPT_PHYSICS], videos=[str(video)],
                           num_frames=evaluation['num_frames'], return_tensors='pt')
        pixels = inputs['video_pixel_values']
        indices = get_index(len(VideoReader(str(video))), evaluation['num_frames']).tolist()
        frames = load_video(str(video), evaluation['num_frames'])
        panel = Image.new('RGB', (8 * 224, 4 * 248), 'white')
        draw = ImageDraw.Draw(panel)
        for i, (index, frame) in enumerate(zip(indices, frames)):
            x, y = i % 8 * 224, i // 8 * 248
            draw.text((x + 4, y + 4), f'{case} verifier frame {index}', fill='black')
            panel.paste(frame, (x, y + 24))
        panel_path = destination / f'{case}_actual32.png'
        panel.save(panel_path)
        tokens = inputs['input_ids']
        report['cases'][case] = dict(video=str(video),
            video_mtime=video.stat().st_mtime, sampled_frame_indices=indices,
            video_pixel_values_shape=list(pixels.shape), finite=bool(torch.isfinite(pixels).all()),
            input_keys=list(inputs), negative_visual_token_ids=sorted(set(tokens[tokens < 0].tolist())),
            sample_panel=str(panel_path),
            raw_scores={task:json.loads((folder / f'planned.{task}.json').read_text())
                        for task in ['sa', 'pc']},
            scores_written_after_video=all((folder / f'planned.{task}.json').stat().st_mtime > video.stat().st_mtime
                                           for task in ['sa', 'pc']))
    report['scope'] = ('Reproduces the original local processor on the exact generated MP4 files. '
        'Checks video selection, decoded frames, visual token and input tensor; '
        'does not rerun the evaluator or establish that its quality judgment is reliable.')
    (destination / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
