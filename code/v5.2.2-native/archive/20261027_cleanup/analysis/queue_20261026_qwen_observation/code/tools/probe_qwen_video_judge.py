"""Read existing failed/supported judge pairs; generate no new Wan videos."""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import imageio.v2 as imageio
import numpy as np
import torch
from physgen_native.config import read_config
from physgen_native.knowledge import QwenKnowledge


def main(args):
    cfg = read_config(args.config)
    torch.manual_seed(42); torch.cuda.set_device(0)
    model = QwenKnowledge(cfg).eval().requires_grad_(False)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    with (output/'judgments.jsonl').open('a', buffering=1) as stream:
        for path in map(Path, args.pairs):
            metadata = json.loads(path.read_text())
            with imageio.get_reader(str(path.with_suffix('.mp4'))) as reader:
                frames = np.stack([frame for frame in reader])
            video = torch.from_numpy(frames).to('cuda').permute(0,3,1,2).float()/127.5-1
            half = video.shape[-1]//2
            first, second = video[:,:,:,:half], video[:,:,:,half:]
            times = torch.tensor(metadata['timestamps'], device='cuda')
            for mode, reverse in (('original', False), ('observation_first', False), ('observation_first', True)):
                model.cfg['video_observation_first'] = mode == 'observation_first'
                started = time.monotonic()
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    preference, probabilities = model.preference(second if reverse else first,
                        first if reverse else second, times, metadata['caption'], metadata['clauses'])
                record = dict(sample=metadata['sample'], pair=str(path), mode=mode, reverse=reverse,
                    reference_preference=float(1-preference if reverse else preference),
                    probabilities=probabilities[0].tolist(),
                    observations=model.last_comparison_observations,
                    reasoning=model.last_comparison_reasoning,
                    seconds=time.monotonic()-started)
                text = json.dumps(record, ensure_ascii=False)
                stream.write(text+'\n');print(text, flush=True)
    print(json.dumps(dict(kind='complete', generated_wan_videos=0, pairs=len(args.pairs))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT/'configs/native_qwen_process_wisa3857.yaml'))
    parser.add_argument('--pairs', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    main(parser.parse_args())
