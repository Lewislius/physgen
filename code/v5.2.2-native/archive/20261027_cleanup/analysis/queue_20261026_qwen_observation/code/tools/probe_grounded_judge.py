"""Test whether measured temporal evidence fixes a scene-only video judgment.

Measurements establish observed pixel change, never physical validity. This is
a frozen-teacher experiment, not a change to the running training objectives.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from physgen_native.config import read_config
from physgen_native.data import VideoDataset, move
from physgen_native.decoder import VideoDecoder
from physgen_native.knowledge import QwenKnowledge
from physgen_native.runtime import Logger, seed_all
from physgen_native.verifier import evidence_indices, rgb_times


class MeasuredQwen(QwenKnowledge):
    use_measurements = False

    def _inputs(self, clips, body, readouts=(), **kwargs):
        if self.use_measurements and kwargs.get('video') and len(clips) == 2:
            facts = []
            for index, (frames, times) in enumerate(clips):
                # Compare corresponding ends of adjacent temporal patches.
                x = frames[1::2].float()
                change = (x[1:]-x[:-1]).abs().mean((1, 2, 3))
                facts.append(f'Video {index+1}: mean absolute RGB change between successive sampled times '
                             f'(RGB range [-1,1]) = {[round(float(v), 5) for v in change]}.')
            body += ('\nDirect measurements from these video pixels, separate from the requested description:\n'
                     +'\n'.join(facts)+'\nAn exactly repeated image has zero change. Nonzero change alone '
                     'does not establish physical plausibility; camera changes, noise and artifacts also change pixels. '
                     'Use these measurements to check claims of stillness, then inspect what is changing and '
                     'whether its development and continuing conditions are physically appropriate.')
        return super()._inputs(clips, body, readouts, **kwargs)


@torch.no_grad()
def main():
    cfg = read_config(ROOT/'configs/native_physical_wisa3857.yaml')
    seed_all(42); torch.cuda.set_device(0)
    log = Logger(ROOT/'analysis/grounded_judge/metrics.jsonl')
    model = MeasuredQwen(cfg, 'cuda').cuda().eval()
    decoder = VideoDecoder(cfg, 'cuda').cuda().eval()
    dataset = VideoDataset(cfg, 'validation')
    for index in (0, 1, 2, 4):
        sample = move(dataset[index], 'cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
            truth = decoder(sample['latent'], resize=cfg['knowledge']['video_size'])
            late = truth.clone(); late[49:] = truth[48]
            ids = evidence_indices(sample['times'], 7); times = rgb_times(ids, sample['times'])
            for view, alternative in [('stagnant', truth[:1].expand_as(truth)), ('late_stagnant', late)]:
                for measured in (False, True):
                    model.use_measurements = measured
                    a, ap = model.preference(truth[ids], alternative[ids], times, sample['caption'], sample['clauses'])
                    reason = model.last_comparison_reasoning
                    b, bp = model.preference(alternative[ids], truth[ids], times, sample['caption'], sample['clauses'])
                    log.write('preference', index, sample=sample['id'], view=view, measurements=measured,
                              preference=(a+1-b).mean()/2, order_gap=(a+b-1).abs().mean(),
                              tie=(ap[:, 2]+bp[:, 2]).mean()/2,
                              forward_reasoning=reason, reverse_reasoning=model.last_comparison_reasoning)
    log.write('complete', 0, peak_gib=torch.cuda.max_memory_allocated()/2**30)
    log.close()


if __name__ == '__main__': main()
