"""Separate temporal input coverage from the frozen judge's discrimination.

Only existing held-out caches are read. A pause is an observed edit, not an
automatically assigned physical-invalid label. Save the actual pixels as well
as judgments, so a caption cannot stand in for evidence of visible movement.
"""
import argparse
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


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(ROOT/'archive/20261027_cleanup/analysis/temporal_evidence'))
    args = parser.parse_args()
    cfg = read_config(ROOT/'configs/native_physical_wisa3857.yaml')
    seed_all(42); torch.cuda.set_device(0)
    out = Path(args.output); log = Logger(out/'metrics.jsonl')
    model = QwenKnowledge(cfg, 'cuda').cuda().eval()
    decoder = VideoDecoder(cfg, 'cuda').cuda().eval()
    dataset = VideoDataset(cfg, 'validation')
    # Four visibly different processes, including two previous weak judgments.
    for index in (0, 1, 2, 4):
        sample = move(dataset[index], 'cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
            truth = decoder(sample['latent'], resize=(288, 512))
        from PIL import Image, ImageDraw
        contact = Image.new('RGB', (512*4, 308*3))
        draw = ImageDraw.Draw(contact)
        for cell, frame in enumerate(torch.linspace(0, 80, 12).round().long().tolist()):
            rgb = ((truth[frame].permute(1, 2, 0)+1)*127.5).clamp(0, 255).byte().cpu().numpy()
            x, y = cell % 4*512, cell//4*308
            contact.paste(Image.fromarray(rgb), (x, y))
            draw.text((x+3, y+288), f'RGB frame {frame}', fill='white')
        contact.save(out/f'{sample["id"]}.jpg')
        late = truth.clone(); late[49:] = truth[48]
        alternatives = dict(stagnant=truth[:1].expand_as(truth), late_stagnant=late)
        log.write('reference_changes', index, sample=sample['id'],
                  early_rms=(truth[4:41:4]-truth[:37:4]).square().mean().sqrt(),
                  late_rms=(truth[44::4]-truth[40:-4:4]).square().mean().sqrt())
        for mode in ('adjacent7', 'uniform14', 'adjacent21', 'larger7'):
            model.cfg['video_size'] = [288, 512] if mode == 'larger7' else [192, 320]
            ids = (torch.linspace(0, 80, 14).round().long().tolist() if mode == 'uniform14'
                   else evidence_indices(sample['times'], 21 if mode == 'adjacent21' else 7))
            times = rgb_times(ids, sample['times'])
            with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
                for name, alternative in alternatives.items():
                    a, ap = model.preference(truth[ids], alternative[ids], times, sample['caption'], sample['clauses'])
                    reason_a = model.last_comparison_reasoning
                    b, bp = model.preference(alternative[ids], truth[ids], times, sample['caption'], sample['clauses'])
                    log.write('preference', index, sample=sample['id'], mode=mode, view=name,
                              rgb_indices=ids, sampled_change_rms=(truth[ids]-alternative[ids]).square().mean().sqrt(),
                              preference=(a+1-b).mean()/2, order_gap=(a+b-1).abs().mean(),
                              tie=(ap[:, 2]+bp[:, 2]).mean()/2,
                              forward_reasoning=reason_a, reverse_reasoning=model.last_comparison_reasoning)
    log.write('complete', 0, peak_gib=torch.cuda.max_memory_allocated()/2**30)
    log.close()


if __name__ == '__main__': main()
