"""Audit the four message verifiers on held-out reference/generated videos.

Scores and weak-teacher agreement are diagnostics, not physical accuracy.
Each video uses its original requirements; C and generator analysis are excluded.
"""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from physgen_native.config import CRITERION_MESSAGE_VARIANTS
from physgen_native.criterion_verifier import CRITERIA
from physgen_native.data import VideoDataset, load_tensor, move
from physgen_native.generator import load_wan
from physgen_native.runtime import load_checkpoint, restore
from physgen_native.system import PhysGenNative
from train.train import export_video


@torch.no_grad()
def evaluate_video(model, frames, times, sample, *, with_teacher=False):
    result = dict(timestamps=times.cpu().tolist(), criteria=[name for name, _ in CRITERIA],
                  classes=['supported', 'violated', 'unknown'])
    modes = (False, True) if model.cfg['verifier']['adapter']['enabled'] else (False,)
    for active in modes:
        logits = model.knowledge.criterion_logits(frames, times, sample['caption'], sample['clauses'], adapt=active)
        p = model.critic.probabilities(logits)
        result['adapter' if active else 'base'] = dict(
            probabilities=p.mean(-2).cpu().tolist(), per_label_order=p.cpu().tolist(),
            metrics={k: float(v) for k, v in model.critic.diagnostics(logits).items()})
    if with_teacher:
        logits, trace = model.knowledge.criterion_teacher(frames, times, sample['caption'], sample['clauses'])
        p, valid = model.critic.teacher_targets(logits)
        result['weak_teacher'] = dict(**trace, probabilities=p.cpu().tolist(),
                                       accepted=(valid & trace.get('observation_usable', trace['observation_complete'])).cpu().tolist())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True, help='New directory for generated videos and evidence JSONL')
    parser.add_argument('--samples', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--with-teacher', action='store_true', help='Also record observation-first weak labels; costs extra Qwen passes')
    args = parser.parse_args()
    if args.samples < 1: raise ValueError('Use a positive held-out sample count')
    state = load_checkpoint(args.checkpoint); cfg = state['config']
    if cfg['revision'] not in CRITERION_MESSAGE_VARIANTS:
        raise ValueError('Use one of the four criterion-verifier message checkpoints')
    if not torch.cuda.is_available(): raise RuntimeError('Evaluation requires an allocated GPU')
    output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
    torch.cuda.set_device(0)
    model = PhysGenNative(cfg, load_wan(cfg), device='cuda').cuda()
    completed, _ = restore(model, None, None, state, cfg); del state
    model.eval().requires_grad_(False)
    cfg['inference']['latent_feedback']['enabled'] = False
    data = VideoDataset(cfg, 'validation')
    null = load_tensor(Path(cfg['paths']['cache'])/'base/null_text.pt').float()[None].cuda()
    (output/'protocol.json').write_text(json.dumps(dict(revision=cfg['revision'], checkpoint=args.checkpoint,
        step=completed, verifier_calibration=model.verifier_state, samples=min(args.samples, len(data)),
        interpretation='Weak-model diagnostics; not audited physical accuracy. Reference videos receive no automatic positive labels.'), indent=2))
    with (output/'evidence.jsonl').open('w') as stream, torch.no_grad():
        for index in range(min(args.samples, len(data))):
            sample = move(data[index], 'cuda')
            with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
                latent = model.sample(model.conditions(sample), sample['text'], null, seed=args.seed+index)
                export_video(model, latent, sample, output/f'{sample["id"]}.mp4', completed, args.seed+index)
                for kind, z in (('reference', sample['latent']), ('generated', latent)):
                    frames, times = model.decode_qwen_view(z, sample)
                    record = dict(id=sample['id'], caption=sample['caption'], clauses=sample['clauses'], kind=kind,
                                  observation_size_hw=list(frames.shape[-2:]), geometry=cfg.get('geometry', {'mode': 'legacy'}),
                                  **evaluate_video(model, frames, times, sample, with_teacher=args.with_teacher))
                    stream.write(json.dumps(record, ensure_ascii=False)+'\n'); stream.flush()
            print(sample['id'], flush=True)


if __name__ == '__main__': main()
