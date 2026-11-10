"""Measure competing objectives on final weights; never an historical-step replay."""
import json
import math
import time
from pathlib import Path
import torch
from physgen_native.criterion_verifier import CriterionVerifier
from physgen_native.data import VideoDataset, SampleOrder, load_tensor, move
from physgen_native.runtime import restore, seed_all
from train.train import export_video


def run_probes(model, source, output):
    output = Path(output); output.mkdir(exist_ok=True, parents=True)
    state = torch.load(source, map_location='cpu', weights_only=False, mmap=True)
    model.training_observers = True
    model.critic = CriterionVerifier(model.cfg['verifier'])
    restore(model, None, None, state, model.cfg)
    del state
    cfg = model.cfg; model.train()
    data = VideoDataset(cfg)
    order = iter(SampleOrder(len(data), cfg['train']['seed'], 0))
    indices = [next(order) for _ in range(804)]
    null = load_tensor(Path(cfg['paths']['cache'])/'base/null_text.pt').float()[None].cuda()
    parameters = {n:p for n,p in model.named_parameters() if n.startswith('conditioner.')}
    original_completion = model.completion
    retained = {}
    def capture(*args, **kwargs):
        result = original_completion(*args, **kwargs)
        retained['latent'] = result.detach()
        return result
    model.completion = capture
    done = set()
    file = output/'metrics.jsonl'
    if file.exists(): done = {json.loads(l)['sample_group_from_original_step'] for l in file.read_text().splitlines()}
    try:
        with file.open('a', buffering=1) as stream:
            for original_step in (65,129,201):
                if original_step in done: continue
                started = time.monotonic()
                step = original_step - 1; begin = step * 4
                model.zero_grad(set_to_none=True); model.set_phase(step)
                paired_values = []; sample_ids = []
                for micro in range(4):
                    sample = move(data[indices[begin+micro]], 'cuda')
                    sample_ids.append(sample['id'])
                    seed_all(42 + begin + micro)
                    with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
                        losses, weights, metrics = model(sample, null, step, micro=micro)
                    model.backward_losses(losses, weights, 4)
                    paired_values.append({k:float(v.detach()) for k,v in losses.items()})
                    del losses, weights, metrics
                paired = {n:p.grad.detach().float().clone() for n,p in parameters.items() if p.grad is not None}
                model.zero_grad(set_to_none=True)
                seed_all(420000 + original_step)
                with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
                    losses, weights, metrics = model.feedback_losses(sample, step, null_text=null)
                model.backward_losses(losses, weights)
                buckets = {}
                for n,p in parameters.items():
                    a = paired.get(n); b = p.grad
                    if a is None and b is None: continue
                    aa = float(a.square().sum()) if a is not None else 0.
                    bb = float(b.float().square().sum()) if b is not None else 0.
                    ab = float((a*b.float()).sum()) if a is not None and b is not None else 0.
                    part = '.'.join(n.split('.')[:3]) if n.startswith('conditioner.block.') else '.'.join(n.split('.')[:2])
                    for name in ['all_C', part]:
                        v = buckets.setdefault(name, dict(paired_squared=0., feedback_squared=0., dot=0.))
                        v['paired_squared'] += aa; v['feedback_squared'] += bb; v['dot'] += ab
                for v in buckets.values():
                    a = math.sqrt(v.pop('paired_squared')); b = math.sqrt(v.pop('feedback_squared'))
                    v.update(paired_norm=a, feedback_norm=b, ratio=b/max(a,1e-30), cosine=v['dot']/max(a*b,1e-30),
                             data_gradient_dot_combined=a*a+v['dot'])
                record = dict(checkpoint=str(source), checkpoint_step=224,
                    scope='Final-checkpoint diagnostic on three predeclared original sample groups; no optimizer update, not historical gradient replay.',
                    sample_group_from_original_step=original_step, training_samples=sample_ids,
                    feedback_sample=sample['id'], paired_losses=paired_values,
                    feedback_metrics={k:float(v.detach()) if torch.is_tensor(v) else v for k,v in metrics.items()},
                    gradients=buckets, seconds=time.monotonic()-started)
                stream.write(json.dumps(record,ensure_ascii=False)+'\n')
                print(json.dumps(dict(gradient_probe=original_step, **buckets['all_C'], seconds=record['seconds'])),flush=True)
                del losses, weights, metrics, paired
                model.zero_grad(set_to_none=True)
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
                    path=output/f'train_{sample["id"]}_group{original_step}_feedback16.mp4'
                    export_video(model,retained.pop('latent'),sample,path,224,420000+original_step)
                meta=json.loads(path.with_suffix('.json').read_text())
                meta.update(diagnostic_training_feedback=True,sampling_steps=16,condition_uses_reference_future=False,
                    initial_noise_seed=420000+original_step,scope='Current checkpoint feedback sample; not saved historical training generation.')
                meta['sampler']=dict(steps=16,shift=5.,guidance=5.,condition_strength=1.)
                path.with_suffix('.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
                del sample
    finally:
        model.completion=original_completion
    (output/'complete.json').write_text(json.dumps(dict(status='complete',cases=3)))
