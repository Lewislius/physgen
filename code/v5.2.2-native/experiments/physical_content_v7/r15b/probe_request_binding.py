"""CPU audit of the exact inner request-to-C binding at three actual fusion layers."""
from pathlib import Path
from datetime import datetime, timezone
import json
import math
import sys
import torch
from torch.nn import functional as F
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'snapshots/20261105T152223Z'
sys.path.insert(0, str(SOURCE))
import bootstrap
from physical_content.model import Enhancement, group_log_prior
from physical_content.teacher import Teacher
from benchmark_inputs import reference, case_input


def geometry(value):
    value = value.float()
    unit = F.normalize(value, dim=-1)
    n = value.shape[-2]
    cosine = ((unit.sum(-2).square().sum(-1) - n) / (n * (n - 1))).mean() if n > 1 else value.new_tensor(1.)
    return dict(token_cosine=float(cosine), rms=float(value.square().mean().sqrt()),
                common_token_energy_fraction=float(value.mean(-2, keepdim=True).square().sum() * n /
                                                  value.square().sum().clamp_min(1e-20)))


def norm(x, weight, eps):
    # Exact WanRMSNorm sequence, including cast before the float32 norm weight.
    return (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)).to(x.dtype) * weight


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    checkpoint = ROOT / 'runs/physical_content_67829/step0400.pt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    cfg = saved['config']; weights = saved['model']; layers = [2, 14, 29]
    assert saved['step'] == 400 and Path(cfg['source_snapshot']) == SOURCE
    foundation = Path(cfg['paths']['wan_checkpoint'])
    config = json.loads((foundation / 'config.json').read_text())
    mapping = json.loads((foundation / 'diffusion_pytorch_model.safetensors.index.json').read_text())['weight_map']
    def tensor(name, dtype):
        with safe_open(str(foundation / mapping[name]), framework='pt', device='cpu') as reader:
            return reader.get_tensor(name).to(dtype).clone()
    text_weights = {name: tensor('text_embedding.' + name, torch.bfloat16)
                    for name in ['0.weight', '0.bias', '2.weight', '2.bias']}
    native_keys = {layer: {name: tensor(f'blocks.{layer}.cross_attn.' + name,
                                      torch.float32 if 'norm_' in name else torch.bfloat16)
                          for name in ['k.weight', 'k.bias', 'norm_k.weight']} for layer in layers}
    enhancer = Enhancement(cfg).eval().requires_grad_(False)
    enhancer.load_state_dict({k.removeprefix('enhancement.'): v
                             for k, v in {**weights, **saved['constants']}.items()
                             if k.startswith('enhancement.')}, strict=True)
    enhancer.checkpointing = False
    teacher = Teacher(cfg, device='cpu')
    ids = {case['id'] for case in json.loads((SOURCE / 'subset16.json').read_text())['cases']}
    cases = [case for case in reference()['cases'] if case['id'] in ids]
    rows = []; means = {layer: [] for layer in layers}
    heads = config['num_heads']; dim = config['dim'] // heads
    split = lambda x: x.reshape(1, -1, heads, dim).transpose(1, 2)
    for case in cases:
        inputs, _ = case_input(case)
        text = inputs['text'][:, :config['text_len']]
        evidence = teacher.input(dict(id='videophy40_' + case['id']))
        with torch.autocast('cpu', dtype=torch.bfloat16):
            content, groups = enhancer(evidence)
            context = F.linear(F.gelu(F.linear(text, text_weights['0.weight'], text_weights['0.bias']),
                                      approximate='tanh'), text_weights['2.weight'], text_weights['2.bias'])
            prior = group_log_prior(groups)[:-1]
            for layer in layers:
                prefix = f'fusions.{layer}.'; native = native_keys[layer]
                request_keys = split(norm(F.linear(context, native['k.weight'], native['k.bias']),
                                          native['norm_k.weight'], config['eps'])).float()
                physical_keys = split(norm(F.linear(content['routing'], weights[prefix + 'key.weight']),
                                           weights[prefix + 'key_norm.weight'], config['eps'])).float()
                values = split(F.linear(content['execution'], weights[prefix + 'value.weight'])).float()
                with torch.autocast('cpu', enabled=False):
                    binding = ((request_keys @ physical_keys.transpose(-1, -2)) / math.sqrt(dim) + prior).softmax(-1)
                    bound = binding @ values
                    entropy = -(binding * binding.clamp_min(1e-30).log()).sum(-1) / math.log(binding.shape[-1])
                    prior_probability = prior.softmax(-1)
                    kl = (binding * (binding.clamp_min(1e-30).log() - prior_probability.log())).sum(-1)
                    joined = bound.transpose(1, 2).flatten(2)
                    rows.append(dict(id=case['id'], layer=layer, request_tokens=text.shape[1],
                                     normalized_binding_entropy=float(entropy.mean()),
                                     binding_top1_mass=float(binding.amax(-1).mean()),
                                     kl_to_relation_prior=float(kl.mean()),
                                     physical_values=geometry(values.transpose(1, 2).flatten(2)),
                                     request_bound_values=geometry(joined),
                                     per_head_bound_values=geometry(bound)))
                    means[layer].append(joined.mean(1))
    aggregate = {}
    for layer in layers:
        selected = [row for row in rows if row['layer'] == layer]
        aggregate[str(layer)] = {
            key: sum(row[key] for row in selected) / len(selected)
            for key in ['normalized_binding_entropy', 'binding_top1_mass', 'kl_to_relation_prior']}
        for key in ['physical_values', 'request_bound_values', 'per_head_bound_values']:
            aggregate[str(layer)][key] = {metric: sum(row[key][metric] for row in selected) / len(selected)
                                          for metric in selected[0][key]}
        aggregate[str(layer)]['cross_sample_bound_means'] = geometry(torch.cat(means[layer], 0)[None])
    report = dict(utc=datetime.now(timezone.utc).isoformat(), checkpoint=str(checkpoint), source=str(SOURCE),
                  layers=layers, count=len(cases), aggregate=aggregate, cases=rows,
                  scope='Exact inner request-key/physical-key binding equation on fixed16 positive conditions, CPU cached-input audit. No Wan generation, new training, scoring, API or download.',
                  limitations='Does not measure the outer video-query attention, output/state multiplication or CFG contribution. CPU autocast is a representation diagnostic, not a replacement for generation evidence.')
    (ROOT / 'reports/own400_fixed16_request_binding.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(aggregate))


if __name__ == '__main__':
    main()
