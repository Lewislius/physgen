"""CPU-only fixed-input C analysis; no video generation or checkpoint mutation."""
import argparse
import importlib.util
import json
import math
import statistics
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from safetensors import safe_open


def module_from(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cosine(value):
    value = value.reshape(-1, value.shape[-1]).float()
    normalized = F.normalize(value, dim=-1)
    count = len(value)
    return float((normalized.sum(0).square().sum() - count) / (count * (count - 1)))


def stats(value):
    value = value.reshape(-1, value.shape[-1]).float()
    return dict(token_cosine=cosine(value), rms=float(value.square().mean().sqrt()),
                centered_energy=float((value - value.mean(0)).square().sum()
                                      / value.square().sum().clamp_min(1e-20)))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True, mmap=True)
    cfg = saved['config']
    encoder_module = module_from(args.snapshot/'physical_content/grounded_experts.py', 'ct_enc')
    evidence_module = module_from(args.snapshot/'physical_content/evidence.py', 'ct_evidence')
    model = cfg['model']
    with torch.device('meta'):
        encoder = encoder_module.SemanticPhysicalEncoder(
            model['qwen_width'], model['width'], model['q_heads'],
            model['q_tokens'], model['content_rounds'])
    weights = {k.removeprefix('enhancement.'): v for k, v in saved['model'].items()
               if k.startswith('enhancement.')}
    encoder.load_state_dict(weights, strict=True, assign=True)
    encoder.eval().requires_grad_(False)
    # Actual trained K/RMS maps can amplify or suppress the small differences
    # in a high-cosine C. Inspect them instead of equating C cosine with routing.
    read_parameters = {k: v for k, v in saved['model'].items()
                       if k.startswith('mechanisms.') and
                       any(s in k for s in ['content_key.', 'key_norm.'])}
    constants = saved['constants']
    mean, scale = constants['condition_mean'], constants['condition_scale']
    del saved

    official = Path(cfg['paths']['wan_checkpoint'])
    index = json.loads((official/'diffusion_pytorch_model.safetensors.index.json').read_text())
    mapping = {k: v for k, v in index['weight_map'].items() if k.startswith('text_embedding.')}
    native = nn.Sequential(nn.Linear(4096, 3072), nn.GELU(approximate='tanh'),
                           nn.Linear(3072, 3072))
    for filename in set(mapping.values()):
        with safe_open(official/filename, framework='pt', device='cpu') as source:
            for key, file in mapping.items():
                if file == filename:
                    native.state_dict()[key.removeprefix('text_embedding.')].copy_(source.get_tensor(key))
    native.eval().requires_grad_(False)
    base = Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native')
    protocol = json.loads((base/'experiments/physical_content_v7/videophy60/protocol.json').read_text())
    paired = ['VP002','VP004','VP005','VP009','VP010','VP013','VP015','VP016',
              'VP017','VP018','VP024','VP025','VP026','VP027','VP034','VP038']
    cases = {case['id']: case for case in protocol['cases']}
    rows, means = [], {name: [] for name in ['native_execution', 'execution', 'execution_delta', 'route']}
    for case in paired:
        evidence = torch.load(Path(cfg['paths']['inference_evidence'])/f'videophy40_{case}'/'positive.pt',
                              map_location='cpu', weights_only=True)
        stored = torch.load(Path(cases[case]['prepared_inputs'])/'conditions.pt',
                            map_location='cpu', weights_only=True, mmap=True)
        text = stored['text'].float()
        if text.ndim == 2:
            text = text[None]
        text = text[:, :512]
        native_caption = native(text)
        valid = torch.ones(text.shape[:2], dtype=torch.bool)
        words = (evidence['words'].float() - mean) / scale
        anchor, _, _ = evidence_module.resample_words(words, evidence['word_groups'], 256)
        physics, _, _ = evidence_module.resample_words(
            evidence['native_condition'].float(), evidence['word_groups'], 256)
        initial_binding = (F.normalize(physics, dim=-1) @ F.normalize(native_caption, dim=-1)
                           .transpose(-1, -2) * math.sqrt(3072)).softmax(-1)
        initial_route = initial_binding @ native_caption
        trained = encoder(anchor, words[None], text, native_caption, valid, physics)
        delta = trained['execute'] - physics
        route_delta = encoder.route_delta(trained['physical_queries'])
        row = dict(case=case, native_execution=stats(physics), initial_route=stats(initial_route),
                   execution=stats(trained['execute']), route=stats(trained['route']),
                   physical_queries=stats(trained['physical_queries']), execution_delta=stats(delta),
                   execution_delta_to_native_norm=float(delta.norm()/physics.norm().clamp_min(1e-20)),
                   route_delta_to_initial_norm=float(route_delta.norm()/initial_route.norm().clamp_min(1e-20)),
                   execution_native_mean_cosine=float(F.cosine_similarity(
                       trained['execute'].mean(1), physics.mean(1)).mean()))
        row['physical_routing_keys'] = {}
        for site in model['injection_layers']:
            prefix = f'mechanisms.{site}.'
            key = F.linear(trained['route'], read_parameters[prefix+'content_key.weight'],
                           read_parameters[prefix+'content_key.bias'])
            key = key * torch.rsqrt(key.square().mean(-1, keepdim=True) + 1e-6)
            key = key * read_parameters[prefix+'key_norm.weight']
            head_keys = key[0].unflatten(-1, (model['native_heads'], -1)).transpose(0,1)
            head_cos = [cosine(h) for h in head_keys]
            row['physical_routing_keys'][str(site)] = dict(
                token_cosine=cosine(key), head_cosine_mean=statistics.mean(head_cos),
                head_cosine_min=min(head_cos), head_cosine_max=max(head_cos))
        rows.append(row)
        for name, value in [('native_execution', physics), ('execution', trained['execute']),
                            ('execution_delta', delta), ('route', trained['route'])]:
            means[name].append(value.mean((0,1)))
        print(json.dumps(dict(case=case, execute_cos=row['execution']['token_cosine'],
                              delta_norm_ratio=row['execution_delta_to_native_norm'])), flush=True)
    aggregate = {}
    for name in ['native_execution','initial_route','execution','route','physical_queries','execution_delta']:
        aggregate[name] = {key: statistics.mean(row[name][key] for row in rows)
                           for key in ['token_cosine','rms','centered_energy']}
    for name in ['execution_delta_to_native_norm','route_delta_to_initial_norm','execution_native_mean_cosine']:
        aggregate[name] = statistics.mean(row[name] for row in rows)
    result = dict(checkpoint=str(args.checkpoint), snapshot=str(args.snapshot), cases=paired,
                  device='CPU float32; descriptive representation audit, not a GPU inference identity check',
                  initial_definition='Exact zero-refinement encoder formula on these same native inputs; no new model trained',
                  aggregate=aggregate, between_sample_mean_cosine={k:cosine(torch.stack(v)) for k,v in means.items()},
                  physical_routing_keys={str(site): {key:statistics.mean(
                      row['physical_routing_keys'][str(site)][key] for row in rows)
                      for key in ['token_cosine','head_cosine_mean','head_cosine_min','head_cosine_max']}
                      for site in model['injection_layers']},
                  rows=rows, causal_visual_failure_proven=False,
                  video_controls_required='Use the already declared same-checkpoint no_c/mean/wrong/module controls')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output), aggregate=aggregate,
                         between_sample_mean_cosine=result['between_sample_mean_cosine'],
                         physical_routing_keys=result['physical_routing_keys'])), flush=True)


if __name__ == '__main__':
    main()
