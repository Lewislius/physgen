"""Frozen native Wan, original text versus physics-expanded native T5 text.

No learning, latent C, trained LoRA, reward optimization, or video selection.
Both arms replay identical saved first latents/noise and the existing Euler CFG
sampler. AutoEval receives the ORIGINAL captions. This is a development I2V
subset of VideoPhy2 captions, not the official T2V benchmark protocol.
"""
import argparse
import copy
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ARMS = ('original', 'physics_text')


def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def plan(config):
    import yaml
    cfg = yaml.safe_load(Path(config).read_text())
    source, out = Path(cfg['source']), Path(cfg['output'])
    original = json.loads((source / 'manifest.json').read_text())
    by_id = {c['id']: c for c in original['cases']}
    ids = cfg['case_ids']
    if not ids or len(ids) != len(set(ids)) or not set(ids) <= set(by_id):
        raise ValueError('Case IDs must be unique members of the fixed input manifest')
    cases = [by_id[k] for k in ids]
    if any(c['seed'] != cfg['seed'] for c in cases):
        raise ValueError('Every case must use the requested seed, without incrementing it')
    expected = dict(width=832, height=480, frames=81, fps=24., steps=50, shift=5., guidance=5.)
    if any(original['settings'].get(k) != v for k, v in expected.items()):
        raise ValueError('Shared input sampling protocol changed')
    for case in cases:
        for filename in ('conditions.pt', 'noise.pt', 'first.png'):
            if not (source / 'cases' / case['id'] / filename).is_file():
                raise FileNotFoundError((case['id'], filename))
    from physgen_native.config import read_config
    model_cfg = read_config(cfg['model_config'])
    evaluation = copy.deepcopy(original['evaluation'])
    # Use the existing independent evaluator with deterministic bounded output.
    evaluation['generation'] = dict(do_sample=False, max_new_tokens=16)
    for arm in ARMS:
        manifest = dict(cases=cases, settings=expected, evaluation=evaluation,
            scope='Fixed development subset: VideoPhy2 captions adapted to shared-first-image I2V; not official T2V leaderboard',
            arm=arm, selection=cfg['selection'], source=str(source), seed=cfg['seed'],
            base_model=model_cfg['paths']['wan_checkpoint'], trained_adapter=None,
            conditioning='Native T5 text only; latent C disabled',
            generation_caption='original caption' if arm == 'original' else 'original caption plus frozen-Qwen visible process detail',
            evaluation_caption='original caption, unchanged',
            rewrite_instruction=cfg['rewrite_instruction'], rewrite_tokens=cfg['rewrite_tokens'],
            rewrite_input=cfg['rewrite_input'],
            text_consistency_check='Frozen Qwen; reject changed actions/outcomes and fall back to the original caption; no video scores used')
        path = out / arm / 'manifest.json'
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError(f'Output already belongs to a different protocol: {path}')
        save_json(path, manifest)
    return cfg, model_cfg, cases, expected


def prepare(cfg, model_cfg, cases):
    import torch
    from physgen_native.knowledge import QwenKnowledge
    source, out = Path(cfg['source']), Path(cfg['output'])
    missing = [c for c in cases if not (out / 'prompts' / f'{c["id"]}.json').exists()]
    if missing:
        knowledge = QwenKnowledge(model_cfg, device='cuda').eval()
        def ask(body, budget):
            rendered = knowledge.tokenizer.apply_chat_template(
                [{'role': 'user', 'content': body}], tokenize=False, add_generation_prompt=True)
            inputs = {k:v.cuda() for k,v in knowledge.tokenizer(rendered, add_special_tokens=False,
                       return_tensors='pt').items() if k in ('input_ids', 'attention_mask')}
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                return knowledge._reason(inputs, budget, return_complete=True)
        for case in missing:
            body = cfg['rewrite_instruction'] + '\nOriginal request: ' + case['caption']
            addition, complete = ask(body, cfg['rewrite_tokens'])
            addition = addition.strip().strip('"')
            if not addition:
                raise ValueError(f'Empty description: {case["id"]}')
            verdict, _ = ask('Check whether the ADDITION preserves the ORIGINAL requested video event. '
                'This is a future-generation instruction, not a claim that an image already shows the action. '
                'Reject contradictions, unrequested extra events or outcomes, invented actors or objects, '
                'changes to requested success/failure, or mechanically implausible details. '
                'Allow visible details necessarily or reasonably implied by the original action. '
                'Examples: stretching a cord does not imply releasing it or an attached weight; '
                'steady skateboarding does not imply a trick; turning a bolt does not visibly compress its threads. '
                'Return exactly OK if acceptable; otherwise REJECT followed by a short reason.\n'
                f'ORIGINAL: {case["caption"]}\nADDITION: {addition}', 128)
            accepted = verdict.strip().rstrip('.').upper() == 'OK'
            generation_prompt = case['caption'] + (' ' + addition if accepted else '')
            save_json(out / 'prompts' / f'{case["id"]}.json', dict(id=case['id'],
                original_caption=case['caption'], addition=addition,
                generation_prompt=generation_prompt, complete=complete, accepted=accepted,
                consistency_verdict=verdict, input_scope='original caption only; first latent remains an unchanged Wan input',
                model=model_cfg['paths']['qwen_checkpoint']))
            print(json.dumps(dict(kind='prompt', id=case['id'], addition=addition, accepted=accepted,
                                  verdict=verdict, complete=complete)), flush=True)
        del ask, knowledge
        gc.collect(); torch.cuda.empty_cache()
    missing = [c for c in cases if not (out / 'text' / f'{c["id"]}.pt').exists()]
    if missing:
        sys.path.insert(0, model_cfg['paths']['wan_code'])
        from wan.modules.t5 import T5EncoderModel
        root = Path(model_cfg['paths']['wan_checkpoint'])
        t5 = T5EncoderModel(text_len=512, dtype=torch.bfloat16, device='cuda',
            checkpoint_path=str(root / 'models_t5_umt5-xxl-enc-bf16.pth'), tokenizer_path=str(root / 'google/umt5-xxl'))
        (out / 'text').mkdir(parents=True, exist_ok=True)
        for case in missing:
            prompt = json.loads((out / 'prompts' / f'{case["id"]}.json').read_text())['generation_prompt']
            with torch.inference_mode():
                encoded = t5([prompt], 'cuda')[0].float().cpu()[None]
            # Exact context length at the limit may indicate truncation; don't
            # silently discard the original task or the added process details.
            if encoded.shape[1] >= 512:
                raise ValueError(f'Expanded prompt reaches the native text limit: {case["id"]}')
            path = out / 'text' / f'{case["id"]}.pt'; temp = path.with_suffix('.tmp.pt')
            torch.save(encoded, temp); temp.replace(path)
        del t5
        gc.collect(); torch.cuda.empty_cache()
    save_json(out / 'preparation.json', dict(status='complete', cases=len(cases), training=False))


def strip_lora(module):
    from physgen_native.lora import LoRALinear
    for name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            setattr(module, name, child.base)
        else:
            strip_lora(child)


def generate(cfg, model_cfg, cases, settings):
    import torch
    from physgen_native.generator import load_wan, ConditionedWan
    from physgen_native.decoder import VideoDecoder
    from physgen_native.system import PhysGenNative
    from physgen_native.data import move
    from train.train import export_video
    source, out = Path(cfg['source']), Path(cfg['output'])
    generator = ConditionedWan(load_wan(model_cfg), model_cfg).eval().requires_grad_(False)
    strip_lora(generator.wan)
    decoder = VideoDecoder(model_cfg, device='cpu').eval().requires_grad_(False)
    metadata = copy.deepcopy(model_cfg)
    metadata['revision'] = 'native_wan_physics_prompt_baseline_v2'
    metadata['knowledge']['latent_reasoning_steps'] = 0
    policy = dict(steps=settings['steps'], shift=settings['shift'], guidance=settings['guidance'], condition_strength=0.)
    metadata['inference'] = policy
    harness = SimpleNamespace(generator=generator, decoder=decoder, cfg=metadata)
    for case in cases:
        folder = source / 'cases' / case['id']
        stored = torch.load(folder / 'conditions.pt', map_location='cpu', weights_only=True, mmap=True)
        inputs = {key: stored[key].float().clone() for key in ('first', 'text', 'null_text', 'times', 'valid')}
        noise = torch.load(folder / 'noise.pt', map_location='cpu', weights_only=True)
        if noise.shape != (1, 48, 21, 30, 52) or noise.dtype != torch.float32:
            raise ValueError('Expected the original FP32 saved video noise')
        prompt = json.loads((out / 'prompts' / f'{case["id"]}.json').read_text())
        for arm in ARMS:
            path = out / arm / 'cases' / case['id'] / 'planned.mp4'
            done = path.with_suffix('.complete.json')
            if done.exists() and path.is_file() and path.with_suffix('.json').is_file():
                continue
            if arm == 'physics_text' and not prompt['accepted']:
                original = out / 'original' / 'cases' / case['id'] / 'planned.mp4'
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(original, path)
                shutil.copyfile(original.with_suffix('.json'), path.with_suffix('.json'))
                record = json.loads(original.with_suffix('.complete.json').read_text())
                record.update(arm=arm, fallback_to_original=True, consistency_verdict=prompt['consistency_verdict'],
                              seconds=0., reused_video=str(original))
                save_json(done, record)
                print(json.dumps(record), flush=True)
                continue
            start = time.monotonic()
            generator.cuda(); decoder.cpu(); torch.cuda.empty_cache()
            x = move(inputs, 'cuda')
            text = x['text'] if arm == 'original' else torch.load(out / 'text' / f'{case["id"]}.pt', weights_only=True).cuda()
            dummy = x['first'].new_zeros(1, 1, model_cfg['model']['width'])
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
                latent = PhysGenNative._sample_tokens(harness, x, text, x['null_text'], cfg['seed'], policy,
                    strength=0., initial_noise=noise.cuda(), token_override=dummy, negative_token_override=dummy)
                generator.cpu(); torch.cuda.empty_cache(); decoder.cuda()
                harness.last_sample_analysis = '' if arm == 'original' else prompt['addition']
                harness.last_sample_reasoning_round = 0; harness.last_sample_condition_tokens = 0
                export_video(harness, latent, dict(x, id=case['id'], caption=case['caption']), path, 0, cfg['seed'])
            # Audit decoded frame count and decoding, including resumed outputs.
            audit = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
                '-show_entries', 'stream=width,height,nb_read_frames', '-of', 'json', str(path)],
                check=True, capture_output=True, text=True)
            stream = json.loads(audit.stdout)['streams'][0]
            if (stream['width'], stream['height'], int(stream['nb_read_frames'])) != (832, 480, 81):
                raise ValueError(f'Bad generated video geometry: {path}')
            subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-f', 'null', '-'], check=True)
            record = dict(id=case['id'], arm=arm, seed=cfg['seed'], seconds=time.monotonic()-start,
                peak_gib=torch.cuda.max_memory_allocated()/2**30, source_noise=str(folder / 'noise.pt'),
                generation_prompt=case['caption'] if arm == 'original' else prompt['generation_prompt'],
                evaluation_caption=case['caption'], status='complete')
            save_json(done, record); print(json.dumps(record), flush=True)
            decoder.cpu(); del latent, x, text, dummy
            gc.collect(); torch.cuda.empty_cache()


def compare(cfg):
    root = Path(cfg['output'])
    scores = {arm: json.loads((root / arm / 'scores.json').read_text()) for arm in ARMS}
    by_id = {arm: {r['id']: r for r in s['cases']} for arm, s in scores.items()}
    if any(set(by_id[a]) != set(cfg['case_ids']) for a in ARMS):
        raise ValueError('Both arms must have every planned case scored')
    paired = []
    for cid in cfg['case_ids']:
        a, b = (by_id[arm][cid] for arm in ARMS)
        paired.append(dict(id=cid, sa_delta=b['sa']-a['sa'], pc_delta=b['pc']-a['pc'],
            joint_delta=int(b['sa']>=4 and b['pc']>=4)-int(a['sa']>=4 and a['pc']>=4)))
    result = dict(scope='Development pilot only; not an untouched benchmark or proof of latent-C effectiveness',
        count=len(paired), seed=cfg['seed'], arms={arm:s['summary'] for arm,s in scores.items()},
        paired=paired, joint_wins=sum(r['joint_delta']>0 for r in paired),
        joint_losses=sum(r['joint_delta']<0 for r in paired),
        sa_mean_delta=statistics.mean(r['sa_delta'] for r in paired),
        pc_mean_delta=statistics.mean(r['pc_delta'] for r in paired),
        recommendation='Inspect complete videos and prompt fidelity; expand only if paired generation results justify it')
    save_json(root / 'comparison.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'inference/native_physics_prompt_baseline.yaml'))
    parser.add_argument('--stage', choices=('prepare', 'generate', 'compare'), default='prepare')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    args = parser.parse_args(); cfg, model_cfg, cases, settings = plan(args.config)
    if args.shards < 1 or not 0 <= args.shard_index < args.shards:
        raise ValueError('Invalid generation shard')
    if args.check_only:
        print(json.dumps(dict(status='ready', cases=len(cases), videos=len(cases)*2,
            seed=cfg['seed'], no_training=True, output=cfg['output']), indent=2)); return
    if args.stage == 'compare':
        compare(cfg); return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Use a single allocated GPU with the supplied launcher')
    torch.cuda.set_device(0)
    if os.environ.get('BASELINE_GPU_FAMILY') == 'rtx_pro_6000':
        from tools.gpu_policy import require_rtx_pro_6000
        require_rtx_pro_6000()
    print(json.dumps(dict(kind='start', stage=args.stage, gpu=torch.cuda.get_device_name(0),
                         time=datetime.now(timezone.utc).isoformat())), flush=True)
    if args.stage == 'prepare': prepare(cfg, model_cfg, cases)
    else: generate(cfg, model_cfg, cases[args.shard_index::args.shards], settings)


if __name__ == '__main__':
    main()
