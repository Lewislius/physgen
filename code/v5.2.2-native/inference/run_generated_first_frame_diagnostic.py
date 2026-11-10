"""Independent input diagnostic. Does not modify training or model implementations.

Two fixed prompts x two text arms; replace only the first image with a reviewed
built-in image-generation asset. Reuse exact saved seed42 noise and T5 contexts.
Old-input counterparts remain in the original twelve-case experiment.
"""
import argparse
import copy
import gc
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'inference'))
from infer_native_physics_prompt_baseline import save_json


def prepare(config, cfg):
    import numpy as np
    from PIL import Image
    import torch
    import yaml
    from physgen_native.config import read_config

    old_cfg = yaml.safe_load(Path(cfg['reference_config']).read_text())
    old_source, old_output = Path(old_cfg['source']), Path(old_cfg['output'])
    source, out = Path(cfg['source']), Path(cfg['output'])
    ids = cfg['case_ids']
    old_manifest = json.loads((old_source / 'manifest.json').read_text())
    by_id = {c['id']: c for c in old_manifest['cases']}
    prompt_dir = Path(cfg['generated_image_directory'])
    image_prompts = json.loads((prompt_dir / 'prompts.json').read_text())
    selected = []
    for cid in ids:
        case = copy.deepcopy(by_id[cid])
        case['first_prompt'] = image_prompts[cid]
        case['image_seed'] = None  # The built-in image tool exposes no seed control.
        case['image_source'] = 'built-in image_gen; reviewed initial state; independent diagnostic'
        selected.append(case)
    source_manifest = copy.deepcopy(old_manifest)
    source_manifest['cases'] = selected
    source_manifest['scope'] = 'Two deliberately selected input-mismatch examples; not representative benchmark results'
    save_json(source / 'manifest.json', source_manifest)

    model_cfg = read_config(cfg['model_config'])
    spec = importlib.util.spec_from_file_location('diagnostic_wan_vae',
        Path(model_cfg['paths']['wan_code']) / 'wan/modules/vae2_2.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # Same FP32 native VAE encoding as the shared input preparation pipeline.
    vae = module.Wan2_2_VAE(vae_pth=str(Path(model_cfg['paths']['wan_checkpoint']) / 'Wan2.2_VAE.pth'),
                           dtype=torch.float32, device='cuda')
    records = []
    for cid in ids:
        folder = source / 'cases' / cid
        folder.mkdir(parents=True, exist_ok=True)
        original = old_source / 'cases' / cid
        image_path = prompt_dir / f'{cid}.png'
        with Image.open(image_path) as image:
            raw_size = image.size
            # Routine model input sizing only; keep the unaltered generated asset.
            first_image = image.convert('RGB').resize((832, 480), Image.Resampling.LANCZOS)
            first_image.save(folder / 'first.png')
            rgb = torch.from_numpy(np.array(first_image)).permute(2, 0, 1)[:, None].float().cuda() / 127.5 - 1
        with torch.inference_mode():
            first = vae.encode([rgb])[0].float().cpu()[None]
        if first.shape != (1, 48, 1, 30, 52) or not torch.isfinite(first).all():
            raise ValueError(f'Invalid first-image latent: {cid}')
        stored = torch.load(original / 'conditions.pt', map_location='cpu', weights_only=True, mmap=True)
        inputs = {k: stored[k].clone() for k in ('text', 'null_text', 'times', 'valid')}
        inputs['first'] = first
        torch.save(inputs, folder / 'conditions.pt')
        shutil.copyfile(original / 'noise.pt', folder / 'noise.pt')
        for sub, suffix in [('prompts', '.json'), ('text', '.pt')]:
            (out / sub).mkdir(parents=True, exist_ok=True)
            shutil.copyfile(old_output / sub / f'{cid}{suffix}', out / sub / f'{cid}{suffix}')
        # Equality checks test the actual paired input contract, not file hashes.
        replay = torch.load(folder / 'conditions.pt', map_location='cpu', weights_only=True)
        if any(not torch.equal(replay[k], stored[k]) for k in ('text', 'null_text', 'times', 'valid')):
            raise ValueError('A non-image condition changed')
        if not torch.equal(torch.load(original / 'noise.pt', weights_only=True),
                           torch.load(folder / 'noise.pt', weights_only=True)):
            raise ValueError('Video noise changed')
        records.append(dict(id=cid, generated_image=str(image_path), generated_size=list(raw_size),
            video_seed=cfg['seed'], image_seed=None, first_latent_changed=not torch.equal(first, stored['first']),
            text_and_noise_identical=True, original_conditions=str(original / 'conditions.pt')))
        print(json.dumps(dict(kind='prepared_generated_first', **records[-1])), flush=True)
        del rgb, first, inputs, replay, stored
    del vae
    gc.collect(); torch.cuda.empty_cache()
    save_json(out / 'input_diagnostic.json', dict(mode='built-in image_gen', training=False,
        existing_design_modified=False, original_pilot=str(old_output), cases=records,
        selection='VP001 and VP006 selected after inspecting initial images, before scoring generated videos',
        limitation='Targeted input diagnostic; cannot estimate overall VideoPhy2 improvement'))
    save_json(out / 'preparation.json', dict(status='complete', cases=len(ids), training=False,
        prompts_reused_from=str(old_output), changed_condition='first image only'))


def main():
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if not 0 <= args.index < args.shards:
        raise ValueError('Invalid shard')
    if cfg['seed'] != 42 or cfg['case_ids'] != ['VP001', 'VP006']:
        raise ValueError('Keep this two-case diagnostic fixed')
    reference = yaml.safe_load(Path(cfg['reference_config']).read_text())
    if Path(cfg['source']).resolve() == Path(reference['source']).resolve() or Path(cfg['output']).resolve() == Path(reference['output']).resolve():
        raise ValueError('Diagnostic must not overwrite existing input or output directories')
    for cid in cfg['case_ids']:
        for path in [Path(cfg['generated_image_directory']) / f'{cid}.png',
                     Path(reference['output']) / 'prompts' / f'{cid}.json',
                     Path(reference['output']) / 'text' / f'{cid}.pt']:
            if not path.is_file():
                raise FileNotFoundError(path)
    if args.check_only:
        print(json.dumps(dict(status='ready', new_videos=4, video_seed=42, source=cfg['source'], output=cfg['output']))); return
    marker = Path(cfg['output']) / 'preparation.json'
    if args.index == 0 and not marker.exists():
        prepare(args.config, cfg)
    deadline = time.monotonic() + 1200
    while not marker.exists():
        if time.monotonic() >= deadline:
            raise RuntimeError('First-image preparation did not finish; existing outputs preserved')
        time.sleep(10)
    subprocess.run([sys.executable, '-I', '-B', str(ROOT / 'inference/run_native_physics_prompt_shard.py'),
        '--config', args.config, '--index', str(args.index), '--shards', str(args.shards)], check=True)


if __name__ == '__main__':
    main()
