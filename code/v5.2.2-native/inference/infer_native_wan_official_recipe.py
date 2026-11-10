"""Frozen Wan using its official I2V sampling method with persisted noise replay.

Independent baseline: no C, trained adapters, or learning. Reuse the existing
mixed-precision checkpoint loader to fit one 24 GiB GPU. The official method
handles T5, first-image VAE encoding, UniPC, native negative prompt and decoding.
An isolated module-local wrapper substitutes the saved seed42 initial noise.
This is a recipe-level control, not an isolated solver ablation.
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'inference'))
from infer_native_physics_prompt_baseline import save_json


def plan(config):
    import yaml
    from physgen_native.config import read_config
    cfg = yaml.safe_load(Path(config).read_text())
    reference_cfg = yaml.safe_load(Path(cfg['reference_config']).read_text())
    reference = json.loads((Path(reference_cfg['output'])/'original/manifest.json').read_text())
    if cfg['seed'] != 42 or cfg['case_ids'] != reference_cfg['case_ids']:
        raise ValueError('Keep the preselected twelve cases and seed42')
    source = Path(reference_cfg['source'])
    cases = []
    for case in reference['cases']:
        item = copy.deepcopy(case)
        item.update(image=str(source/'cases'/case['id']/'first.png'),
                    noise=str(source/'cases'/case['id']/'noise.pt'), group='fixed12')
        cases.append(item)
    for cid in cfg['generated_first_case_ids']:
        item = copy.deepcopy(next(c for c in reference['cases'] if c['id'] == cid))
        item.update(id=cid+'_generated_first', input_id=cid, group='targeted_input_diagnostic',
                    image=str(Path(cfg['generated_first_inputs'])/'cases'/cid/'first.png'),
                    noise=str(source/'cases'/cid/'noise.pt'), image_seed=None)
        cases.append(item)
    for case in cases:
        for key in ['image', 'noise']:
            if not Path(case[key]).is_file():
                raise FileNotFoundError(case[key])
    manifest = dict(cases=cases, settings=reference['settings'], evaluation=reference['evaluation'],
        scope='Fixed 12 VideoPhy2 captions adapted to I2V, plus 2 separately reported input diagnostics; not official T2V benchmark',
        recipe='WanTI2V.i2v method with saved initial-noise replay: UniPC50 shift5 CFG5, official negative prompt, native VAE',
        precision='Existing immutable load_wan: BF16 weights with native critical FP32 parameters',
        trained_adapter=None, latent_C=False, seed=42, reference=str(Path(reference_cfg['output'])/'original'),
        caution='Recipe-level comparison: solver, negative prompt and native encode/decode path change together')
    out = Path(cfg['output'])
    path = out/'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Output directory belongs to another protocol')
    save_json(path, manifest)
    return cfg, read_config(cfg['model_config']), cases, manifest


def generate(cfg, model_cfg, cases):
    import gc
    import imageio.v2 as imageio
    import torch
    from PIL import Image
    from physgen_native.generator import load_wan
    sys.path.insert(0, model_cfg['paths']['wan_code'])
    from wan.configs import WAN_CONFIGS
    from wan.textimage2video import WanTI2V
    import wan.textimage2video as native_module
    from wan.modules.t5 import T5EncoderModel
    from wan.modules.vae2_2 import Wan2_2_VAE
    from wan.utils.utils import best_output_size
    torch.cuda.set_device(0)
    native = WAN_CONFIGS['ti2v-5B']
    base = Path(model_cfg['paths']['wan_checkpoint'])
    # Construct exactly the official method's state, using the existing loader
    # for the DiT to retain its FP32 time embedding on a 24 GiB allocation.
    pipeline = WanTI2V.__new__(WanTI2V)
    pipeline.device = torch.device('cuda:0')
    pipeline.config = native
    pipeline.rank, pipeline.sp_size = 0, 1
    pipeline.t5_cpu, pipeline.init_on_cpu = False, True
    pipeline.num_train_timesteps, pipeline.param_dtype = native.num_train_timesteps, native.param_dtype
    pipeline.patch_size, pipeline.vae_stride = native.patch_size, native.vae_stride
    pipeline.sample_neg_prompt = native.sample_neg_prompt
    pipeline.text_encoder = T5EncoderModel(text_len=native.text_len, dtype=native.t5_dtype,
        device='cpu', checkpoint_path=str(base/native.t5_checkpoint), tokenizer_path=str(base/native.t5_tokenizer))
    pipeline.vae = Wan2_2_VAE(vae_pth=str(base/native.vae_checkpoint), device=pipeline.device)
    pipeline.model = load_wan(model_cfg)
    print(json.dumps(dict(kind='ready', gpu=torch.cuda.get_device_name(0), solver='unipc',
                          negative_prompt=native.sample_neg_prompt)), flush=True)
    out = Path(cfg['output'])
    for case in cases:
        path = out/'cases'/case['id']/'planned.mp4'
        marker = path.with_suffix('.complete.json')
        if marker.exists() and path.is_file():
            continue
        with Image.open(case['image']) as image:
            first = image.convert('RGB')
        if best_output_size(first.width, first.height, 32, 32, 832*480) != (832, 480):
            raise ValueError('Official I2V would change output dimensions')
        saved = torch.load(case['noise'], map_location='cpu', weights_only=True)
        if saved.shape != (1,48,21,30,52) or saved.dtype != torch.float32:
            raise ValueError('Invalid persisted FP32 noise')
        # Official I2V has no initial_noise argument. Replace only this module's
        # torch name during its call; all other modules keep the real torch.
        # Consume the official RNG draw as usual, but return the saved tensor.
        # This modifies no source files, global torch API, or training objects.
        class ReplayNoise:
            calls = 0
            max_difference = None
            def __getattr__(self, name):
                return getattr(torch, name)
            def randn(self, *shape, **kwargs):
                if self.calls or tuple(shape) != (48,21,30,52) or kwargs['generator'].initial_seed() != cfg['seed']:
                    raise ValueError('Unexpected native initial-noise draw')
                drawn = torch.randn(*shape, **kwargs)
                replay = saved[0].to(device=drawn.device, dtype=drawn.dtype).clone()
                self.max_difference = float((drawn-replay).abs().max())
                self.calls += 1
                return replay
        replay_noise = ReplayNoise()
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        original_torch = native_module.torch
        try:
            native_module.torch = replay_noise
            video = pipeline.i2v(input_prompt=case['caption'], img=first, max_area=832*480,
                frame_num=81, shift=5., sample_solver='unipc', sampling_steps=50,
                guide_scale=5., n_prompt='', seed=cfg['seed'], offload_model=True)
        finally:
            native_module.torch = original_torch
        if replay_noise.calls != 1:
            raise ValueError('The official sampler did not consume the saved noise')
        if tuple(video.shape) != (3, 81, 480, 832) or not torch.isfinite(video).all():
            raise ValueError('Invalid native video')
        frames = ((video.permute(1,2,3,0).clamp(-1,1)+1)*127.5).round().byte().cpu().numpy()
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix('.partial.mp4')
        with imageio.get_writer(str(temp), fps=24., codec='libx264', pixelformat='yuv420p',
            macro_block_size=None, quality=None, ffmpeg_params=['-crf','18','-profile:v','baseline']) as stream:
            for frame in frames:
                stream.append_data(frame)
        temp.replace(path)
        audit = subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-count_frames',
            '-show_entries','stream=width,height,nb_read_frames','-of','json',str(path)],
            check=True,capture_output=True,text=True)
        stream_info = json.loads(audit.stdout)['streams'][0]
        if (stream_info['width'],stream_info['height'],int(stream_info['nb_read_frames'])) != (832,480,81):
            raise ValueError('Video encoding lost frames')
        subprocess.run(['ffmpeg','-v','error','-i',str(path),'-f','null','-'],check=True)
        record = dict(id=case['id'], caption=case['caption'], seed=cfg['seed'], solver='unipc',
            steps=50, shift=5., guidance=5., negative_prompt=native.sample_neg_prompt,
            exact_saved_noise_match=True, noise_policy='module-local replay of persisted seed42 FP32 tensor',
            native_redraw_max_difference=replay_noise.max_difference, first_image=case['image'], group=case['group'],
            seconds=time.monotonic()-start, peak_gib=torch.cuda.max_memory_allocated()/2**30,
            latent_C=False, trained_adapter=None, status='complete')
        save_json(path.with_suffix('.json'), record); save_json(marker, record)
        print(json.dumps(record),flush=True)
        del video,frames
        gc.collect();torch.cuda.empty_cache()


def compare(cfg):
    import statistics
    import yaml
    ref_cfg = yaml.safe_load(Path(cfg['reference_config']).read_text())
    ref = json.loads((Path(ref_cfg['output'])/'original/scores.json').read_text())
    current = json.loads((Path(cfg['output'])/'scores.json').read_text())
    old = {r['id']:r for r in ref['cases']}
    new = {r['id']:r for r in current['cases']}
    paired = []
    for cid in cfg['case_ids']:
        a,b = old[cid],new[cid]
        paired.append(dict(id=cid, old_sa=a['sa'],new_sa=b['sa'],old_pc=a['pc'],new_pc=b['pc'],
            joint_delta=int(b['sa']>=4 and b['pc']>=4)-int(a['sa']>=4 and a['pc']>=4)))
    result = dict(scope='Fixed 12-case I2V development pilot; recipe-level control, not an official T2V result',
        count=12, seed=42, original=ref['summary'],
        native_recipe=dict(sa_mean=statistics.mean(r['new_sa'] for r in paired),
            pc_mean=statistics.mean(r['new_pc'] for r in paired),
            joint_pass_count=sum(r['new_sa']>=4 and r['new_pc']>=4 for r in paired)),
        paired=paired, joint_wins=sum(r['joint_delta']>0 for r in paired),
        joint_losses=sum(r['joint_delta']<0 for r in paired),
        generated_first_diagnostics=[new[cid+'_generated_first'] for cid in cfg['generated_first_case_ids']])
    save_json(Path(cfg['output'])/'comparison.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--index',type=int,default=0)
    parser.add_argument('--shards',type=int,default=1)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    cfg,model_cfg,cases,manifest=plan(args.config)
    if not 0 <= args.index < args.shards:
        raise ValueError('Invalid shard')
    if args.check_only:
        print(json.dumps(dict(status='ready',fixed_cases=12,input_diagnostics=2,seed=42,output=cfg['output'])));return
    generate(cfg,model_cfg,cases[args.index::args.shards])
    if args.index == 0:
        expected=[Path(cfg['output'])/'cases'/c['id']/'planned.complete.json' for c in cases]
        deadline=time.monotonic()+3600
        while not all(p.is_file() for p in expected):
            if time.monotonic()>deadline:
                raise RuntimeError('Other shards incomplete; preserve completed videos')
            print(json.dumps(dict(kind='waiting_for_shards',completed=sum(p.is_file() for p in expected),total=len(expected))),flush=True)
            time.sleep(30)
        subprocess.run(['/home/liuzhirui/miniconda3/envs/physvid/bin/python','-I','-B',
            str(ROOT/'tools/evaluate_videophy2.py'),cfg['output']],check=True)
        compare(cfg)


if __name__=='__main__':
    main()
