"""Read only existing first-frame/T5 inputs and persisted comparison noise."""
from pathlib import Path
from fractions import Fraction
import copy
import json
import math
import subprocess
from settings import JOBS, Q_REFERENCE, DEMO_REFERENCE, REVISION

INPUT_KEYS = ('first', 'text', 'null_text', 'times', 'valid')
LATENT_SHAPE = (1, 48, 21, 30, 52)


def read(path):
    return json.loads(Path(path).read_text())


def require(ok, message):
    if not ok:
        raise ValueError(message)


def checkpoint_state(path, variant):
    import torch
    state = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    cfg = state['config']
    require(state['revision'] == cfg['revision'] == REVISION, 'Wrong checkpoint revision')
    require(state['variant'] == cfg['variant'] == variant, 'Wrong variant checkpoint')
    require(state['step'] == sum(cfg['stages'].values()) == 224, 'Use the final 224-step weights')
    require(cfg['verifier'] == {'kind': 'none'}, 'Generation must have no verifier')
    require(cfg['model']['rounds'] == 2 and cfg['model']['q_tokens'] == 128, 'Unexpected Q design')
    require(cfg['inference'] == dict(steps=50, shift=5., guidance=5., sampler='shifted_euler'), 'Sampler differs')
    require(cfg['train']['seed'] == 42, 'Unexpected training seed')
    # Do not materialize the large optimizer state for generation.
    return {k: state[k] for k in ('revision', 'variant', 'step', 'config', 'model')}


def restore_parameters(model, state):
    import torch
    require(set(state['model']) == set(model.learned_names), 'Checkpoint parameter ownership differs')
    named = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state['model'].items():
            require(named[name].shape == value.shape, f'Checkpoint shape differs: {name}')
            named[name].copy_(value)
    model.eval().requires_grad_(False)


def manifest_for(variant, suite):
    reference = read((DEMO_REFERENCE if suite == 'demo21' else Q_REFERENCE)/'manifest.json')
    result = copy.deepcopy(reference)
    job = JOBS[variant]
    result.update(source_checkpoint=job['checkpoint'], checkpoint=job['checkpoint'],
        revision=REVISION, variant=variant, training_experiment_id=job['training_experiment_id'],
        training_source=job['training_source'], message_block_calls_per_condition=2,
        conditioning='Fresh frozen-Qwen first-image/request reasoning; two Q enhancement passes; native text retained',
        original_clauses_used=False, physical_input_image='prepared raw first-frame canvas',
        architecture=dict(c_tokens=128, enhancement_width=4096, wan_width=3072,
                          fusion_layers=list(range(3,31,3)), relation_groups=variant=='v6_assoc'))
    return result


def load_inputs(case):
    import torch
    from PIL import Image
    folder = Path(case['prepared_inputs'])
    stored = torch.load(folder/'conditions.pt', map_location='cpu', weights_only=True, mmap=True)
    # Historical extra fields, future latents and annotations are never consumed.
    inputs = {k: stored[k].float().clone() for k in INPUT_KEYS}
    for key, shape in dict(first=(1,48,1,30,52), times=(21,), valid=(30,52)).items():
        require(tuple(inputs[key].shape) == shape, f'{case["id"]}: wrong {key} shape')
    for key in ('text','null_text'):
        t = inputs[key]
        require(t.ndim==3 and t.shape[0]==1 and 0<t.shape[1]<=512 and t.shape[2]==4096,
                f'{case["id"]}: wrong native T5 dimensions')
    require(all(bool(torch.isfinite(t).all()) for t in inputs.values()), 'Nonfinite prepared inputs')
    require(torch.allclose(inputs['times'],torch.arange(21).float()/6,atol=1e-5,rtol=0), 'PTS must match 24 fps')
    info = read(folder/'input.json')
    require(info['prompt'] == case['caption'], 'Prepared prompt differs')
    canvas = Path(case.get('canvas',folder/'canvas.png'))
    with Image.open(canvas) as image:
        require(image.size == (832,480), 'First canvas must be 832x480')
        image.verify()
    return inputs, canvas


def load_noise(case):
    import torch
    if 'initial_noise' not in case:
        return None
    value = torch.load(case['initial_noise'],map_location='cpu',weights_only=True,mmap=True)
    require(tuple(value.shape)==LATENT_SHAPE and value.dtype==torch.float32, 'Saved noise shape/dtype differs')
    require(bool(torch.isfinite(value).all()), 'Nonfinite saved noise')
    return value


def check_inputs():
    for suite, source, count in [('demo21',DEMO_REFERENCE,21),('videophy40',Q_REFERENCE,40)]:
        plan = read(source/'manifest.json')
        expected = ['reference']+[f'P{i:02d}' for i in range(1,21)] if count==21 else [f'VP{i:03d}' for i in range(1,41)]
        require([c['id'] for c in plan['cases']]==expected, 'Wrong case selection/order')
        require(plan['sampling']==dict(steps=50,shift=5.,guidance=5.,condition_strength=1.), 'Changed comparison sampler')
        for case in plan['cases']:
            load_inputs(case)
            load_noise(case)
    return dict(demo_cases=21,videophy_cases=40,fps=24,frames=81,steps=50,shift=5.,guidance=5.)


def verify_video(path):
    value = subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-count_frames',
        '-show_entries','stream=width,height,nb_read_frames,avg_frame_rate','-of','json',str(path)],
        check=True,capture_output=True,text=True)
    stream, = json.loads(value.stdout)['streams']
    require((stream['width'],stream['height'],int(stream['nb_read_frames']))==(832,480,81), 'Invalid generated geometry/length')
    require(math.isclose(float(Fraction(stream['avg_frame_rate'])),24.,rel_tol=1e-6), 'Wrong generated fps')
    return stream

