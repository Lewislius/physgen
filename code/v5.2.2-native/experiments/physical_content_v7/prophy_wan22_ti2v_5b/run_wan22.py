"""Run the local Wan2.1 VideoPhy2 protocol with official Wan2.2-TI2V-5B."""

import argparse
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT.with_name('prophy_wan21_t2v_1_3b')
sys.path.insert(0, str(REFERENCE))
from common import case_seed, gpu_guard, read, save
from run import evaluator_environment

MODEL = dict(
    name='wan22_ti2v_5b',
    source='/home/liuzhirui/model/Wan2.2',
    checkpoint='/home/liuzhirui/model/Wan2.2/Wan2.2-TI2V-5B',
    task='t2v',
    precision='bf16',
    project_checkpoint=None,
    prompt_extension=False,
)
METRICS = ('sa_mean', 'pc_mean', 'sa_pass_rate', 'pc_pass_rate',
           'joint_pass_rate', 'joint_pass_count')


def replay_protocol():
    protocol = read(REFERENCE / 'protocol.json')
    protocol['scope'] = (
        'Official Wan2.2-TI2V-5B T2V. Matches the local Wan2.1-T2V-1.3B '
        'ProPhy VideoPhy2 replay: 600 ALL, 180 HARD, same prompts, seeds, '
        'sampling, video encoding, and AutoEval settings.'
    )
    protocol['model'] = MODEL
    return protocol


def generate(protocol, job):
    import torch
    from generate import write_video

    gpu = gpu_guard(job)
    spec = protocol['model']
    sampling = protocol['sampling']
    reference_model = read(REFERENCE / 'protocol.json')['model']
    reference_config = Path(reference_model['source']) / 'wan/configs/shared_config.py'
    negative_prompt = runpy.run_path(str(reference_config))['wan_shared_cfg'].sample_neg_prompt
    output = ROOT / 'outputs' / spec['name']
    manifest = dict(
        model=spec, sampling=sampling, source_prompts=protocol['dataset'],
        scope=protocol['scope'], cases=protocol['cases'],
        protocol_reference=str(REFERENCE / 'protocol.json'),
        negative_prompt_text=negative_prompt,
        offload_model=True,
    )
    save(output / 'manifest.json', manifest)
    pending = []
    for case in protocol['cases']:
        folder = output / 'cases' / case['id']
        seed = case_seed(sampling['seed'], case['index'])
        record = dict(
            case=case['id'], caption=case['caption'], is_hard=case['is_hard'],
            source_case=case['source_id'], seed=seed, model=spec, sampling=sampling,
            task='t2v', first_frame_used=False, scope=protocol['scope'],
        )
        if (folder / 'video.mp4').is_file() and (folder / 'video.json').is_file():
            existing = read(folder / 'video.json')
            if any(existing.get(key) != value for key, value in record.items()):
                raise ValueError(f'Existing video has different inputs: {case["id"]}')
        else:
            pending.append((folder, record))

    total = len(protocol['cases'])
    completed = total - len(pending)
    status_path = output / 'generation_status.json'
    status = dict(requested=total, complete=completed, model=spec['name'], gpu=gpu['actual'])
    if pending:
        save(status_path, dict(status, status='loading_model'))
        print(json.dumps(dict(stage='loading_wan22', pending=len(pending))), flush=True)
        sys.path.insert(0, spec['source'])
        from wan.configs import WAN_CONFIGS
        from wan.textimage2video import WanTI2V

        torch.set_num_threads(4)
        model = WanTI2V(
            WAN_CONFIGS['ti2v-5B'], spec['checkpoint'], device_id=0,
            t5_cpu=False, convert_model_dtype=True,
        )
        size = tuple(sampling['size'])
        expected = (3, sampling['frames'], sampling['height'], sampling['width'])
        for folder, record in pending:
            begin = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            video = model.t2v(
                record['caption'].strip(), size=size, frame_num=sampling['frames'],
                shift=sampling['shift'], sample_solver=sampling['solver'],
                sampling_steps=sampling['steps'], guide_scale=sampling['guidance'],
                n_prompt=negative_prompt, seed=record['seed'], offload_model=True,
            )
            if tuple(video.shape) != expected:
                raise ValueError(f'Unexpected Wan2.2 output shape {tuple(video.shape)}')
            rgb = ((video.float().clamp(-1, 1).permute(1, 2, 3, 0) + 1) * 127.5
                   ).round().byte().cpu().numpy()
            folder.mkdir(parents=True, exist_ok=True)
            write_video(rgb, folder / 'video.mp4', sampling['fps'])
            record['seconds'] = time.monotonic() - begin
            record['peak_gpu_gib'] = torch.cuda.max_memory_allocated() / 2**30
            save(folder / 'video.json', record)
            completed += 1
            status.update(complete=completed, last_case=record['case'])
            save(status_path, dict(status, status='running'))
            print(json.dumps(dict(stage='video_complete', case=record['case'],
                                  complete=completed, seconds=record['seconds'],
                                  peak_gpu_gib=record['peak_gpu_gib'])), flush=True)
            del video, rgb
            torch.cuda.empty_cache()
    save(status_path, dict(status, status='completed'))


def compare():
    baseline = read(REFERENCE / 'summary.json')
    current = read(ROOT / 'summary.json')
    case_fields = ('id', 'caption', 'is_hard', 'source_case')
    before = [tuple(row[key] for key in case_fields) for row in baseline['cases']]
    after = [tuple(row[key] for key in case_fields) for row in current['cases']]
    if before != after or len(after) != 600 or current['hard']['count'] != 180:
        raise ValueError('Expected identical VideoPhy2 ALL and HARD cases')
    result = {}
    for label, key in (('all', 'summary'), ('hard', 'hard')):
        base_metrics, new_metrics = baseline[key], current[key]
        result[label] = dict(
            wan21=base_metrics, wan22=new_metrics,
            wan22_minus_wan21={metric: new_metrics[metric] - base_metrics[metric]
                               for metric in METRICS},
        )
    save(ROOT / 'comparison.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('describe', 'generate', 'evaluate'))
    args = parser.parse_args()
    protocol = replay_protocol()
    if args.stage == 'describe':
        description = {key: value for key, value in protocol.items() if key != 'cases'}
        description['all_cases'] = len(protocol['cases'])
        description['hard_cases'] = sum(str(row['is_hard']) == '1' for row in protocol['cases'])
        print(json.dumps(description, indent=2, ensure_ascii=False))
        return

    job = ROOT / 'jobs' / os.getenv('DET_EXPERIMENT_ID', 'local')
    if args.stage == 'generate':
        generate(protocol, job)
        return
    if args.stage == 'evaluate':
        from evaluate import main as evaluate

        evaluate([], protocol=protocol, root=ROOT)
        compare()
        return

    evaluator = Path('/home/liuzhirui/miniconda3/envs/physvid/bin/python')
    for stage, interpreter, env in (
        ('generate', sys.executable, None),
        ('evaluate', evaluator, evaluator_environment()),
    ):
        save(job / 'status.json', dict(status='running', stage=stage, model=MODEL['name']))
        print(json.dumps(dict(stage=stage, model=MODEL['name'])), flush=True)
        try:
            subprocess.run([interpreter, '-I', '-B', '-u', __file__, '--stage', stage],
                           env=env, check=True)
        except subprocess.CalledProcessError as exc:
            save(job / 'status.json', dict(status='failed', stage=stage,
                                          model=MODEL['name'], exit_code=exc.returncode))
            raise
    save(job / 'status.json', dict(status='completed', model=MODEL['name']))


if __name__ == '__main__':
    main()
