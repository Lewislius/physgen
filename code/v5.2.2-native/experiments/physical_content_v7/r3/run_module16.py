"""Expand R1-400 module controls to16, reusing the completed compatible four."""
from pathlib import Path
import argparse
import json
import os
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap
from guidance import description
from review_existing400 import subset_reference


def reuse_four(source, destination, checkpoint, mode):
    plan = json.loads((source/'manifest.json').read_text())
    if Path(plan['checkpoint']).resolve() != checkpoint.resolve() or plan['checkpoint_step'] != 400 or plan['mode'] != mode:
        raise ValueError('Existing controls must use the same R1-400 checkpoint and intervention')
    scale = plan['sampling'].get('guidance', plan['sampling'].get('guidance_scale'))
    if scale != 5:
        raise ValueError('Only the original guidance5 controls can be reused')
    for case in plan['cases']:
        origin = source/'cases'/case['id']; target = destination/'cases'/case['id']
        record = json.loads((origin/'planned.json').read_text())
        if record['seed'] != 42 or record['mode'] != mode or record['step'] != 400:
            raise ValueError('Saved control sampling differs')
        target.mkdir(parents=True, exist_ok=True)
        for name in ['planned.mp4', 'planned.sa.json', 'planned.pc.json']:
            if not (target/name).exists():
                os.link(origin/name, target/name)
        record.update(guidance=description(5., 5.), adopted_from=str(origin/'planned.json'))
        if not (target/'planned.json').exists():
            (target/'planned.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--modes', nargs='+', default=['no_c','reasoning_swapped','empty_reasoning','mean','global_train_mean','anchor_only'])
    p.add_argument('--prepare-only', action='store_true')
    args=p.parse_args()
    source=Path(__file__).resolve().parent
    root=bootstrap.BASE/'experiments/physical_content_v7'
    work=root/'module_controls16/r1_step0400_guidance5_seed42'
    checkpoint=root/'runs/physical_content_66778/step0400.pt'
    mean=root/'cache/global_conditions/r1_step0400_train64.pt'
    selected=[x['id'] for x in json.loads((source/'subset16.json').read_text())['cases']]
    allowed={'no_c','reasoning_swapped','empty_reasoning','mean','global_train_mean','anchor_only'}
    if set(args.modes)-allowed:raise ValueError('Unknown module control')
    for run,label in [('physical_content_66778','real'),('native_lora_66781','native400')]:
        subset_reference(root/'inference_outputs'/run/'step0400_seed42_r1/videophy40',work/'references'/label,selected)
    old=root/'inference_outputs/physical_content_66778/step0400_controls_seed42_r1/controls'
    for mode in args.modes:
        if mode!='global_train_mean':
            reuse_four(old/mode/'videophy40',work/'controls'/mode/'videophy40',checkpoint,mode)
    if args.prepare_only:
        print(json.dumps(dict(output=str(work), cases=selected, modes=args.modes,
            reused_per_mode={mode:(0 if mode=='global_train_mean' else 4) for mode in args.modes},generated_new_videos=0)))
        return
    from physical_scaling.common import gpu_guard,atomic_json
    job=work/'reports/jobs'/os.getenv('DET_EXPERIMENT_ID','local');gpu_guard(job)
    if 'global_train_mean' in args.modes and not mean.is_file():
        raise FileNotFoundError('Prepare the training-only fixed C on CPU before submitting this job')
    sys.path.insert(0,str(bootstrap.BASE/'experiments/physical_scaling_q256/inference_snapshots/20261030T034046Z'))
    from pipeline import run_stage
    for mode in args.modes:
        command=[sys.executable,'-I','-B','-u',source/'infer_pipeline.py','--checkpoint',checkpoint,
            '--output',work,'--suite','videophy40','--physical-guidance','5','--cases',*selected,'--modes',mode]
        if mode=='global_train_mean':command+=['--global-condition',mean]
        run_stage(command,job/f'{mode}.log')
        run_stage([sys.executable,'-I','-B',root/'compare_iteration.py',work/'controls'/mode/'videophy40',
            work/'references/real','--output',work/'reports'/f'correct_vs_{mode}.json'],job/f'compare_{mode}.log')
    atomic_json(dict(status='completed',checkpoint=str(checkpoint),cases=selected,modes=args.modes,
        guidance=description(5.,5.),automatic_training=False),job/'status.json')


if __name__=='__main__':main()
