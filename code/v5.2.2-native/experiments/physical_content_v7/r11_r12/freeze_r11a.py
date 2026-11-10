"""Freeze the revised R11A; leave the already submitted R10A controls untouched."""
from pathlib import Path
from datetime import datetime,timezone
import json
import shutil
import yaml


def main():
    base=Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native')
    root=base/'experiments/physical_content_v7'
    candidate=root/'request_adapt_r11a';diagnostic=root/'r11_r12/r10a_controls16'
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    template=yaml.safe_load((root/'physvid_i2v_599/recovery/20261102T052708Z/r10a.yaml').read_text())
    prepared={}
    for name,work in [('r11a',candidate)]:
        source=work/'snapshots'/stamp;source.mkdir(parents=True)
        for file in work.glob('*.py'):shutil.copy2(file,source/file.name)
        for file in work.glob('*.txt'):shutil.copy2(file,source/file.name)
        shutil.copy2(work/'subset16.json',source/'subset16.json')
        if name=='r11a':
            shutil.copytree(work/'physical_content',source/'physical_content',ignore=shutil.ignore_patterns('__pycache__'))
            (source/'configs').mkdir()
            cfg=yaml.safe_load((work/'configs/physical_content.yaml').read_text())
            cfg['source_snapshot']=str(source)
            (source/'configs/physical_content.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
            command=f'{source}/run_candidate.py --config {source}/configs/physical_content.yaml'
        else:command=f'{source}/run_controls.py'
        cfg=copy_config(template)
        cfg['name']=f'physical-{name}-v2-semantic-execution-own400-fixed40-60-controls16-{stamp}'
        cfg['description']='Revised R11A: native text Q/K/V/O LoRA from update1, physical read/write after native text using frozen base Q; fresh400, original40+20/60, seven16 controls. Supersedes user-canceled67212; retain R10A PC and improve SA/Joint.'
        cfg['entrypoint']=f'bash {base}/tools/runtime_python.sh {command}'
        sub=work/'submissions'/f'{name}_{stamp}.yaml';sub.parent.mkdir(exist_ok=True)
        sub.write_text(yaml.safe_dump(cfg,sort_keys=False))
        context=work/'submission_context';context.mkdir(exist_ok=True)
        record=dict(prepared_utc=datetime.now(timezone.utc).isoformat(),status='prepared_unsubmitted',
            source=str(source),yaml=str(sub),empty_context=str(context),gpu='RTX6000 Blackwell only; guard before foundation loading',
            resource_pool='blk-96g',slots_per_trial=1,checkpoint_step=400,seed=42,gamma=5,
            no_model_downloads=True,automatic_next_version=False)
        if name=='r11a':record.update(config=str(source/'configs/physical_content.yaml'),
            tests=dict(passed=10,scope='CPU installed Wan block/head/RoPE velocity, generation gradients, exact prior initializer/RNG, phase/optimizer/checkpoint ownership and paired control restoration; NOT quality evidence'),
            additional_parameters=23592960,main_cases=40,extended_cases=60,control_cases=16,
            control_modes=['text_adapter_off','parallel_fusion','reasoning_swapped','empty_reasoning','global_train_mean','anchor_only','no_c'],
            training='Fresh official foundations;400 native text/physical updates,272 backbone LoRA updates; same1600 inputs/noise/dropout as R1/R10A',
            supersedes_experiment=67212,supersedes_trial=68599,
            data_update='User chose architecture first; API/caption revision deferred',
            architecture='Full native-text LoRA plus post-text physical execution, without new physical parameters/losses',
            quality_status='Unmeasured; hypotheses and targets are not claimed improvements')
        (work/'prepared.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
        prepared[name]=record
    plan=root/'r11_r12/plan.json';p=json.loads(plan.read_text());p['status']='Revised R11A frozen and validated, ready for submission';p.setdefault('prepared',{}).update(prepared)
    plan.write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(prepared,ensure_ascii=False,indent=2))


def copy_config(value):
    import copy
    return copy.deepcopy(value)


if __name__=='__main__':main()
