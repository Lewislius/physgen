"""Preserve the completed R1-400 reference and its reproduction inputs locally."""
from datetime import datetime, timezone
from pathlib import Path
import json
import shutil


BASE = Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native')
HOME_ROOT = Path('/home/liuzhirui')
ROOT = BASE / 'experiments/physical_content_v7'


def main():
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    bundle = ROOT / 'backups' / f'r1_step0400_pc3.625_gamma5_{stamp}'
    bundle.mkdir(parents=True, exist_ok=False)
    copied = {}

    def copy(source):
        source = Path(source)
        if source.is_dir():
            for item in sorted(source.rglob('*')):
                if item.is_file() and '__pycache__' not in item.parts:
                    copy(item)
            return
        if str(source) in copied:
            return
        if not source.is_file():
            raise FileNotFoundError(source)
        target = bundle / 'workspace' / source.relative_to(HOME_ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        original, saved = source.stat(), target.stat()
        if original.st_size != saved.st_size:
            raise RuntimeError(f'Incomplete copy: {source}')
        if (original.st_dev, original.st_ino) == (saved.st_dev, saved.st_ino):
            raise RuntimeError(f'Expected independent file: {source}')
        copied[str(source)] = dict(backup=str(target.relative_to(bundle)), bytes=saved.st_size)

    candidate = ROOT / 'inference_outputs/physical_content_66778/step0400_seed42_r1/videophy40'
    plan = json.loads((candidate / 'manifest.json').read_text())
    scores = json.loads((candidate / 'scores.json').read_text())
    cases = scores['cases']
    assert len(cases) == 40 and sum(x['pc'] for x in cases) / 40 == 3.625
    assert plan['checkpoint_step'] == 400 and plan['sampling']['guidance'] == 5
    assert plan['training_experiment_id'] == '66778'
    print(json.dumps(dict(stage='copy_checkpoints', backup=str(bundle))), flush=True)
    for run in ['physical_content_66778', 'native_lora_66781']:
        folder = ROOT / 'runs' / run
        for name in ['step0400.pt', 'config.json', 'train.jsonl', 'status.json', 'gpu.json']:
            copy(folder / name)
        if (folder / 'videos').is_dir():
            copy(folder / 'videos')
        for output in sorted((ROOT / 'inference_outputs' / run).glob('step0400*')):
            copy(output)

    print(json.dumps(dict(stage='copy_code_and_configuration')), flush=True)
    for source in [
        ROOT / 'snapshots/20261030T141432Z',
        ROOT / 'inference_snapshots/20261030T151158Z',
        ROOT / 'review_snapshots/20261031T042117Z',
        ROOT / 'module_controls16/snapshots/20261031T053400Z',
        BASE / 'experiments/physical_variants/snapshots/20261029T064043Z',
        BASE / 'experiments/physical_scaling_q256/snapshots/20261029T140406Z',
        BASE / 'experiments/physical_scaling_q256/inference_snapshots/20261030T034046Z',
        BASE / 'physgen_native', ROOT / 'train', ROOT / 'inference',
        ROOT / 'recovery_20261031',
        BASE / 'tools/evaluate_videophy2.py', BASE / 'tools/runtime_python.sh',
        HOME_ROOT / 'Project/physGen/code/v5.2/tools/runtime_env.sh',
        HOME_ROOT / 'model/Wan2.2/wan',
    ]:
        copy(source)
    for name in ['AGENTS.md', 'MODULE_EFFECTIVENESS.zh.txt', 'compare_iteration.py',
                 'summarize_physvid_style.py', 'summarize_first_wave.py',
                 'prepare_words.py', 'prepare_native.py', 'fit_interface.py',
                 'prepare_reasoning_controls.py', 'bootstrap.py']:
        copy(ROOT / name)
    copy(BASE / 'AGENTS.md')
    for source in sorted((HOME_ROOT / 'benchmark/videophy/VIDEOPHY2').rglob('*')):
        if source.is_file() and source.suffix in {'.py', '.json', '.yaml', '.yml', '.txt', '.md'}:
            copy(source)
    for source in sorted((ROOT / 'reports').iterdir()):
        if source.is_file() and ('400' in source.name or source.name in {
            'videophy40_physvid_style.json', 'videophy40_physvid_style.csv',
            'first_wave_evidence.json', 'first_wave_evidence.cases.csv', 'module_subset16.json'}):
            copy(source)
    copy(ROOT / 'reports/recovery_20261031/66855')
    copy(BASE / 'analysis/physical_scaling_audit_20261030/REPORT.zh.txt')

    print(json.dumps(dict(stage='copy_saved_inputs_and_interface')), flush=True)
    copy(ROOT / 'cache')
    input_roots = [BASE / path for path in [
        'experiments/physical_scaling_q256/supervision_cache/r1/input',
        'experiments/physical_variants/supervision_cache/r1/input',
        'experiments/physical_v5_scaling_q256/supervision_cache/r1/input',
        'experiments/physical_variants/inference_cache/physical_r1_first_image_request/input',
    ]]
    for case in plan['cases']:
        prepared = Path(case['prepared_inputs'])
        for name in ['conditions.pt', 'noise.pt', 'first.png', 'text.pt', 'v52_geometry.pt', 'input.json']:
            copy(prepared / name)
        copy(case['image'])
        for branch in ['text', 'null']:
            name = f'videophy40_{case["id"]}_{branch}.pt'
            found = next((root / name for root in input_roots if (root / name).is_file()), None)
            if found is None:
                raise FileNotFoundError(name)
            copy(found)
    copy(plan['case_ids_file'])
    reference_manifest = Path(plan['prepared_root']) / 'manifest.json'
    if reference_manifest.is_file():
        copy(reference_manifest)
    for name in ['manifest.json', 'null_text.pt']:
        copy(HOME_ROOT / 'Project/physGen/data_cache/wisa_4000/base' / name)

    metrics = dict(count=40, sa_mean=sum(x['sa'] for x in cases)/40, pc_mean=3.625,
        sa_pass_rate=sum(x['sa'] >= 4 for x in cases)/40,
        pc_pass_rate=sum(x['pc'] >= 4 for x in cases)/40,
        joint_pass_rate=sum(x['sa'] >= 4 and x['pc'] >= 4 for x in cases)/40)
    record = dict(status='completed', created_utc=stamp, backup=str(bundle),
        training_experiment=66778, checkpoint_step=400, generation_recovery_experiment=66855,
        revision='physical_content_v7_r1', gamma=5, seed=42, metrics=metrics,
        checkpoint=str(plan['checkpoint']), training_snapshot=plan['training_source'],
        inference_snapshot=str(ROOT / 'inference_snapshots/20261030T151158Z'),
        baseline_experiment=66781, baseline_step=400,
        copy_method='independent copies; no hardlinks or symlinks',
        bytes=sum(row['bytes'] for row in copied.values()), files=len(copied),
        external_dependencies=dict(
            wan_foundation=str(HOME_ROOT / 'model/Wan2.2/Wan2.2-TI2V-5B'),
            qwen_foundation=str(HOME_ROOT / 'model/Qwen3-VL/Qwen3-VL-8B-Instruct'),
            evaluator_weights=str(HOME_ROOT / 'benchmark/videophy/videophy_2_auto'),
            full_training_data=str(HOME_ROOT / 'Project/physGen/data_cache/wisa_4000'),
            full_training_qwen_evidence=[str(p) for p in input_roots],
            note='Existing local shared models/data remain in place; no model downloads or copies of full foundation weights. Fixed40 evaluation evidence is copied.'),
        inventory=copied)
    (bundle / 'manifest.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    readme = f'''R1-400 / PC 3.625 参考版本备份

这是用户要求保留、供后续消融的原始参考设计。训练实验66778，第400步；完整40例由66855恢复完成。固定gamma=5，seed=42，50步shifted Euler，shift=5。它与R3短训修订是不同版本。

完整VideoPhy2-40：SA={metrics['sa_mean']}，PC=3.625；SA>=4为11/40，PC>=4为18/40，Joint为10/40。配对native LoRA基线实验66781第400步：SA=3.075，PC=3.45。

独立保存了两方400步checkpoint（包含原有训练状态）、训练配置/日志、实际训练与推理源码快照及共享代码、语义接口与词特征缓存、固定40例首图/条件/噪声/Qwen证据、两方生成视频和逐例评分、已经完成的原400步模块对照及对照实现。manifest.json逐文件列出来源与副本路径。所有文件是独立副本。

workspace/保持各文件相对于/home/liuzhirui的原路径。原始配置与checkpoint内部的绝对路径保留，用于明确复现来源；后续消融应从这个版本的代码与配置另建实验目录，保持本备份不变。原训练架构位于workspace/Project/physGen/code/v5.2.2-native/experiments/physical_content_v7/snapshots/20261030T141432Z；原推理入口在inference_snapshots/20261030T151158Z。新增固定gamma5的16例模块对照代码单独存于module_controls16/snapshots/20261031T053400Z，不替代原实现。

完整官方Wan/Qwen/评估模型、已有完整训练视频缓存和全部训练Qwen证据仍复用本机原目录，详见manifest.json的external_dependencies；本备份不是携带全部基础模型和训练数据的可迁移环境。已单独保存固定40例的生成输入与Qwen证据。没有启动训练、下载或重评估。
'''
    (bundle / 'README.zh.txt').write_text(readme)
    pointer = {k:v for k,v in record.items() if k != 'inventory'}
    pointer['readme'] = str(bundle / 'README.zh.txt')
    (ROOT / 'reports/pc3625_reference_backup.json').write_text(json.dumps(pointer, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(pointer, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
