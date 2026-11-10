"""Close the final R15G evidence bundle; this script never submits or trains."""
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT.parent
OUT = ROOT / 'inference_outputs/physical_content_68103/step0400_original_fixed40_controls16_seed42'
REPORT = ROOT / 'reports'
BENCH = EXPERIMENTS / 'cross_benchmark_70_60'


def read(path):
    return json.loads(path.read_text())


def link(label, path):
    return f'[{label}]({path.resolve()})'


def score40(path):
    rows = read(path)['cases']
    assert len(rows) == 40 and {r['id'] for r in rows} == {f'VP{i:03d}' for i in range(1, 41)}
    return dict(SA=statistics.mean(r['sa'] for r in rows),
        PC=statistics.mean(r['pc'] for r in rows),
        Joint=sum(r['sa'] >= 4 and r['pc'] >= 4 for r in rows), source=str(path.resolve()))


def main():
    result = read(OUT / 'comparison_report.json')
    cross = read(BENCH / 'reports/current_comparison.json')
    wmb = read(REPORT / 'wmb70_review.json')
    piq = read(REPORT / 'physics_iq60_review.json')
    visual = read(REPORT / 'control_visual_review.json')
    assert not result['pending_main'] and len(result['controls']) == 7
    assert result['main']['original40']['count'] == 40 and result['main']['full60']['count'] == 60
    assert all(x['summary']['count'] == 16 for x in result['controls'].values())
    assert wmb['scores']['r15g']['count'] == 70 and piq['per_view_mean_reproduced']
    assert visual['reviewed_control_panels'] == 35
    assert read(ROOT / 'prepared.json')['state'] == 'STATE_COMPLETED'
    assert read(ROOT / 'cross_prepared.json')['status'] == 'STATE_COMPLETED'
    refs = read(EXPERIMENTS / 'reports/r10_final_evidence.json')['source_paths']
    history_paths = {'Wan+LoRA': Path(refs['native400/original40']),
        'R1': Path(refs['R1/original40']), 'R10A': Path(refs['R10-A/original40'])}
    r9 = list((EXPERIMENTS / 'r1_content_reader_r9/inference_outputs/physical_content_66968').glob('*/videophy40/scores.json'))
    if len(r9) == 1:
        history_paths['R9'] = r9[0]
    for name, directory, experiment in [
        ('R15B', 'request_bound_fusion_r15b', 67829),
        ('R15E', 'semantic_addressing_r15e', 67955),
        ('R15F', 'grounded_semantic_fusion_r15f', 68038),
        ('R15G', 'source_residual_fusion_r15g', 68103),
    ]:
        history_paths[name] = (EXPERIMENTS / directory / 'inference_outputs' /
            f'physical_content_{experiment}/step0400_original_fixed40_controls16_seed42/videophy40/scores.json')
    history = {k: score40(v) for k, v in history_paths.items()}
    acceptance = cross['cross_benchmark_numeric_acceptance']['r15g']
    assert not result['primary_numeric_targets_met'] and not acceptance['both_numeric_requirements_met']
    closure = dict(utc=datetime.now(timezone.utc).isoformat(), candidate='R15G',
        work_status='all_declared_evaluations_completed', quality_status='failed_not_accepted',
        goal_achieved=False, last_authorized_model='R15G', further_training_versions_authorized=0,
        pause_requested_after_report=True, main=result['main'], history40=history,
        normal16=result['normal_fixed16'], controls16=result['controls'],
        primary_numeric_targets=result['primary_numeric_targets'], cross_acceptance=acceptance,
        cross_scores={k: dict(WMB70=wmb['scores'][k]['total_score'],
            PhysicsIQ_Verified=piq['scores'][k]) for k in ['wan_lora', 'r1', 'r10a', 'r15g']},
        checkpoint=str(ROOT / 'runs/physical_content_68103/step0400.pt'),
        frozen_source=str(ROOT / 'snapshots/20261107T040316Z'),
        training_experiment=68103, training_trial=69490, trained_steps=400,
        completed_evaluation_experiment=68181, completed_evaluation_trial=69568,
        completed_cross_experiment=68142, completed_cross_trial=69529,
        recovery_added_training_steps=0, new_downloads=0, new_API_calls=0,
        monitoring_next_check_utc=None,
        visual_reviews=[str(REPORT / 'initial_visual_review.json'), str(REPORT / 'control_visual_review.json'),
            str(REPORT / 'cross_visual/review.json'), str(REPORT / 'cross_visual/post_score_thermal_selection.json')])
    (REPORT / 'closure.json').write_text(json.dumps(closure, ensure_ascii=False, indent=2) + '\n')

    lines = [
        '**R15G 最终评测与复盘：未达标**', '',
        'R15G 已完成独立400步训练、固定40/60例、七组固定16例配对消融、WMB-70和Physics-IQ 20场景×3视角。全部硬性量化目标均未满足，实际视频也有明确退化。按授权上限到R15G停止新增训练，并在报告交付后暂停；Goal不能标记为完成。', '',
        '验收标准仍为 VideoPhy2-40：PC≥3.8（优选3.85）、SA≥3.125（继续争取3.3/3.35）、Joint至少12/40，且WMB-70总分、Physics-IQ Verified均严格超过R1，同时改善真实生成。', '',
        '**完整 VideoPhy2 结果**', '',
        '| 模型（400步、同固定40例） | SA /5 | PC /5 | Joint |',
        '|---|---:|---:|---:|',
    ]
    for name in ['Wan+LoRA', 'R1', 'R9', 'R10A', 'R15B', 'R15E', 'R15F', 'R15G']:
        if name in history:
            x = history[name]
            lines.append(f"| {name} | {x['SA']:.4f} | {x['PC']:.4f} | {x['Joint']}/40 |")
    lines += ['', 'R15G补充20例为SA 2.9000、PC 3.4500、Joint 3/20；合并同协议60例为 **SA 3.0000、PC 3.3667、Joint 12/60（20%）**。12/60不能视为满足12/40目标。', '',
        '40例相对R1，PC下降0.3000：0例提高、9例下降、31例持平。60例相对R1仍下降0.3000：2例提高、17例下降、41例持平；没有新增Joint通过例，失去VP034。与Wan+LoRA相比，60例PC也低0.1500。', '',
        '按SA≥4、PC≥4分别通过且Joint要求同一视频两项同时通过，R15G的40例SA/PC/Joint通过率为 **0.2500 / 0.3250 / 0.2250**；60例为 **0.2167 / 0.3833 / 0.2000**。这是固定I2V子集的通过率，不是官方590提示词T2V全量分数。', '',
        '**同权重16例消融**', '',
        '| 设置 | SA | PC | Joint |', '|---|---:|---:|---:|']
    normal = result['normal_fixed16']
    lines.append(f"| 正常 | {normal['sa_mean']:.4f} | {normal['pc_mean']:.4f} | {normal['joint_pass_count']}/16 |")
    labels = {'no_c':'关闭注入', 'execution_prior_only':'仅公共执行项', 'native_physical_query':'替换为原生物理Q',
        'reasoning_swapped':'无关reasoning', 'empty_reasoning':'空reasoning', 'global_train_mean':'固定训练均值C',
        'native_guidance':'原生文本CFG＋一次物理修正'}
    for key, label in labels.items():
        x = result['controls'][key]['summary']
        lines.append(f"| {label} | {x['sa_mean']:.4f} | {x['pc_mean']:.4f} | {x['joint_pass_count']}/16 |")
    lines += ['',
        '正常模型相对关闭注入：SA无胜例、3例下降；PC无胜例、2例下降；Joint少1例。这证明当前训练权重的物理通路在这组配对测试中是净负收益。正确reasoning比无关reasoning的PC高0.1875、比空reasoning高0.0625，但仍未超过关闭注入，不能据此宣称模块整体有效。', '',
        '固定16例中，物理注入的PC净收益从R10A的+0.2500、R15F的+0.3125，变成R15G的−0.1250；三者SA净代价均为−0.1875。G关闭注入的SA/PC/Joint又高于F关闭注入，证据指向物理通路的联合执行退化，而非简单归因于基础LoRA变差。', '',
        '公共执行项对照同时大幅降低了执行幅度，不能当作纯粹的“差异信息有害”证明。均值C、原生Q、引导分解均没有建立满足全部目标的替代方案；原生Q替换的PC均分不变，不能据此否定独立物理Q。各对照保留原始结果，没有逐例择优拼接或改写正常模型分数。', '',
        '**跨基准结果**', '',
        '| 模型 | WMB指令 /3 | 物理 /5 | 常识 /2 | 总分 /10 | Physics-IQ Verified |',
        '|---|---:|---:|---:|---:|---:|']
    for key, label in [('wan_lora','Wan+LoRA'), ('r1','R1'), ('r10a','R10A'), ('r15g','R15G')]:
        x=wmb['scores'][key]
        lines.append(f"| {label} | {x['instruction_mean']:.4f} | {x['physical_sum']:.4f} | {x['common_sense_sum']:.4f} | {x['total_score']:.4f} | {piq['scores'][key]:.6f} |")
    lines += ['',
        'WMB原始总积分R15G为553、R1为600；相对R1，8例提高、26例下降、36例持平，七个领域总分均下降，机器人−1.4、动画−1.3最明显。逐帧质量通过率0.7000（R1为0.8714），时序一致性0.6857（R1为0.8857）。', '',
        'Physics-IQ Verified为0.1700723025，低于R1的0.2651032258；20个场景中4个提高、16个下降，五类均下降。热学0.1529对0.5528降幅最大，蜡烛场景三视角均值0.0388对0.7216。已核对全部60视角及20场景，并从逐视角分数复算总均值；三个视角不当作三个独立场景。', '',
        '**机制与真实视频证据**', '',
        'G保留原生文本Q/K/V/O，使用独立可训练物理Q/K/V/O；末次Q读取真实输入词值，并分开路由C与无原生anchor的执行C。400步采用128步接口训练＋272步联合训练，生成损失仍为0.8平均FM＋0.2多尺度高误差区域项，另有路由词对齐辅助。架构和梯度检查通过，但生成效果失败。', '',
        '固定16输入中，Q更新token余弦从F的约0.99796下降到G的0.81895；G执行C的token余弦约0.87323、跨样本均值余弦约0.68740。这里比较的是明确命名的张量，不能将包含原生anchor的总C与学习修正项混为一谈。表征差异确实改善，PC却下降，说明它没有转化为有效物理执行。', '',
        '无关reasoning会把气球改成彩色球体、改变揉面和滚筒过程，表明条件并未被完全忽略。问题是输入变化未被稳定绑定到原请求的主体、角色和物理过程。刷漆样例中正常G的滚筒形状比部分对照稳定，但涂覆过程仍错误；气球样例多个对照SA=4、PC=5，却主要漂移而非持续膨胀，不能把这一Joint提高直接当作真实改善。', '',
        '跨基准中，G在饮料分配器样例新增另一容器倒水，替换了因果主体；撕纸时夹具脱离并新增人手；镜面动画出现画风和人物身份跳变。蜡烛要求平稳燃烧，G三个视角都新增手、工具或人物，部分还改变/熄灭火焰。稳定物理现象不需要额外动作，更多变化本身不是优化。R1右视角也出现手，不能把参考版本描述成完美。', '',
        '能够确定的是：当前G整体设计不应替换较好的历史模型，物理分支在配对测试中有负贡献，C差异不能充当有效性指标。尚不能将全部退化单独归于某一个投影或独立Q，因为正式版本同时改变了输入值执行与融合初始化；这些机制只能保留为已失败组合的证据，不能宣称已验证优越。', '',
        '**保留内容与停止状态**', '',
        '原始训练Search 68103／Run 69490完成400步；服务器中断后，Search 68181／Run 69568只恢复剩余评测，新增训练0步；跨基准Search 68142／Run 69529已完成。主评测恢复时曾分到H100，被GPU检查在模型评测前拒绝，之后在RTX6000完成。已生成视频和原始评分保留；未下载模型、未新增API调用。', '',
        '保留R1作为综合效果参照，R10A/R15B作为物理收益与代价的参照；R15G不作为成功版本推荐。所有声明评测已完成，授权上限已到，后续训练等待用户重新授权。', '',
        '原始与分析文件：', '',
    ]
    for label,path in [
        ('完整40/60及16例原始评分索引',OUT/'comparison_report.json'),
        ('各版本配对物理贡献',REPORT/'physical_contribution_comparison16.json'),
        ('35组控制视频抽帧审查',REPORT/'control_visual_review.json'),
        ('WMB逐领域与逐例对比',REPORT/'wmb70_review.json'),
        ('Physics-IQ逐场景/视角对比',REPORT/'physics_iq60_review.json'),
        ('跨基准真实视频审查',REPORT/'cross_visual/review.json'),
        ('训练记录',REPORT/'completed_training.json'),
        ('表征审计',REPORT/'representation16.json'),
        ('原400步权重',ROOT/'runs/physical_content_68103/step0400.pt'),
        ('冻结源码',ROOT/'snapshots/20261107T040316Z'),
        ('完整机器可读结论',REPORT/'closure.json')]:
        lines.append('- '+link(label,path))
    (REPORT / 'final_report.zh.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(report=str(REPORT/'final_report.zh.md'), goal_achieved=False,
        history40=list(history), all_declared_evaluations_complete=True),ensure_ascii=False))


if __name__ == '__main__':
    main()
