"""Report the frozen B400 inference change separately from fresh-trained R15I."""
from pathlib import Path
from collections import Counter
import json
import math
import statistics

ROOT = Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_v7')
EVAL = ROOT / 'b400_joint_read_eval'
CROSS = ROOT / 'cross_benchmark_70_60/outputs'


def read(path):
    return json.loads(path.read_text()) if path.is_file() else None


def scores(rows):
    n = len(rows)
    joint = sum(r['sa'] >= 4 and r['pc'] >= 4 for r in rows)
    return dict(count=n, sa_mean=statistics.mean(r['sa'] for r in rows),
                pc_mean=statistics.mean(r['pc'] for r in rows), joint_count=joint,
                sa_pass_rate=sum(r['sa'] >= 4 for r in rows)/n,
                pc_pass_rate=sum(r['pc'] >= 4 for r in rows)/n, joint_pass_rate=joint/n)


def paired(rows, reference):
    refs = {r['id']: r for r in reference}
    for row in rows:
        previous = refs[row['id']]
        if (row['caption'] != previous['caption']
                or row.get('generation_seed', 42) != previous.get('generation_seed', 42)):
            raise ValueError('Paired caption/seed mismatch')
    result = {}
    for name in ('sa', 'pc'):
        changes = [r[name]-refs[r['id']][name] for r in rows]
        result[name] = dict(mean_change=statistics.mean(changes),
                           wins=sum(x > 0 for x in changes), ties=sum(x == 0 for x in changes),
                           losses=sum(x < 0 for x in changes))
    passed = lambda r: r['sa'] >= 4 and r['pc'] >= 4
    result['joint'] = dict(gains=[r['id'] for r in rows if passed(r) and not passed(refs[r['id']])],
                           losses=[r['id'] for r in rows if not passed(r) and passed(refs[r['id']])])
    return result


def paired_cross(candidate, reference):
    result = {}
    for suite, field in [('wmb70', 'cases'), ('physics_iq60', 'per_view')]:
        if suite not in candidate or suite not in reference:
            continue
        rows = {row['id']: row for row in candidate[suite][field]}
        refs = {row['id']: row for row in reference[suite][field]}
        if rows.keys() != refs.keys():
            raise ValueError('Different cross-benchmark subsets')
        metrics = {'Verified': lambda row: row['verified_score']} if suite == 'physics_iq60' else {
            'instruction': lambda row: row['instruction'],
            'physical': lambda row: sum(row['physical_laws']),
            'common_sense': lambda row: sum(row['common_sense']),
            'total': lambda row: row['instruction'] + sum(row['physical_laws']) + sum(row['common_sense'])}
        result[suite] = {}
        for name, value in metrics.items():
            differences = {key: value(rows[key]) - value(refs[key]) for key in rows}
            result[suite][name] = dict(mean_change=statistics.mean(differences.values()),
                wins=sum(d > 1e-12 for d in differences.values()),
                ties=sum(abs(d) <= 1e-12 for d in differences.values()),
                losses=sum(d < -1e-12 for d in differences.values()), by_case=differences)
    return result


def verify_completed_cross(candidate):
    if not all(key in candidate for key in ['wmb70', 'physics_iq60']):
        return {'complete': False}
    protocol = read(EVAL / 'snapshots/20261108T130636Z/cross/protocol.json')
    out = CROSS / 'r15b_joint_i/seed42'
    reference = CROSS / 'r1/seed42'
    for case in protocol['cases']:
        local = out / 'cases' / case['id']
        metadata = read(local / 'video.json')
        other = read(reference / 'cases' / case['id'] / 'video.json')
        if metadata is None or other is None or not (local / 'video.mp4').is_file():
            raise ValueError('Missing cross-benchmark video or provenance')
        for key in ['case', 'caption', 'seed', 'noise_seed', 'first_image', 'original_image', 'output_frames', 'sampling']:
            if metadata[key] != other[key]:
                raise ValueError(f'Cross-benchmark input/protocol mismatch: {case["id"]}/{key}')
        if (metadata['checkpoint'] != protocol['models']['r15b_joint_i']['checkpoint']
                or metadata['inference_operator'] != 'b400_joint_direct_and_request_memory_v1'
                or metadata['new_training_updates'] != 0):
            raise ValueError('Wrong B400 checkpoint/operator')
    wmb = candidate['wmb70']
    official = read(out / 'worldmodelbench_official.json')
    expected = {case['official_id'] for case in protocol['cases'] if case['benchmark'] == 'worldmodelbench70'}
    if (len(wmb['cases']) != 70 or {row['id'] for row in wmb['cases']} != expected
            or set(official['preds']) != expected
            or any(len(official['accs'][key]) != n for key, n in [('instruction', 70), ('physical_laws', 350), ('common_sense', 140)])
            or sorted(Counter(row['domain'] for row in wmb['cases']).values()) != [10] * 7):
        raise ValueError('Incomplete official WMB70 evidence')
    totals = [row['instruction'] + sum(row['physical_laws']) + sum(row['common_sense']) for row in wmb['cases']]
    if not math.isclose(statistics.mean(totals), wmb['overall']['total_score'], abs_tol=1e-12):
        raise ValueError('WMB case scores differ from aggregate')
    piq = candidate['physics_iq60']
    views = piq['per_view']
    expected = {case['id'] for case in protocol['cases'] if case['benchmark'] == 'physics_iq60'}
    if (len(views) != 60 or {row['id'] for row in views} != expected
            or sorted(Counter(row['scene'] for row in views).values()) != [3] * 20
            or not Path(piq['raw_metrics_csv']).is_file()
            or not math.isclose(statistics.mean(row['verified_score'] for row in views), piq['scores']['final_score_view'], abs_tol=1e-12)):
        raise ValueError('Incomplete Physics-IQ per-view/CSV evidence')
    return dict(complete=True, matched_R1_inputs_and_sampler=130, wmb_domains=7,
                wmb_cases_per_domain=10, physics_iq_scenes=20, physics_iq_views_per_scene=3,
                official_raw_scores_and_aggregates_agree=True)


def main():
    report = dict(model='B400 + I proposed joint physical read', new_training_updates=0,
                  not_trained_R15I=True, videophy={}, paired={}, cross={},
                  scoring='1-to5 means; SA/PC >=4 pass rates and same-video joint rate, not mean/5.',
                  actual_video_review='Pending expanded-suite inspection; prior16 visual limitations remain.',
                  goal_achieved=False)
    references = read(ROOT / 'reports/r10_final_evidence.json')['source_paths']
    references.update({'R15B/original40': str(ROOT / 'request_bound_fusion_r15b/inference_outputs/physical_content_67829/step0400_original_fixed40_controls16_seed42/videophy40/scores.json'),
                       'R15B/full60': str(ROOT / 'request_bound_fusion_r15b/inference_outputs/physical_content_67829/step0400_original_fixed40_controls16_seed42/full60/videophy60/scores.json')})
    for name, folder, ids in [('original40', 'videophy40', range(1, 41)),
                              ('extra20', 'videophy60', range(41, 61)),
                              ('full60', 'full60/videophy60', range(1, 61))]:
        data = read(EVAL / 'videophy_outputs' / folder / 'scores.json')
        if data is None:
            continue
        rows = data['cases']
        if len(rows) != len(ids) or {r['id'] for r in rows} != {f'VP{i:03d}' for i in ids}:
            raise ValueError('Require a complete declared suite')
        report['videophy'][name] = scores(rows)
        for label in ['native400', 'R1', 'R10-A', 'R15B']:
            path = references.get(label + '/' + name)
            previous = read(Path(path)) if path else None
            if previous:
                report['paired'][name + '/' + label] = paired(rows, previous['cases'])
        if name == 'original40':
            selected = {r['id'] for r in read(ROOT / 'r15i_design/subset16.json')['cases']}
            report['videophy']['reused_fixed16'] = scores([r for r in rows if r['id'] in selected])
            report['videophy']['remaining24'] = scores([r for r in rows if r['id'] not in selected])
    for model in ['r15b_joint_i', 'wan_lora', 'r1', 'r10a', 'r15b']:
        result = {}
        for key, filename in [('wmb70', 'worldmodelbench_summary.json'), ('physics_iq60', 'physics_iq_summary.json')]:
            data = read(CROSS / model / 'seed42' / filename)
            if data:
                result[key] = data
        if result:
            report['cross'][model] = result
    main40 = report['videophy'].get('original40')
    report['targets40'] = None if main40 is None else dict(PC=main40['pc_mean'] >= 3.8,
        SA=main40['sa_mean'] >= 3.125, Joint=main40['joint_count'] >= 12)
    visual = read(EVAL / 'reports/actual_video_review.json')
    if visual:
        report['actual_video_review'] = visual
    candidate = report['cross'].get('r15b_joint_i', {})
    reference = report['cross'].get('r1', {})
    report['cross_provenance'] = verify_completed_cross(candidate)
    report['cross_paired_references'] = {name: paired_cross(candidate, value)
        for name, value in report['cross'].items() if name != 'r15b_joint_i'}
    report['cross_acceptance'] = {}
    report['wmb_domain_differences'] = {}
    report['physics_category_differences'] = {}
    if 'wmb70' in candidate and 'wmb70' in reference:
        report['cross_acceptance']['wmb_strictly_above_R1'] = candidate['wmb70']['overall']['total_score'] > reference['wmb70']['overall']['total_score']
        for name, row in candidate['wmb70']['domains'].items():
            other = reference['wmb70']['domains'][name]
            report['wmb_domain_differences'][name] = {key: row[key] - other[key]
                for key in ['instruction_mean', 'physical_sum', 'common_sense_sum', 'total_score']}
    if 'physics_iq60' in candidate and 'physics_iq60' in reference:
        report['cross_acceptance']['physics_iq_strictly_above_R1'] = candidate['physics_iq60']['scores']['final_score_view'] > reference['physics_iq60']['scores']['final_score_view']
        for name, row in candidate['physics_iq60']['categories'].items():
            other = reference['physics_iq60']['categories'][name]
            report['physics_category_differences'][name] = dict(Verified=row['final_score_view'],
                R1=other['final_score_view'], difference=row['final_score_view'] - other['final_score_view'])
    report['conclusions'] = [
        '本次是冻结B400的推理读取替换，保留B原有物理Q和全部权重；正式I从官方基础重新训练并学习独立物理Q，两者不能混称。',
        'VideoPhy2-40相对B原版：PC提高0.075（4胜35平1负），SA降低0.025（2胜35平3负），Joint仍为相同10例。联合读取有少量PC收益，没有实现SA和联合成功的提升。',
        '固定16例PC3.9375没有延续为完整40例同样水平；其余24例PC3.6667、SA2.7500。完整60例PC3.8167，但SA2.8167、Joint12/60，不可拿60例PC替代40例硬目标。']
    if report['cross_provenance']['complete']:
        report['conclusions'] += [
            'WMB8.600高于R1的8.5714，但逐例总分13胜44平13负；指令均分2.0429低于R1的2.1000。工业、自动驾驶、机器人领域总分分别低0.2、0.3、0.2。',
            'Physics-IQ Verified0.269586高于R1的0.265103，差0.004483；60视角27胜33负，20场景均值10胜10负。光学0.030265明显低于R1的0.094849，三个光学场景均值全部下降；其他四类别均值提高。',
            '两个跨基准总分的点估计都超过R1，这是保留下来的收益；但SA/Joint、分领域退化和真实过程缺失仍未解决，整体目标未达成。未运行B原始读取的同批跨基准结果，不能把对R1的全部差值单独归因于联合读取。']
    out = EVAL / 'reports'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    lines = ['B400＋联合物理读取：独立推理评测', '',
             '固定 R15B 的400步权重，不更新参数；仅使用已完成16例试验的直接C＋请求关联C联合读取。此结果不属于正式训练的R15I。', '',
             '复用原16条视频和原始评分，其余44条新生成；40＋不重叠20构成60。统一输入、seed42、CFG5、Euler50/shift5。0.xxxx表示阈值通过率，不是5分均值除以5。', '',
             '| 范围 | SA /5 | PC /5 | Joint | SA通过率 | PC通过率 | Joint通过率 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name, value in report['videophy'].items():
        lines.append(f"| {name} | {value['sa_mean']:.4f} | {value['pc_mean']:.4f} | {value['joint_count']}/{value['count']} | {value['sa_pass_rate']:.4f} | {value['pc_pass_rate']:.4f} | {value['joint_pass_rate']:.4f} |")
    lines += ['', 'WMB为同一7领域各10例，Physics-IQ为同一20场景全部3视角。保留官方评分及分领域结果。', '',
              '| 模型 | WMB /10 | Physics-IQ Verified |', '|---|---:|---:|']
    for name, value in report['cross'].items():
        wmb = value.get('wmb70', {}).get('overall', {}).get('total_score')
        piq = value.get('physics_iq60', {}).get('scores', {}).get('final_score_view')
        lines.append(f"| {name} | {f'{wmb:.4f}' if wmb is not None else '待完成'} | {f'{piq:.6f}' if piq is not None else '待完成'} |")
    if report['cross_provenance']['complete']:
        lines += ['', '130条视频已逐例核对与R1一致的首帧、提示词、噪声种子和采样参数；WMB原始70条评分、Physics-IQ官方CSV及60视角结果与汇总一致。', '',
                  '相对R1的同例胜／平／负：', '', '| 指标 | 胜 | 平 | 负 | 均值差 |', '|---|---:|---:|---:|---:|']
        for suite, metrics in report['cross_paired_references'].get('r1', {}).items():
            for name, row in metrics.items():
                lines.append(f"| {suite}/{name} | {row['wins']} | {row['ties']} | {row['losses']} | {row['mean_change']:+.6f} |")
    if report['wmb_domain_differences']:
        lines += ['', 'WMB各领域相对R1的变化：', '', '| 领域 | 指令Δ | 物理Δ | 常识Δ | 总分Δ |', '|---|---:|---:|---:|---:|']
        for name, row in report['wmb_domain_differences'].items():
            lines.append(f"| {name} | {row['instruction_mean']:+.4f} | {row['physical_sum']:+.4f} | {row['common_sense_sum']:+.4f} | {row['total_score']:+.4f} |")
    if report['physics_category_differences']:
        lines += ['', 'Physics-IQ各类别：', '', '| 类别 | 联合读取 Verified | R1 | 差值 |', '|---|---:|---:|---:|']
        for name, row in report['physics_category_differences'].items():
            lines.append(f"| {name} | {row['Verified']:.6f} | {row['R1']:.6f} | {row['difference']:+.6f} |")
    if visual:
        lines += ['', '实际视频：' + visual['overall'], '', '具体配对视频观察及局限见 [视频记录](actual_video_review.json)。']
    for conclusion in report['conclusions']:
        lines += ['', conclusion]
    lines += ['', '40例及跨基准目标逐项判断见 [原始报告](results.json)。当前不宣称达成综合目标。']
    (out / 'results.zh.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(videophy=report['videophy'], cross_completed=list(report['cross'].get('r15b_joint_i', {})), targets40=report['targets40']), ensure_ascii=False))


if __name__ == '__main__':
    main()
