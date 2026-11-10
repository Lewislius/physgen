"""Report the frozen B400 inference change separately from fresh-trained R15I."""
from pathlib import Path
import json
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
    lines += ['', '实际视频和物理过程仍须结合完成后的视频审查，当前不宣称达成综合目标。']
    (out / 'results.zh.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(videophy=report['videophy'], cross_completed=list(report['cross'].get('r15b_joint_i', {})), targets40=report['targets40']), ensure_ascii=False))


if __name__ == '__main__':
    main()
