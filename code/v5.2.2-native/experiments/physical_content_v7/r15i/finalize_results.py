"""Read the completed final I evaluations, validate provenance, and write the Chinese report.

No generation, model loading, scoring, job submission, or Goal status change.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import statistics

ROOT = Path(__file__).resolve().parent
CONTROL_NAMES = ['no_c', 'direct_read_only', 'request_bound_only', 'native_physical_query',
                 'reasoning_swapped', 'empty_reasoning', 'global_train_mean']


def read(path):
    return json.loads(Path(path).read_text())


def summarize(rows):
    n = len(rows)
    return dict(count=n, SA=statistics.mean(r['sa'] for r in rows), PC=statistics.mean(r['pc'] for r in rows),
                SA_rate=sum(r['sa'] >= 4 for r in rows)/n, PC_rate=sum(r['pc'] >= 4 for r in rows)/n,
                Joint=sum(r['sa'] >= 4 and r['pc'] >= 4 for r in rows),
                Joint_rate=sum(r['sa'] >= 4 and r['pc'] >= 4 for r in rows)/n)


def verify_suite(path, expected_ids, checkpoint, source, mode, reference):
    manifest = read(path / 'manifest.json')
    rows = read(path / 'scores.json')['cases']
    expected_ids = set(expected_ids)
    if (len(rows) != len(expected_ids) or {r['id'] for r in rows} != expected_ids
            or len(manifest['cases']) != len(expected_ids)
            or {c['id'] for c in manifest['cases']} != expected_ids):
        raise ValueError(f'Incomplete or duplicated cases: {path}')
    required = dict(checkpoint=str(checkpoint), checkpoint_step=400, training_source=str(source), mode=mode)
    if any(manifest.get(k) != value for k, value in required.items()):
        raise ValueError(f'Wrong suite provenance: {path}')
    if manifest['evaluation'] != reference['evaluation']:
        raise ValueError('Different evaluator protocol')
    refs = {c['id']: c for c in reference['cases']}
    by_id = {r['id']: r for r in rows}
    for case in manifest['cases']:
        case_id = case['id']
        row = by_id[case_id]
        for field in ('caption', 'prepared_inputs', 'initial_noise', 'generation_seed'):
            if case.get(field) != refs[case_id].get(field):
                raise ValueError(f'Changed input {case_id}/{field}')
        if row['caption'] != case['caption'] or row.get('generation_seed') != 42:
            raise ValueError('Score caption or seed mismatch')
        video = path / 'cases' / case_id / 'planned.mp4'
        metadata = read(video.with_suffix('.json'))
        expected = dict(checkpoint=str(checkpoint), step=400, mode=mode, case=case_id,
                        seed=42, inference_source=str(source), caption=case['caption'])
        if any(metadata.get(k) != value for k, value in expected.items()):
            raise ValueError(f'Wrong generated-video provenance: {video}')
        if (metadata['sampler']['steps'] != 50 or metadata['sampler']['shift'] != 5
                or metadata['guidance']['text_scale'] != 5
                or metadata['guidance']['physical_scale'] != 5):
            raise ValueError('Changed sampling or guidance')
        if not video.is_file() or video.stat().st_size == 0:
            raise ValueError(f'Missing video: {video}')
        for metric in ('sa', 'pc'):
            score = read(video.with_suffix(f'.{metric}.json'))
            if (score['case_id'] != case_id or score['task'] != metric
                    or score['score'] != row[metric] or row[metric] not in range(1, 6)):
                raise ValueError('Raw scorer output and aggregate differ')
    return rows


def paired(normal, control):
    previous = {r['id']: r for r in control}
    after = {r['id']: r for r in normal}
    result = {}
    for metric in ('sa', 'pc'):
        differences = [after[k][metric] - previous[k][metric] for k in previous]
        result[metric] = dict(normal_minus_control=statistics.mean(differences),
                             wins=sum(d > 0 for d in differences), ties=sum(d == 0 for d in differences),
                             losses=sum(d < 0 for d in differences))
    success = lambda r: r['sa'] >= 4 and r['pc'] >= 4
    result['joint'] = dict(gains=[k for k in previous if success(after[k]) and not success(previous[k])],
                           losses=[k for k in previous if not success(after[k]) and success(previous[k])])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    run = ROOT / 'runs' / args.run_id
    checkpoint = run / 'step0400.pt'
    source = Path(read(run / 'config.json')['source_snapshot'])
    output = ROOT / 'inference_outputs' / args.run_id / 'step0400_original_fixed40_controls16_seed42'
    reference = read(ROOT.parent / 'videophy60/protocol.json')
    ids16 = {c['id'] for c in read(ROOT / 'subset16.json')['cases']}
    rows40 = verify_suite(output / 'videophy40', [f'VP{i:03d}' for i in range(1, 41)], checkpoint, source, 'real', reference)
    rows20 = verify_suite(output / 'videophy60', [f'VP{i:03d}' for i in range(41, 61)], checkpoint, source, 'real', reference)
    rows60 = verify_suite(output / 'full60/videophy60', [f'VP{i:03d}' for i in range(1, 61)], checkpoint, source, 'real', reference)
    if {r['id']: r for r in rows60} != {r['id']: r for r in rows40 + rows20}:
        raise ValueError('Full60 differs from the disjoint40+20 results')
    normal16 = [r for r in rows40 if r['id'] in ids16]
    controls = {}
    for mode in CONTROL_NAMES:
        rows = verify_suite(output / 'controls' / mode / 'videophy40', ids16, checkpoint, source, mode, reference)
        controls[mode] = dict(summary=summarize(rows), paired=paired(normal16, rows))
    report = dict(utc=datetime.now(timezone.utc).isoformat(), candidate='R15I fresh400',
                  checkpoint=str(checkpoint), frozen_source=str(source), output=str(output),
                  main40=summarize(rows40), extra20=summarize(rows20), full60=summarize(rows60),
                  normal16=summarize(normal16), remaining24=summarize([r for r in rows40 if r['id'] not in ids16]),
                  controls=controls, training=read(ROOT / 'reports/completed_training_match.json'),
                  actual_video_review=read(ROOT / 'reports/actual_video_review.json'),
                  module_conclusions=read(ROOT / 'reports/module_conclusions.json'),
                  control_visual_review=read(ROOT / 'reports/control_visual_review.json'),
                  paired_reference_review=read(ROOT / 'reports/main40_review.json'),
                  joint_threshold_analysis=read(ROOT / 'reports/joint_bottleneck40.json'),
                  formal_I_cross_benchmarks='Deferred by user; not counted as passed',
                  distinct_generated_videos=172, distinct_raw_metric_records=344,
                  all_required_videophy_outputs_verified=True, quality_goal_achieved=False)
    hard = report['main40']
    report['hard_targets'] = dict(PC=hard['PC'] >= 3.8, SA=hard['SA'] >= 3.125, Joint=hard['Joint'] >= 12)
    (ROOT / 'reports/final_review.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    lines = ['R15I 最终训练与 VideoPhy2 评测', '',
             '本报告对应从官方基础模型重新训练400步的R15I，区别于冻结B400权重的联合读取评测。', '',
             '| 范围 | SA /5 | PC /5 | Joint | SA通过率 | PC通过率 | Joint通过率 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name in ['main40', 'extra20', 'full60', 'normal16', 'remaining24']:
        s = report[name]
        lines.append(f"| {name} | {s['SA']:.4f} | {s['PC']:.4f} | {s['Joint']}/{s['count']} | {s['SA_rate']:.4f} | {s['PC_rate']:.4f} | {s['Joint_rate']:.4f} |")
    lines += ['', '40例目标：PC≥3.8、SA≥3.125、Joint≥12/40。结果：' + json.dumps(report['hard_targets'], ensure_ascii=False),
              '0.xxxx为SA/PC各自≥4或同一视频同时≥4的通过率；不是5分均值除以5。60例由原40与不重叠20组成，不替代40例目标。', '',
              '| 同一I400的固定16例 | SA /5 | PC /5 | Joint | 正常−对照SA | 正常−对照PC |',
              '|---|---:|---:|---:|---:|---:|']
    baseline = report['normal16']
    lines.append(f"| 正常联合读取 | {baseline['SA']:.4f} | {baseline['PC']:.4f} | {baseline['Joint']}/16 | — | — |")
    for name, row in controls.items():
        s = row['summary']; comparison = row['paired']
        lines.append(f"| {name} | {s['SA']:.4f} | {s['PC']:.4f} | {s['Joint']}/16 | {comparison['sa']['normal_minus_control']:+.4f} | {comparison['pc']['normal_minus_control']:+.4f} |")
    lines += ['', '全部对照固定同一训练权重、输入与seed42，改变两路CFG中的指定读取／条件。它们不是分别重训的结构消融；正常减对照的正值表示正常分支更高。无关／空reasoning仍保留该样本自身的首帧及请求输入。', '']
    for finding in report['module_conclusions']['findings']:
        lines += [finding, '']
    lines += [
              '训练记录：400次更新、1600个样本；128步接口训练后272步联合训练。数据顺序、噪声级别、条件丢弃和学习率安排与B一致。FM和高误差项是单样本单噪声步损失，不作为生成质量证明。', '',
              'Joint为何没有提高：I、B原版和B400联合读取的SA通过样例完全相同，均为11例；Joint也完全相同，均为10例。I提高了一些SA均分，但没有新增跨过SA≥4的样例；PC通过数却从B的23下降到17。即使当前11个SA通过样例全部物理通过，也不足以达到12/40，必须同时增加真正语义通过的样例。', '',
              '实际画面：' + report['actual_video_review']['overall'], '',
              '相对历史与B400只读结果的逐例变化见 [40例复盘](main40_review.json)，视频对照见 [实际视频记录](actual_video_review.json) 和 [控制变量视频记录](control_visual_review.json)。', '',
              '所有输入、源代码、权重、视频及原始分数保留。完整逐例来源验证与配对结果见 [最终原始报告](final_review.json)。正式I未运行WMB/Physics-IQ，不能声称跨基准目标已达到。', '',
              '本报告完成不代表目标成功；最终结论须同时依据数值、实际过程及跨基准证据。']
    (ROOT / 'reports/final_report.zh.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({k: report[k] for k in ['main40', 'full60', 'hard_targets', 'all_required_videophy_outputs_verified']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
