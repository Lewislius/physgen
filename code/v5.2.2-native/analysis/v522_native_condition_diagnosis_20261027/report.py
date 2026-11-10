"""Collect the completed diagnostics and paired generation results."""
from collections import defaultdict
import csv
import html
import json
import os
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GEN = ROOT/'inference_outputs'/'v522_native_C_use4_seed42_20261027'
IDS = ['VP001', 'VP006', 'VP026', 'VP031']


def read(path):
    return json.loads(path.read_text())


def average(rows, keys):
    return {k: statistics.mean(row[k] for row in rows) for k in keys}


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio
    stages = read(HERE/'representation_summary.json')
    response = read(HERE/'native_response'/'results.json')['records']
    scaling = [json.loads(line) for line in (HERE/'Q'/'query_scale'/'metrics.jsonl').read_text().splitlines()]
    scores, details = {}, []
    for mode in ('H', 'Q'):
        for arm in ('full', 'C_off'):
            records = []
            for shard in (0, 1):
                folder = GEN/mode/f'shard{shard}'/arm
                records.extend(read(folder/'scores.json')['cases'])
                if read(folder.parent/'generation_complete.json')['videos'] != 4:
                    raise ValueError('Incomplete generation shard')
                for case in read(folder/'manifest.json')['cases']:
                    row = next(r for r in records if r['id'] == case['id'])
                    details.append(dict(mode=mode, arm=arm, id=case['id'], caption=case['caption'],
                        sa=row['sa'], pc=row['pc'], video=str(folder/'cases'/case['id']/'planned.mp4')))
            if set(r['id'] for r in records) != set(IDS) or len(records) != 4:
                raise ValueError('Missing or duplicate evaluation cases')
            scores[f'{mode}/{arm}'] = dict(count=4, **average(records, ['sa', 'pc']),
                joint_pass=sum(r['sa'] >= 4 and r['pc'] >= 4 for r in records), cases=records)
    comparisons = {}
    for mode in ('H', 'Q'):
        a = {r['id']: r for r in scores[f'{mode}/full']['cases']}
        b = {r['id']: r for r in scores[f'{mode}/C_off']['cases']}
        comparisons[mode] = dict(
            sa_win=sum(a[k]['sa'] > b[k]['sa'] for k in IDS),
            sa_loss=sum(a[k]['sa'] < b[k]['sa'] for k in IDS),
            pc_win=sum(a[k]['pc'] > b[k]['pc'] for k in IDS),
            pc_loss=sum(a[k]['pc'] < b[k]['pc'] for k in IDS),
            joint_win=sum((a[k]['sa'] >= 4 and a[k]['pc'] >= 4) and not (b[k]['sa'] >= 4 and b[k]['pc'] >= 4) for k in IDS),
            joint_loss=sum((b[k]['sa'] >= 4 and b[k]['pc'] >= 4) and not (a[k]['sa'] >= 4 and a[k]['pc'] >= 4) for k in IDS))
    native = {f'{mode}/{text}': average([r for r in response if r['mode'] == mode and r['native_text'] == text],
               ['rms', 'native_response_rms', 'norm_ratio', 'cosine', 'repeat_output_rms'])
              for mode in ('H', 'Q') for text in ('present', 'absent')}
    scale = {name: average([r for r in scaling if r['variant'] == name],
                ['fm', 'delta_vs_original', 'difference_over_family_C_effect'])
             for name in dict.fromkeys(r['variant'] for r in scaling)}
    result = dict(status='completed', generation=scores, paired_C_full_versus_off=comparisons,
        query_scale=scale, native_response=native, representation=stages,
        findings=[
            'Both H and Q already have a learned 4096-to-3072 projector followed by full-width normalization/affine.',
            'Q slot collapse is already present before the projector, starting at its initial grounding read R0.',
            'Frozen-checkpoint query rescaling increases slot spread but did not materially strengthen measured caption/cross-case selectivity.',
            'H retains substantially more token structure and round sensitivity, which does not by itself prove generation benefit.',
            'No arm passed the joint SA/PC threshold on the four original-input diagnostic cases.'],
        interpretation_limits=[
            'Four fixed I2V cases are diagnostic, not a benchmark-wide improvement claim.',
            'Token variance and one-step FM reconstruction do not establish video quality.',
            'Native T5 response is an interface reference, not physical ground truth.',
            'Manual first-image inspection found prompt/setup mismatches in VP026 and VP006; low SA cannot be assigned solely to C.',
            'C-off preserves each checkpoint LoRA but is an inference intervention outside its training recipe.',
            'The connector response loss is an isolated, untrained prototype; it did not produce these videos.'])
    (HERE/'final_summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    with (HERE/'generation_scores.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for mode in ('H', 'Q'):
        rows = stages[mode]['per_round']
        axes[0].plot([r['round'] for r in rows], [r['message_state'] for r in rows], 'o-', label=mode+' before P')
        axes[0].plot([r['round'] for r in rows], [r['final_C'] for r in rows], 's--', label=mode+' final C')
    axes[0].set(yscale='log', xticks=[0, 1, 2], xlabel='Readout round', ylabel='Across-token centered RMS / total RMS', title='Token spread (not video quality)')
    axes[0].legend(fontsize=8)
    axes[1].bar(['H C', 'H native T5', 'Q C', 'Q native T5'],
        [native['H/present']['rms'], native['H/present']['native_response_rms'],
         native['Q/present']['rms'], native['Q/present']['native_response_rms']],
        color=['#3679b7', '#a3bad2', '#b75c36', '#d6b6a3'])
    axes[1].set(ylabel='Denoising-output RMS difference', title='Response to caption change')
    axes[1].tick_params(axis='x', labelrotation=20)
    labels = list(scores)
    axes[2].bar(range(4), [scores[k]['joint_pass'] for k in labels])
    axes[2].set(xticks=range(4), xticklabels=labels, ylim=(0, 4.3), yticks=range(5),
                ylabel='Cases with SA >= 4 and PC >= 4', title='Full generation: joint passes / 4')
    axes[2].tick_params(axis='x', labelrotation=20)
    fig.savefig(HERE/'diagnosis_summary.png', dpi=160)
    plt.close(fig)
    for cid in IDS:
        rows = [r for r in details if r['id'] == cid]
        fig, axes = plt.subplots(4, 5, figsize=(15, 8.3))
        for ri, row in enumerate(rows):
            with imageio.get_reader(row['video']) as reader:
                for ci, index in enumerate([0, 20, 40, 60, 80]):
                    axes[ri, ci].imshow(reader.get_data(index))
                    axes[ri, ci].set(xticks=[], yticks=[])
                    if ri == 0: axes[ri, ci].set_title(f'{index/24:.2f}s')
                    if ci == 0: axes[ri, ci].set_ylabel(f'{row["mode"]} {row["arm"]}\nSA {row["sa"]} / PC {row["pc"]}')
        fig.suptitle(cid+' | '+rows[0]['caption'], fontsize=10)
        fig.tight_layout()
        fig.savefig(HERE/f'{cid}_paired_frames.jpg', dpi=130)
        plt.close(fig)
    rows_html = ''.join(f'<tr><td>{html.escape(k)}</td><td>{v["sa"]:.2f}</td><td>{v["pc"]:.2f}</td><td>{v["joint_pass"]}/4</td></tr>' for k, v in scores.items())
    sections = []
    for cid in IDS:
        rows = [r for r in details if r['id'] == cid]
        videos = ''.join(f'<div><p>{r["mode"]} / {r["arm"]} · SA {r["sa"]}, PC {r["pc"]}</p><video controls preload="none" src="{html.escape(os.path.relpath(r["video"], HERE))}"></video></div>' for r in rows)
        sections.append(f'<h2>{cid}</h2><p>{html.escape(rows[0]["caption"])}</p><div class="videos">{videos}</div><img src="{cid}_paired_frames.jpg">')
    document = '''<!doctype html><meta charset="utf-8"><title>V5.2.2-native C diagnosis</title>
<style>body{max-width:1250px;margin:30px auto;font:16px/1.55 system-ui;background:#fafafa;color:#222}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:8px 20px}.videos{display:grid;grid-template-columns:1fr 1fr;gap:16px}video,img{width:100%}h2{margin-top:40px}</style>
<h1>V5.2.2-native C 诊断与生成对照</h1>
<p>当前 H 66438 / Q 66443，final224；训练与推理 seed42；两轮 refinement，Q 另含 R0。所有对照保留同一 checkpoint 的 Wan LoRA，使用相同首帧、文本和保存的噪声。原模型设计与训练代码未修改。</p>
<p>四个案例仅用于诊断，不能据此宣称完整 VideoPhy2 提升。token 差异与 FM loss 也不是生成质量指标。新 connector loss 尚未接入训练，本页视频均来自原 checkpoint。</p>
<p><strong>机制检查：</strong>H/Q 已有可训练的 4096→3072 projector 与全宽归一化/仿射。Q 在 projector 之前的 R0 已出现 token 趋同；单独将 query 初始 RMS 提到 1 虽增加 token 差异，却未明显增强文本或跨场景响应。H 保留更多 token 结构和轮次影响，但仍须用实际生成验证收益。</p>
<p><strong>输入限制：</strong>VP026 的提示要求两个人拉伸弹力绳，但旧首帧是海上悬挂装置；VP006 的旧首帧未清楚呈现手持可拧的湿画布。C 开关对照仍使用相同输入，但低 SA 不能全部归因于 connector。<a href="input_audit.json">人工检查记录</a> · <a href="input_followup.html">更换首帧后的独立补测</a></p>
<table><tr><th>当前权重 / C</th><th>SA 均值</th><th>PC 均值</th><th>联合通过</th></tr>'''+rows_html+'''</table>
<img src="diagnosis_summary.png"><p><a href="final_summary.json">完整数据</a> · <a href="generation_scores.csv">逐视频得分</a> · <a href="connector_candidate.yaml">独立 connector 候选设计</a></p>'''+''.join(sections)
    document += '''<p>复现：<a href="generation_ablation.sh">SH 入口</a> · <a href="generation_H_0.yaml">H shard0 YAML</a> · <a href="generation_H_1.yaml">H shard1 YAML</a> · <a href="generation_Q_0.yaml">Q shard0 YAML</a> · <a href="generation_Q_1.yaml">Q shard1 YAML</a>。新首帧补测使用同一入口的 --generated-first 参数，配置为 generated_first_H_0/1.yaml 与 generated_first_Q_0/1.yaml。</p>'''
    (HERE/'review.html').write_text(document)
    print(json.dumps(dict(generation=scores, paired=comparisons), ensure_ascii=False))


if __name__ == '__main__':
    main()
