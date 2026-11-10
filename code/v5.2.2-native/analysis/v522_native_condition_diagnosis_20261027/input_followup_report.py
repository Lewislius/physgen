"""Report targeted first-image replacement separately from the four-case audit."""
import csv
import html
import json
import os
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NEW = ROOT/'inference_outputs'/'v522_native_C_use2_generated_first_seed42_20261027'
OLD = ROOT/'inference_outputs'/'v522_native_C_use4_seed42_20261027'
IDS = ['VP001', 'VP006']


def read(path):
    return json.loads(path.read_text())


def main():
    import imageio.v2 as imageio
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows, summaries = [], {}
    for mode in ('H', 'Q'):
        for arm in ('full', 'C_off'):
            for input_name, source in [('original', OLD), ('generated_first', NEW)]:
                selected = []
                for shard, cid in enumerate(IDS):
                    folder = source/mode/f'shard{shard}'/arm
                    score = next(r for r in read(folder/'scores.json')['cases'] if r['id'] == cid)
                    video = folder/'cases'/cid/'planned.mp4'
                    if not video.is_file():
                        raise FileNotFoundError(video)
                    row = dict(mode=mode, arm=arm, inputs=input_name, **score, video=str(video))
                    selected.append(row)
                    rows.append(row)
                summaries[f'{mode}/{arm}/{input_name}'] = dict(count=2,
                    sa=statistics.mean(r['sa'] for r in selected),
                    pc=statistics.mean(r['pc'] for r in selected),
                    joint_pass=sum(r['sa'] >= 4 and r['pc'] >= 4 for r in selected))
    result = dict(status='complete', generated_videos=8, cases=IDS, summary=summaries, records=rows,
        scope='Two previously selected input-mismatch examples; not representative VideoPhy2 results',
        input_change='Only first-image tensor changes. Native text/null text/timestamps/valid coverage and exact noise verified equal; original process clauses retained.',
        model='Unmodified current H/Q final224; seed42 and two trained refinements; LoRA retained in C-off',
        design_candidate_used=False)
    baseline = ROOT/'inference_outputs'/'generated_first_frame_diagnostic_seed42_20261027'/'original'/'scores.json'
    if baseline.exists():
        result['earlier_native_Wan_same_generated_first_reference'] = read(baseline)
    (HERE/'input_followup_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    with (HERE/'input_followup_scores.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    sections = []
    for cid in IDS:
        chosen = [r for r in rows if r['id'] == cid and r['arm'] == 'full']
        fig, axes = plt.subplots(4, 5, figsize=(15, 8.3))
        for ri, row in enumerate(chosen):
            with imageio.get_reader(row['video']) as reader:
                for ci, index in enumerate([0, 20, 40, 60, 80]):
                    axes[ri, ci].imshow(reader.get_data(index))
                    axes[ri, ci].set(xticks=[], yticks=[])
                    if ri == 0: axes[ri, ci].set_title(f'{index/24:.2f}s')
                    if ci == 0: axes[ri, ci].set_ylabel(f'{row["mode"]} {row["inputs"]}\nSA {row["sa"]} / PC {row["pc"]}')
        fig.suptitle(cid+' | '+chosen[0]['caption'], fontsize=10)
        fig.tight_layout()
        fig.savefig(HERE/f'{cid}_input_followup.jpg', dpi=130)
        plt.close(fig)
        videos = ''.join(f'<div><p>{r["mode"]} / {r["inputs"]} / {r["arm"]} · SA {r["sa"]}, PC {r["pc"]}</p><video controls preload="none" src="{html.escape(os.path.relpath(r["video"],HERE))}"></video></div>' for r in rows if r['id'] == cid)
        sections.append(f'<h2>{cid}</h2><p>{html.escape(chosen[0]["caption"])}</p><img src="{cid}_input_followup.jpg"><div class="videos">{videos}</div>')
    table = ''.join(f'<tr><td>{html.escape(k)}</td><td>{v["sa"]:.2f}</td><td>{v["pc"]:.2f}</td><td>{v["joint_pass"]}/2</td></tr>' for k,v in summaries.items())
    page = '''<!doctype html><meta charset="utf-8"><title>H/Q 首帧补测</title><style>body{max-width:1250px;margin:30px auto;font:16px/1.55 system-ui}td,th{border:1px solid #ccc;padding:8px}table{border-collapse:collapse}.videos{display:grid;grid-template-columns:1fr 1fr;gap:16px}img,video{width:100%}</style>
<h1>当前 H/Q 权重：更换首帧后的独立补测</h1><p>复用此前生成的 VP001/VP006 首帧；原文本、要求、时间与同一份 seed42 噪声保持一致。Qwen 根据新首帧重新编码。原设计与模型权重均未修改，新 connector 原型未用于这些视频。</p>
<p>两例是针对输入问题选定的诊断案例，不能据此估计整体 VideoPhy2 提升。<a href="review.html">返回原输入诊断</a> · <a href="input_followup_summary.json">完整数据</a></p>
<table><tr><th>模式 / C / 输入</th><th>SA</th><th>PC</th><th>联合通过</th></tr>'''+table+'</table>'+''.join(sections)
    (HERE/'input_followup.html').write_text(page)
    print(json.dumps(summaries))


if __name__ == '__main__':
    main()
