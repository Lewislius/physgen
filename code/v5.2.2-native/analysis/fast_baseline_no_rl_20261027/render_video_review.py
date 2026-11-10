"""Build a local video review page from completed experiments; no new scoring."""
import html
import json
import os
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
PILOT = ROOT/'inference_outputs/native_physics_prompt_baseline12_seed42_20261027_v2'
NATIVE = ROOT/'inference_outputs/native_wan_official_recipe12_seed42_20261027'
GENERATED = ROOT/'inference_outputs/generated_first_frame_diagnostic_seed42_20261027'


def local(path):
    return html.escape(os.path.relpath(path, OUT), quote=True)


def panel(title, folder, poster):
    video = folder/'planned.mp4'
    if not video.with_suffix('.complete.json').exists():
        return f'<article><h3>{html.escape(title)}</h3><p>待完成</p></article>'
    scores = []
    for task in ['sa', 'pc']:
        path = folder/f'planned.{task}.json'
        scores.append(json.loads(path.read_text())['score'] if path.exists() else None)
    label = f'SA {scores[0]} / PC {scores[1]}' if None not in scores else '待评分'
    if None not in scores:
        label += ' · 联合通过' if min(scores)>=4 else ' · 未联合通过'
    return (f'<article><h3>{html.escape(title)}</h3><p>{label}</p>'
            f'<video controls muted playsinline preload="none" poster="{local(poster)}" src="{local(video)}"></video>'
            f'<p><a href="{local(video)}">打开视频</a></p></article>')


def main():
    manifest = json.loads((PILOT/'original/manifest.json').read_text())
    source = Path(manifest['source'])
    body = ['<h1>冻结 Wan：推理对照</h1>',
        '<p>固定 12 题，视频 seed=42、保存的初始噪声一致。832×480、81 帧、24 FPS、50 步、CFG=5。'
        '这是 VideoPhy2 文本改成 I2V 的开发对照，不是官方 T2V 结果，也不证明 H/Q 的 C 有效。</p>',
        '<p>SA：语义符合度；PC：物理常识；两项均 ≥4 才联合通过。请同时检查动作是否完成、材料是否变形。</p>']
    for case in manifest['cases']:
        cid = case['id']
        body.append(f'<section><h2>{cid}</h2><p>{html.escape(case["caption"])}</p>'
                    '<button onclick="sync(this)">三组从头同时播放</button><div class="grid">')
        poster = source/'cases'/cid/'first.png'
        for title, base in [('项目 Euler＋原提示词', PILOT/'original'),
                            ('项目 Euler＋Qwen 扩写', PILOT/'physics_text'),
                            ('Wan 官方 I2V 流程＋原提示词', NATIVE)]:
            body.append(panel(title, base/'cases'/cid, poster))
        body.append('</div></section>')
    body.append('<h2>生成首帧的定向诊断</h2><p>以下两题因首帧明显不贴题而选出，不能代表总体提升。原题文本和保存的视频噪声保持一致。</p>')
    for cid in ['VP001', 'VP006']:
        body.append(f'<section><h2>{cid}：新首帧</h2><button onclick="sync(this)">从头同时播放</button><div class="grid">')
        poster = OUT/'generated_first_frames'/f'{cid}.png'
        for title, folder in [('项目 Euler＋原提示词', GENERATED/'original/cases'/cid),
                              ('项目 Euler＋Qwen 扩写', GENERATED/'physics_text/cases'/cid),
                              ('Wan 官方 I2V 流程＋原提示词', NATIVE/'cases'/(cid+'_generated_first'))]:
            body.append(panel(title, folder, poster))
        body.append('</div></section>')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>冻结 Wan 推理对照</title><style>body{font:16px/1.6 system-ui,sans-serif;margin:24px;background:#f6f7f9;color:#17202a}h1,h2,h3{line-height:1.3}h3{font-size:16px}section{margin:24px 0;padding:18px;background:white;border:1px solid #dfe3e8;border-radius:8px}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}video{width:100%;aspect-ratio:832/480;background:#111}button{padding:8px 12px;margin:4px 0 12px;cursor:pointer}a{color:#1261a0}@media(max-width:850px){.grid{grid-template-columns:1fr}}</style><body>'''
    page += '\n'.join(body)
    page += '''<script>function sync(button){for(const v of button.closest('section').querySelectorAll('video')){v.currentTime=0;v.play().catch(()=>{});}}</script></body></html>'''
    (OUT/'video_review.html').write_text(page)
    print(OUT/'video_review.html')


if __name__ == '__main__':
    main()
