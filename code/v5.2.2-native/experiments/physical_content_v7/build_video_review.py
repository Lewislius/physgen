"""Build a local, synchronized viewer of completed paired generation artifacts."""
from pathlib import Path
from datetime import datetime, timezone
import json
import os

ROOT=Path(__file__).resolve().parent


def main():
    destination=ROOT/'reports/paired_video_review.html'
    def complete_or_subset(full,subset):
        return ROOT/full if (ROOT/full/'scores.json').exists() else ROOT/subset
    locations=[
        ('native400','原生 Wan＋LoRA · 400 步',ROOT/'inference_outputs/native_lora_66781/step0400_seed42_r1/videophy40'),
        ('r1_400','R1 增强 · 400 步 · 保留参照',ROOT/'inference_outputs/physical_content_66778/step0400_seed42_r1/videophy40'),
        ('r1_1000','R1 增强 · 1000 步',ROOT/'inference_outputs/physical_content_66856/step1000_seed42_r1/videophy40'),
        ('r1_off','R1-400 · 关闭注入',ROOT/'module_controls16/r1_step0400_guidance5_seed42/controls/no_c/videophy40'),
        ('r1_anchor','R1-400 · 仅保留输入语义锚点',ROOT/'module_controls16/r1_step0400_guidance5_seed42/controls/anchor_only/videophy40'),
        ('r1_unrelated','R1-400 · 无关 reasoning',ROOT/'module_controls16/r1_step0400_guidance5_seed42/controls/reasoning_swapped/videophy40'),
        ('r1_empty','R1-400 · 空 reasoning',ROOT/'module_controls16/r1_step0400_guidance5_seed42/controls/empty_reasoning/videophy40'),
        ('r1_mean','R1-400 · 本样本 token 均值 C',ROOT/'module_controls16/r1_step0400_guidance5_seed42/controls/mean/videophy40'),
        ('r1_global','R1-400 · 跨训练样本固定均值 C',ROOT/'module_controls16/r1_step0400_guidance5_seed42/controls/global_train_mean/videophy40'),
        ('native64','原生 Wan＋LoRA · 64 步',complete_or_subset('early_execution_r3/inference_outputs/native_lora_66866/step0064_full40_text5_seed42/videophy40','early_execution_r3/inference_outputs/native_lora_66866/step0064_subset16_physical1_seed42/videophy40')),
        ('r3_64','R3 增强 · 64 步',ROOT/'early_execution_r3/inference_outputs/physical_content_66870/step0064_subset16_physical5_seed42/videophy40'),
        ('r3_off','R3 · 同一模型关闭注入',ROOT/'early_execution_r3/inference_outputs/physical_content_66870/step0064_subset16_physical5_seed42/controls/no_c/videophy40'),
        ('r3_unrelated','R3 · 无关 reasoning',ROOT/'early_execution_r3/inference_outputs/physical_content_66870/step0064_subset16_physical5_seed42/controls/reasoning_swapped/videophy40'),
        ('r4_64','R4 增强 · 64 步',complete_or_subset('grounded_read_r4/inference_outputs/physical_content_66871/step0064_full40_physical5_seed42/videophy40','grounded_read_r4/inference_outputs/physical_content_66871/step0064_subset16_physical5_seed42/videophy40')),
        ('r4_anchor','R4 · 仅保留输入语义锚点',ROOT/'grounded_read_r4/inference_outputs/physical_content_66871/step0064_subset16_physical5_seed42/controls/anchor_only/videophy40'),
        ('r4_unrelated','R4 · 无关 reasoning',ROOT/'grounded_read_r4/inference_outputs/physical_content_66871/step0064_subset16_physical5_seed42/controls/reasoning_swapped/videophy40'),
        ('r4_off','R4 · 同一模型关闭注入',ROOT/'grounded_read_r4/inference_outputs/physical_content_66871/step0064_subset16_physical5_seed42/controls/no_c/videophy40'),
        ('r1_request_anchor','R1-400 · 请求直接进入 C 锚点',ROOT/'request_anchor_r1/inference_outputs/r1_step0400_full40_gamma5_seed42/controls/request_anchored/videophy40'),
    ]
    prepared=ROOT/'content_modulation_r6/prepared.json'
    if prepared.exists():
        current=json.loads(prepared.read_text())
        if current.get('successful_native128_run'):
            locations.append(('native128','原生 Wan＋LoRA · 128 步',
                ROOT/'content_modulation_r6/inference_outputs'/current['successful_native128_run']/
                'step0128_full40_text5_seed42/videophy40'))
        if current.get('successful_candidate_run'):
            folder=ROOT/'content_modulation_r6/inference_outputs'/current['successful_candidate_run']/'step0128_full40_controls16_physical5_seed42'
            locations.append(('r6_128','R6 增强 · 128 步',folder/'videophy40'))
            for mode,label in [
                ('mean_query_update','训练均值 Q 更新，保留本样本锚点'),
                ('global_train_mean','跨样本固定均值 C'),
                ('anchor_only','仅保留输入语义锚点'),
                ('reasoning_swapped','无关 reasoning'),
                ('no_c','关闭注入')]:
                locations.append(('r6_'+mode,'R6-128 · '+label,folder/'controls'/mode/'videophy40'))
    for key,directory in [('r7','native_coordinate_r7'),('r8','content_reader_r8')]:
        prepared=ROOT/directory/'prepared.json'
        if prepared.exists():
            current=json.loads(prepared.read_text())
            if current.get('successful_candidate_run'):
                folder=ROOT/directory/'inference_outputs'/current['successful_candidate_run']/'step0128_full40_controls16_physical5_seed42'
                locations.append((key+'_128',key.upper()+' 增强 · 128 步',folder/'videophy40'))
                for mode,label in [('mean_query_update','训练均值 Q 更新'),('global_train_mean','跨样本固定均值 C'),
                    ('anchor_only','仅保留输入语义锚点'),('reasoning_swapped','无关 reasoning'),('no_c','关闭注入')]:
                    locations.append((key+'_'+mode,key.upper()+'-128 · '+label,folder/'controls'/mode/'videophy40'))
    locations.append(('r1_250','R1 增强 · 250 步 · 固定16例',
        ROOT/'training_budget_reference/inference_outputs/r1_step0250_subset16_physical5_seed42/videophy40'))
    labels=dict(reasoning_swapped='无关 reasoning',empty_reasoning='空 reasoning',
        global_train_mean='跨训练样本固定均值 C',shared_only='保留锚点与通用修正，关闭内容修正',
        routing_only='保留学习路由，执行值仅用语义锚点',
        core_only='保留执行核心，关闭额外内容路由修正',
        anchor_only='仅保留输入语义锚点',no_c='关闭注入')
    expanded=[('native400_60','原生 Wan＋LoRA · 400 步 · 完整60例','native400'),
              ('r1_400_60','R1 增强 · 400 步 · 完整60例','R1_400')]
    for key,title,directory in [('r9','R9','r1_content_reader_r9'),('r9_1','R9.1','content_value_r9_1'),('r10','R10-A','execution_value_r10'),('r10b','R10-B','grounded_execution_r10b')]:
        prepared=ROOT/directory/'prepared.json'
        if not prepared.exists():continue
        current=json.loads(prepared.read_text())
        run=current.get('successful_candidate_run')
        if not run:continue
        output=ROOT/directory/'inference_outputs'/run
        folder=output/'step0400_full40_controls16_physical5_seed42'
        locations.append((key+'_250',title+' · 250 步 · 固定16例',output/'step0250_subset16_physical5_seed42/videophy40'))
        locations.append((key+'_400',title+' · 400 步 · 完整40例',folder/'videophy40'))
        for mode in current['control_modes']:
            locations.append((key+'_'+mode,title+'-400 · '+labels[mode],folder/'controls'/mode/'videophy40'))
        expanded.append((key+'_400_60',title+' · 400 步 · 完整60例',
                         {'r9':'R9_','r9_1':'R9_1_','r10':'R10_','r10b':'R10B_'}[key]+run))
    for key,label,model_name in expanded:
        folder=ROOT/'videophy60/inference_outputs'/model_name/'full60_gamma5_seed42/videophy60'
        score_file=folder/'scores.json'
        if score_file.exists() and json.loads(score_file.read_text())['summary']['count']==60:
            locations.append((key,label,folder))
    versions=[];cases={}
    for key,label,folder in locations:
        manifest=folder/'manifest.json'
        if not manifest.exists():continue
        plan=json.loads(manifest.read_text())
        scores=json.loads((folder/'scores.json').read_text()) if (folder/'scores.json').exists() else {}
        scored={x['id']:x for x in scores.get('cases',[])}
        videos={}
        for case in plan['cases']:
            video=folder/'cases'/case['id']/'planned.mp4'
            if not video.exists() or not video.with_suffix('.json').exists():continue
            # Cross-version inputs must agree before the viewer presents a pair.
            if case['id'] in cases:
                for field in ['caption','prepared_inputs','initial_noise','generation_seed']:
                    if cases[case['id']].get(field)!=case.get(field):raise ValueError((key,case['id'],field))
            cases[case['id']]=case
            videos[case['id']]={'src':os.path.relpath(video,destination.parent),'score':scored.get(case['id'])}
        if videos:versions.append(dict(id=key,label=label,videos=videos,summary=scores.get('summary')))
    criteria=json.loads((ROOT/'evaluation/process_criteria16.json').read_text())['cases']
    available={version['id']:version for version in versions}
    preferred=[('r1_400_60','r10b_400_60'),('r1_400','r10b_400'),
               ('r1_400_60','r10_400_60'),('r1_400','r10_400'),('r1_250','r10b_250'),('r1_250','r10_250'),
               ('r1_400_60','r9_1_400_60'),('r1_400','r9_1_400'),
               ('r1_400_60','r9_400_60'),('r1_400','r9_400'),('r1_250','r9_1_250'),('r1_250','r9_250'),
               ('native128','r8_128'),('native128','r7_128'),('native128','r6_128'),('native64','r4_64'),
               ('r1_400','r1_request_anchor'),('native400','r1_400')]
    left,right=next(pair for pair in preferred if all(key in available for key in pair))
    shared=sorted(set(available[left]['videos']) & set(available[right]['videos']))
    data=dict(updated=datetime.now(timezone.utc).isoformat(),versions=versions,
        defaults=dict(left=left,right=right,case=shared[0] if shared else sorted(cases)[0]),
        cases=[dict(id=k,caption=v['caption'],criteria=criteria.get(k,'')) for k,v in sorted(cases.items())])
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>配对生成视频复核</title>
<style>
body{font:16px/1.6 system-ui,sans-serif;background:#101923;color:#edf2f7;margin:0;padding:26px;max-width:1700px;margin:auto}h1{font-size:24px;margin:0}p{margin:8px 0}.muted{color:#b6c5d2;font-size:14px}.toolbar{display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin:20px 0}select,button,input{font:inherit}select,button{padding:7px 12px;background:#203245;color:white;border:1px solid #456078;border-radius:6px}button{cursor:pointer}.pair{display:grid;grid-template-columns:1fr 1fr;gap:18px}.panel{background:#172635;padding:14px;border-radius:10px}video{width:100%;background:#080d12;aspect-ratio:832/480;margin-top:12px}.score{min-height:30px}.caption{padding:16px;background:#203245;border-left:4px solid #65c4b3}.cue{margin-top:10px}.notes{width:100%;min-height:85px;box-sizing:border-box;background:#172635;color:white;font:inherit;padding:10px;border:1px solid #456078}#seek{flex:1;min-width:170px}a{color:#8dd8ed}@media(max-width:900px){body{padding:12px}.pair{grid-template-columns:1fr}}
</style><h1>配对生成视频复核</h1>
<p class="muted">相同输入、首帧与噪声。保留 PC 收益，同时检查请求的物理过程、因果先后、结果持续性和不合理形变。</p>
<p class="muted">VP001–VP040 为原40例，VP041–VP060 为新增20例；选择“完整60例”版本可同步复核两部分视频。模块对照仍使用既定16例。</p>
<div class="toolbar"><label>案例 <select id="case"></select></label><button id="prev">上一例</button><button id="next">下一例</button><span id="updated" class="muted"></span></div>
<div class="caption"><div id="caption"></div><div id="criteria" class="cue muted"></div></div>
<div class="pair"><div class="panel"><select id="left"></select><p id="leftScore" class="score"></p><video id="a" controls muted playsinline preload="metadata"></video><a id="leftLink">打开原视频</a></div><div class="panel"><select id="right"></select><p id="rightScore" class="score"></p><video id="b" controls muted playsinline preload="metadata"></video><a id="rightLink">打开原视频</a></div></div>
<div class="toolbar"><button id="play">同步播放</button><button id="pause">暂停</button><button id="reset">从头播放</button><label>速度 <select id="speed"><option value="1">1 倍</option><option value="0.5">0.5 倍</option><option value="0.25">0.25 倍</option></select></label><input id="seek" type="range" min="0" max="3.375" step="0.01" value="0"><span id="time">0.00 秒</span></div>
<p class="muted">短暂接触、滴落和连续形变请结合完整播放判断。未看清记为不确定；有动作不等于物理正确，保持静止也不等于完成请求。不同训练步数的比较会同时受到训练预算影响。</p>
<textarea id="note" class="notes" placeholder="可记录：哪些事件实际发生、何时出现异常、哪些内容尚不确定。"></textarea><div class="toolbar"><button id="save">导出观察记录</button><span class="muted">记录只保存在当前页面，导出后保留；不参与训练。</span></div>
<script>const DATA=__DATA__;
const $=id=>document.getElementById(id),A=$('a'),B=$('b'),notes={};
for(const item of DATA.cases)$('case').add(new Option(item.id,item.id));
for(const id of ['left','right'])for(const v of DATA.versions)$(id).add(new Option(v.label,v.id));
$('left').value=DATA.defaults.left;$('right').value=DATA.defaults.right;$('case').value=DATA.defaults.case;
$('updated').textContent='结果快照：'+DATA.updated.replace('T',' ').slice(0,19)+' UTC';
const noteKey=()=>[$('case').value,$('left').value,$('right').value].join('|');let previousKey=noteKey();
function loadSide(id,video){const v=DATA.versions.find(x=>x.id===$(id).value),entry=v.videos[$('case').value];video.pause();video.removeAttribute('src');$(id+'Link').removeAttribute('href');if(!entry){video.load();$(id+'Score').textContent='该版本未提供此例视频';return;}video.src=entry.src;$(id+'Link').href=entry.src;const s=entry.score;$(id+'Score').textContent=s?'此例 SA '+s.sa+' / 5　PC '+s.pc+' / 5':'视频已完成，评分待完成';video.playbackRate=Number($('speed').value);video.load();}
function load(){notes[previousKey]=$('note').value;const row=DATA.cases.find(x=>x.id===$('case').value);$('caption').textContent=row.caption;$('criteria').textContent=row.criteria?'可见过程要求：'+row.criteria:'';loadSide('left',A);loadSide('right',B);$('seek').value=0;$('time').textContent='0.00 秒';previousKey=noteKey();$('note').value=notes[previousKey]||'';}
for(const id of ['case','left','right'])$(id).onchange=load;
async function play(){await Promise.all([A,B].filter(x=>x.hasAttribute('src')).map(v=>v.play().catch(()=>{})));}
$('play').onclick=play;$('pause').onclick=()=>{A.pause();B.pause();};$('reset').onclick=()=>{for(const v of [A,B])if(v.readyState)v.currentTime=0;play();};
$('speed').onchange=()=>{A.playbackRate=B.playbackRate=Number($('speed').value);};
$('seek').oninput=()=>{const t=Number($('seek').value);for(const v of [A,B])if(v.readyState)v.currentTime=Math.min(t,v.duration||t);$('time').textContent=t.toFixed(2)+' 秒';};
A.ontimeupdate=()=>{if(!A.seeking){$('seek').value=A.currentTime;$('time').textContent=A.currentTime.toFixed(2)+' 秒';if(!A.paused&&B.readyState&&Math.abs(B.currentTime-A.currentTime)>.15)B.currentTime=Math.min(A.currentTime,B.duration);}};
A.onloadedmetadata=()=>{$('seek').max=A.duration;};
function step(n){const i=$('case').selectedIndex;$('case').selectedIndex=(i+n+DATA.cases.length)%DATA.cases.length;load();}
$('prev').onclick=()=>step(-1);$('next').onclick=()=>step(1);
$('save').onclick=()=>{notes[noteKey()]=$('note').value;const content={recorded:new Date().toISOString(),scope:'User-entered visible observations; not automatically validated scores',notes};const url=URL.createObjectURL(new Blob([JSON.stringify(content,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='paired-video-observations.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),5000);};
load();</script></html>'''
    encoded=json.dumps(data,ensure_ascii=False).replace('</','<\\/')
    destination.write_text(page.replace('__DATA__',encoded))
    (destination.with_suffix('.catalog.json')).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(output=str(destination),cases=len(cases),versions={x['id']:len(x['videos']) for x in versions}),ensure_ascii=False))


if __name__=='__main__':main()
