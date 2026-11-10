"""Inspect existing matched outputs against take1 reference outcomes; no generation."""
from pathlib import Path
import datetime
import json

ROOT=Path(__file__).resolve().parent
SELECTED={
    'trimmed-cut-paper.mp4':'Material tearing, tool continuity and execution of the requested interaction.',
    'trimmed-juice-in-water.mp4':'Liquid transfer and mixing, container geometry and fluid response.',
    'trimmed-light-on-mug-block.mp4':'Object rotation and consistency of the cast shadows.',
}


def main():
    import decord
    from PIL import Image,ImageDraw,ImageFont
    protocol=json.loads((ROOT/'protocol.json').read_text())
    folder=ROOT/'reports/physics_iq_visual_predeclared';folder.mkdir(parents=True,exist_ok=True)
    selected=[c for c in protocol['cases'] if c.get('scene') in SELECTED
              and c['perspective']=='perspective-center']
    indices=[0,15,31,47,63,79]
    plan=dict(selected_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scenes=SELECTED,perspective='perspective-center',frames=indices,
        models=['wan_lora','r1','r10a'],reference='Official take1 testing outcome, not a generation input',
        scope='Three qualitative cases selected before official model PhysicsIQ scores; not a representative aggregate visual metric.',
        new_generation=0,new_scoring=0)
    selection=folder/'selection.json'
    if not selection.exists():selection.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    for c in selected:
        reference=ROOT/'physics_iq_gt/testing'/f"{c['official_id']}_testing-videos_16FPS_{c['perspective']}_take-1_{c['scene']}"
        videos={'GT take1':reference};metadata={}
        for model in plan['models']:
            out=ROOT/'outputs'/model/'seed42'
            videos[model]=out/'physics_iq60'/c['output_filename']
            metadata[model]=json.loads((out/'cases'/c['id']/'video.json').read_text())
        first=metadata['wan_lora']
        for model,data in metadata.items():
            if data['caption']!=c['caption']:raise ValueError('Different official caption')
            for key in ['case','seed','noise_seed','first_image','original_image','output_frames','sampling']:
                if data[key]!=first[key]:raise ValueError(f'{model}: mismatched {key}')
        panel=Image.new('RGB',(1656,816),'white');draw=ImageDraw.Draw(panel)
        for row,(model,video) in enumerate(videos.items()):
            reader=decord.VideoReader(str(video),ctx=decord.cpu(0),num_threads=1)
            frames=reader.get_batch(indices).asnumpy()
            y=row*204;draw.text((5,y+2),f"{c['scene']} | {model} | center view",font=font,fill='black')
            for col,(frame,index) in enumerate(zip(frames,indices)):
                x=col*276;panel.paste(Image.fromarray(frame).resize((276,159)),(x,y+22))
                draw.text((x+4,y+182),f'frame {index}',font=font,fill='black')
        output=folder/(Path(c['scene']).stem+'.png');panel.save(output)
        record=dict(case=c,reason=SELECTED[c['scene']],frames=indices,
            videos={k:str(v) for k,v in videos.items()},metadata=metadata,panel=str(output),
            ground_truth_used_for_review_only=True)
        (folder/(Path(c['scene']).stem+'_selection.json')).write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({'case':c['id'],'panel':str(output)}))


if __name__=='__main__':main()
