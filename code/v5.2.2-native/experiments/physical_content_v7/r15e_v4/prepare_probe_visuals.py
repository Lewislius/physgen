"""Matched actual-frame panels for predefined R10A execution failure review."""
from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageDraw
from decord import VideoReader,cpu

ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent
reference=json.loads((BASE/'reports/r10_final_evidence.json').read_text())['source_paths']
roots=[('R10A original',Path(reference['R10-A/original40']).parent),
       ('R10A execution probe',ROOT/'diagnostic_outputs/r10a400_native_video_response_fixed16/videophy40'),
       ('R1 reference',Path(reference['R1/original40']).parent)]
destination=ROOT/'reports/visual_probe';destination.mkdir(parents=True,exist_ok=True)
for case in ['VP004','VP010','VP027','VP034','VP038']:
    videos=[(label,root/'cases'/case/'planned.mp4') for label,root in roots]
    if not all(path.exists() and path.with_suffix('.json').exists() for _,path in videos):continue
    if (destination/(case+'.png')).exists() and (destination/(case+'.json')).exists():continue
    panel=Image.new('RGB',(8*312,3*204),'white');draw=ImageDraw.Draw(panel);records=[]
    for row,(label,path) in enumerate(videos):
        reader=VideoReader(str(path),ctx=cpu(0),num_threads=1)
        indices=np.linspace(0,len(reader)-1,8).round().astype(int).tolist()
        frames=reader.get_batch(indices).asnumpy()
        for col,frame in enumerate(frames):panel.paste(Image.fromarray(frame).resize((312,180)),(col*312,row*204+24))
        draw.text((4,row*204+5),label,fill='black');records.append(dict(label=label,path=str(path),indices=indices))
    panel.save(destination/(case+'.png'))
    (destination/(case+'.json')).write_text(json.dumps(dict(case=case,rows=records,
        scope='Sampled actual frames, not a complete motion metric; same fixed inputs and seed'),indent=2)+'\n')
print(json.dumps(dict(panels=len(list(destination.glob('VP*.png'))),output=str(destination))))
