"""Targeted complete-run paired video review; no evaluator/model/API calls."""
from pathlib import Path
import json
import decord
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).parent/'api400_visual_review'
OUT.mkdir(exist_ok=True)
ref=ROOT/'r10a_api400/inference_outputs/physical_content_67271/step0400_api_fixed60_controls16_seed42'
new=ROOT/'aligned_request_r11a/inference_outputs/physical_content_67265/step0400_api_fixed60_controls16_seed42'
folders={'R10A-API full':ref/'videophy40','R10A-API no-C':ref/'controls/no_c/videophy40','R11A-API full':new/'videophy40'}
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
records=[]
for case in ('VP026','VP038'):
    panel=Image.new('RGB',(1664,1650),'white');draw=ImageDraw.Draw(panel)
    record=dict(case=case,videos={},frames={},selection='Score-selected diagnostic cases: VP026 SA/PC tradeoff; VP038 lost Joint with PC5. Not representative sampling.')
    for row,(name,folder) in enumerate(folders.items()):
        path=folder/'cases'/case/'planned.mp4';reader=decord.VideoReader(str(path),ctx=decord.cpu(0))
        ids=np.linspace(0,len(reader)-1,8,dtype=int).tolist();frames=reader.get_batch(ids).asnumpy()
        draw.text((8,row*550+4),f'{case} | {name}',fill='black',font=font)
        for i,(index,frame) in enumerate(zip(ids,frames)):
            x=(i%4)*416;y=row*550+30+(i//4)*260
            panel.paste(Image.fromarray(frame).resize((416,240)),(x,y))
            draw.text((x+6,y+240),f'frame {index}',fill='black',font=font)
        record['videos'][name]=str(path);record['frames'][name]=ids
    path=OUT/f'{case}.png';panel.save(path);record['panel']=str(path);records.append(record)
(OUT/'selection.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps([v['panel'] for v in records]))
