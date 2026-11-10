from pathlib import Path
import json,textwrap
from PIL import Image,ImageDraw
from decord import VideoReader,cpu
ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'inference_outputs/physical_content_68514/step0400_original_fixed40_controls16_seed42'
DEST=ROOT/'reports/physical_content_68514/visual_review_sheets';DEST.mkdir(parents=True,exist_ok=True)
frames=[0,11,22,34,45,57,68,80]
records=[]
for suite in ['videophy40','videophy60']:
 for meta in sorted((OUT/suite/'cases').glob('*/planned.json')):
  video=meta.with_suffix('.mp4')
  if not video.exists():continue
  data=json.loads(meta.read_text());records.append((meta.parent.name,video,data['caption']))
created=[]
for start in range(0,len(records),4):
 group=records[start:start+4]
 if len(group)<4 and len(records) not in [40,60]:continue
 name=DEST/(group[0][0]+'_'+group[-1][0]+'.jpg')
 if name.exists():continue
 panel=Image.new('RGB',(1024,352*len(group)),(245,245,245));draw=ImageDraw.Draw(panel)
 for row,(case,video,caption) in enumerate(group):
  y=row*352;draw.text((4,y+1),case+' | '+textwrap.fill(caption,width=150),fill=(0,0,0))
  reader=VideoReader(str(video),ctx=cpu(0),num_threads=1);images=reader.get_batch(frames).asnumpy()
  for i,(index,frame) in enumerate(zip(frames,images)):
   x=(i%4)*256;yy=y+36+(i//4)*157
   panel.paste(Image.fromarray(frame).resize((256,148)),(x,yy));draw.text((x+2,yy),str(index),fill=(255,0,0))
 panel.save(name,quality=95);created.append(str(name))
print(json.dumps({'completed_videos':len(records),'created_sheets':created}))
