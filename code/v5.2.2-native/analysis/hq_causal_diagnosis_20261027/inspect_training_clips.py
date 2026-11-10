"""Extract every actual verifier-feedback training clip for direct content review."""
import json
from pathlib import Path
import cv2
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
audit=json.loads((HERE/'training_content_audit.json').read_text())
rows=audit['feedback_sources']
out=HERE/'training_montages';out.mkdir(exist_ok=True)
for start in range(0,len(rows),4):
    selected=rows[start:start+4]
    canvas=Image.new('RGB',(5*276,len(selected)*195),'#171c22');draw=ImageDraw.Draw(canvas)
    for line,row in enumerate(selected):
        y=line*195
        draw.text((5,y+3),f'step {row["feedback_step"]} | {row["id"]} | {row["duration"]:.2f}s | {row["caption"][:115]}',fill='white')
        cap=cv2.VideoCapture(row['video']);g=row['geometry']
        for j,offset in enumerate([0,20,40,60,80]):
            cap.set(cv2.CAP_PROP_POS_FRAMES,row['indices'][offset]);ok,bgr=cap.read();assert ok,(row['id'],offset)
            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            rgb=cv2.resize(rgb,(g['resize_w'],g['resize_h']),interpolation=cv2.INTER_AREA)
            rgb=rgb[g['crop_top']:g['crop_top']+g['h'],g['crop_left']:g['crop_left']+g['w']]
            canvas.paste(Image.fromarray(rgb).resize((276,159)),(276*j,y+30))
        cap.release()
    canvas.save(out/f'feedback_{start:02d}.jpg',quality=91)
    print(out/f'feedback_{start:02d}.jpg',flush=True)
