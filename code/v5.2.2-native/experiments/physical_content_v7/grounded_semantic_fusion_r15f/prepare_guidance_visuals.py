"""Sample the same five predeclared cases from the frozen F guidance probe."""
from pathlib import Path
import json
from PIL import Image,ImageDraw
from decord import VideoReader,cpu

ROOT=Path(__file__).resolve().parent
CASES=['VP004','VP010','VP027','VP034','VP038']
FRAMES=[0,11,22,34,45,57,68,80]


def main():
    prepared=json.loads((ROOT/'native_guidance_prepared.json').read_text())
    suite=Path(prepared['output'])/'controls/native_guidance/videophy40'
    normal=ROOT/'inference_outputs/physical_content_68038/step0400_original_fixed40_controls16_seed42/videophy40'
    destination=ROOT/'reports/native_guidance16/visual_predeclared';destination.mkdir(parents=True,exist_ok=True)
    records=[]
    for case in CASES:
        path=suite/'cases'/case/'planned.mp4';metadata=path.with_suffix('.json')
        if not metadata.is_file():continue
        actual=json.loads(metadata.read_text());previous=json.loads((normal/'cases'/case/'planned.json').read_text())
        if any(actual[k]!=previous[k] for k in ['checkpoint','step','caption','canvas','seed','sampler']):
            raise ValueError('Mismatched video or reference: '+case)
        panel_path=destination/f'{case}.png'
        if not panel_path.is_file():
            frames=VideoReader(str(path),ctx=cpu(0),num_threads=1).get_batch(FRAMES).asnumpy()
            panel=Image.new('RGB',(1536,500),(245,245,245));draw=ImageDraw.Draw(panel)
            for i,(index,frame) in enumerate(zip(FRAMES,frames)):
                x,y=i%4*384,i//4*250;panel.paste(Image.fromarray(frame).resize((384,222)),(x,y+24))
                draw.text((x+6,y+5),f'F native-guidance {case} | frame {index}',fill=(0,0,0))
            panel.save(panel_path)
        records.append(dict(case=case,video=str(path),panel=str(panel_path),caption=actual['caption'],
            guidance=actual['guidance'],selection='Same five cases predeclared before original F scores',
            frame_indices=FRAMES,scope='Sampled-frame diagnostic, not a full temporal quality rating'))
    (destination/'selection.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(prepared=[x['case'] for x in records],expected=CASES)))


if __name__=='__main__':main()
