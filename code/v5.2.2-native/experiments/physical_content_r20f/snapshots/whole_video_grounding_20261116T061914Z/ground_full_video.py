"""Whole-clip visible-process region annotation, training targets only.

One local frozen VLM call observes the complete training excerpt. Sparse region
observations are spatial supervision, not temporal chunk conditions or tracks.
They cover material/medium/contact/light regions as well as moving bodies.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
import torch

BASE = Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native')
REVISION = 'R20-whole-video-visible-physics-regions-v1'
OBSERVE = list(range(0, 81, 4))
ANCHORS = [0, 20, 40, 60, 80]


def read_frames(row):
    spec = importlib.util.spec_from_file_location('_exact_crop', BASE.parent/'v5.2/physgen_v52/preparation.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.read_frames(row, OBSERVE)


def parse_answer(answer):
    start, end = answer.find('{'), answer.rfind('}')
    value = json.loads(answer[start:end+1])
    rows = value['frames']
    if [r['frame_offset'] for r in rows] != ANCHORS:
        raise ValueError('Whole-video observation offsets differ')
    for row in rows:
        if not isinstance(row['regions'], list) or len(row['regions']) > 4:
            raise ValueError('Invalid process-region list')
        for box in row['regions']:
            if len(box) != 4 or not all(isinstance(v,(int,float)) for v in box):
                raise ValueError('Invalid region coordinates')
            x1,y1,x2,y2 = box
            if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
                raise ValueError('Region outside image or degenerate')
    return value


def annotate(model, processor, frames, row):
    anchor_times = [round(row['pts'][i]-row['pts'][0],3) for i in ANCHORS]
    question = '''Observe this entire video excerpt. Identify the VISIBLE regions
directly participating in its physical process, such as interacting/deforming
material, pouring liquid, rising gas, an illuminated or reflecting surface,
and relevant contact/support regions. Include the visible process's extent,
not only its center. Exclude unrelated scenery. This is not object tracking.
The excerpt starts and ends mid-process; do not invent unseen causes, a start,
or a completed outcome. A region may be still (e.g. a support or reflection).
Use the actual images, not the caption alone. Mark up to 3 tight rectangles per
requested observation. Coordinates [x1,y1,x2,y2] are normalized to 0..1000 in
the displayed image. Use [] if no relevant visible region can be located.
Return ONLY compact JSON with a short process phrase and these five observations:
{"process":"visible process","frames":[{"frame_offset":0,"regions":[[x1,y1,x2,y2]]},
{"frame_offset":20,"regions":[]},{"frame_offset":40,"regions":[]},
{"frame_offset":60,"regions":[]},{"frame_offset":80,"regions":[]}]}
The five frame offsets correspond, respectively, to seconds: '''+str(anchor_times)+'''.
Existing description (context, not overriding the video):
'''+row['caption']
    messages=[dict(role='system',content='You annotate only visually supported physical-process regions. Output compact JSON.'),
              dict(role='user',content=[dict(type='video',video='decoded_whole_training_excerpt'),dict(type='text',text=question)])]
    chat=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    # Preserve the actual 832x480 training crop; both dimensions are divisible
    # by this local Qwen3-VL encoder's 16x2 spatial merge.
    video=np.stack([np.asarray(im) for im in frames])
    inputs=processor(text=[chat],videos=[video],do_resize=False,do_sample_frames=False,
        video_metadata=[dict(fps=float(row['source_fps']),total_num_frames=81,frames_indices=OBSERVE.copy())],return_tensors='pt').to('cuda')
    with torch.inference_mode():
        out=model.generate(**inputs,max_new_tokens=480,do_sample=False,use_cache=True)
    answer=processor.tokenizer.decode(out[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True).strip()
    return answer


def panel(frames, result, path):
    canvas=Image.new('RGB',(1600,208),'#202020');draw=ImageDraw.Draw(canvas)
    for c,item in enumerate(result['frames']):
        im=frames[item['frame_offset']//4].resize((320,184));overlay=ImageDraw.Draw(im)
        for x1,y1,x2,y2 in item['regions']:
            overlay.rectangle((x1*.32,y1*.184,x2*.32,y2*.184),outline=(255,220,0),width=2)
        canvas.paste(im,(c*320,24));draw.text((c*320+3,4),f"frame {item['frame_offset']}",fill='white')
    canvas.save(path,quality=92)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--selection',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    selection=json.loads(a.selection.read_text())
    if selection['scope']!='training_only_actual_full_clip' or selection['chunk_conditioning']:
        raise ValueError('Only whole-video training evidence is permitted')
    sys.path.insert(0,str(BASE/'tools'));from gpu_policy import require_rtx_pro_6000
    gpu=require_rtx_pro_6000();(a.output/'gpu.json').write_text(json.dumps(gpu)+'\n')
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    torch.set_num_threads(4)
    name='/home/liuzhirui/model/Qwen3-VL/Qwen3-VL-8B-Instruct'
    model=Qwen3VLForConditionalGeneration.from_pretrained(name,dtype=torch.bfloat16,device_map={'':'cuda'},attn_implementation='sdpa',local_files_only=True).eval().requires_grad_(False)
    processor=AutoProcessor.from_pretrained(name,local_files_only=True)
    start=time.monotonic();completed=[];failures=[]
    for row in selection['cases']:
        dest=a.output/row['id'];dest.mkdir(exist_ok=True)
        frames=read_frames(row)
        try:
            answer=annotate(model,processor,frames,row)
            (dest/'answer.txt').write_text(answer+'\n')
            result=parse_answer(answer)
            record=dict(id=row['id'],revision=REVISION,caption=row['caption'],video=row['video'],
                source_indices=[row['indices'][i] for i in OBSERVE],observed_offsets=OBSERVE,
                region_offsets=ANCHORS,geometry=row['geometry'],model=name,raw_answer=answer,result=result,
                training_only=True,chunk_conditioning=False,weak_supervision=True,verified_localization=False)
            (dest/'record.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
            panel(frames,result,dest/'panel.jpg');completed.append(row['id'])
        except (ValueError,KeyError,TypeError,json.JSONDecodeError) as error:
            failures.append(dict(id=row['id'],error=str(error)));(dest/'failure.json').write_text(json.dumps(failures[-1])+'\n')
        progress=dict(revision=REVISION,completed=len(completed),failed=len(failures),total=len(selection['cases']),seconds=time.monotonic()-start,last=row['id'],gpu=gpu)
        (a.output/'progress.json').write_text(json.dumps(progress)+'\n');print(json.dumps(progress),flush=True)
    (a.output/'complete.json').write_text(json.dumps(dict(revision=REVISION,completed=completed,failures=failures,seconds=time.monotonic()-start,api_calls=0),indent=2)+'\n')


if __name__=='__main__':main()
