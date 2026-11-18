"""Training-only whole-video physical localization from a frozen local VLM.

Inspired by ProPhy's generated-text-query/video-key contrast, with Qwen3-VL
native normalization and actual video-token masks. No chunk captions, optical
flow, tracked objects, invented event times, or generated-evaluation targets.
The output is a weak localization signal, not ground-truth physical laws.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
import torch

BASE = Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native')
REVISION = 'R20-whole-video-physical-localization-v1'
OFFSETS = list(range(0, 81, 4))
SYSTEM = 'You describe only what is visibly supported by the supplied video.'
NEUTRAL = 'Describe the visible scene briefly in at most 35 words.'
PHYSICAL = '''Describe the visible physical process in at most 45 words: what
interacts or changes and how it visibly evolves. Include the relevant material,
body or medium only when visible. Do not invent causes, forces, numerical
properties, events outside the view, or a completed outcome. This is a cropped
excerpt whose beginning and end need not show a complete process. If no clear
physical change is visible, say so. The existing clip description is context,
not evidence overriding the video:
'''


class Capture:
    """Compute mean generated-query/video-key logits without dense attention.

Mean over all language layers, heads and decoded input tokens, matching the
reference contraction. Hooks observe the actual normalized, rotary Q/K. They
do not change model outputs, weights or the installed Transformers package.
"""
    def __init__(self, model):
        self.handles = []
        self.active = False
        for layer in model.model.language_model.layers:
            self.handles.append(layer.self_attn.register_forward_pre_hook(
                self.observe, with_kwargs=True))

    def reset(self, inputs, video_id):
        self.positions = (inputs['input_ids'][0] == video_id).nonzero().flatten()
        if not len(self.positions):
            raise ValueError('No actual video tokens')
        self.prefix = inputs['input_ids'].shape[1]
        self.keys, self.sums, self.counts = {}, {}, {}
        self.active = True

    @torch.no_grad()
    def observe(self, module, args, kwargs):
        if not self.active:
            return
        from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb
        h = kwargs.get('hidden_states', args[0] if args else None)
        pos = kwargs.get('position_embeddings', args[1] if len(args) > 1 else None)
        if h is None or pos is None:
            raise ValueError('Missing actual Qwen attention inputs')
        shape = (*h.shape[:-1], -1, module.head_dim)
        q = module.q_norm(module.q_proj(h).view(shape)).transpose(1, 2)
        k = module.k_norm(module.k_proj(h).view(shape)).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, *pos)
        layer = module.layer_idx
        if h.shape[1] == self.prefix:
            self.keys[layer] = k[:, :, self.positions].detach()
            return
        if layer not in self.keys:
            raise ValueError('Decode reached without cached video keys')
        key = self.keys[layer]
        kv_heads = key.shape[1]
        query = q.reshape(1, kv_heads, module.num_key_value_groups,
                          q.shape[2], module.head_dim).float().mean((2, 3))
        score = torch.einsum('bhd,bhnd->bn', query, key.float())
        score = score[0] / (kv_heads * math.sqrt(module.head_dim))
        count = q.shape[2]
        self.sums[layer] = self.sums.get(layer, torch.zeros_like(score)) + score * count
        self.counts[layer] = self.counts.get(layer, 0) + count

    def result(self):
        self.active = False
        if len(self.sums) != len(self.handles) or any(n < 2 for n in self.counts.values()):
            raise ValueError('Incomplete generated-token attention observation')
        out = torch.stack([v / self.counts[k] for k, v in self.sums.items()]).mean(0).cpu()
        count = min(self.counts.values())
        self.keys.clear(); self.sums.clear()
        return out, count


def read_frames(row):
    path = BASE.parent/'v5.2/physgen_v52/preparation.py'
    spec = importlib.util.spec_from_file_location('_exact_training_frame_geometry', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.read_frames(row, OFFSETS)


def annotate(model, processor, capture, frames, row, question):
    messages = [dict(role='system', content=SYSTEM), dict(role='user', content=[
        dict(type='video', video='already_decoded_training_frames'),
        dict(type='text', text=question)])]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # The exact training crop is preserved before a single uniform downsample.
    # 21 observations cover the whole clip; no local/chunk prompts are created.
    video = np.stack([np.asarray(im.resize((384, 224), Image.Resampling.BILINEAR)) for im in frames])
    fps = float(row['source_fps'])
    inputs = processor(text=[text], videos=[video], return_tensors='pt',
        do_sample_frames=False, do_resize=False,
        video_metadata=[dict(fps=fps, total_num_frames=81, frames_indices=OFFSETS.copy())])
    grid = inputs['video_grid_thw'][0].tolist()
    merge = processor.video_processor.merge_size
    expected = grid[0]*(grid[1]//merge)*(grid[2]//merge)
    inputs = inputs.to('cuda')
    video_id = model.config.video_token_id
    capture.reset(inputs, video_id)
    if len(capture.positions) != expected:
        raise ValueError(f'Video key count/grid mismatch: {len(capture.positions)} vs {expected}')
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=80, do_sample=False,
                                   use_cache=True)
    answer = processor.tokenizer.decode(generated[0, inputs['input_ids'].shape[1]:],
                                        skip_special_tokens=True).strip()
    score, count = capture.result()
    if not torch.isfinite(score).all() or not answer:
        raise ValueError('Invalid localization observation')
    return dict(answer=answer, scores=score.reshape(grid[0],grid[1]//merge,grid[2]//merge),
                video_grid=grid, token_count=expected, decoded_queries=count)


def panel(frames, contrast, path, name, legend='positive physics-minus-scene logits'):
    import cv2
    canvas = Image.new('RGB', (4*320, 2*208), '#202020'); draw = ImageDraw.Draw(canvas)
    heat = contrast.clamp_min(0)
    peak = heat.max().clamp_min(1e-8)
    for col, i in enumerate([0, 7, 14, 20]):
        image = frames[i].resize((320,184)); canvas.paste(image,(col*320,24))
        frame_map = heat[min(i//2,heat.shape[0]-1)]/peak
        v = cv2.resize(frame_map.numpy(), (320,184))
        color = cv2.applyColorMap(np.uint8(np.clip(v,0,1)*255), cv2.COLORMAP_JET)
        color = cv2.cvtColor(color,cv2.COLOR_BGR2RGB)
        blend = np.uint8(np.asarray(image)*.65+color*.35)
        canvas.paste(Image.fromarray(blend),(col*320,232))
        draw.text((col*320+4,4),f'{name} f{OFFSETS[i]}',fill='white')
        draw.text((col*320+4,212),legend,fill='white')
    canvas.save(path, quality=90)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model',default='/home/liuzhirui/model/Qwen3-VL/Qwen3-VL-8B-Instruct')
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    selection=json.loads(args.selection.read_text())
    if selection['scope']!='training_only_actual_full_clip' or selection['chunk_conditioning']:
        raise ValueError('Training-only whole-video evidence required')
    sys.path.insert(0,str(BASE/'tools'))
    from gpu_policy import require_rtx_pro_6000
    gpu=require_rtx_pro_6000()
    (args.output/'gpu.json').write_text(json.dumps(gpu,indent=2)+'\n')
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from transformers.utils import logging
    logging.disable_progress_bar();torch.set_num_threads(4)
    model=Qwen3VLForConditionalGeneration.from_pretrained(args.model,dtype=torch.bfloat16,
        device_map={'':'cuda'},attn_implementation='sdpa',local_files_only=True).eval().requires_grad_(False)
    processor=AutoProcessor.from_pretrained(args.model,local_files_only=True)
    capture=Capture(model);started=time.monotonic(); completed=[]
    for row in selection['cases']:
        dest=args.output/row['id'];dest.mkdir(exist_ok=True)
        if (dest/'localization.pt').exists() and (dest/'record.json').exists():
            record=json.loads((dest/'record.json').read_text())
            if record['revision']!=REVISION or record['caption']!=row['caption']:
                raise ValueError('Previous localization provenance differs')
            completed.append(row['id']);continue
        frames=read_frames(row)
        neutral=annotate(model,processor,capture,frames,row,NEUTRAL)
        physical=annotate(model,processor,capture,frames,row,PHYSICAL+row['caption'])
        contrast=physical['scores']-neutral['scores']
        pts=[float(row['pts'][i]-row['pts'][0]) for i in OFFSETS]
        pts=pts+[pts[-1]] if len(pts)%2 else pts
        merged_pts=[(pts[i]+pts[i+1])/2 for i in range(0,len(pts),2)]
        if len(merged_pts)!=contrast.shape[0]:raise ValueError('Temporal patch mismatch')
        torch.save(dict(physical=physical['scores'],neutral=neutral['scores'],contrast=contrast,
            video_times=torch.tensor(merged_pts),video_grid=torch.tensor(physical['video_grid']),
            frame_offsets=torch.tensor(OFFSETS)),dest/'localization.pt')
        record=dict(id=row['id'],revision=REVISION,caption=row['caption'],video=row['video'],
            source_indices=[row['indices'][i] for i in OFFSETS],frame_offsets=OFFSETS,
            geometry=row['geometry'],model=args.model,physical_answer=physical['answer'],
            neutral_answer=neutral['answer'],physical_queries=physical['decoded_queries'],
            neutral_queries=neutral['decoded_queries'],video_tokens=physical['token_count'],
            contrast_min=float(contrast.min()),contrast_max=float(contrast.max()),
            contrast_std=float(contrast.std()),positive_fraction=float((contrast>0).float().mean()),
            training_only=True,chunk_conditioning=False,api_calls=0,weak_supervision=True,
            verified_localization=False)
        (dest/'record.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
        panel(frames,contrast,dest/'panel.jpg',row['id']);completed.append(row['id'])
        status=dict(revision=REVISION,completed=len(completed),total=len(selection['cases']),
            seconds=time.monotonic()-started,last=row['id'],gpu=gpu,api_calls=0)
        (args.output/'progress.json').write_text(json.dumps(status,indent=2)+'\n')
        print(json.dumps(status),flush=True)
    (args.output/'complete.json').write_text(json.dumps(dict(revision=REVISION,
        completed=completed,seconds=time.monotonic()-started,api_calls=0),indent=2)+'\n')


if __name__=='__main__':main()
