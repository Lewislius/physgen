"""Whole-video physical evidence from the VLM family used by ProPhy.

The observer follows ProPhy's post-RoPE generated-Q/video-K contraction. It
uses the installed Transformers implementation without replacing its forward.
All observations are from actual training clips; no boxes or video chunks.
Full input reasoning remains unchanged and is encoded by the existing Qwen3-VL.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap
from physical_content.common import atomic_json, atomic_torch, gpu_guard, load_config
from physical_content.data import TrainingData
from physical_content.teacher import Teacher

REVISION = 'FT-Qwen25VL32B-whole-video-physical-distribution-v1'
OFFSETS = list(range(0, 81, 4))
MODEL = '/home/liuzhirui/model/Qwen2.5-VL/Qwen2.5-VL-32B-Instruct'


class Capture:
    """Read actual Q/K before attention; leave the VLM computation unchanged."""
    def __init__(self, model):
        self.active = False
        self.handles = [layer.self_attn.register_forward_pre_hook(self.observe, with_kwargs=True)
                        for layer in model.model.language_model.layers]

    def reset(self, inputs, video_id):
        self.positions = (inputs['input_ids'][0] == video_id).nonzero().flatten()
        self.prefix = inputs['input_ids'].shape[1]
        self.keys, self.sums, self.counts = {}, {}, {}
        self.active = True

    @torch.no_grad()
    def observe(self, module, args, kwargs):
        if not self.active:
            return
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb
        hidden = kwargs.get('hidden_states', args[0] if args else None)
        positions = kwargs.get('position_embeddings')
        if hidden is None or positions is None:
            raise ValueError('Actual Qwen2.5-VL attention inputs are unavailable')
        shape = (*hidden.shape[:-1], -1, module.head_dim)
        query = module.q_proj(hidden).view(shape).transpose(1, 2)
        key = module.k_proj(hidden).view(shape).transpose(1, 2)
        query, key = apply_multimodal_rotary_pos_emb(
            query, key, *positions, module.config.rope_parameters['mrope_section'])
        layer = module.layer_idx
        if hidden.shape[1] == self.prefix:
            self.keys[layer] = key[:, :, self.positions].detach()
            return
        if layer not in self.keys:
            raise ValueError('Generated queries have no corresponding video keys')
        video_key = self.keys[layer]
        kv_heads = video_key.shape[1]
        repeated = query.shape[1] // kv_heads
        # Averaging Q first is algebraically identical to averaging scaled
        # dot products over groups and generated-token queries. No softmax.
        reduced = query.reshape(1, kv_heads, repeated, query.shape[2], module.head_dim).float().mean((2, 3))
        scores = torch.einsum('bhd,bhnd->bn', reduced, video_key.float())[0]
        scores = scores / (kv_heads * math.sqrt(module.head_dim))
        count = query.shape[2]
        self.sums[layer] = self.sums.get(layer, torch.zeros_like(scores)) + count * scores
        self.counts[layer] = self.counts.get(layer, 0) + count

    def result(self):
        self.active = False
        if len(self.sums) != len(self.handles) or min(self.counts.values()) < 2:
            raise ValueError('Incomplete generated-query observation')
        value = torch.stack([v / self.counts[k] for k, v in self.sums.items()]).mean(0).cpu()
        count = min(self.counts.values())
        self.keys.clear(); self.sums.clear()
        return value, count


def read_frames(row):
    spec = importlib.util.spec_from_file_location('_exact_training_clip',
        bootstrap.BASE.parent / 'v5.2/physgen_v52/preparation.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.read_frames(row, OFFSETS)


@torch.inference_mode()
def observe(model, processor, capture, frames, row, question):
    message = [dict(role='system', content='You are a helpful assistant.'),
        dict(role='user', content=[dict(type='video', video='decoded_full_training_clip'),
                                 dict(type='text', text=question)])]
    text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    # Qwen2.5 uses14-pixel patches and2x2 merging. These dimensions are multiples28.
    video = np.stack([np.asarray(frame.resize((392, 224), Image.Resampling.BILINEAR))
                      for frame in frames])
    inputs = processor(text=[text], videos=[video], return_tensors='pt',
        do_sample_frames=False, do_resize=False,
        video_metadata=[dict(fps=float(row['source_fps']), total_num_frames=81,
                             frames_indices=OFFSETS.copy())]).to('cuda')
    grid = inputs['video_grid_thw'][0].tolist()
    merge = processor.video_processor.merge_size
    geometry = (grid[0], grid[1] // merge, grid[2] // merge)
    capture.reset(inputs, model.config.video_token_id)
    if len(capture.positions) != math.prod(geometry):
        raise ValueError('Whole-video token geometry differs from actual video tokens')
    output = model.generate(**inputs, max_new_tokens=112, do_sample=False, use_cache=True)
    answer_tokens = output[0, inputs['input_ids'].shape[1]:]
    answer = processor.tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
    scores, queries = capture.result()
    if not answer or not torch.isfinite(scores).all():
        raise ValueError('Empty or nonfinite physical observation')
    return dict(answer=answer, raw_scores=scores.reshape(geometry), decoded_queries=queries,
                generated_tokens=len(answer_tokens), generation_limit=112,
                second_per_grid_ts=inputs['second_per_grid_ts'].detach().cpu().tolist())


def panel(frames, contrast, relations, path):
    canvas = Image.new('RGB', (1600, 210*(len(relations)+1)), '#202020')
    draw = ImageDraw.Draw(canvas)
    for col, index in enumerate([0,5,10,15,20]):
        canvas.paste(frames[index].resize((320,184)), (col*320,26))
        draw.text((col*320+4,5), f'real training frame {OFFSETS[index]}', fill='white')
    for group, relation in enumerate(relations):
        positive = contrast[group].clamp_min(0)
        scale = max(float(positive.max()), 1e-8)
        for col, index in enumerate([0,5,10,15,20]):
            heat = (positive[min(index//2,positive.shape[0]-1)] / scale).numpy()
            mask = Image.fromarray(np.uint8(np.clip(heat,0,1)*255)).resize((320,184),Image.Resampling.BILINEAR)
            image = np.asarray(frames[index].resize((320,184))).astype(float)
            alpha = np.asarray(mask,dtype=float)[...,None]/255*.6
            overlay = image*(1-alpha)+np.array([255,40,0])*alpha
            canvas.paste(Image.fromarray(np.uint8(np.clip(overlay,0,255))), (col*320,210*(group+1)+26))
        draw.text((4,210*(group+1)+5), f'relation {group+1}: {relation[:175]}', fill='white')
    canvas.save(path, quality=92)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model',default=MODEL)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    cfg=load_config(args.config); gpu_guard(args.output)
    selection=json.loads(args.selection.read_text())
    if selection['scope']!='training_only_actual_full_clip' or selection.get('chunk_conditioning',False):
        raise ValueError('Only whole observed training clips are accepted')
    data=TrainingData(cfg); source=Teacher(cfg,device='cpu'); by_id={r['id']:i for i,r in enumerate(data.rows)}
    ids=[r['id'] for r in selection['cases']]
    if not set(ids)<=set(by_id): raise ValueError('Non-training case requested')
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from transformers.utils import logging
    logging.disable_progress_bar(); torch.set_num_threads(4)
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model,dtype=torch.bfloat16,
        device_map={'':'cuda'},attn_implementation='sdpa',local_files_only=True).eval().requires_grad_(False)
    processor=AutoProcessor.from_pretrained(args.model,local_files_only=True)
    capture=Capture(model); started=time.monotonic(); records=[]
    for sid in ids:
        dest=args.output/sid; dest.mkdir(exist_ok=True)
        if (dest/'record.json').is_file() and (dest/'observations.pt').is_file():
            record=json.loads((dest/'record.json').read_text())
            if record['revision']!=REVISION: raise ValueError('Existing observation revision differs')
            records.append(record);continue
        sample=data.get(by_id[sid],device='cpu',targets=False); evidence=source.input(sample,drop=False)
        row=sample['row']; relations=evidence['relations']; frames=read_frames(row)
        neutral=observe(model,processor,capture,frames,row,'Describe the video briefly in30–50 words.')
        physical=[]
        for relation in relations:
            question=(f"Describe the visible evidence for the physical relation '{relation}' briefly in30–50 words. "
                'Do not describe anything else. If it is not visible, say so. '
                'This video is a cropped excerpt; its beginning and end need not show a complete process.')
            physical.append(observe(model,processor,capture,frames,row,question))
        contrasts=torch.stack([value['raw_scores']-neutral['raw_scores'] for value in physical])
        times=[float(row['pts'][i]-row['pts'][0]) for i in OFFSETS]
        padded=times+([times[-1]] if len(times)%2 else [])
        centers=torch.tensor([(padded[i]+padded[i+1])/2 for i in range(0,len(padded),2)])
        if len(centers)!=contrasts.shape[1]: raise ValueError('Actual full-video time mapping differs')
        record=dict(id=sid,revision=REVISION,model=args.model,relations=relations,
            caption=sample['caption'],neutral_answer=neutral['answer'],
            physical_answers=[v['answer'] for v in physical],
            physical_generated_tokens=[v['generated_tokens'] for v in physical],
            positive_fraction=[float((v>0).float().mean()) for v in contrasts],
            contrast_range=[[float(v.min()),float(v.max())] for v in contrasts],
            source_frame_indices=[row['indices'][i] for i in OFFSETS],
            second_per_grid_ts=neutral['second_per_grid_ts'],
            scope='training_only_actual_full_clip',future_inference_condition=False,
            chunk_conditioning=False,bounding_boxes=False,api_calls=0,approved_for_training=False)
        atomic_torch(dict(revision=REVISION,neutral=neutral['raw_scores'],
            physical=torch.stack([v['raw_scores'] for v in physical]),contrast=contrasts,
            video_times=centers,relations=relations),dest/'observations.pt')
        panel(frames,contrasts,relations,dest/'panel.jpg');atomic_json(record,dest/'record.json')
        records.append(record)
        progress=dict(completed=len(records),total=len(ids),seconds=time.monotonic()-started,last=sid)
        atomic_json(progress,args.output/'progress.json');print(json.dumps(progress),flush=True)
    atomic_json(dict(complete=True,revision=REVISION,count=len(records),
        seconds=time.monotonic()-started,records=records,approved_for_training=False),args.output/'complete.json')


if __name__=='__main__':main()
