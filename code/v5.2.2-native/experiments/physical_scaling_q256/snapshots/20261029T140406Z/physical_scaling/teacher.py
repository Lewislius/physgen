"""Frozen input evidence and separately cached, training-only weak supervision.

No generated video is read, scored, or differentiated by this teacher.
Only requested text + first image enter the inference evidence cache.
"""
from pathlib import Path
from contextlib import contextmanager
import fcntl
import json
import re
import torch
from physgen_native.knowledge import QwenKnowledge
from .common import atomic_torch, cpu_tree
from .data import read_teacher_video
from .supervision import association_target, write_support, resample_maps

INPUT_TASK = (
    'Give up to four short physical relation sentences relevant to this first image and requested video. '
    'One numbered sentence per line, each at most 20 words: prerequisite or source -> physical response '
    'and what may change or should persist. Distinguish visible facts from conditional future responses. '
    'Include fluids or other continuous processes when relevant. Do not invent measurements, positions '
    'or exact event times. No introduction or list of named laws.'
)
VIDEO_TASK = (
    'For this actual video clip, give up to four short physical relation sentences supported by visible evidence. '
    'One numbered sentence per line, at most 20 words each: prerequisite or source -> observed response '
    'and allowed or preserved properties. Do not assert events outside the clip or invented measurements.'
)


def split_relations(text):
    lines = [re.sub(r'^\s*(?:[-*]|\d+[.)、:])\s*','',line).strip() for line in text.splitlines()]
    lines = [s for s in lines if s and not s.endswith(':')]
    if not lines:
        raise ValueError('Frozen Qwen returned no usable physical relation text')
    return list(dict.fromkeys(lines))[:4]


class Teacher:
    def __init__(self,cfg,device='cuda',log=None):
        self.cfg,self.device,self.log = cfg,device,log
        self.knowledge = QwenKnowledge(cfg,device=device).eval()
        self.root = Path(cfg['paths']['supervision_cache'])/'r1'
        self.root.mkdir(parents=True,exist_ok=True)
        self.readonly_root = Path(cfg['paths']['supervision_cache_readonly'])/'r1'
        self.t5 = None

    @contextmanager
    def cache(self,kind,name):
        folder = self.root/kind
        folder.mkdir(parents=True,exist_ok=True)
        path = folder/(name+'.pt')
        # Multiple requested variants may need the same input evidence. Advisory
        # locking avoids duplicate GPU annotation and atomically publishes it.
        with path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            old = self.readonly_root/kind/(name+'.pt')
            source = path if path.is_file() else old
            value = torch.load(source,map_location='cpu',weights_only=True) if source.is_file() else None
            yield path,value

    def _report(self,kind,sample,**extra):
        if self.log:
            self.log(kind,sample=sample,**extra)

    @torch.no_grad()
    def input(self,sample,drop=False):
        caption = '' if drop else sample['caption']
        key = sample['id']+('_null' if drop else '_text')
        with self.cache('input',key) as (path,value):
            if value is None:
                self._report('prepare_input',sample['id'],text_dropped=drop)
                first = sample['first_rgb']
                duration = float(sample['times'][0,-1])
                body = f'Request: {caption or "No text request. Infer only conditional relations supported by the image."}\nPresentation interval: {duration:.3f} seconds.\n'+INPUT_TASK
                inputs,_ = self.knowledge._inputs([(first,first.new_zeros(1))],body)
                analysis,complete = self.knowledge._reason(inputs,128,return_complete=True)
                relations = split_relations(analysis)
                spans = [f'[RELATION {i+1}] {s}' for i,s in enumerate(relations)]
                body = f'Request: {caption or "No text request."}\nPresentation interval: {duration:.3f} seconds.\nConditional physical analysis:\n'+'\n'.join(spans)
                inputs,positions = self.knowledge._inputs([(first,first.new_zeros(1))],body,extract_spans=spans)
                hidden = self.knowledge.backbone.model(**inputs,use_cache=False,return_dict=True).last_hidden_state
                if any(not p for p in positions):
                    raise ValueError('Relation span has no tokens')
                vectors = torch.stack([hidden[:,p].mean(1) for p in positions],1)
                value = dict(memory=hidden.cpu().to(torch.bfloat16),memory_mask=inputs['attention_mask'].bool().cpu(),
                             relation_vectors=vectors.cpu().to(torch.bfloat16),relations=relations,
                             caption=caption,analysis=analysis,analysis_complete=complete,
                             scope='first_image_and_request_only',sample=sample['id'],
                             model=self.cfg['paths']['qwen_checkpoint'],prompt=INPUT_TASK)
                atomic_torch(value,path)
            if value['caption'] != caption or value['scope'] != 'first_image_and_request_only' or value['prompt'] != INPUT_TASK:
                raise ValueError('Input evidence cache does not match the legal conditioning request')
        return {k:(v.to(self.device) if torch.is_tensor(v) else v) for k,v in value.items()}

    @torch.no_grad()
    def _attention_maps(self,frames,times,relations):
        """Selected answer query rows only; never materialize N-by-N attention."""
        from transformers.models.qwen3_vl.modeling_qwen3_vl import rotate_half
        spans = [f'RELATION {i+1}: {r}' for i,r in enumerate(relations)]
        body = ('Read the visible video evidence for the following physical relation descriptions. '
                'A description can be conditional; attention is only a soft association, not proof of its occurrence.')
        inputs,positions = self.knowledge._inputs([(frames,times)],body,video=True,
                                                  assistant_prefix='\n'.join(spans),extract_spans=spans)
        visual = (inputs['input_ids'][0] == self.knowledge.backbone.config.video_token_id).nonzero().flatten()
        if any(not p for p in positions):
            raise ValueError('Teacher phrase has no selected text query rows')
        selected = sorted({i for group in positions for i in group})
        query_indices = torch.tensor(selected,device=frames.device)
        remap = {p:i for i,p in enumerate(selected)}
        groups = [[remap[p] for p in group] for group in positions]
        maps,handles = [],[]

        def capture(module,args,kwargs):
            hidden = kwargs.get('hidden_states',args[0] if args else None)
            cos,sin = kwargs['position_embeddings']
            hd = module.head_dim
            q = module.q_norm(module.q_proj(hidden[:,query_indices]).view(1,len(selected),-1,hd)).transpose(1,2)
            k = module.k_norm(module.k_proj(hidden).view(1,hidden.shape[1],-1,hd)).transpose(1,2)
            q = q*cos[:,None,query_indices]+rotate_half(q)*sin[:,None,query_indices]
            k = k*cos[:,None]+rotate_half(k)*sin[:,None]
            k = k.repeat_interleave(module.num_key_value_groups,dim=1)
            logits = torch.matmul(q.float(),k.float().transpose(-1,-2)).float()*module.scaling
            causal = torch.arange(hidden.shape[1],device=hidden.device)[None,:] <= query_indices[:,None]
            logits = logits.masked_fill(~causal[None,None],float('-inf'))
            probabilities = logits.softmax(-1)[0].index_select(-1,visual)
            read = torch.stack([probabilities[:,idx].mean((0,1)) for idx in groups])
            read = read/read.sum(-1,keepdim=True).clamp_min(1e-12)
            maps.append(read.detach())

        language = self.knowledge.backbone.model.language_model
        try:
            for layer in language.layers[-4:]:
                handles.append(layer.self_attn.register_forward_pre_hook(capture,with_kwargs=True))
            self.knowledge.backbone.model(**inputs,use_cache=False,return_dict=True)
        finally:
            for handle in handles:
                handle.remove()
        grid = inputs['video_grid_thw'][0].tolist()
        shape = (grid[0],grid[1]//self.knowledge.merge,grid[2]//self.knowledge.merge)
        result = torch.stack(maps).mean(0).reshape(len(relations),*shape)
        stamps = times.reshape(-1,self.knowledge.temporal).mean(-1)
        return result,stamps

    @torch.no_grad()
    def local_target(self,sample,evidence,model):
        if sample['row']['split'] != 'train':
            raise ValueError('Physical auxiliary targets are prepared for training only')
        mode = self.cfg['variant']
        with self.cache(mode,sample['id']) as (path,value):
            if value is None:
                self._report('prepare_video_target',sample['id'],variant=mode)
                frames,times = read_teacher_video(sample['row'],self.device)
                if mode == 'v6_write':
                    inputs,_ = self.knowledge._inputs([(frames,times)],VIDEO_TASK+'\nClip description: '+sample['caption'],video=True)
                    relations = split_relations(self.knowledge._reason(inputs,128))
                else:
                    # EXACT input-side sentences and order: no future-derived C.
                    relations = evidence['relations']
                raw,observed = self._attention_maps(frames,times,relations)
                bg,_ = self._attention_maps(frames,times,['The overall visible scene, its background and general appearance.'])
                target_times = sample['times'][0]
                if mode == 'v6_write':
                    raw_flat = raw.flatten(1).T
                    support = write_support(raw_flat,bg.flatten())
                    support = support.T.reshape_as(raw)
                    support = resample_maps(support,observed,target_times,15,26)[390:][None]
                    anchors = self._text_anchors(relations,model)
                    value = dict(support=support.cpu(),text_anchors=anchors.cpu(),relations=relations,
                                 supported=bool(support.sum()>0),
                                 scope='real_training_video_only_synthetic_residual_target',sample=sample['id'])
                else:
                    mapped = resample_maps(torch.cat((raw,bg),0),observed,target_times,15,26)
                    mapped = mapped/mapped.sum(0,keepdim=True).clamp_min(1e-12)
                    target = association_target(mapped[:,:-1],mapped[:,-1],self.cfg['loss']['temperature'])[390:][None]
                    value = dict(association=target.cpu(),relations=relations,
                                 scope='real_training_video_weak_association_for_input_relations',sample=sample['id'])
                value.update(observed_pts=times.cpu(),teacher_grid=list(raw.shape[1:]),
                             teacher_layers='last_4',frame_indices=sorted(set(range(0,81,3))|{80}),
                             temperature=self.cfg['loss']['temperature'],model=self.cfg['paths']['qwen_checkpoint'])
                atomic_torch(value,path)
                self._report('video_target_ready',sample['id'],variant=mode,relations=relations)
            if mode=='v6_assoc' and value['relations']!=evidence['relations']:
                raise ValueError('Teacher relations and actual C groups differ')
        if mode=='v6_write' and not value.get('supported',True):
            self._report('no_positive_write_support',sample['id'])
            return None
        return {k:(v.to(self.device) if torch.is_tensor(v) else v) for k,v in value.items()}

    @torch.no_grad()
    def _text_anchors(self,relations,model):
        if self.t5 is None:
            from wan.modules.t5 import T5EncoderModel
            root = Path(self.cfg['paths']['wan_checkpoint'])
            self.t5 = T5EncoderModel(text_len=512,dtype=torch.bfloat16,device=self.device,
                                    checkpoint_path=str(root/'models_t5_umt5-xxl-enc-bf16.pth'),
                                    tokenizer_path=str(root/'google/umt5-xxl'))
        contexts = self.t5(relations,self.device)
        projected = []
        for context in contexts:
            # T5's final EOS is not physical content; no padding participates.
            content = context[:-1] if len(context)>1 else context
            projected.append(model.wan.text_embedding(content).float().mean(0))
        return torch.stack(projected)[None]
