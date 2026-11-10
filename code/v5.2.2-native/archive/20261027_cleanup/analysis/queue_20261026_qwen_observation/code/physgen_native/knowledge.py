"""Shared Qwen base: legal conditioning and native-video training evidence.

Knowledge LoRA is active only for initial conditions. The fixed teacher and
verifier feature extractor preserve RGB derivatives without updating the base.
"""
from collections import OrderedDict
import hashlib
import json
import torch
from torch import nn
from torch.nn import functional as F
from .lora import install_qwen_lora, knowledge_role
from .latent_reasoning import capture_native_memory, refine_dependency_latent
from .execution_trace import traced_call

def visual_patches(rgb, size, patch=16, merge=2, temporal=2, *, video=False):
    x = F.interpolate(rgb.float(), size=tuple(size), mode='bicubic', align_corners=False, antialias=True).clamp(-1, 1)
    frames, channels, height, width = x.shape
    gh, gw = height//patch, width//patch
    if height % (patch*merge) or width % (patch*merge): raise ValueError('Invalid visual patch geometry')
    if video:
        padding = (-frames) % temporal
        if padding: x = torch.cat((x, x[-1:].expand(padding, -1, -1, -1)), 0)
        groups = x.shape[0]//temporal
        x = x.reshape(groups, temporal, channels, height, width)
        grid = torch.tensor([[groups, gh, gw]], device=x.device, dtype=torch.long)
    else:
        groups = frames
        x = x[:, None].expand(-1, temporal, -1, -1, -1)
        grid = torch.tensor([[1, gh, gw]]*frames, device=x.device, dtype=torch.long)
    x = x.reshape(groups, temporal, channels, gh//merge, merge, patch, gw//merge, merge, patch)
    patches = x.permute(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(groups*gh*gw, channels*temporal*patch*patch)
    return patches, grid


def clause_context(caption, clauses):
    lines = [f'Requested scene/process, not evidence of completion: {caption or "No text condition."}']
    for i, clause in enumerate(clauses):
        lines.append(f'Clause {i}: {clause["text"]} [source={clause.get("basis", "prompt")}; '
                     f'phase={clause.get("phase", "evolution")}; scope={clause.get("scope", "unspecified")}]')
    return '\n'.join(lines)


EVIDENCE_QUERIES = (
    'Represent visible prerequisites, interactions and continuing conditions, and how they support the observed responses.',
    'Represent the actual intermediate developments over time, including delayed changes, transient changes and completion.',
    'Represent continuity of attributes and sources through occlusion, deformation, diffusion, splitting, merging and disappearance.',
)


DEPENDENCY_QUERIES = (
    'Encode the observed initial conditions and distinguish them from requested future events and unknown conditions.',
    'Encode the applicable physical interactions: trigger or prerequisite, source of influence, and resulting response.',
    'Encode how the influence persists or stops, and which attributes and material dependencies must remain consistent over time.',
    'Encode a conditional intervention: if the relevant trigger or continuing condition were absent or changed, which response would change and which properties would remain.',
)


class QwenKnowledge(nn.Module):
    def __init__(self, cfg, device='cuda', backbone=None, tokenizer=None):
        super().__init__()
        from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration
        self.cfg = cfg['knowledge']
        self.reasoning_cache = OrderedDict()
        root = cfg['paths']['qwen_checkpoint']
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(root, local_files_only=True)
        self.backbone = backbone or Qwen3VLForConditionalGeneration.from_pretrained(
            root, dtype=torch.bfloat16, local_files_only=True, attn_implementation='sdpa',
            device_map={'': str(device)}, low_cpu_mem_usage=True)
        self.backbone.requires_grad_(False)
        install_qwen_lora(self.backbone, self.cfg)
        if self.cfg.get('latent_reasoning_steps', 0):
            depth = self.cfg['latent_reasoning_layers']
            # DeepStack is injected only into the initial language layers. All
            # replayed late layers must operate after those visual injections.
            available = len(self.backbone.model.language_model.layers)-len(self.backbone.config.vision_config.deepstack_visual_indexes)
            if not 1 <= depth <= min(available, self.cfg['lora_last_layers']):
                raise ValueError('Latent reasoning must reuse late knowledge-adapted Qwen layers after DeepStack')
        self.backbone.config.use_cache = False
        vision = self.backbone.config.vision_config
        self.patch, self.merge, self.temporal = vision.patch_size, vision.spatial_merge_size, vision.temporal_patch_size
        ids = [self.tokenizer.encode(x, add_special_tokens=False) for x in ('A', 'B', 'C')]
        if any(len(x) != 1 for x in ids): raise ValueError('Comparison labels must be native single tokens')
        self.register_buffer('answer_ids', torch.tensor([x[0] for x in ids]), persistent=False)

    def _truncate(self, text, maximum=1000):
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        return self.tokenizer.decode(ids[:maximum], skip_special_tokens=False)

    def _inputs(self, clips, body, readouts=(), *, video=False, assistant_prefix='', extract_spans=()):
        pixels, grids, descriptions = [], [], []
        for index, (frames, times) in enumerate(clips):
            patches, grid = visual_patches(frames, self.cfg['video_size' if video else 'image_size'],
                                           self.patch, self.merge, self.temporal, video=video)
            pixels.append(patches); grids.append(grid)
            count = int(grid[0, 1:].prod())//self.merge**2
            descriptions.append(f'Video {index+1}:\n' if video else 'Initial observation:\n')
            if video:
                timestamps = times
                padding = (-len(times)) % self.temporal
                if padding: timestamps = torch.cat((times, times[-1:].expand(padding)))
                timestamps = timestamps.reshape(-1, self.temporal).mean(-1)
                for stamp in timestamps:
                    descriptions.append(f'<{float(stamp):.1f} seconds><|vision_start|>'+ '<|video_pad|>'*count+'<|vision_end|>')
            else:
                for _ in range(len(frames)):
                    descriptions.append('<|vision_start|>'+'<|image_pad|>'*count+'<|vision_end|>')
            descriptions.append('\n')
        content = ''.join(descriptions)+body
        for i, query in enumerate(readouts): content += f'\n[EVIDENCE {i}] {query}\n'
        rendered = self.tokenizer.apply_chat_template([{'role': 'user', 'content': content}], tokenize=False, add_generation_prompt=True)
        rendered += assistant_prefix
        encoded = self.tokenizer(rendered, add_special_tokens=False, return_tensors='pt', return_offsets_mapping=True)
        offsets = encoded.pop('offset_mapping')[0]
        ids = encoded['input_ids']
        visual_id = self.backbone.config.video_token_id if video else self.backbone.config.image_token_id
        if int((ids != visual_id).sum()) > self.cfg['max_text_tokens']:
            raise ValueError('Qwen text budget exceeded')
        ends = []
        for i, query in enumerate(readouts):
            segment = f'[EVIDENCE {i}] {query}'; start = rendered.index(segment); end = start+len(segment)
            selected = ((offsets[:, 1] > start) & (offsets[:, 1] <= end)).nonzero().flatten()
            ends.append(int(selected[-1]))
        if extract_spans:
            if readouts:raise ValueError('Select either query endpoints or semantic content spans')
            for segment in extract_spans:
                start=rendered.index(segment);end=start+len(segment)
                selected=((offsets[:,1]>start)&(offsets[:,0]<end)).nonzero().flatten()
                ends.append(selected.tolist())
        device = clips[0][0].device
        inputs = {k: v.to(device) for k, v in encoded.items() if k in ('input_ids', 'attention_mask')}
        grid_key, pixel_key = ('video_grid_thw', 'pixel_values_videos') if video else ('image_grid_thw', 'pixel_values')
        inputs.update({grid_key: torch.cat(grids), pixel_key: torch.cat(pixels)})
        inputs['position_ids'] = self.backbone.model.get_rope_index(
            inputs['input_ids'], **{grid_key: inputs[grid_key]}, attention_mask=inputs['attention_mask'])[0]
        return inputs, ends

    @torch.no_grad()
    def _reason(self, inputs, budget):
        # Native generation manages its own 4-axis positions and KV cache.
        inputs = {k:v for k,v in inputs.items() if k != 'position_ids'}
        training = self.backbone.training
        self.backbone.eval()
        try:
            with knowledge_role(False):
                result = self.backbone.generate(**inputs, max_new_tokens=budget, do_sample=False,
                    # The pretrained config enables sampling (0.7/0.8/20).
                    # This path deliberately uses greedy decoding; unset its
                    # sampling-only settings without changing the shared config.
                    temperature=None, top_p=None, top_k=None,
                    use_cache=True, return_dict_in_generate=False, pad_token_id=self.tokenizer.eos_token_id)
            return self.tokenizer.decode(result[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True).strip()
        finally:
            self.backbone.train(training)

    def dependency_reasoning(self, first_rgb, caption, clauses, *, duration=None):
        if self.cfg['reasoning_tokens'] == 0: return ''
        key = hashlib.sha256(first_rgb.detach().float().cpu().numpy().tobytes()+
            json.dumps([caption,clauses,duration],sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        if key in self.reasoning_cache:
            self.reasoning_cache.move_to_end(key)
            return self.reasoning_cache[key]
        body = clause_context(self._truncate(caption),clauses)+(
            '\nConstruct a compact physical dependency analysis for generating the future video. '
            'Use four short lines: OBSERVED: visible starting facts and unknowns. '
            'INTERACTION: relevant prerequisite or source -> response. '
            'PERSISTENCE: what sustains, propagates or terminates that response. '
            'INTERVENTION: if one relevant condition changes or is absent, what response changes and what remains. '
            'Separate observed facts from conditional predictions. Include continuous media and transformations when relevant. '
            'Give the applicable dependencies, not a scene caption or a list of generic physics laws. '
            'Do not invent a uniquely determined trajectory from one image.')
        if duration is not None:
            body += f' The requested interval is {duration:.3f} seconds; reason about processes on this time scale.'
        inputs,_ = self._inputs([(first_rgb,first_rgb.new_zeros(len(first_rgb)))],body)
        analysis = traced_call('Qwen_analysis', self._reason, inputs, self.cfg['reasoning_tokens'])
        self.reasoning_cache[key] = analysis
        while len(self.reasoning_cache)>self.cfg['reasoning_cache']: self.reasoning_cache.popitem(last=False)
        return analysis

    def encode(self, first_rgb, caption, clauses, *, adapt=True, readout=None, duration=None):
        if self.cfg['dependency_queries']:
            queries = [f'For clause {i}, represent its observed basis, applicable prerequisites and possible consequences.'
                       for i in range(len(clauses))]+list(DEPENDENCY_QUERIES)
        else:
            queries = [f'Represent the image and the text in clause {i}.' for i in range(len(clauses))]+[
                'Represent the initial appearance.','Represent the scene description.',
                'Represent the requested video content.','Represent the overall multimodal context.']
        provenance = {'image': 0, 'prompt': 1, 'both': 2}
        roles = [provenance.get(c.get('basis'), 1) for c in clauses]+[3]*len(DEPENDENCY_QUERIES)
        body = clause_context(self._truncate(caption), clauses)+'\nFuture goals are not already observed facts.'
        if duration is not None: body += f' Requested interval: {duration:.3f} seconds.'
        reasoning = (self.dependency_reasoning(first_rgb,caption,clauses,duration=duration)
                     if not adapt or self.cfg.get('generation_reasoning', True) else '')
        if reasoning: body += '\nConditional physical analysis:\n'+reasoning
        readout=readout or self.cfg.get('condition_readout','queries')
        if readout=='content':
            # Preserve the contextualized content of the physical dependencies,
            # rather than reducing each to the last token of a generic question.
            spans=[f'Clause {i}: {clause["text"]}' for i,clause in enumerate(clauses)]
            roles=[provenance.get(c.get('basis'),1) for c in clauses]
            if self.cfg.get('include_prompt_span') and caption:
                spans.insert(0, f'Requested scene/process, not evidence of completion: {self._truncate(caption)}')
                roles.insert(0, 1)
            if reasoning:spans.append(reasoning);roles.append(3)
            if not spans:spans=[body];roles=[1]
            inputs,ends=self._inputs([(first_rgb,first_rgb.new_zeros(len(first_rgb)))],body,extract_spans=spans)
        elif readout=='queries':
            inputs, ends = self._inputs([(first_rgb, first_rgb.new_zeros(len(first_rgb)))], body, queries)
        else:raise ValueError('Unknown knowledge readout')
        steps = self.cfg.get('latent_reasoning_steps', 0) if adapt else 0
        language = self.backbone.model.language_model
        if steps:
            depth = self.cfg['latent_reasoning_layers']
            with capture_native_memory(language, depth) as captured, knowledge_role(adapt):
                output = traced_call('Qwen_encode', self.backbone.model, **inputs, use_cache=False, return_dict=True)
            source = captured['raw_output']
        else:
            with knowledge_role(adapt): output = self.backbone.model(**inputs, use_cache=False, return_dict=True)
            source = output.last_hidden_state
        if readout=='content':
            remaining=self.cfg.get('max_condition_tokens',128);total=sum(map(len,ends))
            chunks=[];expanded_roles=[]
            for index,(positions,role) in enumerate(zip(ends,roles)):
                budget=min(len(positions),max(1,round(remaining*len(positions)/total)))
                budget=min(budget,remaining-(len(ends)-index-1))
                states=source[:,positions]
                if budget<len(positions):states=F.adaptive_avg_pool1d(states.transpose(1,2),budget).transpose(1,2)
                chunks.append(states);expanded_roles += [role]*budget
                remaining-=budget;total-=len(positions)
            contextual=torch.cat(chunks,1);roles=expanded_roles
        else:contextual=source[:,ends]
        if steps:
            with knowledge_role(adapt):
                contextual = traced_call('Qwen_latent_reasoning', refine_dependency_latent, language, captured['memory'], contextual,
                    inputs['position_ids'], inputs['attention_mask'], steps=steps, depth=depth,
                    attention=self.cfg.get('latent_reasoning_attention', 'causal'))
        return dict(contextual=contextual,roles=torch.tensor(roles,device=first_rgb.device),reasoning=reasoning,
                    latent_reasoning_steps=steps)

    def evidence(self, frames, times, caption, clauses):
        body = clause_context(self._truncate(caption), clauses)+(
            '\nRepresent what the time-ordered video actually shows. Distinguish prerequisites, progression and '
            'attribute/source continuity. Allow continuous media and justified appearance or disappearance.')
        inputs, ends = self._inputs([(frames, times)], body, EVIDENCE_QUERIES, video=True)
        with knowledge_role(False):
            output = self.backbone.model(**inputs, use_cache=False, return_dict=True)
        return output.last_hidden_state[:, ends].float()

    def process_features(self, frames):
        """Frozen visual features at EVERY latent-time endpoint, including late times.

        The two-frame visual encoder sees adjacent RGB observations; no causal-VAE
        temporal delta is interpreted as physical velocity. These are observed
        generated-video features, never an independent prediction of the future.
        """
        indices = [0,0]+[i for end in range(4,len(frames),4) for i in (end-1,end)]
        patches,grid = visual_patches(frames[indices],self.cfg['video_size'],
                                     self.patch,self.merge,self.temporal,video=True)
        with knowledge_role(False):
            output = self.backbone.model.visual(patches.to(next(self.backbone.model.visual.parameters()).dtype),grid_thw=grid)
        t,h,w = grid[0].tolist()
        return output.pooler_output.float().reshape(t,h//self.merge,w//self.merge,-1)

    @torch.no_grad()
    def preference(self, first, second, times, caption, clauses):
        observations = []
        if self.cfg.get('video_observation_first'):
            # Separate visual observation from the requested result. A caption
            # saying "suspended" must not override a visibly changing ball;
            # more vigorous change alone is not evidence of a better process.
            for frames in (first, second):
                observation_prompt = (
                    'Inspect only this time-ordered video. Report EARLY, MIDDLE, and LATE visible states with '
                    'approximate timestamps, then describe what changed between them. Follow the same materials, '
                    'objects, shapes and sources across time, including diffuse media, deformation and appearance. '
                    'Distinguish actual visible contact or response from a merely possible interaction. '
                    'Describe any abrupt unexplained replacement, reversal, shape change or loss of continuity; '
                    'do not assume all deformation or disappearance is wrong. If evidence is unclear, say so. '
                    'Do not invent events between sampled frames. Use at most 100 words; no quality verdict.')
                inputs, _ = self._inputs([(frames, times)], observation_prompt, video=True)
                observations.append(self._reason(inputs, self.cfg.get('video_observation_tokens', 192)))
        body = clause_context(self._truncate(caption), clauses)+(
            '\nCompare the observable development of the requested process in Video 1 and Video 2. '
            'Check (1) required interactions or sources and their responses, (2) actual intermediate and late progression, '
            '(3) continuity of attributes and of the physical influence. '
            'First describe the concrete temporal evidence and any missing development in each video in at most 80 words. '
            'Accept different physically plausible realizations. Treat stillness as appropriate only when the requested process allows it. '
            'Do not decide from the scene description alone. Do not give the verdict yet.')
        if observations:
            body += ('\nIndependent video observations, to check against the actual frames:\n'
                     f'VIDEO 1: {observations[0]}\nVIDEO 2: {observations[1]}\n'
                     'The requested scene describes intended conditions, not proof that an event happened. '
                     'Prioritize supported interactions, coherent intermediate and late development, and continuity '
                     'of the same materials and sources. More activity or closer caption wording cannot compensate '
                     'for unsupported changes. Judge the observed interval, not an imagined completed event. '
                     'Use concrete temporal evidence; correct any mistaken observation above.')
        clips = [(first,times),(second,times)]
        inputs, _ = self._inputs(clips, body, video=True)
        reasoning = self._reason(inputs,self.cfg['comparison_tokens']) if self.cfg['comparison_tokens'] else ''
        decision = ('\nOn that evidence, choose the better fulfillment of the physical process. '
                    'A means Video 1; B means Video 2; C means both have comparable process evidence. '
                    'Return one letter only.')
        inputs, _ = self._inputs(clips, body+decision, video=True, assistant_prefix=reasoning+'\nVerdict: ')
        self.last_comparison_reasoning = reasoning
        self.last_comparison_observations = observations
        with knowledge_role(False):
            output = self.backbone.model(**inputs, use_cache=False, return_dict=True)
            weight = self.backbone.lm_head.weight.index_select(0, self.answer_ids.to(first.device))
            logits = F.linear(output.last_hidden_state[:, -1].to(weight.dtype), weight).float()
        probabilities = logits.softmax(-1)
        return probabilities[:, 0]+.5*probabilities[:, 2], probabilities
