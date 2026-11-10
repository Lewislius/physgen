"""v14 content-preserving reasoning and optional supervised reasoning APIs."""
import hashlib
import json
import torch
from torch.nn import functional as F
from .lora import knowledge_role, verifier_role
from .recurrent_reasoning import PrefixMemory, capture_prefix_projections


ANALYSIS_TEMPLATE = (
    'Construct physical dependencies for the requested video. Distinguish observations, '
    'requests and unknowns. Explain applicable prerequisites -> responses, persistence '
    'and termination, and how a changed prerequisite would alter the response. '
    'Include continuous media and transformations when relevant. Add information rather '
    'than repeating the caption. Do not invent measurements or a uniquely determined '
    'future from one image. Counterfactual alternatives are analysis, not extra requested events.'
)

COMPACT_ANALYSIS_TEMPLATE = (
    'Write a compact physical dependency brief for generating this video. '
    'Return exactly FOUR short lines, one sentence per line, at most 25 words per line '
    'and at most 100 words total. Use these labels in order: '
    'OBSERVED: relevant initial facts; '
    'RESPONSE: the main prerequisite or source and its expected response; '
    'PERSISTENCE: what sustains, propagates, or ends that response; '
    'CHANGE: how changing one relevant prerequisite would alter the response. '
    'Use process-specific physical commonsense, including fluids, waves, deformation and appearance when relevant. '
    'Keep observations separate from requests and conditional predictions. '
    'The original caption and clauses are retained separately, so do not repeat them. '
    'No introduction, extended explanation, alternative scenarios, equations or generic list of laws. '
    'Finish immediately after the fourth sentence.'
)


def content_sections(caption, clauses, duration=None, analysis=''):
    """Source markers PRECEDE content; deduplicate only identical records."""
    sections, seen = [], set()
    if caption: sections.append('[REQUEST / 要求] '+caption)
    sources = {'image': 'OBSERVED / 观察', 'prompt': 'REQUEST / 要求',
               'both': 'MIXED SOURCE / 观察与要求须区分'}
    for clause in clauses:
        key = json.dumps(clause, sort_keys=True, ensure_ascii=False)
        if key in seen: continue
        seen.add(key)
        source = sources.get(clause.get('basis', 'prompt'), 'UNKNOWN / 未知来源')
        sections.append(f'[{source}; clause={len(seen)}; phase={clause.get("phase", "unspecified")}; '
                        f'scope={clause.get("scope", "unspecified")}] {clause["text"]}')
    if analysis: sections.append('[CONDITIONAL ANALYSIS / 条件分析与未知] '+analysis)
    if not sections: sections.append('[UNKNOWN / 未知] No textual requirement supplied.')
    header = 'Requests are not observed facts. Preserve uncertainty about unobserved conditions.\n'
    if duration is not None: header += f'Requested interval: {duration:.3f} seconds.\n'
    return header+'\n'.join(sections), sections


class MultiroundKnowledge:
    @torch.no_grad()
    def _encode_message(self, first_rgb, caption, clauses, *, duration=None):
        analysis = (self._multiround_analysis(first_rgb, caption, clauses, duration, adapt=False)
                    if self.cfg.get('generation_reasoning', True) else '')
        body, spans = content_sections(caption, clauses, duration, analysis)
        inputs, ends = self._inputs([(first_rgb, first_rgb.new_zeros(len(first_rgb)))], body, extract_spans=spans)
        with knowledge_role(False), verifier_role(False):
            output = self.backbone.model(**inputs, use_cache=False, return_dict=True)
        memory = output.last_hidden_state.detach()
        positions = sorted({p for segment in ends for p in segment})
        if not positions:
            raise ValueError('Message content cannot be empty')
        return dict(memory=memory, memory_mask=inputs['attention_mask'].bool(),
                    contextual=memory[:, positions],
                    content_positions=torch.tensor(positions, device=memory.device), reasoning=analysis)

    def _multiround_analysis(self, first_rgb, caption, clauses, duration=None, *, adapt=True):
        if not self.cfg['reasoning_tokens']: return ''
        budget, limit = self.cfg['reasoning_tokens'], self.cfg['reasoning_max_tokens']
        keep_partial = self.cfg.get('analysis_on_token_limit') == 'use_partial'
        body, _ = content_sections(caption, clauses, duration)
        body += '\n'+(COMPACT_ANALYSIS_TEMPLATE if self.cfg.get('analysis_format') == 'compact_four_lines'
                      else ANALYSIS_TEMPLATE)
        # Optimizer/copy_ increments Parameter._version. Cache never survives an
        # adapter change, prompt/budget change, or learned-vs-fixed role change.
        versions = [(n, p._version) for n, p in self.backbone.named_parameters()
                    if adapt and n.endswith(('lora_a', 'lora_b'))]
        identity = json.dumps([body, budget, limit, keep_partial, adapt, versions, self.cfg['image_size'],
                               self.tokenizer.chat_template], ensure_ascii=False)
        key = hashlib.sha256(first_rgb.detach().float().cpu().numpy().tobytes()+identity.encode()).hexdigest()
        if key in self.reasoning_cache:
            self.reasoning_cache.move_to_end(key)
            cached = self.reasoning_cache[key]
            if isinstance(cached, tuple):
                text, complete = cached
                self.last_analysis_truncated = not complete
                return text
            self.last_analysis_truncated = False
            return cached
        inputs, _ = self._inputs([(first_rgb, first_rgb.new_zeros(len(first_rgb)))], body)
        text, complete = self._reason(inputs, budget, adapt=adapt, return_complete=True)
        if not complete and limit > budget:
            text, complete = self._reason(inputs, limit, adapt=adapt, return_complete=True)
        if not complete and not keep_partial:
            raise ValueError('Physical analysis hit its token limit without EOS; increase reasoning_max_tokens explicitly')
        if not text: raise ValueError('Physical analysis is empty')
        self.last_analysis_truncated = not complete
        if not complete:
            print('[Qwen analysis] token budget reached; retaining generated analysis and continuing training.', flush=True)
        self.reasoning_cache[key] = (text, complete) if keep_partial else text
        while len(self.reasoning_cache) > self.cfg['reasoning_cache']: self.reasoning_cache.popitem(last=False)
        return text

    def _encode_multiround(self, first_rgb, caption, clauses, *, duration=None, reasoning_round=None, all_rounds=False):
        analysis = (self._multiround_analysis(first_rgb, caption, clauses, duration, adapt=True)
                    if self.cfg.get('generation_reasoning', True) else '')
        body, spans = content_sections(caption, clauses, duration, analysis)
        inputs, ends = self._inputs([(first_rgb, first_rgb.new_zeros(len(first_rgb)))], body, extract_spans=spans)
        language = self.backbone.model.language_model
        limit = self.cfg['latent_reasoning_steps']
        if reasoning_round is None:
            exits = self.cfg.get('train_reasoning_rounds', [limit])
            train_exit = self.training and torch.is_grad_enabled() and any(p.requires_grad for p in self.reasoner.parameters())
            reasoning_round = exits[int(torch.randint(len(exits), ()).item())] if train_exit else self.cfg['inference_reasoning_round']
        if not isinstance(reasoning_round, int) or not 0 <= reasoning_round <= limit:
            raise ValueError('Requested reasoning exit is outside the trained/configured round range')
        with knowledge_role(True):
            with capture_prefix_projections(language) as captured:
                output = self.backbone.model(**inputs, use_cache=False, return_dict=True)
            memory = PrefixMemory.from_projections(language, captured, output.last_hidden_state,
                                                   inputs['position_ids'], inputs['attention_mask'])
            positions = sorted({p for segment in ends for p in segment})
            content = output.last_hidden_state[:, positions]
            if not positions: raise ValueError('Content readout cannot be empty')
            states = self.reasoner(language, memory, content.dtype, steps=limit if all_rounds else reasoning_round,
                                   checkpointing=self.cfg['gradient_checkpointing'])
        contextual = torch.cat((content, states[reasoning_round]), 1)
        return dict(contextual=contextual, content=content, relation_states=states,
                    content_positions=torch.tensor(positions, device=content.device),
                    attention_mask=torch.ones(contextual.shape[:2], dtype=torch.bool, device=content.device),
                    reasoning=analysis, latent_reasoning_steps=reasoning_round,
                    content_tokens=content.shape[1], relation_tokens=states[0].shape[1])

    def analysis_sft_loss(self, first_rgb, caption, clauses, target, *, duration=None):
        """Teacher-forced text loss for an explicit, audited training target.

        The normal generator never receives this target. No pseudo-labels are
        silently fabricated from held-out future frames or from the verifier.
        """
        if not self.multiround: raise ValueError('Reasoning supervision requires v14')
        if not target.strip(): raise ValueError('An explicit nonempty analysis target is required')
        if len(self.tokenizer.encode(target, add_special_tokens=False)) > self.cfg['reasoning_max_tokens']:
            raise ValueError('Analysis supervision exceeds the explicit output budget')
        body, _ = content_sections(caption, clauses, duration)
        inputs, _ = self._inputs([(first_rgb, first_rgb.new_zeros(len(first_rgb)))], body+'\n'+ANALYSIS_TEMPLATE,
            assistant_prefix=target+self.tokenizer.eos_token)
        # Assistant text is at the end; never locate its target by repeated words
        # in the user prompt. Train only target tokens and its terminal EOS.
        plain, _ = self._inputs([(first_rgb, first_rgb.new_zeros(len(first_rgb)))], body+'\n'+ANALYSIS_TEMPLATE)
        start = plain['input_ids'].shape[1]
        if not torch.equal(inputs['input_ids'][:, :start], plain['input_ids']):
            raise ValueError('Assistant boundary retokenized; cannot safely mask this target')
        with knowledge_role(True):
            hidden = self.backbone.model(**inputs, use_cache=False, return_dict=True).last_hidden_state
            logits = self.backbone.lm_head(hidden[:, start-1:-1])
        labels = inputs['input_ids'][:, start:]
        return F.cross_entropy(logits.float().flatten(0, 1), labels.flatten())

    def relation_sft_loss(self, state, question, answer):
        """Probe ONLY R_k plus a question; the raw context is not a shortcut.

        A training-only native language readout, not a second future predictor.
        Call separately on R0/R1/R2 using audited scene-dependent questions.
        """
        if not self.multiround or state.shape[0] != 1 or not answer.strip():
            raise ValueError('An explicit answer and one v14 relation state are required')
        prefix = self.tokenizer.apply_chat_template([{'role': 'user', 'content': question}],
                                                    tokenize=False, add_generation_prompt=True)
        question_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        answer_ids = self.tokenizer.encode(answer+self.tokenizer.eos_token, add_special_tokens=False)
        if len(question_ids)+len(answer_ids) > self.cfg['max_text_tokens']:
            raise ValueError('Relation supervision exceeds text budget')
        ids = torch.tensor([question_ids+answer_ids], device=state.device)
        language = self.backbone.model.language_model
        with knowledge_role(True):
            relation_input = self.reasoner.input_from(state, 1, language.embed_tokens.weight.dtype)
            embedded = torch.cat((relation_input, language.embed_tokens(ids)), 1)
            hidden = language(inputs_embeds=embedded, use_cache=False, return_dict=True).last_hidden_state
            start = state.shape[1]+len(question_ids)
            logits = self.backbone.lm_head(hidden[:, start-1:-1])
        return F.cross_entropy(logits.float().flatten(0, 1), ids[:, len(question_ids):].flatten())
