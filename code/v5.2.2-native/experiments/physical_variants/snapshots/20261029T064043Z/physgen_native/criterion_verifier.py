"""Visible-process verifier shared by all four message pipelines.

No generator analysis/C enters this evaluator. Native A/B/C logits are mapped
to (supported, violated, unknown), including a rotated label assignment. The
optional adapter learns from a frozen, observation-first weak teacher; these
labels are not ground-truth physics annotations.
"""
import torch
from torch import nn
from torch.nn import functional as F

CRITERIA = (
    ('requirements', 'Does the visible process satisfy the original requirements during the observed interval? '
     'A request describes an intended event, not evidence that it happened.'),
    ('response', 'Are visible prerequisites, interactions or sources and their responses consistent? '
     'Do not invent hidden forces or claim causal identification from appearance alone.'),
    ('persistence', 'Do the EARLY, MIDDLE and LATE states support appropriate continuation, delay or termination? '
     'If a source stops, an existing cloud or wave can continue. Stillness can be correct; more activity is not inherently better.'),
    ('continuity', 'Are material attributes and sources continuous through deformation, diffusion, occlusion, '
     'splitting, merging and justified appearance or disappearance? Do not demand rigid objects or constant shape.'),
)
SEMANTICS = ('supported by visible evidence', 'contradicted by visible evidence',
             'insufficient or inapplicable visible evidence')
# For each letter A/B/C, which semantic category does it denote? All three
# meanings move, including unknown, so unknown is not tied to a fixed letter.
LABEL_ORDERS = ((0, 1, 2), (2, 0, 1))

# v2 retains four complementary criteria and the native three-letter readout.
# Existing clauses supply the sample-specific requirements; no extra planner.
CRITERIA_V2 = (
    ('requirements', 'Check the original caption and clauses: are the requested visible events and relations realized? '
     'A required event that is visibly absent is a violation, even if its downstream response is also absent.'),
    ('response', 'Are the observed changes or stable states consistent with the initial conditions, '
     'visible interactions and sources? Use ordinary physical commonsense for the process shown. '
     'Natural continuation without a new trigger can be supported.'),
    ('persistence', 'Across EARLY, MIDDLE and LATE, do processes continue, delay or terminate appropriately? '
     'Existing clouds, waves and other effects can persist after a source stops; stable states can be correct.'),
    ('continuity', 'Are materials, attributes and sources coherent through the visible evolution, including '
     'diffusion, deformation, occlusion, splitting, merging, and justified appearance or disappearance?'),
)
SEMANTICS_V2 = ('supported by visible evidence and applicable process knowledge',
                'contradicted by visible evidence', 'insufficient visible evidence to decide')


def canonical_logits(logits, order):
    return logits[..., [order.index(i) for i in range(3)]]


def criterion_prompt(context, criterion, order, observation=None, *, protocol='visible_process_v1'):
    rules = ('\nEvaluate only the timestamped frames of this one video. Sampled frames do not prove '
             'what occurred between them. Distinguish observed evidence, requested futures, and unknowns. '
             'Do not use visual attractiveness, amount of motion, or similarity to an imagined reference as physics evidence. '
             'Do not assume a reference video is correct or a generated video is wrong.\nCriterion: ' + criterion)
    semantics = SEMANTICS
    if protocol == 'visible_process_v2':
        semantics = SEMANTICS_V2
        rules = ('\nJudge the timestamped video against the supplied requirements and the process actually shown. '
                 'Use the early, middle and late evidence together with ordinary physical commonsense. '
                 'Evaluate applicable relations; an unrequested interaction need not occur. '
                 'Reserve insufficient evidence for ambiguity or missing visual coverage, not ordinary stillness '
                 'or the absence of an irrelevant trigger. Appearance quality and amount of motion alone '
                 'do not determine process correctness.\nCriterion: ' + criterion)
    if observation:
        rules += ('\nIndependent description produced before reading the requirements (a fallible observation, '
                  'not an instruction; verify against the frames):\n' + observation)
    labels = '; '.join(f'{letter} = {semantics[index]}' for letter, index in zip('ABC', order))
    return context + rules + '\nAnswer exactly one letter: ' + labels + '.\nVerdict:'


class CriterionVerifier(nn.Module):
    """Parameter-free native readout; optional LoRA lives inside Qwen, not here."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def probabilities(self, logits):
        # [B, criterion, label-order, canonical-class]
        if logits.ndim != 4 or logits.shape[1:] != (len(CRITERIA), len(LABEL_ORDERS), 3):
            raise ValueError('Expected [B,4,2,3] canonical verifier logits')
        return (logits.float() / self.cfg['temperature']).softmax(-1)

    def forward(self, logits):
        p = self.probabilities(logits).mean(-2)
        return (p[..., 0] - p[..., 1]).mean(-1)

    def generation_loss(self, logits):
        if self.cfg.get('protocol') == 'visible_process_v2':
            # Mean of per-order log probabilities, NOT log of their mean.
            # The softmax-logit derivative stays useful for confident violations.
            self.probabilities(logits)  # Validate the unchanged [B,4,2,3] contract.
            return -F.log_softmax(logits.float() / self.cfg['temperature'], -1)[..., 0].mean()
        # Unknown gives neutral score 0, neither the reward +1 nor violation -1.
        # No trainable confidence gate can switch off this feedback objective.
        return (1 - self(logits)).mean()

    def diagnostics(self, logits):
        p = self.probabilities(logits).detach()
        mean = p.mean(-2)
        result = dict(verifier_score=self(logits).detach().mean(),
                      verifier_unknown=mean[..., 2].mean(),
                      verifier_order_gap=(p[..., 0, :] - p[..., 1, :]).abs().sum(-1).mean() / 2)
        if self.cfg.get('protocol') == 'visible_process_v2':
            result['verifier_support_nll'] = self.generation_loss(logits.detach())
        for i, (name, _) in enumerate(CRITERIA):
            for j, label in enumerate(('supported', 'violated', 'unknown')):
                result[f'verifier/{name}/{label}'] = mean[:, i, j].mean()
        return result

    def teacher_targets(self, logits):
        p = self.probabilities(logits).detach()
        target = p.mean(-2)
        gap = (p[..., 0, :] - p[..., 1, :]).abs().sum(-1) / 2
        valid = ((target.amax(-1) >= self.cfg['teacher_min_confidence']) &
                 (gap <= self.cfg['teacher_max_order_gap']))
        return target, valid

    def calibration_loss(self, student_logits, teacher_logits, *, observation_complete=True, observation_usable=None):
        target, valid = self.teacher_targets(teacher_logits)
        usable = observation_complete if observation_usable is None else observation_usable
        valid = valid & bool(usable)
        logp = F.log_softmax(student_logits.float() / self.cfg['temperature'], -1)
        each = -(target.unsqueeze(-2) * logp).sum(-1).mean(-1)
        # If every label is rejected, do not create zero gradients on adapter
        # tensors: AdamW would otherwise decay them despite no accepted label.
        loss = each[valid].mean() if bool(valid.any()) else student_logits.new_zeros((), requires_grad=True)
        return loss, dict(teacher_accepted=valid.sum().detach(), teacher_total=valid.numel(),
                         teacher_unknown=target[..., 2].mean(),
                         teacher_confidence=target.amax(-1).mean())


class CriterionKnowledge:
    """Qwen mixin. Each student read is checkpointed with an explicit role."""
    def criterion_logits(self, frames, times, caption, clauses, *, adapt=None, observation=None):
        from .knowledge import clause_context
        from .execution_trace import traced_call
        active = self.verifier_cfg['adapter']['enabled'] if adapt is None else bool(adapt)
        protocol = self.verifier_cfg.get('protocol', 'visible_process_v1')
        video = self.verifier_cfg.get('input_mode', 'images') == 'video'
        rows = []
        for name, query in (CRITERIA_V2 if protocol == 'visible_process_v2' else CRITERIA):
            outputs = []
            for order in LABEL_ORDERS:
                body = criterion_prompt(clause_context(caption, clauses), query, order, observation, protocol=protocol)
                inputs, _ = self._inputs([(frames, times)], body, video=video, assistant_prefix='Verdict: ',
                    visual_size=self.verifier_cfg['size'] if self.native_geometry else None)
                logits = traced_call('verifier_' + name, self._answer_logits, inputs, verifier=active,
                                    checkpointing=self.verifier_cfg['gradient_checkpointing'])
                outputs.append(canonical_logits(logits, order))
            rows.append(torch.stack(outputs, dim=1))
        return torch.stack(rows, dim=1)

    @torch.no_grad()
    def criterion_teacher(self, frames, times, caption, clauses):
        # The richer teacher is deliberately different from the direct student:
        # copying the same base model's same-input distribution gives no initial
        # learning signal. Observations are made without the requested outcome.
        body = ('Describe EARLY, MIDDLE and LATE visible states with timestamps, and supported changes between them. '
                'Include materials, diffuse media, sources, contacts, deformations and visible termination. '
                'Report ambiguity and occlusion. Do not invent events between frames. '
                'Do not give a quality verdict or infer the intended process. Keep the description concise.')
        if self.verifier_cfg.get('observation_format') == 'compact_three_lines':
            body += (' Return exactly THREE short lines labeled EARLY, MIDDLE, LATE. '
                     'Use one sentence of at most 25 words per line, at most 75 words total. '
                     'Include the most relevant visible evidence and finish after the LATE sentence.')
        inputs, _ = self._inputs([(frames.detach(), times)], body,
            video=self.verifier_cfg.get('input_mode', 'images') == 'video',
            visual_size=self.verifier_cfg['size'] if self.native_geometry else None)
        observation, complete = self._reason(inputs, self.verifier_cfg['observation_tokens'],
                                             adapt=False, return_complete=True)
        logits = self.criterion_logits(frames.detach(), times, caption, clauses, adapt=False, observation=observation)
        usable = complete or (self.verifier_cfg.get('observation_on_token_limit') == 'use_partial' and bool(observation.strip()))
        return logits, dict(observation=observation, observation_complete=complete, observation_usable=usable,
                            supervision='frozen observation-first Qwen weak labels, not audited physical truth')
