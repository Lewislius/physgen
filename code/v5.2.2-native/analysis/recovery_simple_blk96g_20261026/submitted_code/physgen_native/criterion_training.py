"""Isolated verifier calibration on existing train videos and current samples."""
import torch
from .execution_trace import traced_call


class CriterionTraining:
    def require_calibrated_verifier(self):
        if (getattr(self, 'criterion_verifier', False) and self.cfg['verifier']['adapter']['enabled']
                and not self.verifier_state['accepted']):
            raise RuntimeError('Verifier LoRA has no accepted calibration labels; inspect teacher evidence before feedback')

    def criterion_calibration(self, sample, step, micro, *, capture=False, null_text=None):
        cfg = self.cfg['verifier']
        if not cfg['adapter']['enabled'] or self.phase != 'calibration':
            raise ValueError('Verifier LoRA calibration requires its isolated training stage')
        iteration = step * self.cfg['train']['accumulation'] + micro
        generated = bool(iteration % 2)
        with torch.no_grad():
            if generated:
                first = self.decoder(sample['first'], [0])
                c = self.encode_condition(self.conditions(sample), first)
                latent = traced_call('judge_current_completion', self.completion, sample, c,
                    self._completion_sigma(sample['latent']), torch.randn_like(sample['latent']),
                    differentiable=False, null_text=null_text)
            else:
                latent = sample['latent']
            frames, times = self.decode_qwen_view(latent, sample)
            teacher, trace = traced_call('judge_observation_teacher', self.knowledge.criterion_teacher,
                                         frames, times, sample['caption'], sample['clauses'])
        # Frames and teacher are detached; only the separate adapter is trainable.
        student = self.knowledge.criterion_logits(frames.detach(), times, sample['caption'], sample['clauses'])
        loss, metrics = self.critic.calibration_loss(student, teacher,
                                                    observation_complete=trace['observation_complete'],
                                                    observation_usable=trace.get('observation_usable'))
        if torch.is_grad_enabled():
            self.verifier_state['accepted'] += int(metrics['teacher_accepted'])
            self.verifier_state['seen'] += int(metrics['teacher_total'])
        if capture:
            targets, accepted = self.critic.teacher_targets(teacher)
            accepted &= bool(trace.get('observation_usable', trace['observation_complete']))
            self.last_verifier_comparison = dict(frames=frames.detach().cpu(), metadata=dict(
                sample=sample['id'], caption=sample['caption'], clauses=sample['clauses'],
                candidate_kind='current_generated' if generated else 'cached_reference',
                layout='single observed video; neither kind is automatically correct',
                observation_size_hw=list(frames.shape[-2:]), geometry=self.cfg.get('geometry', {'mode': 'legacy'}),
                timestamps=times.detach().cpu().tolist(), **trace,
                teacher_probabilities=targets.cpu().tolist(), accepted=accepted.cpu().tolist(),
                student_probabilities=self.critic.probabilities(student).detach().mean(-2).cpu().tolist()))
        metrics.update(self.critic.diagnostics(student), verifier_active=1, qwen_verifier_active=1,
                       generator_frozen=1, generated_candidate=int(generated),
                       teacher_observation_complete=int(trace['observation_complete']))
        metrics['teacher_observation_usable'] = int(trace.get('observation_usable', trace['observation_complete']))
        return self._operator_report(dict(judge=loss), metrics)
