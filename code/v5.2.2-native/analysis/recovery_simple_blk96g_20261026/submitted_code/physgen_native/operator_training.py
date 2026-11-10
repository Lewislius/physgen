"""Three progressive stages with explicit verifier/generation gradient ownership.

The causal-response revision has 3 / 2 / 3 named objectives; legacy operator
revisions retain 4 / 2 / 4. Qwen-only v9 calibrates first, then uses 1 / 4 / 3.
Aligned native v13 uses interface / calibration / joint with 3 / 1 / 4 losses;
there is no intensity head, target or loss in that path.
v25-v28 use 2 / 1 / 3 objectives (no RGB process loss), full-width messages,
and the deployment CFG sampler for C-only verifier feedback.
Feedback differentiates a whole short conditional
denoising completion, not a detached suffix of a differently updated full
sampler. All 81 video frames remain present in every completion.
"""
import itertools
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .adjoint import discrete_adjoint
from .sequence import sequence_error, sequence_metrics, weighted_mean, temporal_flow_error
from .spatial import pool_grid
from .process_distribution import ordered_process_distribution
from .execution_trace import traced_call
from .verifier import rgb_times
from .criterion_training import CriterionTraining
from .geometry import uses_native_geometry, require_native_size
from .config import SIMPLE_MESSAGE_VARIANTS


class OperatorTraining(CriterionTraining):
    def set_operator_phase(self, phase):
        generation = phase != 'calibration'
        self.conditioner.requires_grad_(generation)
        self.generator.interface.requires_grad_(generation)
        # Legacy checkpoints retain a; the aligned full-width reader has no
        # auxiliary head, query modulation, or intensity loss.
        if hasattr(self.generator.interface, 'intensity'):
            self.generator.interface.intensity.requires_grad_(phase == 'interface')
        self.critic.requires_grad_(phase == 'calibration')
        self.generator.wan.requires_grad_(False)
        self.knowledge.requires_grad_(False)
        for name, p in self.generator.wan.named_parameters():
            if name.endswith(('lora_a', 'lora_b')):
                p.requires_grad_(phase == 'joint' or (phase == 'interface' and self.cfg['train'].get('wan_lora_in_interface', False)))
        for name, p in self.knowledge.named_parameters():
            if name.endswith(('verifier_lora_a', 'verifier_lora_b')):
                p.requires_grad_(phase == 'calibration')
            elif not self.message_conditioning and (name.endswith(('lora_a', 'lora_b')) or name.startswith('reasoner.')):
                p.requires_grad_(generation and self.cfg['knowledge'].get('condition_source', 'qwen') == 'qwen')
        self.decoder.requires_grad_(False)
        if self.process_encoder is not None: self.process_encoder.requires_grad_(False)
        self.phase = phase
        return phase

    def _operator_report(self, losses, metrics, phase=None):
        weights = {k: self.cfg['loss'][phase or self.phase][k] for k in losses}
        if (phase or self.phase) != 'calibration' and (getattr(self.knowledge, 'multiround', False) or self.message_conditioning) and hasattr(self, 'last_condition_round'):
            metrics.update(reasoning_round=self.last_condition_round, condition_tokens=self.last_condition_token_count)
        metrics.update({f'loss/{k}': v.detach() for k, v in losses.items()})
        if hasattr(self.knowledge, 'last_analysis_truncated'):
            metrics['analysis_truncated'] = int(self.knowledge.last_analysis_truncated)
        metrics.update({f'weighted/{k}': (weights[k]*v).detach() for k, v in losses.items()})
        metrics['loss'] = sum(weights[k]*v for k, v in losses.items()).detach()
        return losses, weights, metrics

    def _alternative_condition(self, sample):
        alternative = sample['alternative']
        # SAME first image and duration. No alternative video, latent, JEPA
        # target, or future state is ever encoded into either C.
        replacements = alternative['clauses'] or [dict(text=alternative['caption'])]
        clauses = []
        prompt_index = 0
        for clause in sample['clauses']:
            if clause.get('basis', 'prompt') == 'prompt':
                other = replacements[prompt_index % len(replacements)]
                clauses.append(dict(clause, text=other['text']))
                prompt_index += 1
            else:
                clauses.append(dict(clause))
        # Preserve clause count/provenance and the actual image observations;
        # otherwise "missing image clauses" becomes a condition-matching cue.
        result = dict(self.conditions(sample), caption=alternative['caption'], clauses=clauses)
        if self.cfg['knowledge'].get('condition_source', 'qwen') == 'cached_t5':
            from pathlib import Path
            from .data import load_tensor
            # Match the text-source intervention without touching the first
            # image, native Wan text, or any cached future evidence.
            result['native_text'] = load_tensor(Path(self.cfg['paths']['cache'])/'base/text'/
                f'{alternative["id"]}.pt').float()[None].to(sample['text'])
        return result

    def operator_forward(self, sample, null_text, step, *, micro=0, validation=False, sigma_override=None):
        self.set_phase(step)
        if self.phase == 'calibration' and not validation:
            return self.operator_calibration(sample, step, micro, null_text=null_text)
        train = self.cfg['train']
        native_only_probability = train.get('native_text_only_dropout', 0.)
        if train.get('dropout_policy') == 'exclusive':
            draw = float(torch.rand(())) if not validation else 1.
            dropped = draw < train['text_dropout']
            native_only = train['text_dropout'] <= draw < train['text_dropout'] + native_only_probability
            native_dropped = dropped or native_only
            # Native-only dropout must preserve the COMPLETE C condition.
            req_drop = not validation and not native_only and float(torch.rand(())) < train['requirements_dropout']
        else:
            dropped = not validation and float(torch.rand(())) < train['text_dropout']
            req_drop = not validation and float(torch.rand(())) < train['requirements_dropout']
            native_dropped = dropped
            if not validation and not dropped and native_only_probability:
                iteration = step*train['accumulation']+micro
                local = torch.Generator().manual_seed(train['seed']*1000003+iteration)
                native_dropped = float(torch.rand((), generator=local)) < native_only_probability
        condition = self.conditions(sample, dropped=dropped, requirements_dropped=req_drop)
        with torch.no_grad(): first_rgb = traced_call('first_frame_decode', self.decoder, sample['first'], [0])
        tokens = traced_call('positive_C', self.encode_condition, condition, first_rgb)
        z = sample['latent']
        sigma = z.new_tensor(sigma_override if sigma_override is not None else
                            train['sigma_min']+(train['sigma_max']-train['sigma_min'])*float(torch.rand(())))
        noise = torch.randn_like(z)
        noisy = torch.cat((sample['first'], ((1-sigma)*z+sigma*noise)[:, :, 1:]), 2)
        text = null_text if native_dropped else sample['text']
        size = self.cfg['sequence']['size']
        temporal = 'flow_temporal' in self.cfg
        intensity_active = self.phase == 'interface' and 'intensity' in self.cfg['loss']['interface']
        report_phase = 'joint' if self.phase == 'calibration' else self.phase
        process_enabled = 'process' in self.cfg['loss'][report_phase]
        rgb_active = process_enabled and (not temporal or float(sigma) <= self.cfg['sequence']['max_sigma'])
        truth = None
        if rgb_active or validation:
            with torch.no_grad(): truth = traced_call('reference_video_decode', self.decoder, z, resize=size)

        def observe(c, *, positive):
            if positive:
                result = traced_call('positive_Wan', self.generator, noisy, sample['first'], text, sigma, c,
                                     sample['times'], auxiliary=intensity_active)
                if intensity_active: velocity, intensity = result
                else: velocity = result
                fm = weighted_mean((velocity[:, :, 1:]-(noise-z)[:, :, 1:]).square().mean(1), sample['valid'])
            else:
                velocity = traced_call('alternate_Wan', self.generator, noisy, sample['first'], text, sigma, c, sample['times'])
                fm = velocity.new_zeros(())
            flow = temporal_flow_error(velocity, noise-z, sample['valid'], self.cfg['flow_temporal']['lags']) if temporal else fm*0
            process = velocity.sum()*0
            detail = dict(sequence_level=process.detach(), sequence_change=process.detach())
            video = velocity.new_empty(0)
            if rgb_active or validation:
                clean = torch.cat((sample['first'], (noisy-sigma*velocity)[:, :, 1:]), 2)
                video = traced_call('positive_video_decode' if positive else 'alternate_video_decode', self.decoder, clean, resize=size)
                if process_enabled:
                    raw_process, detail = sequence_error(video, truth, sample['valid'], self.cfg['sequence'],
                                                         native_geometry=uses_native_geometry(self.cfg))
                    if rgb_active: process = raw_process
            intensity_loss = process.new_zeros(())
            if positive and intensity_active:
                target, pv = pool_grid(sample['intensity'], sample['valid'], intensity.shape[-2:])
                iv, _ = pool_grid(sample['change_valid'], sample['valid'], intensity.shape[-2:])
                intensity_loss = weighted_mean(F.smooth_l1_loss(intensity.float(), target.detach(), reduction='none'), iv*pv)
            return fm, process, intensity_loss, detail['sequence_level'], detail['sequence_change'], video.detach(), flow

        def branch(c, positive):
            # Save C and scalar outputs, not two complete Wan/VAE histories.
            # Backward recomputes one arm at a time; both sides of the contrast
            # retain their exact derivative, including temporal VAE caches.
            if train.get('branch_checkpointing', True) and torch.is_grad_enabled():
                return checkpoint(lambda value: observe(value, positive=positive), c, use_reentrant=False)
            return observe(c, positive=positive)

        fm, process, intensity_loss, level, change, generated, flow = branch(tokens, True)
        metrics = dict(sequence_level=level, sequence_change=change) if process_enabled else {}
        positive_only = self.cfg['condition_use'].get('mode') == 'positive_only'
        losses = dict(fm=fm)
        if process_enabled: losses['process'] = process
        if temporal: losses['flow_temporal'] = flow
        if not positive_only: losses['use'] = process*0
        # Same native text, first frame, noise and diffusion time isolate C's
        # conditional contribution. This is correspondence training, not a
        # ground-truth counterfactual video or a physical-impossibility label.
        contrast = not positive_only and not validation and not dropped
        if contrast:
            alternative_condition = self._alternative_condition(sample)
            if req_drop: alternative_condition['clauses'] = []
            alternate = traced_call('alternate_C', self.encode_condition, alternative_condition, first_rgb)
            _, other_error, _, _, _, other_video, _ = branch(alternate, False)
            losses['use'] = F.softplus((process-other_error)/self.cfg['condition_use']['temperature'])
            metrics['condition/process_margin'] = (other_error-process).detach()
            metrics['condition/output_rms'] = (other_video.detach()-generated.detach()).square().mean().sqrt()
        if intensity_active:
            losses['intensity'] = intensity_loss
        metrics.update(sigma=sigma.detach(), sequence_active=int(rgb_active), verifier_active=0,
                       flow_temporal_active=int(temporal),
                       qwen_condition_active=int(self.cfg['knowledge'].get('condition_source', 'qwen') == 'qwen'),
                       native_text_dropped=int(native_dropped), C_text_dropped=int(dropped),
                       condition_use_active=int(contrast), wan_forward_count=1+int(contrast))
        if validation: metrics.update(sequence_metrics(generated.detach(), truth, sample['valid']))
        # During calibration validation we still measure the frozen generator;
        # the calibration-only loss schema must not be used for these metrics.
        return self._operator_report(losses, metrics, report_phase)

    def completion(self, sample, tokens, sigma, noise, *, differentiable=True, recompute=True,
                   generator_gradients=True, strength=1., null_text=None):
        """Differentiate every step of the configured connected trajectory.

        Legacy revisions start at an exogenous FM draw; v9 starts at pure noise.
        Neither detaches a parameter-dependent prefix. This is the derivative
        of the configured K-step conditional trajectory, not 50-step CFG sampling.
        """
        if self.cfg['revision'] in SIMPLE_MESSAGE_VARIANTS:
            if null_text is None:
                raise ValueError('CFG feedback requires the same cached null text as deployment')
            if float(sigma) != 1.:
                raise ValueError('Simple feedback starts from pure noise')
            c = self.cfg['completion']
            policy = dict(self.cfg['inference'], steps=c['steps'], shift=c['shift'], guidance=c['guidance'],
                          discrete_adjoint=bool(recompute), gradient_steps=c['steps'],
                          branch_checkpointing=c['branch_checkpointing'])
            # Reuse deployment's exact positive/negative condition construction,
            # Euler grid and CFG formula. Only the configured step count differs.
            with torch.set_grad_enabled(differentiable):
                return self._sample_tokens(self.conditions(sample), sample['text'], null_text, 0, policy,
                    differentiable=differentiable, strength=strength, initial_noise=noise,
                    token_override=tokens, generator_gradients=generator_gradients)
        z = sample['latent']
        if self.cfg['completion'].get('from_pure_noise'):
            if float(sigma) != 1.: raise ValueError('Pure-noise process feedback must start at sigma=1')
            initial = torch.cat((sample['first'], noise[:, :, 1:]), 2).detach()
        else:
            initial = torch.cat((sample['first'], ((1-sigma)*z+sigma*noise)[:, :, 1:]), 2).detach()
        count = self.cfg['completion']['steps']
        grid = torch.linspace(1, 0, count+1, device=z.device)
        shift = self.cfg['completion'].get('shift', 1.)
        times = (shift*grid/(1+(shift-1)*grid))*sigma
        def advance(state, c, unused, index):
            velocity = self.generator(state, sample['first'], sample['text'], times[index], c, sample['times'], strength)
            return torch.cat((sample['first'], (state+(times[index+1]-times[index])*velocity)[:, :, 1:]), 2)
        if differentiable and recompute:
            return discrete_adjoint(initial, tokens, tokens.detach(), advance, count,
                                    gradient_steps=count,
                                    parameters=self.generator.parameters() if generator_gradients else ())
        if differentiable and not generator_gradients:
            raise ValueError('Latent-only feedback uses explicit adjoint parameter ownership')
        state = initial
        with torch.set_grad_enabled(differentiable):
            for index in range(count): state = advance(state, tokens, tokens.detach(), index)
        return state

    def _completion_sigma(self, z):
        cfg = self.cfg['completion']
        return z.new_tensor(cfg['sigma_min']+(cfg['sigma_max']-cfg['sigma_min'])*float(torch.rand(())))

    def operator_calibration(self, sample, step, micro, *, null_text=None):
        if self.qwen_process:
            return self.qwen_calibration(sample, step, micro,
                capture=bool(self.cfg.get('smoke')) or step == 0 or
                        (self.criterion_verifier and micro == 0), null_text=null_text)
        cfg = self.cfg['verifier']
        with torch.no_grad():
            first_rgb = self.decoder(sample['first'], [0])
            tokens = self.encode_condition(self.conditions(sample), first_rgb)
            sigma = self._completion_sigma(sample['latent'])
            latent = self.completion(sample, tokens, sigma, torch.randn_like(sample['latent']), differentiable=False)
            generated = self.decoder(latent, resize=cfg['size'])
            truth = self.decoder(sample['latent'], resize=cfg['size'])
            # Omission strength is measured against this reference, not assigned
            # an unconditional "freezing is physically wrong" label.
            starts = (.25, .5, .75)
            start = max(1, int((len(truth)-1)*starts[(step*self.cfg['train']['accumulation']+micro) % len(starts)]))
            held = torch.cat((truth[:start], truth[start:start+1].expand(len(truth)-start, -1, -1, -1)), 0)
            features = [self.process_encoder(video) for video in (truth, generated, held)]
            duration = float(sample['times'][-1]-sample['times'][0])
            positive = self.knowledge.encode(first_rgb, sample['caption'], sample['clauses'], adapt=False, readout='content', duration=duration)['contextual']
            alternate = self._alternative_condition(sample)
            negative = self.knowledge.encode(first_rgb, alternate['caption'], alternate['clauses'], adapt=False, readout='content', duration=duration)['contextual']
            targets = [ordered_process_distribution(x, features[0], sample['valid'], lags=cfg['lags'],
                       scale_floor=cfg['scale_floor'], return_pairs=True)[2] for x in features]
        scores = [self.critic(x, positive, sample['times'], sample['valid']) for x in features]
        other = [self.critic(x, negative, sample['times'], sample['valid']) for x in features]
        rank = torch.stack([self.critic.preference_loss(scores[i], scores[j], targets[i], targets[j], cfg['temperature'])
                            for i, j in itertools.combinations(range(len(features)), 2)]).mean()
        matching = torch.stack([self.critic.matching_loss(scores[0], scores[i], other[0], other[i],
                                targets[0], targets[i], cfg['temperature']) for i in range(1, len(features))]).mean()
        return self._operator_report(dict(rank=rank, matching=matching), dict(
            completion_sigma=sigma, completion_steps=self.cfg['completion']['steps'],
            verifier_active=1, generator_frozen=1, generated_from_pure_noise=False,
            verifier_reference_score=torch.stack([s.mean() for s in scores[0].values()]).mean().detach(),
            verifier_generated_score=torch.stack([s.mean() for s in scores[1].values()]).mean().detach()))

    def should_feedback(self, step):
        if self.qwen_process:
            if 'verifier' not in self.cfg['loss'][self.phase]: return False
            start = (sum(self.cfg['stages'].get(p, 0) for p in ('interface','calibration'))
                     if 'flow_temporal' in self.cfg else self.cfg['stages']['calibration'])
            return self.phase in ('interface', 'joint') and (
                step-start) % self.cfg['completion']['every'] == 0
        return self.operator_training and self.phase == 'joint' and (
            step-sum(self.cfg['stages'][p] for p in ('interface', 'calibration'))) % self.cfg['completion']['every'] == 0

    def feedback_losses(self, sample, step, *, null_text=None):
        self.set_phase(step)
        self.require_calibrated_verifier()
        if self.phase != 'joint' and not (self.qwen_process and self.phase == 'interface'):
            raise ValueError('Frozen-verifier feedback requires a generator-learning phase')
        cfg = self.cfg['verifier']
        with torch.no_grad():
            first_rgb = self.decoder(sample['first'], [0])
            fixed = None if self.qwen_process else self.knowledge.encode(
                first_rgb, sample['caption'], sample['clauses'], adapt=False, readout='content',
                duration=float(sample['times'][-1]-sample['times'][0]))['contextual']
        tokens = self.encode_condition(self.conditions(sample), first_rgb)
        sigma = self._completion_sigma(sample['latent'])
        reader_gradients = self.cfg['completion'].get('feedback_to','joint') != 'latent'
        # For latent feedback the reader is differentiated with respect to C,
        # but is NOT an optimizer input to this objective. The generation
        # objectives still train Wan and the interface. The verifier must improve the knowledge representation
        # instead of satisfying its objective through a separate reader update.
        latent = self.completion(sample, tokens, sigma, torch.randn_like(sample['latent']),
                                 generator_gradients=reader_gradients, null_text=null_text)
        if self.qwen_process:
            frames, times = self.decode_qwen_view(latent, sample)
            features = self.qwen_frame_evidence(frames, times, sample)
            loss = self.critic.generation_loss(features)
        else:
            video = self.decoder(latent, resize=cfg['size'])
            features = self.process_encoder(video)
            loss = self.critic.generation_loss(features, fixed, sample['times'], sample['valid'])
        result = self._operator_report(dict(verifier=loss), dict(verifier_active=1, completion_sigma=sigma,
            completion_steps=self.cfg['completion']['steps'], gradient_steps=self.cfg['completion']['steps'],
            completion_guidance=self.cfg['completion'].get('guidance', 1.),
            generated_from_pure_noise=bool(self.cfg['completion'].get('from_pure_noise')),
            qwen_verifier_active=int(self.qwen_process),
            verifier_frozen=1, full_video_frames=1 + 4 * (latent.shape[2] - 1),
            feedback_updates_reader=reader_gradients))
        if self.qwen_process:
            result[2].update(observer_frames=len(frames), observer_height=frames.shape[-2],
                             observer_width=frames.shape[-1], observer_native_geometry=int(uses_native_geometry(self.cfg)))
        if self.criterion_verifier:
            result[2].update(self.critic.diagnostics(features))
        if tokens.requires_grad:
            # Populated when the actual weighted loss is backpropagated, rather
            # than checking an unrelated probe derivative in a second pass.
            def record(gradient):
                result[2]['C_gradient_rms'] = gradient.detach().float().square().mean().sqrt()
            tokens.register_hook(record)
        return result

    def qwen_video_view(self, video, sample):
        ids = self.qwen_frame_indices(len(video))
        return video[ids], rgb_times(ids, sample['times'])

    def qwen_frame_indices(self, total):
        stride = self.cfg['verifier'].get('frame_stride')
        if stride is None:
            return torch.linspace(0, total-1, min(total, self.cfg['verifier']['frames'])).round().long().tolist()
        ids = list(range(0, total, stride))
        if ids[-1] != total - 1: ids.append(total - 1)
        return ids

    def decode_qwen_view(self, latent, sample):
        if not uses_native_geometry(self.cfg):
            video = self.decoder(latent, resize=self.cfg['verifier']['size'])
            return self.qwen_video_view(video, sample)
        # The native VAE still traverses all preceding chunks and caches. Only
        # selected decoded RGB outputs are retained, with no spatial resampling.
        total = 1 + 4 * (latent.shape[2] - 1)
        ids = self.qwen_frame_indices(total)
        frames = self.decoder(latent, frame_indices=ids)
        require_native_size(frames.shape[-2:], self.cfg['geometry']['rgb_size'])
        return frames, rgb_times(ids, sample['times'])

    def qwen_video_evidence(self, video, sample):
        frames, times = self.qwen_video_view(video, sample)
        return self.qwen_frame_evidence(frames, times, sample)

    def qwen_frame_evidence(self, frames, times, sample):
        if self.criterion_verifier:
            return self.knowledge.criterion_logits(frames, times, sample['caption'], sample['clauses'])
        if self.native_verifier:
            return self.knowledge.verifier_logits(frames, times, sample['caption'], sample['clauses'])
        # knowledge.evidence disables the generation adapter, while preserving
        # the RGB derivative through Qwen's visual AND language transformers.
        return self.knowledge.evidence(frames, times, sample['caption'], sample['clauses'])

    def qwen_calibration(self, sample, step, micro, *, capture=False, null_text=None):
        if self.criterion_verifier:
            return self.criterion_calibration(sample, step, micro, capture=capture, null_text=null_text)
        iteration = step*self.cfg['train']['accumulation']+micro
        with torch.no_grad():
            truth = traced_call('judge_reference_decode', self.decoder, sample['latent'], resize=self.cfg['verifier']['size'])
            if iteration % 2 == 0:
                # Calibrate on native Wan before changing the generation branch.
                # No random C intervention and no reference future in the draw.
                c = sample['first'].new_zeros(1, 1, self.cfg['model']['width'])
                current = self.cfg['verifier'].get('calibrate_current', False)
                if current: c = self.encode_condition(self.conditions(sample), truth[:1])
                latent = traced_call('judge_native_completion', self.completion, sample, c, self._completion_sigma(sample['latent']),
                    torch.randn_like(sample['latent']), differentiable=False, strength=1. if current else 0.)
                candidate = traced_call('judge_candidate_decode', self.decoder, latent, resize=self.cfg['verifier']['size'])
            else:
                fraction = (.25, .5, .75)[(iteration//2) % 3]
                start = max(1, int((len(truth)-1)*fraction))
                candidate = torch.cat((truth[:start], truth[start:start+1].expand(len(truth)-start, -1, -1, -1)), 0)
            reference_features = traced_call('judge_reference_evidence', self.qwen_video_evidence, truth, sample)
            candidate_features = traced_call('judge_candidate_evidence', self.qwen_video_evidence, candidate, sample)
            first, times = self.qwen_video_view(truth, sample)
            second, _ = self.qwen_video_view(candidate, sample)
            # Reverse independently of candidate type to avoid a reference-first
            # or generated/held position cue. Uncertain teacher answers stay soft.
            reverse = (iteration//2) % 2 == 1
            preference, probabilities = traced_call('judge_preference', self.knowledge.preference,
                second if reverse else first, first if reverse else second,
                times, sample['caption'], sample['clauses'])
            if reverse: preference = 1-preference
            if capture:
                # Retain the already computed training evidence, not another
                # generated candidate or a separate evaluation trajectory.
                frames = torch.cat((first, second), -1)
                self.last_verifier_comparison = dict(
                    frames=frames.detach().cpu(),
                    metadata=dict(sample=sample['id'], caption=sample['caption'], clauses=sample['clauses'],
                        candidate_kind='generated' if iteration % 2 == 0 else 'late_hold',
                        reference_preference=float(preference.mean()),
                        teacher_video1='candidate' if reverse else 'reference',
                        layout='reference left; candidate right', timestamps=times.detach().cpu().tolist(),
                        independent_observations=getattr(self.knowledge, 'last_comparison_observations', []),
                        reasoning=getattr(self.knowledge, 'last_comparison_reasoning', '')))
        loss = self.critic.preference_loss(reference_features, candidate_features, preference)
        reference_score = self.critic(reference_features).detach().mean()
        candidate_score = self.critic(candidate_features).detach().mean()
        predicted = (reference_score-candidate_score).sigmoid()
        return self._operator_report(dict(rank=loss), dict(
            verifier_active=1, qwen_verifier_active=1, generator_frozen=1,
            generated_candidate=int(iteration % 2 == 0),
            teacher_reference_preference=preference.mean(), teacher_tie=probabilities[:, 2].mean(),
            predicted_reference_preference=predicted,
            teacher_order_gap=getattr(self.knowledge, 'last_comparison_order_gap', preference.new_zeros(())).mean(),
            teacher_confidence=(2*preference.mean()-1).abs(),
            preference_brier=(predicted-preference.mean()).square(),
            reference_score=reference_score, candidate_score=candidate_score))
