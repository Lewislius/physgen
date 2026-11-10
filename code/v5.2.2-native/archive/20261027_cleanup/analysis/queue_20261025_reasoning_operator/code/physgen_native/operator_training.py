"""Three stages: ground C, fit a conditional verifier, jointly improve generation.

There are 4 / 2 / 4 explicitly named objectives. Feedback differentiates a whole
short conditional denoising completion, not a detached suffix of a differently
updated full sampler. All 81 video frames remain present in every completion.
"""
import itertools
import torch
from torch.nn import functional as F
from .adjoint import discrete_adjoint
from .sequence import sequence_error, sequence_metrics, weighted_mean
from .spatial import pool_grid
from .process_distribution import ordered_process_distribution


class OperatorTraining:
    def set_operator_phase(self, phase):
        generation = phase != 'calibration'
        self.conditioner.requires_grad_(generation)
        self.generator.interface.requires_grad_(generation)
        # Intensity is calibrated with existing RGB-change targets in stage 1.
        # Later its effect on generation remains trainable through the operator
        # and gate, without sneaking a fifth auxiliary loss into joint training.
        self.generator.interface.intensity.requires_grad_(phase == 'interface')
        self.critic.requires_grad_(phase == 'calibration')
        self.generator.wan.requires_grad_(False)
        self.knowledge.requires_grad_(False)
        for name, p in self.generator.wan.named_parameters():
            if name.endswith(('lora_a', 'lora_b')): p.requires_grad_(phase == 'joint')
        for name, p in self.knowledge.named_parameters():
            if name.endswith(('lora_a', 'lora_b')): p.requires_grad_(generation)
        self.decoder.requires_grad_(False)
        self.process_encoder.requires_grad_(False)
        self.phase = phase
        return phase

    def _operator_report(self, losses, metrics, phase=None):
        weights = {k: self.cfg['loss'][phase or self.phase][k] for k in losses}
        metrics.update({f'loss/{k}': v.detach() for k, v in losses.items()})
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
        return dict(self.conditions(sample), caption=alternative['caption'], clauses=clauses)

    def operator_forward(self, sample, null_text, step, *, micro=0, validation=False, sigma_override=None):
        self.set_phase(step)
        if self.phase == 'calibration' and not validation:
            return self.operator_calibration(sample, step, micro)
        train = self.cfg['train']
        dropped = not validation and float(torch.rand(())) < train['text_dropout']
        req_drop = not validation and float(torch.rand(())) < train['requirements_dropout']
        condition = self.conditions(sample, dropped=dropped, requirements_dropped=req_drop)
        with torch.no_grad(): first_rgb = self.decoder(sample['first'], [0])
        tokens = self.encode_condition(condition, first_rgb)
        z = sample['latent']
        sigma = z.new_tensor(sigma_override if sigma_override is not None else
                            train['sigma_min']+(train['sigma_max']-train['sigma_min'])*float(torch.rand(())))
        noise = torch.randn_like(z)
        noisy = torch.cat((sample['first'], ((1-sigma)*z+sigma*noise)[:, :, 1:]), 2)
        text = null_text if dropped else sample['text']
        velocity, intensity = self.generator(noisy, sample['first'], text, sigma, tokens, sample['times'], auxiliary=True)
        fm = weighted_mean((velocity[:, :, 1:]-(noise-z)[:, :, 1:]).square().mean(1), sample['valid'])
        size = self.cfg['sequence']['size']
        with torch.no_grad(): truth = self.decoder(z, resize=size)
        clean = torch.cat((sample['first'], (noisy-sigma*velocity)[:, :, 1:]), 2)
        generated = self.decoder(clean, resize=size)
        process, metrics = sequence_error(generated, truth, sample['valid'], self.cfg['sequence'])
        losses = dict(fm=fm, process=process, use=velocity.sum()*0)
        # Same native text, first frame, noise and diffusion time isolate C's
        # conditional contribution. This is correspondence training, not a
        # ground-truth counterfactual video or a physical-impossibility label.
        if not validation and not dropped:
            alternative_condition = self._alternative_condition(sample)
            if req_drop: alternative_condition['clauses'] = []
            alternate = self.encode_condition(alternative_condition, first_rgb)
            other_v = self.generator(noisy, sample['first'], text, sigma, alternate, sample['times'])
            other_clean = torch.cat((sample['first'], (noisy-sigma*other_v)[:, :, 1:]), 2)
            other_video = self.decoder(other_clean, resize=size)
            other_error, _ = sequence_error(other_video, truth, sample['valid'], self.cfg['sequence'])
            losses['use'] = F.softplus((process-other_error)/self.cfg['condition_use']['temperature'])
            metrics['condition/process_margin'] = (other_error-process).detach()
            metrics['condition/output_rms'] = (other_video.detach()-generated.detach()).square().mean().sqrt()
        if self.phase == 'interface':
            target, pv = pool_grid(sample['intensity'], sample['valid'], intensity.shape[-2:])
            iv, _ = pool_grid(sample['change_valid'], sample['valid'], intensity.shape[-2:])
            losses['intensity'] = weighted_mean(F.smooth_l1_loss(intensity.float(), target.detach(), reduction='none'), iv*pv)
        metrics.update(sigma=sigma.detach(), sequence_active=1, verifier_active=0,
                       condition_use_active=int(not validation and not dropped),
                       wan_forward_count=1+int(not validation and not dropped))
        if validation: metrics.update(sequence_metrics(generated.detach(), truth, sample['valid']))
        # During calibration validation we still measure the frozen generator;
        # the calibration-only loss schema must not be used for these metrics.
        report_phase = 'joint' if self.phase == 'calibration' else self.phase
        return self._operator_report(losses, metrics, report_phase)

    def completion(self, sample, tokens, sigma, noise, *, differentiable=True, recompute=True,
                   generator_gradients=True):
        """Exactly differentiate K connected denoising steps from a sampled FM state.

        z_sigma is an exogenous training draw, independent of model parameters.
        There is no unaccounted model-generated prefix. This objective does NOT
        estimate the gradient of 50-step pure-noise inference. Completion uses
        unguided conditional flow, as does FM training; inference keeps native CFG.
        """
        z = sample['latent']
        initial = torch.cat((sample['first'], ((1-sigma)*z+sigma*noise)[:, :, 1:]), 2).detach()
        count = self.cfg['completion']['steps']
        times = torch.linspace(1, 0, count+1, device=z.device)*sigma
        def advance(state, c, unused, index):
            velocity = self.generator(state, sample['first'], sample['text'], times[index], c, sample['times'])
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

    def operator_calibration(self, sample, step, micro):
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
        return self.operator_training and self.phase == 'joint' and (
            step-sum(self.cfg['stages'][p] for p in ('interface', 'calibration'))) % self.cfg['completion']['every'] == 0

    def feedback_losses(self, sample, step):
        self.set_phase(step)
        if self.phase != 'joint': raise ValueError('Frozen-verifier feedback belongs to joint training')
        cfg = self.cfg['verifier']
        with torch.no_grad():
            first_rgb = self.decoder(sample['first'], [0])
            fixed = self.knowledge.encode(first_rgb, sample['caption'], sample['clauses'], adapt=False, readout='content',
                                          duration=float(sample['times'][-1]-sample['times'][0]))['contextual']
        tokens = self.encode_condition(self.conditions(sample), first_rgb)
        sigma = self._completion_sigma(sample['latent'])
        reader_gradients = self.cfg['completion'].get('feedback_to','joint') != 'latent'
        # In v7 the reader is differentiated with respect to C, but is NOT an
        # optimizer input to this objective. FM/process/use still train Wan and
        # the interface. The verifier must improve the knowledge representation
        # instead of satisfying its objective through a separate reader update.
        latent = self.completion(sample, tokens, sigma, torch.randn_like(sample['latent']),
                                 generator_gradients=reader_gradients)
        video = self.decoder(latent, resize=cfg['size'])
        features = self.process_encoder(video)
        loss = self.critic.generation_loss(features, fixed, sample['times'], sample['valid'])
        result = self._operator_report(dict(verifier=loss), dict(verifier_active=1, completion_sigma=sigma,
            completion_steps=self.cfg['completion']['steps'], gradient_steps=self.cfg['completion']['steps'],
            generated_from_pure_noise=False, verifier_frozen=1, full_video_frames=len(video),
            feedback_updates_reader=reader_gradients))
        if tokens.requires_grad:
            # Populated when the actual weighted loss is backpropagated, rather
            # than checking an unrelated probe derivative in a second pass.
            def record(gradient):
                result[2]['C_gradient_rms'] = gradient.detach().float().square().mean().sqrt()
            tokens.register_hook(record)
        return result
