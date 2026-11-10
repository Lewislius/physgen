"""Joint knowledge injection, complete-process supervision and sampled feedback."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .config import phase_at, OPERATOR_REVISIONS
from .conditioning import KnowledgeCondition
from .generator import ConditionedWan
from .knowledge import QwenKnowledge
from .decoder import VideoDecoder
from .sequence import weighted_mean, sequence_error, sequence_metrics, perceptual_process_error
from .spatial import pool_grid
from .verifier import ProcessCritic, evidence_indices, rgb_times, omitted_process
from .temporal_critic import TemporalProcessCritic
from .process_distribution import ordered_process_distribution
from .evidence_verifier import ProcessEvidenceVerifier
from .conditional_verifier import ConditionalProcessVerifier
from .operator_training import OperatorTraining


def shifted_sigma(value, shift): return shift*value/(1+(shift-1)*value)


class PhysGenNative(OperatorTraining, nn.Module):
    def __init__(self, cfg, wan, knowledge=None, decoder=None, device='cuda', process_encoder=None, *, training_observers=True):
        super().__init__()
        self.cfg = cfg
        self.operator_training = cfg['revision'] in OPERATOR_REVISIONS
        self.training_observers = training_observers
        self.conditioner = KnowledgeCondition(cfg)
        self.generator = ConditionedWan(wan, cfg)
        self.knowledge = knowledge if knowledge is not None else QwenKnowledge(cfg, device)
        self.decoder = decoder if decoder is not None else VideoDecoder(cfg, device)
        self.temporal_verifier = cfg['verifier'].get('kind') == 'temporal_distribution'
        self.ordered_process = cfg['verifier'].get('kind') == 'ordered_process'
        self.calibrated_verifier = cfg['verifier'].get('kind') == 'calibrated_process'
        self.process_encoder = None
        if training_observers and (self.ordered_process or self.calibrated_verifier or self.operator_training):
            if process_encoder is None:
                from .process_encoder import VideoProcessEncoder
                process_encoder=VideoProcessEncoder(cfg['paths']['video_evidence_code'],cfg['paths']['video_evidence_checkpoint'],
                    size=cfg['verifier']['size'],device=device,checkpointing=cfg['train']['gradient_checkpointing'])
            self.process_encoder=process_encoder
        if not training_observers: self.critic = nn.Identity()
        elif self.operator_training:
            self.critic=ConditionalProcessVerifier(cfg['verifier']['feature_width'],cfg['model']['qwen_width'],
                cfg['verifier']['width'],lags=cfg['verifier']['lags'],
                response_only=cfg['verifier'].get('response_only',False))
        elif self.calibrated_verifier:
            self.critic=ProcessEvidenceVerifier(cfg['verifier']['feature_width'],cfg['model']['qwen_width'],
                cfg['verifier']['width'],lags=cfg['verifier']['lags'])
        elif self.ordered_process:self.critic=nn.Identity()
        elif self.temporal_verifier:
            self.critic=TemporalProcessCritic(cfg['verifier'].get('feature_width',cfg['model']['qwen_width']),
                                            cfg['model']['qwen_width'],cfg['verifier'].get('width',128))
        else:self.critic=ProcessCritic(cfg)
        self.set_phase(0)

    def set_phase(self, step):
        phase = phase_at(self.cfg, step)
        if not self.training_observers:
            self.requires_grad_(False)
            self.phase = phase
            return phase
        if self.operator_training: return self.set_operator_phase(phase)
        self.conditioner.requires_grad_(True)
        self.generator.interface.requires_grad_(True)
        self.critic.requires_grad_(True)
        self.generator.wan.requires_grad_(False)
        self.knowledge.requires_grad_(False)
        for name, p in self.generator.wan.named_parameters():
            if name.endswith(('lora_a', 'lora_b')): p.requires_grad_(phase == 'joint')
        for name, p in self.knowledge.named_parameters():
            if name.endswith(('lora_a', 'lora_b')): p.requires_grad_(True)
        self.decoder.requires_grad_(False)
        if self.process_encoder is not None:self.process_encoder.requires_grad_(False)
        self.phase = phase
        return phase

    def conditions(self, sample, *, dropped=False, requirements_dropped=False):
        result = dict(first=sample['first'], times=sample['times'], valid=sample['valid'],
                      caption='' if dropped else sample['caption'],
                      clauses=[] if dropped or requirements_dropped else sample['clauses'])
        if self.cfg['knowledge'].get('condition_source', 'qwen') == 'cached_t5':
            result['native_text'] = sample['text']
        return result

    def encode_condition(self, conditions, first_rgb=None):
        if self.cfg['knowledge'].get('condition_source', 'qwen') == 'cached_t5':
            # The naive ablation uses only existing frozen T5 text features.
            # It shares the learned C projection and Wan state operator with
            # the main model, but executes no Qwen generation-side encoder.
            content = conditions['native_text'].detach()
            if not conditions['caption']:
                content = content.new_zeros(content.shape[0], 1, content.shape[-1])
            maximum = self.cfg['knowledge'].get('max_condition_tokens', 128)
            if content.shape[1] > maximum:
                content = F.adaptive_avg_pool1d(content.transpose(1, 2), maximum).transpose(1, 2)
            self.last_condition_analysis = 'Ablation: cached T5 text mapped to C; no Qwen generation condition.'
            return self.conditioner(dict(contextual=content,
                roles=torch.ones(content.shape[1], device=content.device, dtype=torch.long)))
        if first_rgb is None:
            with torch.no_grad(): first_rgb = self.decoder(conditions['first'], [0])
        options = dict(duration=float(conditions['times'][-1]-conditions['times'][0])) if self.operator_training else {}
        evidence = self.knowledge.encode(first_rgb, conditions['caption'], conditions['clauses'], **options)
        self.last_condition_analysis = evidence.get('reasoning','')
        return self.conditioner(evidence)

    def _verifier(self, generated, truth, sample, condition, iteration, *, alternative=None):
        ids = evidence_indices(sample['times'], self.cfg['knowledge']['evidence_pairs'])
        times = rgb_times(ids, sample['times'])
        caption, clauses = condition['caption'], condition['clauses']
        with torch.no_grad(): reference = self.knowledge.evidence(truth[ids], times, caption, clauses)
        evidence = self.knowledge.evidence(generated[ids], times, caption, clauses)
        if alternative is None:
            alternative = (omitted_process(truth, sample, self.cfg['verifier'], iteration)
                           if iteration % 2 == 0 else generated.detach())
        with torch.no_grad():
            if alternative is generated or alternative.data_ptr() == generated.data_ptr():
                alt_evidence = evidence.detach()
            else:
                alt_evidence = self.knowledge.evidence(alternative[ids], times, caption, clauses)
            forward, fp = self.knowledge.preference(truth[ids], alternative[ids], times, caption, clauses)
            reverse, rp = self.knowledge.preference(alternative[ids], truth[ids], times, caption, clauses)
            preference = (forward+1-reverse)/2
            probabilities = (fp+rp)/2
        calibration, metrics = self.critic.calibration_loss(reference, alt_evidence, preference)
        physical, scores = self.critic.generation_loss(evidence, reference)
        metrics.update(scores, teacher_tie=probabilities[:, 2].mean(),
                       teacher_order_gap=(forward+reverse-1).abs().mean(),
                       evidence_difference=(evidence.detach()-reference).square().mean())
        return calibration, physical, metrics

    def forward(self, sample, null_text, step, *, micro=0, validation=False, sigma_override=None):
        if self.operator_training:
            return self.operator_forward(sample, null_text, step, micro=micro, validation=validation, sigma_override=sigma_override)
        self.set_phase(step)
        train = self.cfg['train']
        dropped = not validation and float(torch.rand(())) < train['text_dropout']
        req_drop = not validation and float(torch.rand(())) < train['requirements_dropout']
        iteration = step*train['accumulation']+micro
        native_dropped = dropped
        native_only_probability = train.get('native_text_only_dropout', 0.)
        if not validation and not dropped and native_only_probability:
            # An independent local RNG leaves matched FM noise/timesteps intact
            # across the reliance control and its baseline. Qwen keeps the legal
            # caption here, so C must sometimes carry information missing in T5.
            local = torch.Generator().manual_seed(train['seed']*1000003+iteration)
            native_dropped = float(torch.rand((),generator=local)) < native_only_probability
        condition = self.conditions(sample, dropped=dropped, requirements_dropped=req_drop)
        with torch.no_grad(): first_rgb = self.decoder(sample['first'], [0])
        tokens = self.encode_condition(condition, first_rgb)
        z = sample['latent']
        sigma = z.new_tensor(sigma_override if sigma_override is not None else
                             train['sigma_min']+(train['sigma_max']-train['sigma_min'])*float(torch.rand(())))
        noise = torch.randn_like(z)
        noisy = torch.cat((sample['first'], ((1-sigma)*z+sigma*noise)[:, :, 1:]), 2)
        velocity, intensity = self.generator(noisy, sample['first'], null_text if native_dropped else sample['text'],
                                             sigma, tokens, sample['times'], auxiliary=True)
        fm = weighted_mean((velocity[:, :, 1:]-(noise-z)[:, :, 1:]).square().mean(1), sample['valid'])
        target, pv = pool_grid(sample['intensity'], sample['valid'], intensity.shape[-2:])
        iv, _ = pool_grid(sample['change_valid'], sample['valid'], intensity.shape[-2:])
        intensity_loss = weighted_mean(F.smooth_l1_loss(intensity.float(), target.detach(), reduction='none'), iv*pv)
        zero = velocity.sum()*0
        losses = dict(fm=fm, sequence=zero, intensity=intensity_loss, critic=zero, physical=zero)
        metrics = dict(sigma=sigma.detach(), verifier_active=0, sequence_active=0, wan_forward_count=1,
                       native_text_dropped=int(native_dropped),knowledge_text_dropped=int(dropped))
        verify = not (self.temporal_verifier or self.calibrated_verifier) and (self.cfg['loss']['critic']>0 or self.cfg['loss']['physical']>0) and (
            validation or iteration % self.cfg['verifier']['every'] == 0)
        sequence = validation or (self.cfg['loss']['sequence']>0 and float(sigma) <= self.cfg['sequence']['max_sigma'])
        if sequence or verify:
            clean = torch.cat((sample['first'], (noisy-sigma*velocity)[:, :, 1:]), 2)
            size = tuple(max(a, b) for a, b in zip(self.cfg['sequence']['size'], self.cfg['knowledge']['video_size']))
            with torch.no_grad(): truth = self.decoder(z, resize=size)
            generated = self.decoder(clean, resize=size)
            if sequence:
                losses['sequence'], process_metrics = sequence_error(generated, truth, sample['valid'], self.cfg['sequence'])
                fraction = self.cfg['sequence']['perceptual_fraction']
                if fraction:
                    with torch.no_grad(): reference_features=self.knowledge.process_features(truth)
                    generated_features=self.knowledge.process_features(generated)
                    perceptual,feature_metrics=perceptual_process_error(generated_features,reference_features,sample['valid'],self.cfg['sequence'])
                    losses['sequence']=(1-fraction)*losses['sequence']+fraction*perceptual
                    metrics.update(feature_metrics)
                metrics.update(process_metrics, sequence_active=1)
                metrics.update(sequence_metrics(generated.detach(), truth, sample['valid']))
            if verify:
                losses['critic'], losses['physical'], verifier_metrics = self._verifier(
                    generated, truth, sample, condition, iteration//self.cfg['verifier']['every'])
                metrics.update(verifier_metrics, verifier_active=1)
        weights = {name: self.cfg['loss'][name] for name in losses}
        warmup = self.cfg['verifier']['warmup']
        weights['physical'] *= min(1., max(0., (step+1-warmup)/warmup)) if warmup else 1.
        metrics.update({f'loss/{k}': v.detach() for k, v in losses.items()})
        metrics.update({f'weighted/{k}': (weights[k]*v).detach() for k, v in losses.items()})
        metrics['loss'] = sum(weights[k]*v for k, v in losses.items()).detach()
        return losses, weights, metrics

    def backward_losses(self, losses, weights, accumulation=1):
        # All generator objectives reach C and Qwen. Only critic calibration owns
        # critic parameters; its features and teacher targets were detached above.
        (sum(weights[k]*v for k, v in losses.items())/accumulation).backward()

    def should_rollout(self, step):
        if self.operator_training: return False
        r = self.cfg['rollout']
        return self.cfg['loss']['rollout']>0 and step >= r['start'] and (step-r['start']) % r['every'] == 0

    def _sample_tokens(self, conditions, text, null_text, seed, policy, *, differentiable=False, strength=1.,
                       initial_noise=None, token_override=None, negative_token_override=None,adjoint_callback=None,
                       prefix_token_override=None,condition_update_start=0):
        if not 0<=condition_update_start<policy['steps']:
            raise ValueError('Condition update boundary must be a sampling step')
        if condition_update_start and prefix_token_override is None:
            raise ValueError('A fixed original C is required before the update boundary')
        first_rgb=None
        if token_override is None or (policy['guidance']!=1 and negative_token_override is None):
            with torch.no_grad(): first_rgb = self.decoder(conditions['first'], [0])
        positive = self.encode_condition(conditions, first_rgb) if token_override is None else token_override
        analysis = getattr(self,'last_condition_analysis','') if token_override is None else 'Explicit C intervention; see experiment record.'
        if policy['guidance']==1:negative=positive
        elif negative_token_override is not None:negative=negative_token_override
        else:negative=self.encode_condition(dict(conditions,caption='',clauses=[]),first_rgb)
        self.last_sample_analysis = analysis
        shape = list(conditions['first'].shape); shape[2] = len(conditions['times'])
        random = torch.Generator(device=conditions['first'].device).manual_seed(seed)
        x = (torch.randn(shape, device=conditions['first'].device, generator=random) if initial_noise is None
             else initial_noise.to(device=conditions['first'].device,dtype=torch.float32).clone())
        if list(x.shape) != shape: raise ValueError('Initial noise shape differs from the requested video')
        x[:, :, :1] = conditions['first']
        sigmas = shifted_sigma(torch.linspace(1, 0, policy['steps']+1, device=x.device), policy['shift'])
        self.last_sample_feedback = None
        if policy.get('latent_feedback', {}).get('enabled', False):
            if differentiable or prefix_token_override is not None:
                raise ValueError('Online C refinement is an inference path, not a training adjoint')
            from .feedback_sampling import sample_with_feedback
            return sample_with_feedback(self, conditions, positive, negative, text, null_text,
                                        x, sigmas, policy, strength)
        def step(value,c,u,index):
            current,following=sigmas[index],sigmas[index+1]
            # For a C-only feedback update, keep its ORIGINAL legal condition in
            # the prefix. This matches a suffix derivative to the actual update
            # applied during complete resampling; it never disables conditioning.
            active=(prefix_token_override.detach() if index<condition_update_start else c)
            v=self.generator(value,conditions['first'],text,current,active,conditions['times'],strength)
            if policy['guidance']!=1:
                uncond=self.generator(value,conditions['first'],null_text,current,u,conditions['times'],strength)
                v=uncond+policy['guidance']*(v-uncond)
            return torch.cat((conditions['first'],(value+(following-current)*v)[:,:,1:]),2)
        if differentiable and policy.get('discrete_adjoint',False):
            from .adjoint import discrete_adjoint
            return discrete_adjoint(x,positive,negative,step,policy['steps'],
                gradient_steps=policy['gradient_steps'],parameters=self.generator.parameters(),vjp_callback=adjoint_callback)
        def integrate(state, c, u, begin, end):
            for i in range(begin, end):
                def advance(value,positive_c,negative_c,index=i):return step(value,positive_c,negative_c,index)
                if policy.get('step_checkpointing',False) and torch.is_grad_enabled():
                    # Store only Euler-step boundaries, then recompute that step's
                    # native Wan blocks during backward. No step link is detached.
                    state=checkpoint(advance,state,c,u,use_reentrant=False)
                else:state=advance(state,c,u)
            return state
        if not differentiable: return integrate(x, positive, negative, 0, policy['steps'])
        boundary = policy['steps']-policy['gradient_steps']
        with torch.no_grad(): x = integrate(x, positive.detach(), negative.detach(), 0, boundary)
        if policy.get('step_checkpointing',False):
            return integrate(x,positive,negative,boundary,policy['steps'])
        # Recompute the differentiable suffix, while preserving all K step links.
        def suffix(state, c, u): return integrate(state, c, u, boundary, policy['steps'])
        return checkpoint(suffix, x, positive, negative, use_reentrant=False)

    def rollout_losses(self, sample, null_text, step, seed):
        self.set_phase(step)
        condition = self.conditions(sample)
        if self.calibrated_verifier:
            return self.calibrated_rollout_losses(sample,null_text,step,seed,condition)
        if self.ordered_process:
            latent=self._sample_tokens(condition,sample['text'],null_text,seed,self.cfg['rollout'],differentiable=True)
            generated=self.decoder(latent,resize=self.cfg['verifier']['size'])
            predicted=self.process_encoder(generated)
            with torch.no_grad():
                reference=self.process_encoder(self.decoder(sample['latent'],resize=self.cfg['verifier']['size']))
            loss,metrics=ordered_process_distribution(predicted,reference,sample['valid'],
                lags=tuple(self.cfg['verifier']['lags']),scale_floor=self.cfg['verifier']['scale_floor'])
            weight=self.cfg['loss']['rollout']
            metrics.update(loss=loss.detach()*weight,sampling_steps=self.cfg['rollout']['steps'],
                           gradient_steps=self.cfg['rollout']['gradient_steps'],sampled_from_noise=True,
                           target_uses_reference_future=True,condition_uses_reference_future=False)
            return dict(rollout=loss),dict(rollout=weight),metrics
        if self.temporal_verifier:
            return self.temporal_rollout_losses(sample,null_text,step,seed,condition)
        latent = self._sample_tokens(condition, sample['text'], null_text, seed, self.cfg['rollout'], differentiable=True)
        size = self.cfg['knowledge']['video_size']
        generated = self.decoder(latent, resize=size)
        with torch.no_grad(): truth = self.decoder(sample['latent'], resize=size)
        calibration, physical, metrics = self._verifier(generated, truth, sample, condition, step, alternative=generated.detach())
        weights = dict(critic=self.cfg['loss']['critic'], rollout=self.cfg['loss']['rollout'])
        losses = dict(critic=calibration, rollout=physical)
        metrics.update(loss=sum(weights[k]*v for k, v in losses.items()).detach(),
                       sampling_steps=self.cfg['rollout']['steps'], gradient_steps=self.cfg['rollout']['gradient_steps'],
                       sampled_from_noise=True)
        return losses, weights, metrics

    def calibrated_rollout_losses(self,sample,null_text,step,seed,condition):
        policy=self.cfg['rollout'];vcfg=self.cfg['verifier']
        latent=self._sample_tokens(condition,sample['text'],null_text,seed,policy,differentiable=True)
        generated=self.decoder(latent,resize=vcfg['size'])
        features=self.process_encoder(generated)
        with torch.no_grad():
            truth=self.decoder(sample['latent'],resize=vcfg['size'])
            reference=self.process_encoder(truth)
            frozen=self.process_encoder(truth[:1].expand_as(truth))
            middle=truth.clone();begin=int(.3*(len(truth)-1));end=int(.5*(len(truth)-1))
            middle[begin+1:end+1]=truth[begin]
            omitted=self.process_encoder(middle)
            first=self.decoder(sample['first'],[0])
            fixed=self.knowledge.encode(first,condition['caption'],condition['clauses'],adapt=False,
                readout='queries')['contextual']
            observations=[reference,features.detach(),frozen,omitted]
            targets=[ordered_process_distribution(x,reference,sample['valid'],lags=tuple(vcfg['lags']),
                scale_floor=vcfg['scale_floor'],return_pairs=True)[2] for x in observations]
        if vcfg.get('calibration','regression')=='paired':
            # Same-scene comparisons remove cross-scene error-scale shortcuts.
            # Targets remain graded process evidence, not hard physics labels.
            calibration=torch.stack([self.critic.paired_calibration_loss(
                observations[i],observations[j],targets[i],targets[j],fixed,sample['times'],sample['valid'])
                for i in range(len(observations)) for j in range(i+1,len(observations))]).mean()
        else:
            calibration=torch.stack([self.critic.evidence_calibration_loss(x,target,fixed,sample['times'],sample['valid'])[0]
                                      for x,target in zip(observations,targets)]).mean()
        feedback=self.critic.generation_loss(features,fixed,sample['times'],sample['valid'])
        losses=dict(critic=calibration,rollout=feedback)
        weights=dict(critic=self.cfg['loss']['critic'],rollout=self.cfg['loss']['rollout'])
        metrics=dict(loss=sum(weights[k]*v for k,v in losses.items()).detach(),
            verifier_calibration=calibration.detach(),verifier_feedback=feedback.detach(),
            verifier_paired_calibration=int(vcfg.get('calibration','regression')=='paired'),
            sampling_steps=policy['steps'],gradient_steps=policy['gradient_steps'],
            sampled_from_noise=True,calibration_uses_reference_future=True,
            verifier_scoring_uses_reference_future=False,condition_uses_reference_future=False,
            observed_generated_process_error=torch.stack([v.mean() for v in targets[1].values()]).mean(),
            observed_freeze_process_error=torch.stack([v.mean() for v in targets[2].values()]).mean())
        return losses,weights,metrics

    def temporal_rollout_losses(self,sample,null_text,step,seed,condition):
        policy=self.cfg['rollout'];warmup=self.cfg['verifier']['warmup']
        update=(step-policy['start'])//policy['every']+1
        scale=min(1.,max(0.,(update-warmup)/warmup)) if warmup else 1.
        # Calibration uses genuine full trajectories, never a one-step estimate
        # already containing the reference future. Early updates train D only.
        with torch.set_grad_enabled(bool(scale) and torch.is_grad_enabled()):
            latent=self._sample_tokens(condition,sample['text'],null_text,seed,policy,
                                       differentiable=bool(scale))
            generated=self.decoder(latent,resize=self.cfg['knowledge']['video_size'])
            generated_features=self.knowledge.process_features(generated)
        with torch.no_grad():
            truth=self.decoder(sample['latent'],resize=self.cfg['knowledge']['video_size'])
            reference_features=self.knowledge.process_features(truth)
            first=self.decoder(sample['first'],[0])
            fixed_condition=self.knowledge.encode(first,condition['caption'],condition['clauses'],adapt=False,readout='queries')['contextual']
        calibration,metrics=self.critic.calibration_loss(reference_features,generated_features,
                                                        fixed_condition,sample['times'],sample['valid'])
        process=self.critic.generation_loss(generated_features,fixed_condition,sample['times'],sample['valid'])
        losses=dict(critic=calibration,rollout=process)
        weights=dict(critic=self.cfg['loss']['critic'],rollout=self.cfg['loss']['rollout']*scale)
        metrics.update(loss=sum(weights[k]*v for k,v in losses.items()).detach(),
                       sampling_steps=policy['steps'],gradient_steps=policy['gradient_steps'] if scale else 0,
                       sampled_from_noise=True,verifier_update=update,generator_feedback_scale=scale)
        return losses,weights,metrics

    @torch.no_grad()
    def sample(self, conditions, text, null_text, seed=42, steps=None, guidance=None, strength=None,
               initial_noise=None, token_override=None,negative_token_override=None):
        policy = dict(self.cfg['inference'])
        if steps is not None: policy['steps'] = steps
        if guidance is not None: policy['guidance'] = guidance
        strength = policy['condition_strength'] if strength is None else strength
        return self._sample_tokens(conditions, text, null_text, seed, policy, strength=strength,
                                   initial_noise=initial_noise,token_override=token_override,
                                   negative_token_override=negative_token_override)
