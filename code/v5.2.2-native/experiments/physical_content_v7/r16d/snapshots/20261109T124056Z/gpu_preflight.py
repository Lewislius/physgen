"""Three disposable full-size training passes; never initialize formal training."""
from pathlib import Path
import argparse, json, random, sys, time
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap
import numpy as np
import torch
from physical_content.common import load_config, gpu_guard, atomic_json
from physical_content.model import PhysicalModel
from physical_content.teacher import Teacher
from physical_content.supervision import generation_loss
from physical_scaling.data import Data, SampleOrder
from physical_scaling.schedule import rates, auxiliary_coefficient


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    cfg = load_config(args.config)
    folder = args.output
    gpu = gpu_guard(folder)
    torch.set_num_threads(4)
    random.seed(42); np.random.seed(42)
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    atomic_json(dict(status='loading_fresh_official_model', gpu=gpu), folder / 'status.json')
    try:
        model = PhysicalModel(cfg).train()
        model.wan.eval()
        teacher = Teacher(cfg)
        named = dict(model.named_parameters())
        groups = model.optim_groups()
        count = sum(named[n].numel() for n in model.learned_names)
        if any(p.requires_grad for block in model.wan.blocks for p in block.cross_attn.parameters()):
            raise AssertionError('Native text projections must stay frozen')
        if any(p.requires_grad for n, p in model.wan.named_parameters()
               if n.endswith(('q.weight', 'k.weight', 'v.weight', 'o.weight'))
               and 'cross_attn' in n):
            raise AssertionError('Frozen native text Q/K/V/O violated')
        for site, fusion in model.fusions.items():
            for layer in (fusion.output, fusion.condition_query, fusion.request_condition_query,
                          fusion.key_condition, fusion.query_residual_up, fusion.time):
                if torch.count_nonzero(layer.weight):
                    raise AssertionError(f'{site}: declared zero-init write/condition is nonzero')
        optimizer = torch.optim.AdamW(groups, betas=(.9, .999), weight_decay=.01, foreach=False)
        data = Data(cfg)
        index = next(iter(SampleOrder(len(data), cfg['train']['seed'], 0)))
        sample = data.get(index)
        evidence = teacher.input(sample)
        noise = torch.randn(sample['latent'].shape, device='cuda',
                            generator=torch.Generator(device='cuda').manual_seed(42))
        sigma = torch.tensor(.6, device='cuda')
        rows = []
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
            initial, relations = model.enhancement(evidence)
            noisy = (1 - sigma) * sample['latent'] + sigma * noise
            initial_full, _ = model.velocity(noisy, sample['first'], sample['text'], sigma,
                                             initial, sample['times'], relations)
            initial_native, _ = model.velocity(noisy, sample['first'], sample['text'], sigma,
                                               initial, sample['times'], relations,
                                               physical_enabled=False)
            initial_native_max_abs = float((initial_full - initial_native).abs().max())
            torch.testing.assert_close(initial_full, initial_native, rtol=2e-5, atol=2e-5)
            del initial, initial_full, initial_native, noisy
        for iteration, step in enumerate((0, 1, 2, 3, 128)):
            phase = model.set_phase(step)
            rates(optimizer, cfg, step)
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            start = time.monotonic()
            with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
                c, groups_i = model.enhancement(evidence)
                noisy = (1 - sigma) * sample['latent'] + sigma * noise
                pred, _ = model.velocity(noisy, sample['first'], sample['text'], sigma,
                                         c, sample['times'], groups_i, None, sample['valid'])
                generation, parts = generation_loss(pred, noise, sample['latent'], sample['valid'],
                                                    **cfg['loss']['generation'])
                aux = model.enhancement.content_loss(c, evidence)
                objective = generation + auxiliary_coefficient(cfg, step) * aux
            if not torch.isfinite(objective):
                raise FloatingPointError('Nonfinite training objective')
            objective.backward()
            fusion = model.fusions['2']
            grad = lambda p: None if p.grad is None else float(p.grad.detach().float().norm())
            relevant = dict(
                output=grad(fusion.output.weight),
                condition_query=grad(fusion.condition_query.weight),
                request_condition_query=grad(fusion.request_condition_query.weight),
                key_condition=grad(fusion.key_condition.weight),
                query_residual_up=grad(fusion.query_residual_up.weight),
                query_residual_down=grad(fusion.query_residual_down.weight),
                video_q=grad(fusion.video_q.weight),
                video_k=grad(fusion.video_k.weight),
                video_v=grad(fusion.video_v.weight),
                key=grad(fusion.key.weight),
                value=grad(fusion.value.weight),
                routing_C=grad(model.enhancement.project.weight),
                execution_C=grad(model.enhancement.execution_project.weight))
            if not relevant['output'] > 0:
                raise AssertionError('FM gradient must reach the zero-initialized local output')
            if iteration >= 1:
                # Step0 opens the zero output; step1 opens the two zero query
                # conditions and the local video Q/K/V.
                for key in ('condition_query', 'request_condition_query',
                            'video_q', 'video_k', 'video_v'):
                    if not relevant[key] > 0:
                        raise AssertionError(f'Generation gradient did not reach {key}')
            if iteration >= 2:
                # The combined condition is nonzero only after step1, so K
                # conditioning, residual-up and C K/V credit open at step2.
                for key in ('key_condition', 'query_residual_up', 'key', 'value', 'execution_C'):
                    if not relevant[key] > 0:
                        raise AssertionError(f'Generation gradient did not reach {key}')
            if iteration >= 3:
                for key in ('query_residual_down',):
                    if not relevant[key] > 0:
                        raise AssertionError(f'Generation gradient did not reach {key}')
            for name, param in model.wan.named_parameters():
                if param.grad is not None and (phase == 'interface' or not name.endswith(('lora_a', 'lora_b'))):
                    raise AssertionError(f'Unexpected native gradient: {phase}/{name}')
            norms = {}
            for group in optimizer.param_groups:
                params = [p for p in group['params'] if p.grad is not None]
                norms[group['name']] = float(torch.nn.utils.clip_grad_norm_(
                    params, cfg['train']['grad_clip'][group['name']], error_if_nonfinite=True)) if params else 0.
            if phase == 'joint' and not norms['wan_lora'] > 0:
                raise AssertionError('Joint phase has no native LoRA gradient')
            optimizer.step()
            torch.cuda.synchronize()
            row = dict(disposable_iteration=iteration + 1, phase=phase, sample=sample['id'],
                       seconds=time.monotonic() - start,
                       peak_gib=torch.cuda.max_memory_allocated() / 2**30,
                       gradients=relevant, group_gradients=norms,
                       generation=float(generation.detach()), aux=float(aux.detach()))
            rows.append(row)
            print(json.dumps(row), flush=True)
            atomic_json(dict(status='running', rows=rows), folder / 'status.json')
        # After one optimizer step the write is nonzero; verify that both C
        # conditions change the prediction and that they are not identical.
        model.eval().requires_grad_(False)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
            c, relations = model.enhancement(evidence)
            noisy = (1 - sigma) * sample['latent'] + sigma * noise
            full, _ = model.velocity(noisy, sample['first'], sample['text'], sigma,
                                     c, sample['times'], relations)
            off, _ = model.velocity(noisy, sample['first'], sample['text'], sigma,
                                    c, sample['times'], relations, condition_enabled=False)
            direct, _ = model.velocity(noisy, sample['first'], sample['text'], sigma,
                                       c, sample['times'], relations,
                                       request_condition_enabled=False)
            request, _ = model.velocity(noisy, sample['first'], sample['text'], sigma,
                                        c, sample['times'], relations,
                                        direct_condition_enabled=False)
            no_key, _ = model.velocity(noisy, sample['first'], sample['text'], sigma,
                                       c, sample['times'], relations,
                                       key_condition_enabled=False)
            no_residual, _ = model.velocity(noisy, sample['first'], sample['text'], sigma,
                                            c, sample['times'], relations,
                                            query_residual_enabled=False)
            scale = full.float().square().mean().sqrt().clamp_min(1e-8)
            checks = dict(full_vs_condition_off=float((full - off).float().square().mean().sqrt() / scale),
                          full_vs_direct_only=float((full - direct).float().square().mean().sqrt() / scale),
                          full_vs_request_only=float((full - request).float().square().mean().sqrt() / scale),
                          direct_vs_request=float((direct - request).float().square().mean().sqrt() / scale),
                          full_vs_key_condition_off=float((full - no_key).float().square().mean().sqrt() / scale),
                          full_vs_query_residual_off=float((full - no_residual).float().square().mean().sqrt() / scale))
        if checks['full_vs_condition_off'] <= 0:
            raise AssertionError('C conditions do not affect the trained local operator')
        if checks['direct_vs_request'] <= 0:
            raise AssertionError('Direct and request-bound conditions are identical')
        if checks['full_vs_key_condition_off'] <= 0:
            raise AssertionError('Local K conditioning does not affect the prediction')
        if checks['full_vs_query_residual_off'] <= 0:
            raise AssertionError('C-read query residual does not affect the prediction')
        report = dict(status='passed', gpu=gpu, parameters=count,
                      parameter_groups={g['name']: sum(p.numel() for p in g['params']) for g in groups},
                      initial_native_max_abs=initial_native_max_abs,
                      condition_dependency=checks, rows=rows, quality_proof=False,
                      temporary_weights_saved=False,
                      formal_training='Separate process reloads official foundations, seed42, cursor0; never uses these temporary weights')
        atomic_json(report, folder / 'result.json')
        atomic_json(report, folder / 'status.json')
    except BaseException as error:
        atomic_json(dict(status='failed', error=repr(error)), folder / 'status.json')
        raise


if __name__ == '__main__':
    main()
