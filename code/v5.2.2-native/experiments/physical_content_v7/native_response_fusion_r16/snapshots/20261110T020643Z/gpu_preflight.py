"""Three disposable full-size training passes; never initialize formal training."""
from pathlib import Path
import argparse, json, os, random, sys, time
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
    p.add_argument('--diagnostic-gpu', action='store_true',
                   help='Allow an idle A100/A800 evaluation GPU for a disposable diagnostic; H100 is always rejected')
    args = p.parse_args()
    cfg = load_config(args.config)
    mode = cfg['model']['response_mode']
    active = cfg['model']['global_execution']
    active_keys = ['request' if name == 'request_bound' else name for name in active]
    folder = args.output
    if args.diagnostic_gpu:
        count = torch.cuda.device_count()
        name = torch.cuda.get_device_name(0) if count else 'NO CUDA DEVICE'
        accepted = count == 1 and 'H100' not in name and ('A100' in name or 'A800' in name)
        gpu = dict(kind='diagnostic_gpu', actual=name, visible_count=count,
                   accepted=accepted, resource_pool='amp-80g',
                   experiment_id=os.getenv('DET_EXPERIMENT_ID'), time=time.time())
        atomic_json(gpu, folder / 'gpu.json')
        print(json.dumps(gpu), flush=True)
        if not accepted:
            raise SystemExit(86)
    else:
        gpu = gpu_guard(folder)
    torch.set_num_threads(4)
    random.seed(42); np.random.seed(42)
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    atomic_json(dict(status='loading_fresh_official_model', gpu=gpu), folder / 'status.json')
    try:
        model = PhysicalModel(cfg).train()
        model.wan.eval()
        model.set_phase(0)
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
                          fusion.joint_condition_query, fusion.time,
                          fusion.direct_output, fusion.request_output, fusion.joint_output,
                          fusion.direct_state_value, fusion.request_state_value,
                          fusion.joint_state_value):
                if torch.count_nonzero(layer.weight):
                    raise AssertionError(f'{site}: declared zero-init write/condition is nonzero')
        if torch.count_nonzero(model.value_correction.weight):
            raise AssertionError('Declared shared request-bound V correction is nonzero at start')
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
        for iteration, step in enumerate((0, 1, 2, 128)):
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
                joint_condition_query=grad(fusion.joint_condition_query.weight),
                direct_output=grad(fusion.direct_output.weight),
                request_output=grad(fusion.request_output.weight),
                joint_output=grad(fusion.joint_output.weight),
                direct_state_value=grad(fusion.direct_state_value.weight),
                request_state_value=grad(fusion.request_state_value.weight),
                joint_state_value=grad(fusion.joint_state_value.weight),
                value_correction=grad(model.value_correction.weight),
                video_q=grad(fusion.video_q.weight),
                video_k=grad(fusion.video_k.weight),
                video_v=grad(fusion.video_v.weight),
                key=grad(fusion.key.weight),
                value=grad(fusion.value.weight),
                routing_C=grad(model.enhancement.project.weight),
                execution_C=grad(model.enhancement.execution_project.weight))
            if not relevant['output'] > 0:
                raise AssertionError('FM gradient must reach the zero-initialized local output')
            for name in active_keys:
                if not relevant[name + '_output'] > 0:
                    raise AssertionError(f'Global {name} output did not receive generation gradient')
            if iteration >= 1:
                for key in ('condition_query', 'video_q', 'video_k', 'video_v', 'key', 'value'):
                    if not relevant[key] > 0:
                        raise AssertionError(f'Generation gradient did not reach {key}')
                for name in active_keys:
                    if not relevant[name + '_state_value'] > 0:
                        raise AssertionError(f'Global {name} state gate did not receive gradient')
                if mode in ('triple_global_joint_local', 'triple_global_full_local'):
                    if not relevant['joint_condition_query'] > 0:
                        raise AssertionError('Joint local query condition did not receive gradient')
                if mode == 'triple_global_full_local':
                    if not relevant['request_condition_query'] > 0:
                        raise AssertionError('Request local query condition did not receive gradient')
                    if not relevant['value_correction'] > 0:
                        raise AssertionError('Request-bound local V correction did not receive gradient')
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
        # Four disposable updates cannot grow every zero-init path above bf16
        # resolution. For the connectivity/behavior check only, replace the
        # zero-init projections with small random values; these weights are
        # disposable and never saved.
        model.eval().requires_grad_(False)
        with torch.no_grad():
            for fusion in model.fusions.values():
                for layer in (fusion.output, fusion.condition_query,
                              fusion.request_condition_query, fusion.joint_condition_query,
                              fusion.direct_output, fusion.request_output,
                              fusion.joint_output, fusion.direct_state_value,
                              fusion.request_state_value, fusion.joint_state_value):
                    layer.weight.normal_(std=.04)
            model.value_correction.weight.normal_(std=.04)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
            c, relations = model.enhancement(evidence)
            noisy = (1 - sigma) * sample['latent'] + sigma * noise
            full, _ = model.velocity(noisy, sample['first'], sample['text'], sigma,
                                     c, sample['times'], relations)
            off, _ = model.velocity(noisy, sample['first'], sample['text'], sigma,
                                    c, sample['times'], relations, condition_enabled=False)
            local_only, _ = model.velocity(noisy, sample['first'], sample['text'], sigma,
                                           c, sample['times'], relations, global_enabled=False)
            global_only, _ = model.velocity(noisy, sample['first'], sample['text'], sigma,
                                            c, sample['times'], relations, local_enabled=False)
            no_direct, _ = model.velocity(noisy, sample['first'], sample['text'], sigma,
                                          c, sample['times'], relations,
                                          global_direct_enabled=False)
            no_request = None
            if 'request' in active_keys:
                no_request, _ = model.velocity(noisy, sample['first'], sample['text'], sigma,
                                               c, sample['times'], relations,
                                               global_request_enabled=False)
            no_joint = None
            if 'joint' in active_keys:
                no_joint, _ = model.velocity(noisy, sample['first'], sample['text'], sigma,
                                             c, sample['times'], relations,
                                             global_joint_enabled=False)
            no_value = None
            if mode == 'triple_global_full_local':
                no_value, _ = model.velocity(noisy, sample['first'], sample['text'], sigma,
                                             c, sample['times'], relations,
                                             value_correction_enabled=False)
            no_temporal, _ = model.velocity(noisy, sample['first'], sample['text'], sigma,
                                            c, sample['times'], relations,
                                            temporal_enabled=False)
            scale = full.float().square().mean().sqrt().clamp_min(1e-8)
            def effect(value):
                return float((full - value).float().square().mean().sqrt() / scale)
            checks = dict(full_vs_condition_off=effect(off), full_vs_global_off=effect(local_only),
                          full_vs_local_off=effect(global_only), full_vs_direct_off=effect(no_direct),
                          full_vs_temporal_off=effect(no_temporal))
            if no_request is not None:
                checks['full_vs_request_off'] = effect(no_request)
            if no_joint is not None:
                checks['full_vs_joint_off'] = effect(no_joint)
            if no_value is not None:
                checks['full_vs_value_correction_off'] = effect(no_value)
        if checks['full_vs_condition_off'] <= 0:
            raise AssertionError('C conditions do not affect the trained operator')
        if checks['full_vs_global_off'] <= 0:
            raise AssertionError('Global execution writes do not affect the prediction')
        if checks['full_vs_local_off'] <= 0:
            raise AssertionError('Local state operator does not affect the prediction')
        if checks['full_vs_direct_off'] <= 0:
            raise AssertionError('Direct global read does not affect the prediction')
        if checks['full_vs_temporal_off'] <= 0:
            raise AssertionError('Local temporal edges do not affect the prediction')
        if 'request' in active_keys and checks['full_vs_request_off'] <= 0:
            raise AssertionError('Request-bound global read does not affect the prediction')
        if 'joint' in active_keys and checks['full_vs_joint_off'] <= 0:
            raise AssertionError('Joint global read does not affect the prediction')
        if mode == 'triple_global_full_local' and checks['full_vs_value_correction_off'] <= 0:
            raise AssertionError('Request-bound local V correction does not affect the prediction')
        report = dict(status='passed', gpu=gpu, parameters=count,
                      parameter_groups={g['name']: sum(p.numel() for p in g['params']) for g in groups},
                      response_mode=mode, global_execution=active,
                      initial_native_max_abs=initial_native_max_abs,
                      condition_dependency=checks, rows=rows, quality_proof=False,
                      condition_dependency_scope='Active-operator connectivity check with disposable small random zero-init projections; not learned magnitude and not a quality claim',
                      temporary_weights_saved=False,
                      formal_training='Separate process reloads official foundations, seed42, cursor0; never uses these temporary weights')
        atomic_json(report, folder / 'result.json')
        atomic_json(report, folder / 'status.json')
    except BaseException as error:
        atomic_json(dict(status='failed', error=repr(error)), folder / 'status.json')
        raise


if __name__ == '__main__':
    main()
