"""Bounded checks of routing/execution gradients, CFG controls and checkpoint ownership."""
from pathlib import Path
import copy
import importlib.util
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE if HERE.name == 'grounded_execution_r10b' else HERE.parents[1]
sys.path.insert(0, str(HERE))
import bootstrap
import torch
import yaml
from torch import nn
from torch.nn import functional as F
from physical_content.model import Enhancement, Fusion, PhysicalModel, map_content
from torch.utils.checkpoint import checkpoint
from physical_content.evidence import resample_words, memory_text, recover_visual_tokens
from physical_content.teacher import Teacher
from physical_content.common import load_config
from physical_content.train import save, restore
from physical_scaling.schedule import learning_rates, auxiliary_coefficient
from infer import enhance_pair

PROJECT = bootstrap.BASE / 'experiments/physical_content_v7'
REFERENCE = PROJECT / 'content_value_r9_1/snapshots/20261031T173355Z/physical_content/model.py'
spec = importlib.util.spec_from_file_location('physical_content.r9_reference', REFERENCE)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
torch.set_num_threads(2)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim, self.num_heads, self.head_dim = 16, 2, 8
        for name in ['q', 'k', 'v', 'o']:
            setattr(self, name, nn.Linear(16, 16, bias=False))
        self.norm_q, self.norm_k = nn.RMSNorm(16), nn.RMSNorm(16)

    def forward(self, x, context):
        return x


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn, self.cross_attn = Attention(), Attention()
        self.ffn = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 16))


class TinyWan(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([TinyBlock()])


class ContentValueTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        self.cfg = dict(model=dict(qwen_width=32, width=16, q_heads=4,
            q_tokens=8, rounds=2, grounded_read_heads=2, visual_read_heads=1),
            train=dict(gradient_checkpointing=True))
        self.interface = dict(mean=torch.randn(32), scale=torch.rand(32) + .5,
            weight=torch.randn(16, 32) * .05, bias=torch.randn(16) * .01,
            target_scale=torch.tensor(.1))
        self.evidence = dict(words=torch.randn(6, 32), word_groups=torch.arange(6) // 3,
            request_words=torch.randn(5, 32), native_words=torch.randn(6,16)*.1,
            visual_memory=torch.randn(1,7,32))

    def test_existing_initializer_rng_and_initial_content(self):
        attention = Attention()
        torch.manual_seed(42)
        old, old_fusion = reference.Enhancement(self.cfg, self.interface), reference.Fusion(attention)
        rng = torch.get_rng_state().clone()
        torch.manual_seed(42)
        new, new_fusion = Enhancement(self.cfg, self.interface), Fusion(attention)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(set(dict(new.named_parameters())) - set(dict(old.named_parameters())),
                         {'execution_project.weight'})
        for name, parameter in old.named_parameters():
            torch.testing.assert_close(parameter, dict(new.named_parameters())[name], rtol=0, atol=0)
        for name, parameter in old_fusion.state_dict().items():
            torch.testing.assert_close(parameter, new_fusion.state_dict()[name], rtol=0, atol=0)
        for value in new(self.evidence)[0].values():
            torch.testing.assert_close(old(self.evidence)[0], value, rtol=0, atol=0)
        self.assertEqual(new.execution_project.weight.numel(),32*16)

    def test_native_word_auxiliary_targets_execution_core(self):
        model = Enhancement(self.cfg, self.interface)
        with torch.no_grad():
            model.project.weight.normal_(std=.02)
            model.execution_project.weight.normal_(std=.02)
        c,_=model(self.evidence)
        c['routing'].retain_grad();c['execution'].retain_grad()
        model.content_loss(c,self.evidence).backward()
        self.assertGreater(float(model.execution_project.weight.grad.norm()),0.)
        self.assertGreater(float(model.seed.grad.norm()),0.)
        self.assertIsNone(model.project.weight.grad)
        self.assertIsNone(model.shared_offset.grad)
        self.assertIsNone(c['routing'].grad)
        self.assertGreater(float(c['execution'].grad.norm()),0.)

    def test_direct_visual_input_affects_execution_with_words_fixed(self):
        model = Enhancement(self.cfg, self.interface).eval()
        with torch.no_grad():
            model.execution_project.weight.normal_(std=.02)
            before = model(self.evidence)[0]['execution']
            changed = dict(self.evidence, visual_memory=torch.randn_like(self.evidence['visual_memory']))
            after = model(changed)[0]['execution']
        self.assertGreater(float((before-after).abs().max()),1e-5)

    def test_actual_cached_visual_selection_and_reasoning_controls(self):
        cfg = load_config(HERE / 'configs/physical_content.yaml')
        teacher = Teacher(cfg,device='cpu')
        original = teacher.input(dict(id='videophy40_VP026'))
        conditions = [original]
        for mode in ['reasoning_swapped','empty_reasoning']:
            path = PROJECT/'cache/reasoning_controls'/mode/'VP026.pt'
            evidence = torch.load(path,map_location='cpu',weights_only=True)
            teacher.add_request(evidence,80/24)
            self.assertEqual(evidence['sample'],original['sample'])
            self.assertEqual(evidence['caption'],original['caption'])
            conditions.append(evidence)
        for evidence in conditions:
            text,_,_ = memory_text(evidence,80/24,teacher._tokenizer)
            ids = teacher._tokenizer(text,add_special_tokens=False)['input_ids']
            image_id = teacher._tokenizer.convert_tokens_to_ids('<|image_pad|>')
            positions = torch.tensor([i for i,value in enumerate(ids) if value==image_id])
            self.assertGreater(len(positions),0)
            torch.testing.assert_close(evidence['visual_token_positions'],positions,rtol=0,atol=0)
            torch.testing.assert_close(evidence['visual_memory'],evidence['memory'].index_select(1,positions),rtol=0,atol=0)
            other = evidence['memory'].clone()
            outside = torch.ones(other.shape[1],dtype=torch.bool);outside[positions]=False
            other[:,outside] = 100
            selected = recover_visual_tokens(dict(evidence,memory=other),80/24,teacher._tokenizer)
            torch.testing.assert_close(selected['visual_memory'],evidence['visual_memory'],rtol=0,atol=0)

    def test_actual_inference_anchor_control_removes_both_corrections(self):
        enhancement = Enhancement(self.cfg, self.interface)
        with torch.no_grad():
            enhancement.project.weight.normal_(std=.02)
            enhancement.execution_project.weight.normal_(std=.02)
            enhancement.shared_offset.fill_(.3)
        model = nn.Module()
        model.enhancement = enhancement
        negative = dict(self.evidence, words=self.evidence['words'].flip(0),
                        request_words=torch.empty(0, 32))
        before = enhance_pair(model, self.evidence, negative, 'real')
        anchor = enhance_pair(model, self.evidence, negative, 'anchor_only')
        routing = enhance_pair(model, self.evidence, negative, 'core_only')
        after = enhance_pair(model, self.evidence, negative, 'real')
        for evidence, index in [(self.evidence, 0), (negative, 2)]:
            words = (evidence['words'] - enhancement.anchor_mean) / enhancement.anchor_scale
            values, _, _ = resample_words(words, evidence['word_groups'], 8)
            expected = F.linear(values, enhancement.anchor_weight, enhancement.anchor_bias)
            for name in ['routing','execution']:
                torch.testing.assert_close(anchor[index][name], expected, rtol=0, atol=0)
                torch.testing.assert_close(before[index][name],after[index][name],rtol=0,atol=0)
                self.assertTrue(torch.isfinite(after[index][name]).all())
            torch.testing.assert_close(routing[index]['execution'],before[index]['execution'],rtol=0,atol=0)
            torch.testing.assert_close(routing[index]['routing'],before[index]['execution'],rtol=0,atol=0)
            self.assertGreater(float((before[index]['routing']-routing[index]['routing']).detach().abs().max()),.001)
            self.assertGreater(float((before[index]['execution']-expected).detach().abs().max()),.001)
        # Mean controls store both arms explicitly; later negative calls cannot overwrite positive keys.
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'mean.pt'
            torch.save(dict(positive=map_content(before[0],lambda x:x.detach().cpu()),
                            negative=map_content(before[2],lambda x:x.detach().cpu())),path)
            mean=torch.load(path,weights_only=True)
            for arm,index in [('positive',0),('negative',2)]:
                for name,value in map_content(mean[arm],lambda x:x.to('cpu')).items():
                    torch.testing.assert_close(value,before[index][name],rtol=0,atol=0)

    def test_original_fusion_generation_gradients_reach_query_selection(self):
        model = Enhancement(self.cfg, self.interface)
        attention = Attention().requires_grad_(False)
        fusion = Fusion(attention)
        optimizer = torch.optim.SGD([*model.parameters(), *fusion.parameters()], lr=.2)
        x, context, target = torch.randn(1, 4, 16), torch.randn(1, 5, 16), torch.randn(1, 2, 16)
        norms = []
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            c, groups = model(self.evidence)
            # Same non-reentrant nested-content checkpoint transport as PhysicalModel.velocity.
            def run(hidden,condition):
                return fusion(hidden,context,condition,torch.tensor([[0.,1.]]),(2,1,2),attention,groups)
            y, _ = checkpoint(run,x,c,use_reentrant=False)
            F.mse_loss(y[:, 2:], target).backward()
            parameters = dict(output=fusion.output.weight, project=model.project.weight,
                              execution=model.execution_project.weight,
                              offset=model.shared_offset, seed=model.seed,
                              query=model.block.cross.q.weight, value=model.block.cross.v.weight)
            norms.append({name: float(p.grad.norm()) for name, p in parameters.items()})
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            optimizer.step()
        self.assertGreater(norms[0]['output'], 0.)
        for name in ['project', 'execution', 'offset']:
            self.assertGreater(norms[1][name], 0.)
        for name in ['seed', 'query', 'value']:
            self.assertGreater(norms[2][name], 0.)

    def test_actual_parameter_ownership_optimizer_and_own_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            torch.save(self.interface, folder / 'interface.pt')
            cfg = copy.deepcopy(self.cfg)
            cfg.update(variant='physical_content', revision='physical_content_v7_grounded_execution_r10b',
                       paths=dict(interface=str(folder / 'interface.pt')),
                       stages=dict(interface=128, joint=272))
            cfg['model'].update(condition_every=1, lora_rank=2)
            cfg['train'].update(seed=42, keep_steps=[1], peak_lrs=dict(
                enhancement=.001, fusion_read=.001, fusion_write=.001, wan_lora=.001))
            model = PhysicalModel(cfg, device='cpu', wan=TinyWan())
            optimizer = torch.optim.AdamW(model.optim_groups(), foreach=False)
            self.assertIn('enhancement.execution_project.weight', model.learned_names)
            owner = next(group for group in optimizer.param_groups if group['name'] == 'enhancement')
            self.assertTrue(any(p is model.enhancement.execution_project.weight for p in owner['params']))
            model.set_phase(0)
            sum(value.square().mean() for value in model.enhancement(self.evidence)[0].values()).backward()
            optimizer.step()
            with patch('torch.cuda.get_rng_state_all', return_value=[]):
                save(model, optimizer, cfg, folder, 1, 4)
            restored = PhysicalModel(cfg, device='cpu', wan=TinyWan())
            resumed_optimizer = torch.optim.AdamW(restored.optim_groups(), foreach=False)
            with patch('torch.cuda.set_rng_state_all'):
                self.assertEqual(restore(restored, resumed_optimizer, folder / 'step0001.pt', cfg), (1, 4))
            left, right = dict(model.named_parameters()), dict(restored.named_parameters())
            for name in model.learned_names:
                torch.testing.assert_close(left[name], right[name], rtol=0, atol=0)
            for key in ['step', 'exp_avg', 'exp_avg_sq']:
                torch.testing.assert_close(optimizer.state[model.enhancement.execution_project.weight][key],
                    resumed_optimizer.state[restored.enhancement.execution_project.weight][key], rtol=0, atol=0)
            bad = copy.deepcopy(cfg)
            bad['revision'] = 'another_revision'
            with self.assertRaises(ValueError):
                restore(restored, resumed_optimizer, folder / 'step0001.pt', bad)

    def test_original_400_recipe_and_declared_evaluation(self):
        original = yaml.safe_load((PROJECT / 'configs/physical_content.yaml').read_text())
        current = load_config(HERE / 'configs/physical_content.yaml')
        self.assertEqual(current['stages'], dict(interface=128, joint=272))
        self.assertEqual(original['loss'],{k:v for k,v in current['loss'].items() if k!='semantic_target'})
        self.assertEqual(current['loss']['semantic_target'],'execution_core')
        self.assertEqual({k: v for k, v in original['train'].items() if k not in ['keep_steps', 'validation_steps']},
                         {k: v for k, v in current['train'].items() if k not in ['keep_steps', 'validation_steps']})
        for step in range(400):
            self.assertEqual(learning_rates(original, step), learning_rates(current, step))
            self.assertEqual(auxiliary_coefficient(original, step), auxiliary_coefficient(current, step))
        self.assertEqual(current['evaluation_budget']['control_modes'],
                         ['reasoning_swapped', 'empty_reasoning', 'global_train_mean', 'core_only', 'anchor_only', 'no_c'])
        self.assertEqual(current['evaluation_budget']['extended_evaluation']['cases'], 60)


if __name__ == '__main__':
    unittest.main()
