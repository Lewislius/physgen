"""CPU audit of the launcher correction and the real greedy generation call."""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main():
    import torch
    from transformers import GenerationConfig, LlamaConfig, LlamaForCausalLM
    from physgen_native.config import read_config
    from physgen_native.data import audit
    from physgen_native.knowledge import QwenKnowledge

    torch.set_num_threads(4)
    report = dict(checked_at_utc=datetime.now(timezone.utc).isoformat())
    probe = '''import json, os, sys, torch, transformers
print(json.dumps(dict(executable=sys.executable, isolated=sys.flags.isolated,
    no_user_site=sys.flags.no_user_site, dont_write_bytecode=sys.dont_write_bytecode,
    torch=torch.__version__, transformers=transformers.__version__,
    environment={k:os.environ.get(k) for k in ('PATH','LD_LIBRARY_PATH','CONDA_PREFIX',
        'CONDA_DEFAULT_ENV','PYTHONPATH','PYTHONHOME','PYTHONUSERBASE',
        'PYTHONNOUSERSITE','TOKENIZERS_PARALLELISM','OMP_NUM_THREADS')})))'''
    commands = {
        'v52': ['bash', '-c', 'source /home/liuzhirui/Project/physGen/code/v5.2/tools/runtime_env.sh; v52_python "$@"', 'runtime-probe', '-c', probe],
        'native_before': ['bash', str(ROOT/'analysis/queue_20261025_blk_validation/code/tools/runtime_python.sh'), '-c', probe],
        'native_after': ['bash', str(ROOT/'tools/runtime_python.sh'), '-c', probe],
    }
    launchers = {}
    for name, command in commands.items():
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=30, check=True)
        launchers[name] = json.loads(result.stdout)
    assert launchers['v52'] == launchers['native_after'] == launchers['native_before']
    report['launcher'] = dict(runtime_contract_equal=True, before_already_equivalent=True,
                             after_reuses_original_function=True, actual=launchers['native_after'])

    cfg = read_config(ROOT/'configs/native_causal_response_wisa3857.yaml')
    generation_file = Path(cfg['paths']['qwen_checkpoint'])/'generation_config.json'
    raw = json.loads(generation_file.read_text())
    greedy = GenerationConfig.from_dict(raw)
    greedy.do_sample = False
    greedy.temperature = greedy.top_p = greedy.top_k = None
    greedy.validate(strict=True)

    # Use an actual Transformers generation implementation on a small CPU
    # model: check token equality through QwenKnowledge._reason, not a mock.
    torch.manual_seed(20261026)
    tiny = LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=32,
        intermediate_size=64, num_hidden_layers=1, num_attention_heads=4,
        num_key_value_heads=2, bos_token_id=1, eos_token_id=2, pad_token_id=2)).eval()
    for name in ('do_sample', 'temperature', 'top_p', 'top_k', 'repetition_penalty'):
        setattr(tiny.generation_config, name, raw[name])
    original = copy.deepcopy(tiny.generation_config.to_dict())
    inputs = dict(input_ids=torch.tensor([[1, 8, 17, 23]]), attention_mask=torch.ones(1, 4, dtype=torch.long))
    with torch.no_grad():
        previous = tiny.generate(**inputs, max_new_tokens=6, do_sample=False,
            use_cache=True, return_dict_in_generate=False, pad_token_id=2)
    knowledge = QwenKnowledge.__new__(QwenKnowledge)
    torch.nn.Module.__init__(knowledge)
    knowledge.backbone = tiny
    knowledge.tokenizer = SimpleNamespace(eos_token_id=2,
        decode=lambda ids, **kwargs: json.dumps(ids.tolist()))
    tiny.train()
    actual = knowledge._reason(inputs, 6)
    expected = json.dumps(previous[0, inputs['input_ids'].shape[1]:].tolist())
    assert actual == expected
    assert tiny.training and tiny.generation_config.to_dict() == original
    report['greedy_generation'] = dict(strict_config_valid=True, token_sequence_equal=True,
        shared_generation_config_unchanged=True, training_mode_restored=True,
        cpu_control_model='one-layer Llama using installed Transformers generate()',
        full_qwen_gpu_generation_tested=False)

    report['cache'] = audit(cfg)
    assert report['cache']['ready']
    report['configs'] = {}
    for path in [ROOT/'configs/native_causal_response_wisa3857.yaml',
                 ROOT/'configs/ablation_native_text_causal_response.yaml',
                 ROOT/'analysis/queue_20261025_blk_validation/qwen_no_latent_reasoning/config.yaml']:
        config = read_config(path)
        report['configs'][str(path.relative_to(ROOT))] = dict(revision=config['revision'],
            stages=config['stages'], loss_counts={k:len(v) for k,v in config['loss'].items()})

    report['checkpoints'] = {}
    for path in [ROOT/'analysis/queue_20261025_causal_response/first_checkpoint_verification/checkpoint.pt',
                 ROOT/'analysis/queue_20261025_blk_validation/code/checkpoints/v522-native-causal-response-naive-text-wisa3857_66201/latest.pt']:
        state = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
        assert state['step'] == 16 and state['revision'] == cfg['revision']
        assert state['revision'] == state['config']['revision'] and state['cursor'] == 128
        report['checkpoints'][str(path.relative_to(ROOT))] = dict(step=state['step'], cursor=state['cursor'],
            revision=state['revision'], learned_tensors=len(state['model']),
            optimizer_groups=len(state['optimizer']['param_groups']), rng_saved='rng' in state)
        del state
    files = [f for folder in ('physgen_native','train','inference','tools','tests') for f in (ROOT/folder).rglob('*.py')]
    for path in files:
        compile(path.read_bytes(), str(path), 'exec')
    report['python_sources_compiled'] = len(files)
    report['status'] = 'passed'
    (OUT/'execution_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('launcher','configs','checkpoints')}))


if __name__ == '__main__':
    main()
