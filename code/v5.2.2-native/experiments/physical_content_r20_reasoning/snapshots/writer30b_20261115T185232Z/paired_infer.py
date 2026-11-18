"""Reuse the frozen BT sampler for revised-text or official-native comparisons."""
import argparse
import copy
import json
from pathlib import Path
import sys

BASE = Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native')
BT = BASE/'experiments/physical_content_r20b/snapshots/20261115T052651Z'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-checkpoint', type=Path, required=True)
    parser.add_argument('--evidence', type=Path)
    parser.add_argument('--official-native', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(BASE/'tools'))
    from gpu_policy import require_rtx_pro_6000
    require_rtx_pro_6000()
    sys.path.insert(0, str(BT))
    import infer
    import torch
    from physgen_native.decoder import VideoDecoder
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    saved = torch.load(args.reference_checkpoint, map_location='cpu', mmap=True,
                       weights_only=False)
    cfg = copy.deepcopy(saved['config'])
    if cfg['task'] != 't2v' or saved['update'] != 400:
        raise ValueError('This comparison uses the completed BT T2V400 reference')
    if args.evidence is not None:
        cfg['paths']['inference_evidence'] = str(args.evidence)
    model = infer.PhysicalModel(cfg)
    if args.official_native:
        # No project weight is loaded. All newly installed native LoRA B maps
        # are zero, and the entire physical branch is skipped below.
        for name, value in model.wan.named_parameters():
            if name.endswith('.lora_b') and torch.count_nonzero(value):
                raise ValueError('Official-native reference has an active LoRA')
        step, revision = 0, 'official-native-matched-project-sampler'
        cfg['variant_id'] = 'Wan2.2-TI2V-5B-official-T2V'
        cfg['plan'] = cfg['design_plan'] = 'official_native'
        cfg['mechanism'] = 0
        cfg['train']['sample_sequence'] = None
        cfg['train']['data_revision'] = 'official_pretrained_no_project_training'
        args.checkpoint = Path(cfg['paths']['wan_checkpoint'])
        mode = 'no_c'
    else:
        parameters, buffers = dict(model.named_parameters()), dict(model.named_buffers())
        if set(saved['model']) != set(model.learned_names):
            raise ValueError('BT checkpoint ownership differs')
        with torch.no_grad():
            for name, value in saved['model'].items():
                parameters[name].copy_(value)
            for name, value in saved['constants'].items():
                buffers[name].copy_(value)
        step, revision = saved['update'], saved['revision']
        cfg['variant_id'] = 'R20BT400-with-revised-local-reasoning'
        args.checkpoint = args.reference_checkpoint
        mode = 'normal'
    del saved
    model.eval().requires_grad_(False)
    decoder = VideoDecoder(cfg).eval()
    cache = infer.InferenceCache(cfg['paths']['inference_evidence'], 't2v')
    protocol = infer.reference()
    selected = [case for case in protocol['cases'] if case['id'] in args.cases]
    if len(selected) != len(set(args.cases)):
        raise ValueError('Unknown or duplicated comparison cases')
    args.suite, args.control_cache = 'videophy40', None
    infer.generate_selected(model, decoder, cache, cfg, protocol, selected, args,
                            step, revision, mode, args.output, None)
    extra = dict(comparison=('official_native' if args.official_native else 'BT400_revised_reasoning'),
                 reference_checkpoint_config=str(args.reference_checkpoint),
                 project_weights_loaded=not args.official_native,
                 inference_evidence=cfg['paths']['inference_evidence'],
                 sampler='unchanged frozen BT sampler; same native prompts/noise/gamma5',
                 new_training=False, paid_api_calls=0)
    folder = args.output/'videophy40'
    for path in [folder/'manifest.json', *folder.glob('cases/*/planned.json')]:
        value = json.loads(path.read_text()); value.update(extra)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
