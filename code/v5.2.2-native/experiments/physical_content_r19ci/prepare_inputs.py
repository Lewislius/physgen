"""Add native input-physics words to matched I2V evidence and controls."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap  # noqa: F401
import torch

from benchmark_inputs import case_input, reference
from condition_controls import validate_control_input
from physical_content.common import atomic_json, atomic_torch, load_config
from physical_content.native_inputs import NativeEncoder, check_native
from run_candidate import PAIRED


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--inspect-only', action='store_true')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    destination = Path(cfg['paths']['inference_evidence'])
    reference_root = Path(cfg['paths']['prepared_evidence_reference'])
    native_root = Path(cfg['paths']['native_words_cache'])
    new_native = Path(cfg['paths']['new_native_evidence'])
    torch.set_num_threads(4)
    encoder, records, pending, checked = None, [], [], 0

    def prepare(value, name, *, donor=None, empty=False):
        nonlocal encoder, checked
        if value.get('scope') != 'first_image_and_request_only':
            raise ValueError('I2V input evidence scope differs')
        if empty:
            if any(r.strip() for r in value['relations']) or any(value['word_strings']):
                raise ValueError('Empty analysis has nonempty physical words')
            value['native_condition'] = torch.zeros(len(value['word_strings']), 3072,
                                                     dtype=torch.bfloat16)
            value['native_condition_source'] = 'empty_input_analysis_zero_words'
            checked += 1
            return value
        source = next((p for p in (native_root/f'{name}.pt', new_native/f'{name}.pt')
                       if p.is_file()), None)
        if source is None:
            if args.inspect_only:
                pending.append(name)
                return None
            if encoder is None:
                encoder = NativeEncoder(cfg, device=args.device)
            source = new_native/f'{name}.pt'
            native = encoder.encode(value)
            atomic_torch(native, source)
        else:
            native = torch.load(source, map_location='cpu', weights_only=True)
        value['native_condition'] = check_native(native, value, sample=donor)
        value['native_condition_source'] = str(source.resolve())
        checked += 1
        return value

    for case in reference()['cases']:
        _, item = case_input(case, 'i2v')
        folder = destination/item['id']
        ready = json.loads((reference_root/item['id']/'ready.json').read_text())
        if (ready['caption'] != item['caption'] or ready['image'] != item['image']
                or ready['scope'] != 'first_image_and_request_only'):
            raise ValueError('The matched I2V input changed')
        complete = True
        for drop in (False, True):
            arm = 'negative' if drop else 'positive'
            source = reference_root/item['id']/f'{arm}.pt'
            value = torch.load(source, map_location='cpu', weights_only=True)
            if (value['sample'] != item['id']
                    or value['caption'] != ('' if drop else item['caption'])
                    or value['first_image'] != item['image']):
                raise ValueError('Wrong owner of an I2V physical input')
            value = prepare(value, item['id'] + ('_null' if drop else '_text'))
            complete = complete and value is not None
            if value is not None and not args.inspect_only:
                atomic_torch(value, folder/f'{arm}.pt')
        if case['id'] in PAIRED:
            for mode in ('reasoning_swapped', 'empty_reasoning'):
                source = (Path(cfg['paths']['original_control_cache'])/item['id']
                          /'controls'/f'{mode}.pt')
                value = torch.load(source, map_location='cpu', weights_only=True)
                validate_control_input(value, item)
                if value.get('sample') != item['id'] or value.get('intervention') != mode:
                    raise ValueError('Prepared control identity differs')
                donor = f'videophy40_{value["donor"]}' if mode == 'reasoning_swapped' else None
                value = prepare(value, f'{donor}_text', donor=donor,
                                empty=mode == 'empty_reasoning')
                complete = complete and value is not None
                if value is not None and not args.inspect_only:
                    atomic_torch(value, folder/'controls'/f'{mode}.pt')
        if complete:
            record = dict(ready, physics_native_encoding='frozen_T5_Wan_input_relations_v1',
                          native_input_from_future_video=False)
            records.append(record)
            if not args.inspect_only:
                atomic_json(record, folder/'ready.json')
    report = dict(complete=not pending, cases=len(records), checked_arms=checked,
                  pending_native_arms=pending, original_input_cache=str(reference_root),
                  api_calls=0, downloads=0, encoder_is_frozen=True,
                  scope='same own image/request; only input reasoning encoded',
                  inspection_only=args.inspect_only)
    path = (Path(cfg['paths']['run_dir']).parent.parent/'reports/input_preflight.json'
            if args.inspect_only else destination/'ready.json')
    atomic_json(report, path)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
