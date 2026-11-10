"""Version the held-out diagnostic's completion budget without truncating requests."""
from pathlib import Path
import argparse
import json
import os


def repair(path):
    original = path.read_text()
    plan = json.loads(original)
    if plan.get('suite') != 'heldout8' or plan.get('source_split') != 'validation':
        raise ValueError('This repair is only for the separately reported real held-out diagnostic')
    generation = plan['evaluation']['generation']
    if generation.get('max_new_tokens') == 8 and 'max_length' not in generation:
        return
    if generation.get('max_length') != 256 or 'max_new_tokens' in generation:
        raise ValueError('Unexpected prior evaluator budget')
    if list((path.parent/'cases').glob('*/planned.sa.json')) or (path.parent/'scores.json').exists():
        raise ValueError('Do not silently reuse existing scores under a changed protocol')
    backup = path.with_name(path.stem+'.evaluation_budget_v1.json')
    if backup.exists() and backup.read_text() != original:
        raise ValueError('An original manifest is already preserved with different contents')
    if not backup.exists():
        with backup.open('x') as stream:
            stream.write(original)
    del generation['max_length']
    generation['max_new_tokens'] = 8
    plan['evaluation_budget_revision'] = {
        'revision': 'heldout8_long_caption_completion_budget_v2',
        'original_manifest': str(backup.resolve()),
        'reason': 'A complete WISA caption exceeds the legacy total max_length=256; use an explicit eight-token score completion budget.',
        'request_truncated': False,
        'fixed_videophy40_protocol_changed': False,
    }
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(plan, ensure_ascii=False, indent=2)+'\n')
    os.replace(temporary, path)
    print(json.dumps({'manifest': str(path), 'revision': plan['evaluation_budget_revision']['revision']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path, nargs='+')
    for path in parser.parse_args().manifest:
        repair(path)
