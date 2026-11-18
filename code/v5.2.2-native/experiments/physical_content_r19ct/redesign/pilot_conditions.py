"""Small request-only condition pilot using the already configured Qwen API.

Never supplies images, real/generated future frames, benchmark labels or scores.
Original training captions and evaluation prompts are not rewritten.
"""
from pathlib import Path
import concurrent.futures
import importlib.util
import json

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parents[2]
SPEC = importlib.util.spec_from_file_location('existing_qwen_client', BASE /
    'experiments/physical_content_v7/data_revision_qwen38/api.py')
CLIENT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLIENT)

INSTRUCTION = '''Prepare conditioning for a video generator from the request alone.
The request may describe a cropped temporal segment: its cause may have begun
before the clip and its consequences may continue afterwards. Do not force a
complete start/contact/end cycle or invent missing actions.
Return JSON with these two fields:
"semantic_focus": one sentence, at most 35 words, preserving the main specified
participants, their roles, and the requested visible phenomenon or process.
"relations": one to four sentences, each at most 24 words, specifying physical
dependencies that are directly relevant to producing that requested process.
Use concrete qualitative cause/response or persistence relationships. Preserve
the request's actual direction, sequence and material state. Do not add numeric
thresholds, hidden material properties, failure or rupture, rest/motion phases,
or strong necessity claims that are not entailed by the request and basic physics.
Do not equate motion with breakage or a continuous visible process with stasis.
No physics category labels, camera aesthetics, instructions to the scorer, or
predicted quality. A short set of useful dependencies is better than four padded
statements. The original request remains the generator's semantic input.
Request:
'''


def one(row):
    destination = ROOT / 'condition_pilot' / (row['id'] + '.json')
    if destination.is_file():
        return json.loads(destination.read_text())
    answer, trace = CLIENT.Client().call(INSTRUCTION + row['caption'], [], max_tokens=520)
    if not isinstance(answer.get('semantic_focus'), str):
        raise ValueError('Missing semantic focus')
    relations = answer.get('relations')
    if not isinstance(relations, list) or not 1 <= len(relations) <= 4:
        raise ValueError('Expected one to four physical dependencies')
    if any(not isinstance(value, str) or not value.strip() for value in relations):
        raise ValueError('Empty physical dependency')
    result = dict(id=row['id'], caption=row['caption'], old_relations=row['relations'],
                  proposed=answer, trace=trace, source='request_only',
                  used_for_training=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


if __name__ == '__main__':
    rows = json.loads((ROOT / 't2v_condition_audit.json').read_text())['records']
    suffixes = ('VP005', 'VP015', 'VP025', 'VP026', 'VP056',
                'source0002263', 'source0000035', 'source0001714')
    selected = [row for row in rows if any(row['id'].endswith(s) for s in suffixes)]
    assert len(selected) == 8
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(one, selected))
    usage = {'prompt_tokens': 0, 'completion_tokens': 0}
    for row in results:
        for key in usage:
            usage[key] += row['trace']['usage'].get(key, 0)
        print(json.dumps({key: row[key] for key in ('id', 'proposed')}, ensure_ascii=False))
    (ROOT / 'condition_pilot/summary.json').write_text(json.dumps(dict(
        records=len(results), usage=usage, evaluation_scores_used=False,
        training_or_evaluation_cache_changed=False), indent=2) + '\n')
