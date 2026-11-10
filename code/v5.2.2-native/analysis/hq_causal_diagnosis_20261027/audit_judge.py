"""Inspect the actual frozen training judge on existing outputs and declared controls."""
import argparse
import json
import sys
import time
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import torch
import yaml
from physgen_native.knowledge import QwenKnowledge
from physgen_native.criterion_verifier import CriterionVerifier


def read_video(path):
    cap = cv2.VideoCapture(str(path)); count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ids = list(range(0, count, 3))
    if ids[-1] != count - 1: ids.append(count - 1)
    assert len(ids) == 28 and count == 81, (path, count)
    frames = []
    for index in ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index); ok, frame = cap.read()
        assert ok, (path, index)
        assert frame.shape[:2] == (480, 832)
        frames.append(torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).permute(2,0,1))
    cap.release()
    return torch.stack(frames).cuda().float() / 127.5 - 1, ids


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--plan', required=True); args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    cfg = yaml.safe_load((ROOT / 'checkpoints/v522-h-content-message-verifier-frozen-simple_66356/config.yaml').read_text())
    torch.cuda.set_device(0)
    model = QwenKnowledge(cfg, 'cuda').eval().requires_grad_(False)
    critic = CriterionVerifier(cfg['verifier'])
    path = HERE / 'judge_scores.jsonl'
    done = {json.loads(l)['key'] for l in path.read_text().splitlines()} if path.exists() else set()
    with path.open('a', buffering=1) as output, torch.no_grad():
        for case in plan:
            if case['key'] in done: continue
            started = time.monotonic()
            frames, ids = read_video(case['video'])
            if case['control'] == 'static_first': frames = frames[:1].expand_as(frames).contiguous()
            elif case['control'] == 'reversed': frames = frames.flip(0)
            times = torch.tensor(ids, device='cuda').float() / case['fps']
            with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
                logits = model.criterion_logits(frames, times, case['caption'], case['clauses'], adapt=False)
                scores = {k: float(v) for k,v in critic.diagnostics(logits).items()}
                probabilities = critic.probabilities(logits).mean(-2).cpu().tolist()
            result = dict(case, scores=scores, probabilities=probabilities, seconds=time.monotonic()-started)
            output.write(json.dumps(result, ensure_ascii=False) + '\n')
            print(json.dumps(dict(key=case['key'], **scores, seconds=result['seconds'])), flush=True)
            del frames, logits
    (HERE / 'judge_complete.json').write_text(json.dumps(dict(status='complete', cases=len(plan))))


if __name__ == '__main__': main()
