"""Check causal premises against an observation made without the future caption."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from physgen_native.config import read_config
from physgen_native.data import VideoDataset, move
from physgen_native.decoder import VideoDecoder
from physgen_native.knowledge import QwenKnowledge, clause_context
from physgen_native.runtime import Logger, seed_all


@torch.no_grad()
def main():
    cfg = read_config(ROOT/'configs/native_physical_wisa3857.yaml')
    seed_all(42); torch.cuda.set_device(0)
    log = Logger(ROOT/'analysis/condition_grounding/metrics.jsonl')
    model = QwenKnowledge(cfg, 'cuda').cuda().eval()
    decoder = VideoDecoder(cfg, 'cuda').cuda().eval()
    data = VideoDataset(cfg, 'validation')
    for index in (0, 1, 3, 4):
        sample = move(data[index], 'cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
            first = decoder(sample['first'], [0])
            original = model.dependency_reasoning(first, sample['caption'], sample['clauses'])
            for size in ([288, 512], [480, 832]):
                model.cfg['image_size'] = size
                clips = [(first, first.new_zeros(1))]
                body = ('Describe only the initial image in at most 80 words. Identify visible material states, '
                        'contacts, sources and effects that are ALREADY present. Inspect small active sources '
                        'as well as the main objects. Distinguish what is visible from what a single image cannot '
                        'establish. Do not invent prior triggers, future events, motion direction, or camera behavior.')
                inputs, _ = model._inputs(clips, body)
                observation = model._reason(inputs, 112)
                body = clause_context(model._truncate(sample['caption']), sample['clauses'])+(
                    '\nInitial observation extracted without the future description:\n'+observation+
                    '\nConstruct ONE coherent physical dependency model of the requested process, grounded in '
                    'the initial image. An effect already present need not wait for a later described trigger. '
                    'Distinguish initiation from maintenance, and distinguish direct physical dependencies from '
                    'coincidental appearance. A condition being removed does not imply an instantaneous response; '
                    'account for persistence where applicable. Do not use camera behavior as a material mechanism. '
                    'Write four short lines: OBSERVED: starting material states, already active effects and unknowns. '
                    'INTERACTION: actual relevant condition or interaction -> affected process. '
                    'PERSISTENCE: what sustains and eventually changes or terminates it. '
                    'INTERVENTION: remove or change that same condition, state the consequence and preserved properties. '
                    'Do not substitute "if the process stops, it stops" for a causal dependency. '
                    'No invented numerical values or unique future trajectory.')
                inputs, _ = model._inputs(clips, body)
                revised = model._reason(inputs, 224)
                log.write('condition_grounding', index, sample=sample['id'], image_size=size,
                          original=original, observation=observation, revised=revised)
            model.cfg['image_size'] = cfg['knowledge']['image_size'] = [288, 512]
    log.write('complete', 0, peak_gib=torch.cuda.max_memory_allocated()/2**30)
    log.close()


if __name__ == '__main__': main()
