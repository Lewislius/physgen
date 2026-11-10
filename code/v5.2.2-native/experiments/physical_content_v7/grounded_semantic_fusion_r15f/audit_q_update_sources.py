"""Locate the common Q correction inside the frozen enhancement, on CPU."""
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from audit_condition_representation import cosine_rows, rms


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    checkpoint=HERE/'runs/physical_content_68038/step0400.pt'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    cfg=saved['config'];sys.path.insert(0,cfg['source_snapshot'])
    import bootstrap
    from physical_content.model import Enhancement
    from condition_controls import InferenceCache
    model=Enhancement(cfg).eval().requires_grad_(False)
    model.load_state_dict({k.removeprefix('enhancement.'):v for k,v in
        {**saved['model'],**saved['constants']}.items() if k.startswith('enhancement.')},strict=True)
    del saved
    reader=InferenceCache(cfg['paths']['inference_evidence'],device='cpu')
    ids=[v['id'] for v in json.loads((HERE/'subset16.json').read_text())['cases']]
    state={};rows=[]

    def stats(x):
        return dict(token_cosine=cosine_rows(x),rms=rms(x))

    def block_before(module,inputs):
        q,anchor,*_=inputs
        state['row']=dict(case=state['case'],round=state['round'],
                          state_before=stats(q),update_before=stats(q-anchor))
        state['anchor']=anchor
        state['norm_calls']=0

    def memory_after(module,inputs,output):
        state['norm_calls']+=1
        name='full_memory_after_layernorm' if state['norm_calls']==1 else 'physical_words_after_layernorm'
        value=output[:,state['memory_mask'][0]] if state['norm_calls']==1 else output
        state['row'][name]=stats(value)

    def read_before(module,inputs):
        read=inputs[0]
        width=model.block.grounded_heads*model.block.cross.head_dim
        state['row']['grounded_head_read']=stats(read[...,:width])
        state['row']['global_head_read']=stats(read[...,width:])

    def capture(name):
        def hook(module,inputs,output):
            state['row'][name]=stats(output)
        return hook

    def norm_before(module,inputs):
        state['row']['update_before_RMSNorm']=stats(inputs[0])

    def block_after(module,inputs,output):
        state['row']['update_after_RMSNorm']=stats(output-state['anchor'])
        rows.append(state['row']);state['round']+=1

    handles=[model.block.register_forward_pre_hook(block_before),
             model.block.memory_norm.register_forward_hook(memory_after),
             model.block.cross.o.register_forward_pre_hook(read_before),
             model.block.cross.o.register_forward_hook(capture('cross_attention_correction')),
             model.block.exchange.o.register_forward_hook(capture('exchange_correction')),
             model.block.ffn.register_forward_hook(capture('ffn_correction')),
             model.block.update_norm.register_forward_pre_hook(norm_before),
             model.block.register_forward_hook(block_after)]
    inputs=[]
    for case in ids:
        ready=json.loads((reader.root/f'videophy40_{case}'/'ready.json').read_text())
        evidence=reader.input(dict(id=ready['id'],caption=ready['caption']),image=ready['image'],duration=ready['duration'])
        state.update(case=case,round=1,memory_mask=evidence['memory_mask'])
        words=evidence['words'].float()
        inputs.append(dict(case=case,raw_physical_words=stats(words),
                           whitened_physical_words=stats((words-model.anchor_mean)/model.anchor_scale),
                           raw_request_words=stats(evidence['request_words']),
                           raw_full_memory=stats(evidence['memory'][:,evidence['memory_mask'][0]])))
        model(evidence)
    for handle in handles:handle.remove()
    summary={}
    for round_id in range(1,model.rounds+1):
        selected=[x for x in rows if x['round']==round_id]
        summary[str(round_id)]={key:{metric:statistics.mean(x[key][metric] for x in selected)
                                    for metric in ['token_cosine','rms']}
                                for key in selected[0] if key not in ['case','round']}
    input_summary={key:{metric:statistics.mean(x[key][metric] for x in inputs)
                        for metric in ['token_cosine','rms']} for key in inputs[0] if key!='case'}
    report=dict(utc=datetime.now(timezone.utc).isoformat(),checkpoint=str(checkpoint),training_source=cfg['source_snapshot'],
                input_summary=input_summary,round_summary=summary,inputs=inputs,rows=rows,
                learned_seed=stats(model.seed),
                scope='Fixed16 normal inputs, exact frozen enhancement with read-only hooks on CPU. '
                      'Component amplitudes and directions locate common corrections; none is a generation-quality metric.',
                quality_proven=False)
    output=HERE/'reports/q_update_source_audit16.json';output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(report=str(output),input_summary=input_summary,round_summary=summary,learned_seed=report['learned_seed']),ensure_ascii=False))


if __name__=='__main__':
    main()
