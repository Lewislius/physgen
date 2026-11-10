"""Official T5/Wan semantic targets for legal input-side physical words."""
from pathlib import Path
import sys,json,os,re,time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
from bootstrap import ROOT
import torch
from torch import nn
from safetensors import safe_open
from physical_scaling.common import atomic_json,atomic_torch,gpu_guard

WAN=Path('/home/liuzhirui/model/Wan2.2/Wan2.2-TI2V-5B')

def native_embedding(device):
    index=json.loads((WAN/'diffusion_pytorch_model.safetensors.index.json').read_text())['weight_map']
    module=nn.Sequential(nn.Linear(4096,3072),nn.GELU(approximate='tanh'),nn.Linear(3072,3072))
    weights={}
    for key in module.state_dict():
        full='text_embedding.'+key
        with safe_open(WAN/index[full],framework='pt',device='cpu') as f:weights[key]=f.get_tensor(full)
    module.load_state_dict(weights)
    return module.to(device=device,dtype=torch.bfloat16).eval().requires_grad_(False)

@torch.no_grad()
def main():
    run=ROOT/'preparation'/os.getenv('DET_EXPERIMENT_ID','local')
    gpu_guard(run);torch.set_num_threads(4)
    sys.path.insert(0,'/home/liuzhirui/model/Wan2.2')
    from wan.modules.t5 import T5EncoderModel
    encoder=T5EncoderModel(text_len=128,dtype=torch.bfloat16,device='cuda',
        checkpoint_path=str(WAN/'models_t5_umt5-xxl-enc-bf16.pth'),tokenizer_path=str(WAN/'google/umt5-xxl'))
    native=native_embedding('cuda')
    inventory=json.loads((ROOT/'cache/inventory.json').read_text())
    out=ROOT/'cache/native_words';out.mkdir(parents=True,exist_ok=True)
    pending=[item for item in inventory if not (out/(item['name']+'.pt')).exists()]
    begin=time.monotonic()
    for start in range(0,len(pending),8):
        batch=pending[start:start+8];texts=[];offsets=[];records=[]
        for item in batch:
            words=torch.load(ROOT/'cache/words'/f'{item["name"]}.pt',map_location='cpu',weights_only=True)
            if words['scope']!='first_image_and_request_only':raise ValueError('Future evidence is forbidden')
            records.append(words);offsets.append(len(texts))
            texts.extend(words['relations'])
        clean=[encoder.tokenizer._clean(t) for t in texts]
        encoded=encoder.tokenizer.tokenizer(clean,return_tensors='pt',return_offsets_mapping=True,
                    add_special_tokens=True,padding=True,truncation=True,max_length=128)
        token_offsets=encoded.pop('offset_mapping')
        ids=encoded['input_ids'].cuda();mask=encoded['attention_mask'].cuda()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            hidden=encoder.model(ids,mask)
            target=native(hidden).float()
        for item,record,offset in zip(batch,records,offsets):
            features=[];tokens=[]
            for group in range(len(record['relations'])):
                i=offset+group;spans=token_offsets[i]
                for match in re.finditer(r'\S+',clean[i]):
                    selected=((spans[:,1]>match.start())&(spans[:,0]<match.end())).nonzero().flatten()
                    if not len(selected):raise ValueError('Native physical word has no token')
                    features.append(target[i,selected.to(target.device)].mean(0));tokens.append(match.group())
            if len(tokens)!=len(record['word_strings']):
                raise ValueError(f'Lexical normalization changed word count: {item["name"]}')
            atomic_torch(dict(native_words=torch.stack(features).cpu().to(torch.bfloat16),
                word_strings=tokens,scope='first_image_request_reasoning_only',sample=item['sample'],
                split=item['split'],teacher='official frozen T5 then official Wan text_embedding'),out/f'{item["name"]}.pt')
        atomic_json(dict(status='preparing',completed=start+len(batch),total=len(pending),seconds=time.monotonic()-begin),run/'status.json')
        print(json.dumps(dict(stage='native_semantic_targets',completed=start+len(batch),total=len(pending),seconds=time.monotonic()-begin)),flush=True)
    atomic_json(dict(status='complete',count=len(inventory),seconds=time.monotonic()-begin),run/'status.json')

if __name__=='__main__':main()
