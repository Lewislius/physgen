"""CPU-only lexical metadata/feature extraction; no foundation model loading."""
from pathlib import Path
import sys,json,torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
from bootstrap import ROOT
from transformers import AutoTokenizer
from physical_scaling.common import atomic_json,atomic_torch
from physical_content.evidence import input_inventory,recover_words

def main():
    torch.set_num_threads(4)
    tokenizer=AutoTokenizer.from_pretrained('/home/liuzhirui/model/Qwen3-VL/Qwen3-VL-8B-Instruct',local_files_only=True)
    out=ROOT/'cache/words';out.mkdir(parents=True,exist_ok=True)
    inventory=input_inventory();completed=[]
    for i,item in enumerate(inventory):
        dest=out/(item['name']+'.pt')
        if not dest.is_file():
            evidence=torch.load(item['path'],map_location='cpu',weights_only=True)
            words=recover_words(evidence,item['duration'],tokenizer)
            atomic_torch({**words,'source':item},dest)
        completed.append(item)
        if i%100==0:print(json.dumps(dict(stage='physical_words',completed=i+1,total=len(inventory))),flush=True)
    atomic_json(completed,ROOT/'cache/inventory.json')
    print(json.dumps(dict(stage='physical_words',status='complete',count=len(completed))),flush=True)

if __name__=='__main__':main()
