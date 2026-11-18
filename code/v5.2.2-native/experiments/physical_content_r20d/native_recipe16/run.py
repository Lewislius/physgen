"""One fixed16 official-Wan inference-recipe comparison; original weights."""
from pathlib import Path
import argparse
import gc
import json
import sys

BASE=Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native')
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'source'
CASES=['VP002','VP004','VP005','VP009','VP010','VP013','VP015','VP016',
       'VP017','VP018','VP024','VP025','VP026','VP027','VP034','VP038']


def main():
    sys.path.insert(0,str(BASE/'tools'))
    from gpu_policy import require_rtx_pro_6000
    require_rtx_pro_6000()
    sys.path.insert(0,str(SOURCE))
    import bootstrap
    import torch
    from pipeline import evaluator_environment, run_stage
    from physical_content.wan_loader import ensure_wan_namespace
    ensure_wan_namespace({'paths':{'wan_code':'/home/liuzhirui/model/Wan2.2'}})
    torch.set_num_threads(4)
    output=ROOT/'outputs';output.mkdir(exist_ok=True)
    negative=ROOT/'native_quality_text.pt'
    if not negative.exists():
        from wan.modules.t5 import T5EncoderModel
        from wan.configs.shared_config import wan_shared_cfg
        wan=Path('/home/liuzhirui/model/Wan2.2/Wan2.2-TI2V-5B')
        encoder=T5EncoderModel(text_len=512,dtype=torch.bfloat16,device='cuda',
            checkpoint_path=str(wan/'models_t5_umt5-xxl-enc-bf16.pth'),
            tokenizer_path=str(wan/'google/umt5-xxl'))
        with torch.inference_mode():
            value=encoder([wan_shared_cfg.sample_neg_prompt],torch.device('cuda'))[0]
        torch.save(value.cpu().unsqueeze(0),negative)
        (ROOT/'negative_prompt.txt').write_text(wan_shared_cfg.sample_neg_prompt+'\n')
        del encoder,value
        gc.collect();torch.cuda.empty_cache()
    checkpoint=ROOT.parent/'runs/r20dt_native_physics_attention400/step0400.pt'
    def stage(name,command,environment=None):
        (ROOT/'status.json').write_text(json.dumps(dict(stage=name),indent=2)+'\n')
        run_stage(command,ROOT/(name+'.log'),environment)
    stage('generation16',[sys.executable,'-I','-B','-u',str(SOURCE/'infer.py'),
        '--checkpoint',str(checkpoint),'--output',str(output),'--suite','videophy40',
        '--native-quality-text',str(negative),'--cases',*CASES])
    stage('score16',['/home/liuzhirui/miniconda3/envs/physvid/bin/python','-I','-B','-u',
        str(BASE/'tools/evaluate_videophy2.py'),str(output/'videophy40')],evaluator_environment())
    (ROOT/'status.json').write_text(json.dumps(dict(stage='complete',training=False,
        comparison='official UniPC plus default quality negative versus original Euler plus empty negative',
        fixed='same DT400 weights/positive request/seed42 persisted noise/50 steps/shift5/gamma5',
        native_physical_negative='original null Qwen C retained',cases=CASES),indent=2)+'\n')


if __name__=='__main__':main()
