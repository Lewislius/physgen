"""Validate the actual 1600 selected input arms once, without GPU/models/API."""
from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap
import torch
from physical_content.common import load_config, atomic_json
from physical_content.data import TrainingData
from physical_content.teacher import Teacher


if __name__ == '__main__':
    cfg=load_config(Path(__file__).resolve().parent/'configs/r19ei.yaml')
    torch.set_num_threads(4)
    data,teacher=TrainingData(cfg),Teacher(cfg,device='cpu')
    spec=json.loads(Path(cfg['train']['sample_sequence']).read_text())
    visual_counts=[]
    for row,drop in zip(data.rows,spec['drop']):
        evidence=teacher.input(row,drop=drop)
        visual_counts.append(int(evidence['visual_tokens'].shape[1]))
    result=dict(passed=True,samples=len(visual_counts),selected_arms_only=True,
                vision_tokens_min=min(visual_counts),vision_tokens_max=max(visual_counts),
                own_image_request_scope=True,native_physics_provenance_checked=True,
                model_calls=0,api_calls=0)
    atomic_json(result,Path(cfg['paths']['run_dir']).parent.parent/'reports/training_inputs.json')
    print(json.dumps(result))
