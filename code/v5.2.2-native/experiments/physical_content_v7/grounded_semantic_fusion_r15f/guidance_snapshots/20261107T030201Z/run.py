"""Frozen R15F matched16 guidance intervention, no training."""
from pathlib import Path
import os,sys,json
sys.path.insert(0,'/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_v7/grounded_semantic_fusion_r15f/snapshots/20261106T202709Z')
import bootstrap
from pipeline import run_stage,evaluator_environment
from allocated_gpu import bind_allocated_gpu
from physical_content.common import gpu_guard,atomic_json
ROOT=Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_v7/grounded_semantic_fusion_r15f'); SOURCE=Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_v7/grounded_semantic_fusion_r15f/guidance_snapshots/20261107T030201Z')
output=Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_v7/grounded_semantic_fusion_r15f/inference_outputs/physical_content_68038/native_guidance16_seed42')
folder=ROOT/'reports/native_guidance16'/os.getenv('DET_EXPERIMENT_ID','local')
gpu_guard(folder);bind_allocated_gpu(folder,evaluator_environment)
atomic_json(dict(status='generating',training_updates=0,cases=16),folder/'status.json')
run_stage([sys.executable,'-I','-B','-u',SOURCE/'infer.py','--checkpoint','/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_v7/grounded_semantic_fusion_r15f/runs/physical_content_68038/step0400.pt','--output',output,'--suite','videophy40','--modes','native_guidance','--cases',*['VP002', 'VP004', 'VP005', 'VP009', 'VP010', 'VP013', 'VP015', 'VP016', 'VP017', 'VP018', 'VP024', 'VP025', 'VP026', 'VP027', 'VP034', 'VP038']],folder/'generation.log')
atomic_json(dict(status='scoring',training_updates=0,cases=16),folder/'status.json')
run_stage(['/home/liuzhirui/miniconda3/envs/physvid/bin/python','-I','-B','-u',bootstrap.BASE/'tools/evaluate_videophy2.py',output/'controls/native_guidance/videophy40'],folder/'scoring.log',evaluator_environment())
atomic_json(dict(status='completed',training_updates=0,cases=16,output=str(output)),folder/'status.json')
